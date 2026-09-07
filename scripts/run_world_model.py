"""
CyberSentinel AI - Cyber World Model Experiment Runner (Phase 8).

Trains and evaluates the CyberWorldModel against baselines on the
same ScenarioBasedSplitter splits used in Phase 6/7.

Saves to experiments/<run_id>/world_model/ :
- world_model.pt          : trained checkpoint
- metrics_world_model.json: full evaluation metrics
- predictions_wm.csv      : per-sequence predictions
- confusion matrices (.png)
- evaluation_report_wm.md : comparison table vs baselines
- rollout_examples.json   : K-step rollout examples from test set
"""

import sys
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from network.flow.csv_loader import CSVFlowLoader
from ml.state.state_builder import NetworkStateBuilder, FEATURE_NAMES
from ml.preprocessing.validator import FeatureValidator
from ml.preprocessing.stage_labeler import StageLabeler, STAGE_TAXONOMY, STAGE_TO_ID
from ml.preprocessing.splitter import ScenarioBasedSplitter
from ml.preprocessing.scaler import FeatureScaler
from ml.preprocessing.sequence_builder import SequenceBuilder
from ml.world_model.cyber_world_model import CyberWorldModelTrainer, INPUT_DIM
from ml.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_forecasting_metrics,
    save_confusion_matrix_plot,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_world_model")


def extract_current_states(x_seq: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Extract the last valid state from each sequence -> (N, input_dim)."""
    lengths = mask.sum(dim=1).clamp(min=1) - 1
    idx = lengths.view(-1, 1, 1).expand(-1, 1, x_seq.shape[-1])
    return x_seq.gather(1, idx).squeeze(1).numpy()


def run_world_model_experiment(run_id: Optional[str] = None) -> Path:
    if run_id is None:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    exp_dir = Path("experiments") / run_id / "world_model"
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== CyberSentinel Phase 8 — Cyber World Model ===")
    logger.info("Output directory: %s", exp_dir)

    # ── Load config ──────────────────────────────────────────────────────────
    with open("configs/default_config.yaml") as f:
        cfg = yaml.safe_load(f)

    SEQ_LEN = cfg["pipeline"]["sequence_length"]      # 8
    K = cfg["pipeline"]["forecast_horizon_k"]         # 4
    HIDDEN = cfg["model"]["hidden_dim"]                # 128
    N_LAYERS = cfg["model"]["num_layers"]              # 2
    N_HEADS = cfg["model"]["num_heads"]                # 4
    DROPOUT = cfg["model"]["dropout"]                  # 0.1
    LR = cfg["model"]["learning_rate"]                 # 0.001
    LAM = cfg["model"]["loss_weights"]

    # ── Ingest data ──────────────────────────────────────────────────────────
    logger.info("Loading CSV traces...")
    loader = CSVFlowLoader()
    all_flows = []
    data_dir = Path("datasets/sample")
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No trace CSVs found in {data_dir}")
    for csv_path in csv_files:
        flows = loader.load_flows(csv_path)
        all_flows.extend(flows)
    logger.info("Loaded %d flows from %d traces", len(all_flows), len(csv_files))

    # ── Build network states ─────────────────────────────────────────────────
    builder = NetworkStateBuilder(window_size_seconds=30.0)
    states_df = builder.build_states(all_flows)
    logger.info("Built %d state windows", len(states_df))

    # ── Validate features ────────────────────────────────────────────────────
    validator = FeatureValidator()
    is_valid = validator.validate(states_df)
    logger.info("Feature validation passed: %s", is_valid)

    # ── Attach labels ────────────────────────────────────────────────────────
    labeler = StageLabeler(fallback_to_heuristics=True)
    states_df = labeler.attach_labels_to_dataframe(states_df)
    logger.info("Stage distribution: %s", states_df["stage_name"].value_counts().to_dict())

    # ── Split scenarios ──────────────────────────────────────────────────────
    splitter = ScenarioBasedSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    train_df, val_df, test_df = splitter.split_dataframe(states_df)

    train_sc = sorted(train_df["scenario_id"].unique().tolist())
    val_sc = sorted(val_df["scenario_id"].unique().tolist())
    test_sc = sorted(test_df["scenario_id"].unique().tolist())
    splitter.verify_no_overlap(train_sc, val_sc, test_sc)
    logger.info("Split: train=%d  val=%d  test=%d scenarios", len(train_sc), len(val_sc), len(test_sc))

    # ── Fit scaler on train only ─────────────────────────────────────────────
    scaler = FeatureScaler(scaler_type="robust")
    scaler.fit(train_df)
    scaler.save(exp_dir / "scaler.pkl")

    X_train_scaled = scaler.transform(train_df)
    X_val_scaled = scaler.transform(val_df)
    X_test_scaled = scaler.transform(test_df)

    # ── Build sequences ──────────────────────────────────────────────────────
    seq_builder = SequenceBuilder(sequence_length=SEQ_LEN, pad_short_sequences=True)
    train_seq = seq_builder.build_sequences(train_df, scaled_features=X_train_scaled)
    val_seq = seq_builder.build_sequences(val_df, scaled_features=X_val_scaled)
    test_seq = seq_builder.build_sequences(test_df, scaled_features=X_test_scaled)
    logger.info("Sequences — train: %d  val: %d  test: %d",
                len(train_seq.x_seq), len(val_seq.x_seq), len(test_seq.x_seq))

    # ── Compute class weights (inverse frequency on train) ───────────────────
    num_classes = len(STAGE_TAXONOMY)
    class_counts = np.bincount(train_seq.y_next_stage.numpy(), minlength=num_classes)
    total_samples = len(train_seq.y_next_stage)
    weights = [total_samples / (num_classes * max(c, 1)) for c in class_counts]
    stage_weights = torch.tensor(weights, dtype=torch.float32)

    # ── Build y_next_state tensors (S_{t+1} raw features for TransitionHead) ─
    y_next_state_train = train_seq.y_next_state if train_seq.y_next_state is not None else None
    y_next_state_val   = val_seq.y_next_state   if val_seq.y_next_state   is not None else None

    # ── Train World Model ────────────────────────────────────────────────────
    logger.info("Training CyberWorldModel...")
    trainer = CyberWorldModelTrainer(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN,
        num_heads=N_HEADS,
        num_layers=N_LAYERS,
        num_stages=len(STAGE_TAXONOMY),
        dropout=DROPOUT,
        learning_rate=LR,
        weight_decay=1e-4,
        batch_size=32,
        epochs=80,
        patience=12,
        lambda_stage=LAM["lambda_stage"],
        lambda_attack=LAM["lambda_attack"],
        lambda_transition=LAM["lambda_transition"],
        lambda_next_stage=LAM["lambda_next_stage"],
        label_smoothing=0.1,
        random_seed=42,
    )

    trainer.fit(
        x_train=train_seq.x_seq,
        mask_train=train_seq.mask,
        y_current_stage_train=train_seq.y_current_stage,
        y_next_stage_train=train_seq.y_next_stage,
        y_attack_train=train_seq.y_attack.float(),
        y_next_state_train=y_next_state_train,
        x_val=val_seq.x_seq,
        mask_val=val_seq.mask,
        y_current_stage_val=val_seq.y_current_stage,
        y_next_stage_val=val_seq.y_next_stage,
        y_attack_val=val_seq.y_attack.float(),
        y_next_state_val=y_next_state_val,
        stage_class_weights=stage_weights,
    )

    trainer.save(str(exp_dir / "world_model.pt"))
    logger.info("Model saved.")

    # ── Evaluate on test set ─────────────────────────────────────────────────
    logger.info("Evaluating on test set...")
    x_test = test_seq.x_seq
    mask_test = test_seq.mask

    # Attack detection
    atk_proba = trainer.predict_attack_proba(x_test, mask_test)
    atk_pred = (atk_proba >= 0.5).astype(int)
    y_atk_true = test_seq.y_attack.numpy()
    wm_atk_metrics = calculate_classification_metrics(y_atk_true, atk_pred, atk_proba, is_binary=True)

    # Next-stage forecasting
    ns_proba = trainer.predict_next_stage_proba(x_test, mask_test)
    ns_pred = np.argmax(ns_proba, axis=1)
    y_ns_true = test_seq.y_next_stage.numpy()
    wm_ns_metrics = calculate_forecasting_metrics(y_ns_true, ns_proba, k=3)
    wm_ns_clf = calculate_classification_metrics(
        y_ns_true, ns_pred, ns_proba, class_names=STAGE_TAXONOMY, is_binary=False
    )

    # Current stage
    cs_proba = trainer.predict_current_stage_proba(x_test, mask_test)
    cs_pred = np.argmax(cs_proba, axis=1)
    y_cs_true = test_seq.y_current_stage.numpy()
    wm_cs_metrics = calculate_classification_metrics(
        y_cs_true, cs_pred, cs_proba, class_names=STAGE_TAXONOMY, is_binary=False
    )

    logger.info("Attack detection — Acc=%.4f  FPR=%.4f  ROC-AUC=%s",
                wm_atk_metrics["accuracy"], wm_atk_metrics["fpr"],
                wm_atk_metrics["roc_auc"])
    logger.info("Next-stage forecast — Top-1=%.4f  Top-3=%.4f  Brier=%.4f",
                wm_ns_metrics["next_stage_accuracy"],
                wm_ns_metrics["top_3_accuracy"],
                wm_ns_metrics["brier_score"])

    # ── K-step rollout examples ──────────────────────────────────────────────
    n_rollout = min(5, len(x_test))
    rollout_results = trainer.rollout(x_test[:n_rollout], mask_test[:n_rollout], k_steps=K)
    rollout_export = []
    for r in rollout_results:
        rollout_export.append({
            "step": int(r["step"]),
            "stage_pred": r["stage_pred"].tolist(),
            "attack_prob": r["attack_prob"].tolist(),
            "top_stage": [STAGE_TAXONOMY[int(i)] for i in r["stage_pred"]],
        })

    with open(exp_dir / "rollout_examples.json", "w") as f:
        json.dump({"k_steps": K, "n_sequences": n_rollout, "rollout": rollout_export}, f, indent=2)
    logger.info("Rollout examples saved.")

    # ── Save predictions CSV ─────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "scenario_id": test_seq.scenario_ids,
        "y_true_stage": [STAGE_TAXONOMY[i] for i in y_cs_true],
        "y_pred_stage_wm": [STAGE_TAXONOMY[i] for i in cs_pred],
        "y_true_next_stage": [STAGE_TAXONOMY[i] for i in y_ns_true],
        "y_pred_next_stage_wm": [STAGE_TAXONOMY[i] for i in ns_pred],
        "y_true_attack": y_atk_true.astype(int),
        "y_pred_attack_wm": atk_pred,
        "attack_prob_wm": atk_proba,
    })
    pred_df.to_csv(exp_dir / "predictions_wm.csv", index=False)

    # ── Confusion matrices ───────────────────────────────────────────────────
    unique_stages = np.unique(y_ns_true)
    save_confusion_matrix_plot(
        wm_ns_clf["confusion_matrix"],
        [STAGE_TAXONOMY[i] for i in unique_stages],
        exp_dir / "confusion_matrix_wm_next_stage.png",
        title="Cyber World Model - Next Stage Forecast",
    )
    save_confusion_matrix_plot(
        wm_atk_metrics["confusion_matrix"],
        ["BENIGN", "ATTACK"],
        exp_dir / "confusion_matrix_wm_attack.png",
        title="Cyber World Model - Attack Detection",
    )

    # ── Save metrics ─────────────────────────────────────────────────────────
    full_metrics = {
        "run_id": run_id,
        "phase": "8_cyber_world_model",
        "dataset_summary": {
            "train_sequences": int(len(train_seq.x_seq)),
            "val_sequences": int(len(val_seq.x_seq)),
            "test_sequences": int(len(test_seq.x_seq)),
        },
        "world_model": {
            "attack_detection": {k: (float(v) if v is not None else None) for k, v in wm_atk_metrics.items() if k != "confusion_matrix"},
            "current_stage": {k: (float(v) if v is not None else None) for k, v in wm_cs_metrics.items() if k != "confusion_matrix"},
            "next_stage_forecasting": {
                **{k: (float(v) if v is not None else None) for k, v in wm_ns_metrics.items()},
                **{k: (float(v) if v is not None else None) for k, v in wm_ns_clf.items() if k not in ("confusion_matrix",)},
            },
        },
        "training_epochs": len(trainer.training_history),
        "final_train_loss": trainer.training_history[-1]["train"]["loss_total"] if trainer.training_history else None,
        "final_val_loss": trainer.training_history[-1]["val"]["loss_total"] if trainer.training_history else None,
    }

    with open(exp_dir / "metrics_world_model.json", "w") as f:
        json.dump(full_metrics, f, indent=2)

    # ── Evaluation report ────────────────────────────────────────────────────
    def fmt(v):
        if v is None: return "N/A"
        return f"{float(v):.4f}"

    report = f"""# CyberSentinel AI - Phase 8 World Model Evaluation Report ({run_id})

## Attack Detection (Binary: Attack vs. Benign)

| Model | Accuracy | Macro F1 | FPR | ROC-AUC | PR-AUC |
|-------|----------|----------|-----|---------|--------|
| Cyber World Model | {fmt(wm_atk_metrics['accuracy'])} | {fmt(wm_atk_metrics['f1_macro'])} | {fmt(wm_atk_metrics['fpr'])} | {fmt(wm_atk_metrics['roc_auc'])} | {fmt(wm_atk_metrics['pr_auc'])} |

## Next-Stage Forecasting

| Model | Top-1 Acc | Top-3 Acc | Macro F1 | Brier Score |
|-------|-----------|-----------|----------|-------------|
| Cyber World Model | {fmt(wm_ns_metrics['next_stage_accuracy'])} | {fmt(wm_ns_metrics['top_3_accuracy'])} | {fmt(wm_ns_clf['f1_macro'])} | {fmt(wm_ns_metrics['brier_score'])} |

## Training Summary

- Epochs trained: {len(trainer.training_history)}
- Final train loss: {fmt(full_metrics['final_train_loss'])}
- Final val loss: {fmt(full_metrics['final_val_loss'])}
- K-step rollout: {K} steps saved in rollout_examples.json
"""
    with open(exp_dir / "evaluation_report_wm.md", "w") as f:
        f.write(report)

    logger.info("=== Phase 8 Experiment Complete. Results in %s ===", exp_dir)
    return exp_dir


if __name__ == "__main__":
    run_world_model_experiment()
