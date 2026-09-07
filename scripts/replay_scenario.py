"""
CyberSentinel AI — Deterministic Trace Replay Script (Phase 9).

Replays an existing test scenario window-by-window through the trained
CyberWorldModelV2, demonstrating the full defensive intelligence pipeline:

  At each window t:
    1. Observe current network state S_t
    2. Predict next stage P(stage_{t+1})
    3. Show confidence and uncertainty
    4. Explain key feature changes (S_hat_{t+1} - S_t)
    5. Deterministic MITRE ATT&CK mapping
    6. K=4 step autoregressive forward simulation
    7. Risk score and defensive priority

Default demo scenario: trace_multistage_03 (BENIGN -> RECON -> CRED_ACCESS -> LATERAL)

Usage:
    python scripts/replay_scenario.py
    python scripts/replay_scenario.py --scenario trace_multistage_theta --k 4
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.world_model.world_model_v2 import (
    CyberWorldModelTrainerV2,
    WorldModelV2,
    RolloutResult,
)
from ml.world_model.explainability import build_explanation
from ml.defense.risk_engine import (
    ForecastEvent,
    RiskEngine,
    STAGE_TAXONOMY,
)
from ml.calibration.temperature_scaling import load_temperature
from mitre.mappings.mitre_mapper import get_mitre_summary
from ml.state.state_builder import FEATURE_NAMES, NetworkStateBuilder
from ml.preprocessing.scaler import FeatureScaler
from network.flow.csv_loader import CSVFlowLoader
from ml.preprocessing.sequence_builder import SequenceBuilder
from ml.preprocessing.stage_labeler import StageLabeler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# Stage taxonomy mapping (index -> name)
_STAGE_NAMES = STAGE_TAXONOMY


def _stage_name(idx: int) -> str:
    if 0 <= idx < len(_STAGE_NAMES):
        return _STAGE_NAMES[idx]
    return "UNKNOWN"


def load_or_train_v2(
    scenario_id: str,
    checkpoint_path: Optional[Path] = None,
    epochs: int = 40,
    seed: int = 42,
):
    """
    Load a trained CyberWorldModelV2 checkpoint if available, else train one on the fly.
    """
    ckpt = checkpoint_path or (WORKSPACE_ROOT / "models" / "world_model_v2.pt")

    if ckpt.exists():
        logger.info("Loading CyberWorldModelV2 from checkpoint %s", ckpt)
        trainer = CyberWorldModelTrainerV2.load(ckpt)
        # Load calibration temperature from validated artifact
        calib_artifact = WORKSPACE_ROOT / "artifacts" / "calibration" / "temperature.json"
        trainer.apply_calibration_from_artifact(calib_artifact)
        return trainer

    logger.info("No checkpoint found at %s — training V2 on fly for replay.", ckpt)
    # Load data
    loader = CSVFlowLoader()
    all_flows = []
    for p in sorted((WORKSPACE_ROOT / "datasets" / "sample").glob("*.csv")):
        all_flows.extend(loader.load_flows(p))

    states_df = NetworkStateBuilder(window_size_seconds=30.0).build_states(all_flows)
    states_df = StageLabeler(fallback_to_heuristics=True).attach_labels_to_dataframe(states_df)

    test_sc = ['trace_multistage_03', 'trace_multistage_theta', 'trace_benign_beta', 'trace_recon_gamma']
    val_sc = ['trace_multistage_01', 'trace_benign_alpha']
    train_sc = [sc for sc in states_df['scenario_id'].unique() if sc not in test_sc and sc not in val_sc]

    train_df = states_df[states_df['scenario_id'].isin(train_sc)].reset_index(drop=True)
    val_df = states_df[states_df['scenario_id'].isin(val_sc)].reset_index(drop=True)

    scaler = FeatureScaler(scaler_type='robust').fit(train_df)
    X_tr = scaler.transform(train_df)
    X_val = scaler.transform(val_df)

    seq_builder = SequenceBuilder(sequence_length=8, pad_short_sequences=True)
    tr_seq = seq_builder.build_sequences(train_df, scaled_features=X_tr)
    val_seq = seq_builder.build_sequences(val_df, scaled_features=X_val)

    # Compute transition weights for training
    is_trans_tr = (tr_seq.y_current_stage != tr_seq.y_next_stage).float()
    weights_tr = 1.0 + 9.0 * is_trans_tr

    num_stages = len(STAGE_TAXONOMY)
    trainer = CyberWorldModelTrainerV2(
        input_dim=len(FEATURE_NAMES),
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        num_stages=num_stages,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=32,
        epochs=epochs,
        patience=10,
        lambda_stage=1.0,
        lambda_attack=1.0,
        lambda_state=1.0,
        lambda_next_stage=2.0,
        random_seed=seed,
    )
    trainer.fit(
        tr_seq.x_seq, tr_seq.mask,
        tr_seq.y_current_stage, tr_seq.y_next_stage, tr_seq.y_attack.float(),
        y_next_state_train=tr_seq.y_next_state,
        sample_weights_train=weights_tr,
        x_val=val_seq.x_seq, mask_val=val_seq.mask,
        y_current_stage_val=val_seq.y_current_stage,
        y_next_stage_val=val_seq.y_next_stage,
        y_attack_val=val_seq.y_attack.float(),
        y_next_state_val=val_seq.y_next_state,
    )

    # Apply calibration
    calib_artifact = WORKSPACE_ROOT / "artifacts" / "calibration" / "temperature.json"
    trainer.apply_calibration_from_artifact(calib_artifact)

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(ckpt)
    logger.info("CyberWorldModelV2 trained and saved to %s", ckpt)

    return trainer, scaler


def replay_scenario(
    scenario_id: str = "trace_multistage_03",
    k_steps: int = 4,
    checkpoint_path: Optional[Path] = None,
    output_json: Optional[Path] = None,
):
    """
    Replay a scenario trace through CyberWorldModelV2, window by window.

    This uses the ACTUAL trained model on the actual test sequences.
    It is NOT a canned animation.
    """
    # Load data
    loader_csv = CSVFlowLoader()
    all_flows = []
    for p in sorted((WORKSPACE_ROOT / "datasets" / "sample").glob("*.csv")):
        all_flows.extend(loader_csv.load_flows(p))

    states_df = NetworkStateBuilder(window_size_seconds=30.0).build_states(all_flows)
    states_df = StageLabeler(fallback_to_heuristics=True).attach_labels_to_dataframe(states_df)

    test_sc = ['trace_multistage_03', 'trace_multistage_theta', 'trace_benign_beta', 'trace_recon_gamma']
    val_sc = ['trace_multistage_01', 'trace_benign_alpha']
    train_sc = [sc for sc in states_df['scenario_id'].unique() if sc not in test_sc and sc not in val_sc]

    train_df = states_df[states_df['scenario_id'].isin(train_sc)].reset_index(drop=True)
    val_df = states_df[states_df['scenario_id'].isin(val_sc)].reset_index(drop=True)
    test_df = states_df[states_df['scenario_id'].isin(test_sc)].reset_index(drop=True)
    scenario_df = test_df[test_df['scenario_id'] == scenario_id].reset_index(drop=True)

    if scenario_df.empty:
        logger.error("Scenario '%s' not found in test split.", scenario_id)
        return []

    scaler = FeatureScaler(scaler_type='robust').fit(train_df)
    X_te = scaler.transform(test_df)
    X_scenario = scaler.transform(scenario_df)

    seq_builder = SequenceBuilder(sequence_length=8, pad_short_sequences=True)
    test_seq = seq_builder.build_sequences(test_df, scaled_features=X_te)
    scenario_indices = [
        i for i, sc in enumerate(test_seq.scenario_ids) if sc == scenario_id
    ]

    if not scenario_indices:
        logger.error("No sequences for scenario '%s' found in test sequences.", scenario_id)
        return []

    # Load or train model
    result = load_or_train_v2(scenario_id, checkpoint_path)
    if isinstance(result, tuple):
        trainer, _ = result
    else:
        trainer = result

    risk_engine = RiskEngine()
    replay_log = []

    logger.info("=" * 72)
    logger.info("CYBERSENTINEL AI — TRACE REPLAY: %s", scenario_id)
    logger.info("=" * 72)

    for win_num, seq_idx in enumerate(scenario_indices):
        x = test_seq.x_seq[seq_idx:seq_idx+1]   # (1, T, D)
        mask = test_seq.mask[seq_idx:seq_idx+1]  # (1, T)

        true_current_stage = _stage_name(int(test_seq.y_current_stage[seq_idx]))
        true_next_stage = _stage_name(int(test_seq.y_next_stage[seq_idx]))

        # ── 1. Forward pass ──────────────────────────────────────────────
        with torch.no_grad():
            out = trainer._forward(x, mask)

        import torch.nn.functional as F
        pred_current = int(torch.argmax(out.logits_current_stage, dim=-1).item())
        T = max(trainer.temperature, 1e-4)
        p_next = F.softmax(out.logits_next_stage / T, dim=-1).cpu().numpy()[0]
        pred_next = int(np.argmax(p_next))
        p_attack = float(torch.sigmoid(out.logits_attack_prob[0]).item())

        confidence = float(np.max(p_next))
        eps = 1e-12
        raw_ent = float(-np.sum(p_next * np.log(p_next + eps)))
        norm_ent = raw_ent / max(np.log(len(p_next)), eps)

        cur_state = x[0, -1, :].numpy()
        pred_state = out.pred_next_state[0].detach().cpu().numpy()

        # ── 2. Explainability ────────────────────────────────────────────
        explanation = build_explanation(
            cur_state, pred_state,
            true_current_stage, _stage_name(pred_next),
            top_k=5,
        )

        # ── 3. ForecastEvent ─────────────────────────────────────────────
        stage_prob_dict = {_stage_name(i): round(float(p_next[i]), 4) for i in range(len(p_next))}
        event = ForecastEvent(
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            model_version=WorldModelV2.MODEL_VERSION,
            horizon_seconds=30,
            current_stage=true_current_stage,
            current_state=cur_state.tolist(),
            predicted_stage=_stage_name(pred_next),
            predicted_next_state=pred_state.tolist(),
            attack_probability=p_attack,
            stage_probabilities=stage_prob_dict,
            confidence=confidence,
            uncertainty_entropy=norm_ent,
            transition_detected=(_stage_name(pred_next) != true_current_stage),
            top_features=explanation["top_k_changed_features"][:5],
            scenario_id=scenario_id,
        )

        # ── 4. MITRE mapping ─────────────────────────────────────────────
        mitre = get_mitre_summary(_stage_name(pred_next))

        # ── 5. Risk assessment ───────────────────────────────────────────
        risk = risk_engine.evaluate(event, horizon_steps=1)

        # ── 6. K-step rollout ────────────────────────────────────────────
        rollout: RolloutResult = trainer.rollout(x, mask, k_steps=k_steps)
        rollout_path = [
            {
                "step": k + 1,
                "predicted_stage": _stage_name(int(rollout.predicted_stages[k][0])),
                "confidence": round(float(rollout.confidence[k][0]), 4),
                "uncertainty": round(float(rollout.uncertainty[k][0]), 4),
                "attack_prob": round(float(rollout.attack_probabilities[k][0]), 4),
            }
            for k in range(rollout.horizon)
        ]

        # ── 7. Print window summary ──────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"Window {win_num+1:>2d} | {scenario_id}")
        print(f"  True stage   : {true_current_stage}")
        print(f"  Pred stage   : {_stage_name(pred_next)} (conf={confidence:.2%}, ent={norm_ent:.3f})")
        print(f"  Attack prob  : {p_attack:.3f}")
        print(f"  Transition?  : {'YES ⚠' if event.transition_detected else 'No'} "
              f"(true next: {true_next_stage})")
        print(f"  Risk score   : {risk.risk_score:.1f}/100 → {risk.severity}")
        print(f"  Priority     : {risk.recommended_priority[:50]}")
        if mitre["primary_technique_id"]:
            print(f"  MITRE        : {mitre['primary_technique_id']} — {mitre['primary_technique_name']}")
        top_feat = explanation["stage_relevant_features"][:2] or explanation["top_k_changed_features"][:2]
        if top_feat:
            print("  Key features :")
            for f in top_feat:
                print(f"    {f['feature']:25s}: {f['current']:+.3f} → {f['predicted']:+.3f} ({f['rel_change_pct']:+.1f}%)")
        print(f"  K={k_steps} rollout path: {' → '.join(r['predicted_stage'] for r in rollout_path)}")

        # Store in log
        window_record = {
            "window": win_num + 1,
            "sequence_index": seq_idx,
            "true_current_stage": true_current_stage,
            "true_next_stage": true_next_stage,
            "forecast": event.to_dict(),
            "mitre": mitre,
            "risk": risk.to_dict(),
            "rollout_path": rollout_path,
            "explanation": explanation,
        }
        replay_log.append(window_record)

    print(f"\n{'='*72}")
    print(f"Replay complete: {len(scenario_indices)} windows for {scenario_id}")
    print(f"{'='*72}")

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(replay_log, f, indent=2)
        logger.info("Replay log saved to %s", output_json)

    return replay_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CyberSentinel AI — Trace Replay")
    parser.add_argument("--scenario", default="trace_multistage_03")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    replay_scenario(
        scenario_id=args.scenario,
        k_steps=args.k,
        checkpoint_path=args.checkpoint,
        output_json=args.output,
    )
