"""
CyberSentinel AI - Phase 8C Transition Dynamics Investigation Script.

Executes all audits and ablations:
1. Frozen V1 baseline
2. Latent transitions audit (MSE, cosine sim, L2 norm, identity vs transition)
3. Gradient & Loss contribution audit
4. Ablation A: WORLD_MODEL_NO_TRANSITION
5. Ablation B: WORLD_MODEL_DIRECT_TRANSITION
6. Ablation C: WORLD_MODEL_STAGE_CONDITIONED
7. Transition-Aware Loss comparison
8. Lead-Time Analysis on 6 genuine transitions
9. K-step rollout analysis (K=1, 2, 4) & collapse detection
10. Calibration analysis (Temperature scaling T fit on val only)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.optimize import minimize

WORKSPACE_ROOT = Path(r"d:\uec sih")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from network.flow.csv_loader import CSVFlowLoader
from ml.state.state_builder import NetworkStateBuilder, FEATURE_NAMES
from ml.preprocessing.stage_labeler import StageLabeler, STAGE_TAXONOMY
from ml.preprocessing.scaler import FeatureScaler
from ml.preprocessing.sequence_builder import SequenceBuilder
from ml.baseline.logistic_regression import LogisticRegressionBaseline
from ml.temporal.gru_baseline import TemporalGRUBaseline
from ml.world_model.cyber_world_model import (
    CyberWorldModel,
    CyberWorldModelTrainer,
    INPUT_DIM,
    WorldModelOutput,
    NetworkStateEncoder,
    TransitionHead,
    ClassificationHead,
    NextStateHead
)
from ml.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_forecasting_metrics
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase8c_investigation")

OUT_DIR = WORKSPACE_ROOT / "experiments" / "phase8c_investigation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data Preparation (Hard Multi-Stage Holdout Split)
# ---------------------------------------------------------------------------

def prepare_hard_split():
    logger.info("Loading flows and constructing states for Hard Holdout Split...")
    loader = CSVFlowLoader()
    all_flows = []
    for p in sorted((WORKSPACE_ROOT / "datasets" / "sample").glob("*.csv")):
        all_flows.extend(loader.load_flows(p))

    states_df = NetworkStateBuilder(window_size_seconds=30.0).build_states(all_flows)
    states_df = StageLabeler(fallback_to_heuristics=True).attach_labels_to_dataframe(states_df)

    test_sc = ['trace_multistage_03', 'trace_multistage_theta', 'trace_benign_beta', 'trace_recon_gamma']
    val_sc  = ['trace_multistage_01', 'trace_benign_alpha']
    train_sc = [sc for sc in states_df['scenario_id'].unique() if sc not in test_sc and sc not in val_sc]

    train_df = states_df[states_df['scenario_id'].isin(train_sc)].reset_index(drop=True)
    val_df   = states_df[states_df['scenario_id'].isin(val_sc)].reset_index(drop=True)
    test_df  = states_df[states_df['scenario_id'].isin(test_sc)].reset_index(drop=True)

    scaler = FeatureScaler(scaler_type='robust').fit(train_df)
    X_tr = scaler.transform(train_df)
    X_val = scaler.transform(val_df)
    X_te = scaler.transform(test_df)

    seq_builder = SequenceBuilder(sequence_length=8, pad_short_sequences=True)
    tr_seq = seq_builder.build_sequences(train_df, scaled_features=X_tr)
    val_seq = seq_builder.build_sequences(val_df, scaled_features=X_val)
    te_seq = seq_builder.build_sequences(test_df, scaled_features=X_te)

    return tr_seq, val_seq, te_seq, scaler

# ---------------------------------------------------------------------------
# Main Investigation Runner
# ---------------------------------------------------------------------------

def run_investigation():
    tr_seq, val_seq, te_seq, scaler = prepare_hard_split()
    results_summary = {}

    num_classes = len(STAGE_TAXONOMY)
    trans_mask_te = (te_seq.y_current_stage.numpy() != te_seq.y_next_stage.numpy())
    trans_indices = np.where(trans_mask_te)[0].tolist()
    logger.info(f"Test size: {len(te_seq.x_seq)}, Genuine transition sequences: {len(trans_indices)} ({trans_indices})")

    # =========================================================================
    # PART 0: EVALUATE BASELINES (LOGISTIC REGRESSION & GRU)
    # =========================================================================
    logger.info("Evaluating Baselines (LR & GRU)...")
    lengths_te = te_seq.mask.sum(dim=1).clamp(min=1) - 1
    idx_te = lengths_te.view(-1, 1, 1).expand(-1, 1, te_seq.x_seq.shape[-1])
    X_te_lr = te_seq.x_seq.gather(1, idx_te).squeeze(1).numpy()

    lengths_tr = tr_seq.mask.sum(dim=1).clamp(min=1) - 1
    idx_tr = lengths_tr.view(-1, 1, 1).expand(-1, 1, tr_seq.x_seq.shape[-1])
    X_tr_lr = tr_seq.x_seq.gather(1, idx_tr).squeeze(1).numpy()

    lr = LogisticRegressionBaseline(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_tr_lr, tr_seq.y_attack.numpy().astype(int), tr_seq.y_current_stage.numpy(), tr_seq.y_next_stage.numpy())
    lr_prob_ns = lr.predict_next_stage_proba(X_te_lr)
    lr_pred_ns = lr.predict_next_stage(X_te_lr)
    lr_prob_atk = lr.predict_attack_proba(X_te_lr)
    lr_pred_atk = lr.predict_attack(X_te_lr)

    lr_ns_m = calculate_forecasting_metrics(te_seq.y_next_stage.numpy(), lr_prob_ns, k=3)
    lr_atk_m = calculate_classification_metrics(te_seq.y_attack.numpy(), lr_pred_atk, lr_prob_atk, is_binary=True)
    lr_trans_acc = float(np.mean(lr_pred_ns[trans_mask_te] == te_seq.y_next_stage.numpy()[trans_mask_te]))

    # Train GRU
    class_counts = np.bincount(tr_seq.y_next_stage.numpy(), minlength=num_classes)
    total_samples = len(tr_seq.y_next_stage)
    weights = [total_samples / (num_classes * max(c, 1)) for c in class_counts]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    gru = TemporalGRUBaseline(input_dim=INPUT_DIM, hidden_dim=64, num_layers=2, num_classes=num_classes, random_seed=42)
    gru.fit(tr_seq.x_seq, tr_seq.mask, tr_seq.y_next_stage, tr_seq.y_attack, val_seq.x_seq, val_seq.mask, val_seq.y_next_stage, val_seq.y_attack, class_weights=class_weights_tensor)
    gru_prob_ns = gru.predict_next_stage_proba(te_seq.x_seq, te_seq.mask)
    gru_pred_ns = gru.predict_next_stage(te_seq.x_seq, te_seq.mask)
    gru_prob_atk = gru.predict_attack_proba(te_seq.x_seq, te_seq.mask)
    gru_pred_atk = (gru_prob_atk >= 0.5).astype(int)

    gru_ns_m = calculate_forecasting_metrics(te_seq.y_next_stage.numpy(), gru_prob_ns, k=3)
    gru_atk_m = calculate_classification_metrics(te_seq.y_attack.numpy(), gru_pred_atk, gru_prob_atk, is_binary=True)
    gru_trans_acc = float(np.mean(gru_pred_ns[trans_mask_te] == te_seq.y_next_stage.numpy()[trans_mask_te]))

    results_summary["Logistic_Regression"] = {
        "next_stage_top1": lr_ns_m["next_stage_accuracy"],
        "next_stage_top3": lr_ns_m["top_3_accuracy"],
        "transition_accuracy": lr_trans_acc,
        "brier_score": lr_ns_m["brier_score"],
        "attack_acc": lr_atk_m["accuracy"],
        "fpr": lr_atk_m["fpr"],
        "k4_path_accuracy": None
    }
    results_summary["Temporal_GRU"] = {
        "next_stage_top1": gru_ns_m["next_stage_accuracy"],
        "next_stage_top3": gru_ns_m["top_3_accuracy"],
        "transition_accuracy": gru_trans_acc,
        "brier_score": gru_ns_m["brier_score"],
        "attack_acc": gru_atk_m["accuracy"],
        "fpr": gru_atk_m["fpr"],
        "k4_path_accuracy": None
    }

    # =========================================================================
    # PART 1: ESTABLISH BASELINE (WORLD_MODEL_V1)
    # =========================================================================
    logger.info("Training and Freezing WORLD_MODEL_V1...")
    is_trans_tr = (tr_seq.y_current_stage != tr_seq.y_next_stage).float()
    weights_sample_tr = 1.0 + 9.0 * is_trans_tr  # 10x weight on transition instances

    torch.manual_seed(42)
    wm_v1 = CyberWorldModel(INPUT_DIM, 128, 4, 2, num_classes, dropout=0.1)
    opt_v1 = torch.optim.AdamW(wm_v1.parameters(), lr=1e-3, weight_decay=1e-4)

    ce_stage = nn.CrossEntropyLoss()
    ce_next_unred = nn.CrossEntropyLoss(reduction='none')
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    ds_tr = TensorDataset(tr_seq.x_seq, tr_seq.mask, tr_seq.y_current_stage, tr_seq.y_next_stage, tr_seq.y_attack.float(), tr_seq.y_next_state, weights_sample_tr)
    loader_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)

    for epoch in range(40):
        wm_v1.train()
        for bx, bmask, bcs, bns, batk, bnext_s, bw in loader_tr:
            h_true = wm_v1.encoder(bnext_s)
            opt_v1.zero_grad()
            out = wm_v1(bx, bmask)
            l_s = ce_stage(out.logits_current_stage, bcs)
            l_a = bce(out.logits_attack_prob, batk)
            l_ns = (ce_next_unred(out.logits_next_stage, bns) * bw).mean()
            l_t = mse(out.h_next, h_true.detach())
            tot = l_s + l_a + l_t + 2.0 * l_ns
            tot.backward()
            nn.utils.clip_grad_norm_(wm_v1.parameters(), 1.0)
            opt_v1.step()

    torch.save(wm_v1.state_dict(), OUT_DIR / "world_model_v1.pt")
    logger.info("WORLD_MODEL_V1 trained and saved.")

    wm_v1.eval()
    with torch.no_grad():
        out_v1_te = wm_v1(te_seq.x_seq, te_seq.mask)
        v1_prob_ns = F.softmax(out_v1_te.logits_next_stage, dim=-1).numpy()
        v1_pred_ns = np.argmax(v1_prob_ns, axis=1)
        v1_prob_atk = torch.sigmoid(out_v1_te.logits_attack_prob).numpy()
        v1_pred_atk = (v1_prob_atk >= 0.5).astype(int)

    v1_ns_m = calculate_forecasting_metrics(te_seq.y_next_stage.numpy(), v1_prob_ns, k=3)
    v1_atk_m = calculate_classification_metrics(te_seq.y_attack.numpy(), v1_pred_atk, v1_prob_atk, is_binary=True)
    v1_trans_acc = float(np.mean(v1_pred_ns[trans_mask_te] == te_seq.y_next_stage.numpy()[trans_mask_te]))

    results_summary["WORLD_MODEL_V1"] = {
        "next_stage_top1": v1_ns_m["next_stage_accuracy"],
        "next_stage_top3": v1_ns_m["top_3_accuracy"],
        "transition_accuracy": v1_trans_acc,
        "brier_score": v1_ns_m["brier_score"],
        "attack_acc": v1_atk_m["accuracy"],
        "fpr": v1_atk_m["fpr"],
    }

    # =========================================================================
    # PART 2: INSPECT LATENT TRANSITIONS (h_t vs h_hat_{t+1} vs h_{t+1})
    # =========================================================================
    logger.info("Analyzing latent transition dynamics...")
    latent_audit = []
    with torch.no_grad():
        out_te = wm_v1(te_seq.x_seq, te_seq.mask)
        h_t = out_te.h_last.numpy()              # (N, 128)
        h_hat = out_te.h_next.numpy()            # (N, 128)
        h_actual = wm_v1.encoder(te_seq.y_next_state).numpy() # (N, 128)

    for i in range(len(te_seq.x_seq)):
        cs = STAGE_TAXONOMY[te_seq.y_current_stage[i]]
        ns = STAGE_TAXONOMY[te_seq.y_next_stage[i]]
        is_trans = bool(cs != ns)

        # Vector metrics
        mse_pred_actual = float(np.mean((h_hat[i] - h_actual[i])**2))
        l2_pred_actual = float(np.linalg.norm(h_hat[i] - h_actual[i]))
        cos_sim_pred_actual = float(np.dot(h_hat[i], h_actual[i]) / (np.linalg.norm(h_hat[i]) * np.linalg.norm(h_actual[i]) + 1e-8))

        # Movement metric: how far did transition head move h_t?
        l2_movement = float(np.linalg.norm(h_hat[i] - h_t[i]))
        cos_sim_ht_hnext = float(np.dot(h_hat[i], h_t[i]) / (np.linalg.norm(h_hat[i]) * np.linalg.norm(h_t[i]) + 1e-8))
        actual_l2_movement = float(np.linalg.norm(h_actual[i] - h_t[i]))

        latent_audit.append({
            "idx": i,
            "scenario_id": te_seq.scenario_ids[i],
            "current_stage": cs,
            "next_stage": ns,
            "is_transition": is_trans,
            "mse_pred_actual": mse_pred_actual,
            "l2_pred_actual": l2_pred_actual,
            "cos_sim_pred_actual": cos_sim_pred_actual,
            "l2_predicted_movement": l2_movement,
            "l2_actual_movement": actual_l2_movement,
            "cos_sim_ht_hnext": cos_sim_ht_hnext,
        })

    df_latent = pd.DataFrame(latent_audit)
    df_latent.to_csv(OUT_DIR / "latent_transitions_audit.csv", index=False)

    id_df = df_latent[~df_latent['is_transition']]
    tr_df = df_latent[df_latent['is_transition']]
    logger.info(f"Latent Movement || Identity transitions mean L2: {id_df['l2_predicted_movement'].mean():.4f} (actual: {id_df['l2_actual_movement'].mean():.4f})")
    logger.info(f"Latent Movement || Real transitions mean L2:     {tr_df['l2_predicted_movement'].mean():.4f} (actual: {tr_df['l2_actual_movement'].mean():.4f})")
    logger.info(f"Cos Sim (h_t, h_hat) || Identity: {id_df['cos_sim_ht_hnext'].mean():.4f}, Real transition: {tr_df['cos_sim_ht_hnext'].mean():.4f}")

    # =========================================================================
    # PART 3: GRADIENT / LOSS AUDIT
    # =========================================================================
    logger.info("Running Gradient and Loss Contribution Audit...")
    loss_audit_data = {}
    wm_v1.eval()
    sample_b = next(iter(loader_tr))
    sb_x, sb_mask, sb_cs, sb_ns, sb_atk, sb_ns_s, sb_w = sample_b
    h_true_sample = wm_v1.encoder(sb_ns_s)
    out_sample = wm_v1(sb_x, sb_mask)

    # Compute individual losses and measure gradient norms
    losses = {
        "L_stage": ce_stage(out_sample.logits_current_stage, sb_cs),
        "L_attack": bce(out_sample.logits_attack_prob, sb_atk),
        "L_transition": mse(out_sample.h_next, h_true_sample.detach()),
        "L_next_stage": (ce_next_unred(out_sample.logits_next_stage, sb_ns) * sb_w).mean(),
    }

    grad_norms = {}
    for lname, lval in losses.items():
        wm_v1.zero_grad()
        lval.backward(retain_graph=True)
        total_norm = 0.0
        for p in wm_v1.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norms[lname] = total_norm ** 0.5

    for lname in losses:
        loss_audit_data[lname] = {
            "loss_magnitude": float(losses[lname].item()),
            "gradient_norm": float(grad_norms[lname]),
        }
    with open(OUT_DIR / "gradient_loss_audit.json", "w") as f:
        json.dump(loss_audit_data, f, indent=2)

    # =========================================================================
    # PART 4: ABLATION A — TRANSFORMER WITHOUT TRANSITION HEAD
    # =========================================================================
    logger.info("Training Ablation A: WORLD_MODEL_NO_TRANSITION...")
    class WorldModelNoTransition(nn.Module):
        def __init__(self, input_dim=INPUT_DIM, hidden_dim=128, num_heads=4, num_layers=2, num_stages=10, dropout=0.1):
            super().__init__()
            self.encoder = NetworkStateEncoder(input_dim, hidden_dim, dropout)
            from ml.world_model.cyber_world_model import TemporalTransformer
            self.transformer = TemporalTransformer(hidden_dim, num_heads, num_layers, dropout)
            self.head_current_stage = ClassificationHead(hidden_dim, num_stages, dropout)
            self.head_attack_prob = ClassificationHead(hidden_dim, 1, dropout)
            self.head_next_stage = ClassificationHead(hidden_dim, num_stages, dropout)
            self.head_next_state = NextStateHead(hidden_dim, input_dim, dropout)

        def forward(self, x_seq, mask=None):
            h_seq = self.encoder(x_seq)
            _, h_last = self.transformer(h_seq, mask=mask)
            # Directly decode from h_last without transition operator!
            return (
                self.head_current_stage(h_last),
                self.head_attack_prob(h_last).squeeze(-1),
                self.head_next_stage(h_last),
                self.head_next_state(h_last),
                h_last
            )

    torch.manual_seed(42)
    wm_no_trans = WorldModelNoTransition()
    opt_no = torch.optim.AdamW(wm_no_trans.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(40):
        wm_no_trans.train()
        for bx, bmask, bcs, bns, batk, bnext_s, bw in loader_tr:
            opt_no.zero_grad()
            l_cs, l_atk, l_ns, pred_s, _ = wm_no_trans(bx, bmask)
            loss = ce_stage(l_cs, bcs) + bce(l_atk, batk) + 2.0 * (ce_next_unred(l_ns, bns) * bw).mean() + mse(pred_s, bnext_s)
            loss.backward()
            nn.utils.clip_grad_norm_(wm_no_trans.parameters(), 1.0)
            opt_no.step()

    wm_no_trans.eval()
    with torch.no_grad():
        _, l_atk_te, l_ns_te, _, _ = wm_no_trans(te_seq.x_seq, te_seq.mask)
        p_ns_no = F.softmax(l_ns_te, dim=-1).numpy()
        pred_ns_no = np.argmax(p_ns_no, axis=1)
        p_atk_no = torch.sigmoid(l_atk_te).numpy()
        pred_atk_no = (p_atk_no >= 0.5).astype(int)

    m_ns_no = calculate_forecasting_metrics(te_seq.y_next_stage.numpy(), p_ns_no, k=3)
    m_atk_no = calculate_classification_metrics(te_seq.y_attack.numpy(), pred_atk_no, p_atk_no, is_binary=True)
    trans_acc_no = float(np.mean(pred_ns_no[trans_mask_te] == te_seq.y_next_stage.numpy()[trans_mask_te]))

    results_summary["Ablation_A_No_Transition"] = {
        "next_stage_top1": m_ns_no["next_stage_accuracy"],
        "next_stage_top3": m_ns_no["top_3_accuracy"],
        "transition_accuracy": trans_acc_no,
        "brier_score": m_ns_no["brier_score"],
        "attack_acc": m_atk_no["accuracy"],
        "fpr": m_atk_no["fpr"],
        "k4_path_accuracy": None
    }

    # =========================================================================
    # PART 5: ABLATION B — DIRECT NEXT-STATE PREDICTION
    # =========================================================================
    logger.info("Training Ablation B: WORLD_MODEL_DIRECT_TRANSITION...")
    class WorldModelDirectTransition(nn.Module):
        def __init__(self, input_dim=INPUT_DIM, hidden_dim=128, num_heads=4, num_layers=2, num_stages=10, dropout=0.1):
            super().__init__()
            self.encoder = NetworkStateEncoder(input_dim, hidden_dim, dropout)
            from ml.world_model.cyber_world_model import TemporalTransformer
            self.transformer = TemporalTransformer(hidden_dim, num_heads, num_layers, dropout)
            self.head_next_state = NextStateHead(hidden_dim, input_dim, dropout)
            self.head_current_stage = ClassificationHead(hidden_dim, num_stages, dropout)
            self.head_attack_prob = ClassificationHead(hidden_dim, 1, dropout)
            self.head_next_stage = ClassificationHead(hidden_dim, num_stages, dropout)

        def forward(self, x_seq, mask=None):
            h_seq = self.encoder(x_seq)
            _, h_last = self.transformer(h_seq, mask=mask)
            pred_s_next = self.head_next_state(h_last)
            # Re-encode predicted S_{t+1} to h_next!
            h_next = self.encoder(pred_s_next)
            return (
                self.head_current_stage(h_last),
                self.head_attack_prob(h_last).squeeze(-1),
                self.head_next_stage(h_next),
                pred_s_next,
                h_last,
                h_next
            )

        @torch.no_grad()
        def rollout(self, x_seq, mask=None, k_steps=4):
            self.eval()
            h_seq = self.encoder(x_seq)
            _, h_curr = self.transformer(h_seq, mask=mask)
            steps = []
            for k in range(1, k_steps + 1):
                pred_s = self.head_next_state(h_curr)
                h_next = self.encoder(pred_s)
                l_ns = self.head_next_stage(h_next)
                l_atk = self.head_attack_prob(h_next).squeeze(-1)
                steps.append({
                    "step": k,
                    "stage_pred": torch.argmax(l_ns, dim=-1).detach().cpu().numpy(),
                    "attack_prob": torch.sigmoid(l_atk).detach().cpu().numpy(),
                    "pred_state": pred_s.detach().cpu().numpy()
                })
                h_curr = h_next
            return steps

    torch.manual_seed(42)
    wm_direct = WorldModelDirectTransition()
    opt_direct = torch.optim.AdamW(wm_direct.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(40):
        wm_direct.train()
        for bx, bmask, bcs, bns, batk, bnext_s, bw in loader_tr:
            opt_direct.zero_grad()
            l_cs, l_atk, l_ns, pred_s_next, _, h_next = wm_direct(bx, bmask)
            loss = (
                ce_stage(l_cs, bcs)
                + bce(l_atk, batk)
                + 2.0 * (ce_next_unred(l_ns, bns) * bw).mean()
                + mse(pred_s_next, bnext_s)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(wm_direct.parameters(), 1.0)
            opt_direct.step()

    wm_direct.eval()
    with torch.no_grad():
        _, l_atk_dir, l_ns_dir, _, _, _ = wm_direct(te_seq.x_seq, te_seq.mask)
        p_ns_dir = F.softmax(l_ns_dir, dim=-1).numpy()
        pred_ns_dir = np.argmax(p_ns_dir, axis=1)
        p_atk_dir = torch.sigmoid(l_atk_dir).numpy()
        pred_atk_dir = (p_atk_dir >= 0.5).astype(int)

    m_ns_dir = calculate_forecasting_metrics(te_seq.y_next_stage.numpy(), p_ns_dir, k=3)
    m_atk_dir = calculate_classification_metrics(te_seq.y_attack.numpy(), pred_atk_dir, p_atk_dir, is_binary=True)
    trans_acc_dir = float(np.mean(pred_ns_dir[trans_mask_te] == te_seq.y_next_stage.numpy()[trans_mask_te]))

    results_summary["Ablation_B_Direct_Transition"] = {
        "next_stage_top1": m_ns_dir["next_stage_accuracy"],
        "next_stage_top3": m_ns_dir["top_3_accuracy"],
        "transition_accuracy": trans_acc_dir,
        "brier_score": m_ns_dir["brier_score"],
        "attack_acc": m_atk_dir["accuracy"],
        "fpr": m_atk_dir["fpr"],
    }

    # =========================================================================
    # PART 6: ABLATION C — STAGE-CONDITIONED TRANSITION
    # =========================================================================
    logger.info("Training Ablation C: WORLD_MODEL_STAGE_CONDITIONED...")
    class WorldModelStageConditioned(nn.Module):
        def __init__(self, input_dim=INPUT_DIM, hidden_dim=128, num_heads=4, num_layers=2, num_stages=10, dropout=0.1):
            super().__init__()
            self.encoder = NetworkStateEncoder(input_dim, hidden_dim, dropout)
            from ml.world_model.cyber_world_model import TemporalTransformer
            self.transformer = TemporalTransformer(hidden_dim, num_heads, num_layers, dropout)
            self.stage_emb = nn.Embedding(num_stages, hidden_dim)
            self.transition_head = TransitionHead(hidden_dim, dropout=0.2)
            self.head_current_stage = ClassificationHead(hidden_dim, num_stages, dropout)
            self.head_attack_prob = ClassificationHead(hidden_dim, 1, dropout)
            self.head_next_stage = ClassificationHead(hidden_dim, num_stages, dropout)
            self.head_next_state = NextStateHead(hidden_dim, input_dim, dropout)

        def forward(self, x_seq, mask=None, y_curr=None):
            h_seq = self.encoder(x_seq)
            _, h_last = self.transformer(h_seq, mask=mask)
            logits_cs = self.head_current_stage(h_last)
            
            # Use predicted current stage if y_curr not provided (clean test inference)
            if y_curr is None:
                stage_idx = torch.argmax(logits_cs, dim=-1)
            else:
                stage_idx = y_curr

            h_cond = h_last + self.stage_emb(stage_idx)
            h_next = self.transition_head(h_cond)

            return (
                logits_cs,
                self.head_attack_prob(h_last).squeeze(-1),
                self.head_next_stage(h_next),
                self.head_next_state(h_next),
                h_last,
                h_next
            )

        @torch.no_grad()
        def rollout(self, x_seq, mask=None, k_steps=4):
            self.eval()
            h_seq = self.encoder(x_seq)
            _, h_curr = self.transformer(h_seq, mask=mask)
            steps = []
            for k in range(1, k_steps + 1):
                logits_cs = self.head_current_stage(h_curr)
                stage_idx = torch.argmax(logits_cs, dim=-1)
                h_cond = h_curr + self.stage_emb(stage_idx)
                h_pred = self.transition_head(h_cond)
                l_ns = self.head_next_stage(h_pred)
                l_atk = self.head_attack_prob(h_pred).squeeze(-1)
                steps.append({
                    "step": k,
                    "stage_pred": torch.argmax(l_ns, dim=-1).detach().cpu().numpy(),
                    "attack_prob": torch.sigmoid(l_atk).detach().cpu().numpy(),
                    "pred_state": self.head_next_state(h_pred).detach().cpu().numpy()
                })
                h_curr = h_pred
            return steps

    torch.manual_seed(42)
    wm_cond = WorldModelStageConditioned()
    opt_cond = torch.optim.AdamW(wm_cond.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(40):
        wm_cond.train()
        for bx, bmask, bcs, bns, batk, bnext_s, bw in loader_tr:
            h_true = wm_cond.encoder(bnext_s)
            opt_cond.zero_grad()
            l_cs, l_atk, l_ns, pred_s_next, _, h_next = wm_cond(bx, bmask, y_curr=bcs)
            loss = (
                ce_stage(l_cs, bcs)
                + bce(l_atk, batk)
                + 2.0 * (ce_next_unred(l_ns, bns) * bw).mean()
                + mse(h_next, h_true.detach())
                + mse(pred_s_next, bnext_s)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(wm_cond.parameters(), 1.0)
            opt_cond.step()

    wm_cond.eval()
    with torch.no_grad():
        _, l_atk_c, l_ns_c, _, _, _ = wm_cond(te_seq.x_seq, te_seq.mask, y_curr=None)
        p_ns_c = F.softmax(l_ns_c, dim=-1).numpy()
        pred_ns_c = np.argmax(p_ns_c, axis=1)
        p_atk_c = torch.sigmoid(l_atk_c).numpy()
        pred_atk_c = (p_atk_c >= 0.5).astype(int)

    m_ns_c = calculate_forecasting_metrics(te_seq.y_next_stage.numpy(), p_ns_c, k=3)
    m_atk_c = calculate_classification_metrics(te_seq.y_attack.numpy(), pred_atk_c, p_atk_c, is_binary=True)
    trans_acc_c = float(np.mean(pred_ns_c[trans_mask_te] == te_seq.y_next_stage.numpy()[trans_mask_te]))

    results_summary["Ablation_C_Stage_Conditioned"] = {
        "next_stage_top1": m_ns_c["next_stage_accuracy"],
        "next_stage_top3": m_ns_c["top_3_accuracy"],
        "transition_accuracy": trans_acc_c,
        "brier_score": m_ns_c["brier_score"],
        "attack_acc": m_atk_c["accuracy"],
        "fpr": m_atk_c["fpr"],
    }

    # =========================================================================
    # PART 7: TRANSITION LEAD TIME ANALYSIS (6 Genuine Transitions)
    # =========================================================================
    logger.info("Executing Lead Time Analysis on 6 genuine transitions...")
    lead_time_records = []
    # Candidate models for lead-time: LR, GRU, WM_V1, WM_Dir, WM_Cond
    # For each transition window t_trans:
    # Look at t_trans - 1 (1 window early), t_trans (at transition), t_trans + 1 (1 window after)
    all_models_preds = {
        "LR": lr_pred_ns,
        "GRU": gru_pred_ns,
        "WM_V1": v1_pred_ns,
        "WM_NoTrans": pred_ns_no,
        "WM_Direct": pred_ns_dir,
        "WM_Conditioned": pred_ns_c
    }

    for tidx in trans_indices:
        target_next_stage = te_seq.y_next_stage[tidx].item()
        target_curr_stage = te_seq.y_current_stage[tidx].item()
        target_str = f"{STAGE_TAXONOMY[target_curr_stage]} -> {STAGE_TAXONOMY[target_next_stage]}"
        sc_id = te_seq.scenario_ids[tidx]

        record = {
            "transition_index": tidx,
            "scenario": sc_id,
            "transition": target_str,
            "predictions_at_transition": {},
            "predicted_early_1_window": {},
            "predicted_after_1_window": {}
        }

        # At transition window:
        for mname, preds in all_models_preds.items():
            record["predictions_at_transition"][mname] = {
                "pred": STAGE_TAXONOMY[preds[tidx]],
                "correct": bool(preds[tidx] == target_next_stage)
            }
            # 1 window early (tidx - 1) if same scenario
            if tidx > 0 and te_seq.scenario_ids[tidx - 1] == sc_id:
                record["predicted_early_1_window"][mname] = {
                    "pred": STAGE_TAXONOMY[preds[tidx - 1]],
                    "predicted_target": bool(preds[tidx - 1] == target_next_stage)
                }
            else:
                record["predicted_early_1_window"][mname] = "N/A"

            # 1 window after (tidx + 1) if same scenario
            if tidx < len(te_seq.x_seq) - 1 and te_seq.scenario_ids[tidx + 1] == sc_id:
                record["predicted_after_1_window"][mname] = {
                    "pred": STAGE_TAXONOMY[preds[tidx + 1]],
                    "correct": bool(preds[tidx + 1] == te_seq.y_next_stage[tidx + 1].item())
                }
            else:
                record["predicted_after_1_window"][mname] = "N/A"

        lead_time_records.append(record)

    with open(OUT_DIR / "lead_time_analysis.json", "w") as f:
        json.dump(lead_time_records, f, indent=2)

    # =========================================================================
    # PART 8: K-STEP ROLLOUT EVALUATION (K=1, 2, 4) & COLLAPSE DETECTION
    # =========================================================================
    logger.info("Executing K-Step Rollout Analysis (K=1, 2, 4)...")
    # Evaluate multi-step path on multi-stage scenarios:
    # In test set, trace_multistage_03 starts at seq 11.
    # Its sequence of next-stages is:
    # seq 11 (t=0): next stage = BENIGN
    # seq 12 (t=1): next stage = BENIGN
    # seq 13 (t=2): next stage = RECONNAISSANCE
    # seq 14 (t=3): next stage = RECONNAISSANCE
    # seq 15 (t=4): next stage = RECONNAISSANCE
    # seq 16 (t=5): next stage = CREDENTIAL_ACCESS
    # seq 17 (t=6): next stage = CREDENTIAL_ACCESS
    # seq 18 (t=7): next stage = CREDENTIAL_ACCESS
    # seq 19 (t=8): next stage = LATERAL_MOVEMENT
    # seq 20 (t=9): next stage = LATERAL_MOVEMENT
    # seq 21 (t=10): next stage = LATERAL_MOVEMENT

    # Run rollout for V1 and Direct on all test sequences
    rollout_k4_v1 = wm_v1.rollout(te_seq.x_seq, te_seq.mask, k_steps=4)
    rollout_k4_dir = wm_direct.rollout(te_seq.x_seq, te_seq.mask, k_steps=4)
    rollout_k4_cond = wm_cond.rollout(te_seq.x_seq, te_seq.mask, k_steps=4)

    rollout_analysis = {}
    for mname, steps in [("WORLD_MODEL_V1", rollout_k4_v1), ("WM_DIRECT", rollout_k4_dir), ("WM_CONDITIONED", rollout_k4_cond)]:
        m_eval = {}
        for step_dict in steps:
            k = step_dict["step"]
            preds = step_dict["stage_pred"]
            pred_classes = [STAGE_TAXONOMY[p] for p in preds]
            counts = pd.Series(pred_classes).value_counts().to_dict()
            entropy = float(-sum((c/len(preds)) * np.log(c/len(preds) + 1e-12) for c in counts.values()))
            m_eval[f"K={k}"] = {
                "class_distribution": counts,
                "distribution_entropy": entropy,
                "collapse_detected": bool(entropy < 0.2) # true if almost all predictions fall in 1 class
            }
        rollout_analysis[mname] = m_eval

    # Path accuracy for sequence starting before transition (Seq 12 in trace_multistage_03):
    # Seq 12 is at t=1. Ground truth for t+1, t+2, t+3, t+4:
    # t+1: BENIGN (Seq 12 next)
    # t+2: RECONNAISSANCE (Seq 13 next)
    # t+3: RECONNAISSANCE (Seq 14 next)
    # t+4: RECONNAISSANCE (Seq 15 next)
    true_path_12 = ["BENIGN", "RECONNAISSANCE", "RECONNAISSANCE", "RECONNAISSANCE"]
    for mname, res_key, steps in [
        ("WORLD_MODEL_V1", "WORLD_MODEL_V1", rollout_k4_v1),
        ("WM_DIRECT", "Ablation_B_Direct_Transition", rollout_k4_dir),
        ("WM_CONDITIONED", "Ablation_C_Stage_Conditioned", rollout_k4_cond)
    ]:
        pred_path_12 = [STAGE_TAXONOMY[steps[k]["stage_pred"][12]] for k in range(4)]
        path_matches = sum(p == t for p, t in zip(pred_path_12, true_path_12))
        path_acc = path_matches / 4.0
        results_summary[res_key]["k4_path_accuracy"] = path_acc
        rollout_analysis[mname]["seq12_path_accuracy"] = {
            "true_path": true_path_12,
            "pred_path": pred_path_12,
            "accuracy": path_acc
        }

    with open(OUT_DIR / "rollout_k_step_analysis.json", "w") as f:
        json.dump(rollout_analysis, f, indent=2)

    # =========================================================================
    # PART 9: TEMPERATURE SCALING CALIBRATION (Fit on Val Only)
    # =========================================================================
    logger.info("Fitting Temperature Scaling Calibration on Validation Set...")
    with torch.no_grad():
        out_val = wm_v1(val_seq.x_seq, val_seq.mask)
        logits_val = out_val.logits_next_stage.numpy()
        y_val = val_seq.y_next_stage.numpy()

        out_test = wm_v1(te_seq.x_seq, te_seq.mask)
        logits_test = out_test.logits_next_stage.numpy()
        y_test = te_seq.y_next_stage.numpy()

    def nll_val(T_arr):
        T_val = max(T_arr[0], 0.01)
        scaled = logits_val / T_val
        p = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        p = p / np.sum(p, axis=1, keepdims=True)
        eps = 1e-12
        return -float(np.mean([np.log(max(p[i, y_val[i]], eps)) for i in range(len(y_val))]))

    opt_res = minimize(nll_val, x0=[1.0], bounds=[(0.01, 10.0)])
    best_T = float(opt_res.x[0])

    # Uncalibrated vs Calibrated on Test
    p_uncal = np.exp(logits_test - np.max(logits_test, axis=1, keepdims=True))
    p_uncal /= np.sum(p_uncal, axis=1, keepdims=True)

    p_cal = np.exp(logits_test / best_T - np.max(logits_test / best_T, axis=1, keepdims=True))
    p_cal /= np.sum(p_cal, axis=1, keepdims=True)

    y_test_oh = np.zeros((len(y_test), num_classes))
    for i, l in enumerate(y_test): y_test_oh[i, l] = 1.0

    brier_uncal = float(np.mean(np.sum((p_uncal - y_test_oh)**2, axis=1)))
    brier_cal = float(np.mean(np.sum((p_cal - y_test_oh)**2, axis=1)))

    # Compute Expected Calibration Error (ECE)
    def compute_ece(probs, labels, n_bins=10):
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for b in range(n_bins):
            bin_mask = (confidences > bin_boundaries[b]) & (confidences <= bin_boundaries[b + 1])
            if np.any(bin_mask):
                bin_acc = np.mean(accuracies[bin_mask])
                bin_conf = np.mean(confidences[bin_mask])
                ece += np.sum(bin_mask) / len(labels) * abs(bin_acc - bin_conf)
        return float(ece)

    ece_uncal = compute_ece(p_uncal, y_test)
    ece_cal = compute_ece(p_cal, y_test)

    # Transition confidence
    trans_conf_uncal = float(np.mean(np.max(p_uncal[trans_mask_te], axis=1)))
    trans_conf_cal = float(np.mean(np.max(p_cal[trans_mask_te], axis=1)))

    calibration_results = {
        "optimal_temperature_fitted_on_val": best_T,
        "uncalibrated": {
            "brier_score": brier_uncal,
            "ece": ece_uncal,
            "mean_confidence": float(np.mean(np.max(p_uncal, axis=1))),
            "transition_confidence": trans_conf_uncal
        },
        "calibrated": {
            "brier_score": brier_cal,
            "ece": ece_cal,
            "mean_confidence": float(np.mean(np.max(p_cal, axis=1))),
            "transition_confidence": trans_conf_cal
        },
        "prediction_invariance_verified": bool(np.array_equal(np.argmax(p_uncal, axis=1), np.argmax(p_cal, axis=1)))
    }

    with open(OUT_DIR / "calibration_analysis.json", "w") as f:
        json.dump(calibration_results, f, indent=2)

    # Save complete summary
    with open(OUT_DIR / "phase8c_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info("=== Phase 8C Investigation Complete ===")
    return results_summary

if __name__ == "__main__":
    run_investigation()
