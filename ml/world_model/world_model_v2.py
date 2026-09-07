"""
CyberSentinel AI — Cyber World Model V2 (Phase 9 Canonical Architecture).

Promoted from Phase 8C Ablation B (Direct Physical State Transition).

Core architectural change vs V1:
    V1: h_t -> TransitionHead -> h_hat_{t+1}  (latent space transition)
    V2: h_t -> PhysicalNextStatePredictor -> S_hat_{t+1} -> Encoder -> h_hat_{t+1}

The transition now occurs in physical network-state space (S ∈ R^24), which:
  - Prevents latent manifold collapse on identity-dominated training sets.
  - Grounds rollout predictions in physically interpretable feature deltas.
  - Enables direct feature attribution from predicted state changes.

Phase 8C empirical benchmark (Hard Multi-Stage Holdout, N=44, 6 genuine transitions):
    Next-stage Top-1 accuracy   : 97.73%
    Next-stage Top-3 accuracy   : 100.0%
    True transition accuracy    : 83.33% (5/6)
    Brier score (uncalibrated)  : 0.0452
    Attack FPR                  : 0.00%
    K=4 path accuracy           : 25.0% (limited by small transition count in dataset)

V1 is preserved unchanged in cyber_world_model.py as WorldModelV1 / CyberWorldModel.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ml.state.state_builder import FEATURE_NAMES
from ml.world_model.cyber_world_model import (
    NetworkStateEncoder,
    TemporalTransformer,
    ClassificationHead,
    INPUT_DIM,
    # V1 aliases preserved for backwards compat
    CyberWorldModel as WorldModelV1,
    CyberWorldModelTrainer as WorldModelV1Trainer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature metadata for physical bounds enforcement
# ---------------------------------------------------------------------------

# Maps each feature index to its physical constraint type:
#   "nonneg"  : must be >= 0 (counts, rates, bytes, durations)
#   "ratio"   : must be in [0, 1] (ratios, shares)
#   "entropy" : must be >= 0, practically < log2(max_items)
#   "free"    : no hard constraint (can be negative in scaled space)
FEATURE_CONSTRAINT = {
    0:  "nonneg",   # flow_count
    1:  "nonneg",   # total_packets
    2:  "nonneg",   # total_bytes
    3:  "nonneg",   # pkt_rate
    4:  "nonneg",   # byte_rate
    5:  "nonneg",   # unique_src_ips
    6:  "nonneg",   # unique_dst_ips
    7:  "nonneg",   # unique_dst_ports
    8:  "nonneg",   # syn_count
    9:  "nonneg",   # rst_count
    10: "nonneg",   # fin_count
    11: "ratio",    # syn_ratio
    12: "ratio",    # rst_ratio
    13: "entropy",  # dst_ip_entropy
    14: "entropy",  # dst_port_entropy
    15: "nonneg",   # failed_flow_count
    16: "ratio",    # failed_flow_ratio
    17: "entropy",  # tcp_flag_diversity
    18: "ratio",    # port_445_share
    19: "ratio",    # port_3389_share
    20: "ratio",    # port_22_share
    21: "ratio",    # port_80_443_share
    22: "nonneg",   # mean_flow_duration
    23: "nonneg",   # bytes_per_packet
}


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

class WorldModelOutputV2(NamedTuple):
    """All predictions from a CyberWorldModelV2 forward pass."""
    logits_current_stage: torch.Tensor   # (B, num_stages)
    logits_attack_prob: torch.Tensor     # (B,)
    logits_next_stage: torch.Tensor      # (B, num_stages)
    pred_next_state: torch.Tensor        # (B, input_dim) - physical feature space
    h_last: torch.Tensor                 # (B, hidden_dim) - context representation
    h_next: torch.Tensor                 # (B, hidden_dim) - re-encoded predicted state


@dataclass
class RolloutResult:
    """Structured container for multi-step autoregressive rollout."""
    current_state: np.ndarray             # (B, input_dim) — last observed state
    predicted_states: List[np.ndarray]    # K arrays, each (B, input_dim)
    attack_probabilities: List[np.ndarray]# K arrays, each (B,)
    stage_probabilities: List[np.ndarray] # K arrays, each (B, num_stages)
    predicted_stages: List[np.ndarray]    # K arrays, each (B,) integer stage index
    confidence: List[np.ndarray]          # K arrays, each (B,) max probability
    uncertainty: List[np.ndarray]         # K arrays, each (B,) normalized entropy
    horizon: int                          # K steps
    safety_flags: Dict[str, Any]          # Rollout health diagnostics


# ---------------------------------------------------------------------------
# Physical Next-State Predictor
# ---------------------------------------------------------------------------

class PhysicalNextStatePredictor(nn.Module):
    """
    Predicts the physical next network state S_{t+1} from contextual h_t.

    Output is in the scaled feature space (same space as model inputs).
    Physical bound enforcement applies Softplus for nonneg features and
    sigmoid for ratio features ONLY when the model is used for rollout
    (bounds are not applied during supervised training since we MSE against
    already-scaled target states which may themselves be unconstrained).

    Rationale for NOT clipping unconditionally:
      - During training the model sees RobustScaler-scaled inputs, so "nonneg"
        features can legitimately appear negative in scaled space for low-count
        windows. We only apply bounds post-hoc during rollout inference.
      - Clipping during training would introduce a discontinuous gradient and
        bias the learned dynamics toward the constraint boundary.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = INPUT_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, hidden_dim) -> S_hat: (B, output_dim)  [unconstrained]"""
        return self.net(h)

    def forward_bounded(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with physical bounds applied for rollout inference.
        Softplus (smooth non-negative) for count/rate/duration features.
        Sigmoid for ratio/share features.
        Entropy features clamped >= 0.
        """
        s = self.net(h)
        out = s.clone()
        for idx, ctype in FEATURE_CONSTRAINT.items():
            if ctype == "nonneg":
                out[:, idx] = F.softplus(s[:, idx])
            elif ctype == "ratio":
                out[:, idx] = torch.sigmoid(s[:, idx])
            elif ctype == "entropy":
                out[:, idx] = F.softplus(s[:, idx])
        return out


# ---------------------------------------------------------------------------
# Main V2 model
# ---------------------------------------------------------------------------

class CyberWorldModelV2(nn.Module):
    """
    CyberSentinel AI Temporal Cyber World Model V2.

    Architecture (canonical Physical State Forward Simulation):
        Historical states S_{1..t}
        ↓
        NetworkStateEncoder:   MLP -> h_t
        ↓
        TemporalTransformer:   Causal self-attn [h_1..h_t] -> ctx_t
        ↓
        Context representation h_t
        ↓
        PhysicalNextStatePredictor: MLP -> S_hat_{t+1}  (in scaled feature space)
        ↓
        NetworkStateEncoder:   MLP -> h_hat_{t+1}       (re-encode predicted state)
        ↓
        ClassificationHead:   h_t   -> P(current_stage)
        ClassificationHead:   h_hat -> P(next_stage)
        ClassificationHead:   h_hat -> P(attack)

    Rollout (autoregressive, no future observations):
        S_t -> S_hat_{t+1} -> Encoder -> S_hat_{t+2} -> Encoder -> ... -> S_hat_{t+K}
    """

    MODEL_VERSION: str = "2.0"

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_stages: int = 10,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_stages = num_stages

        self.encoder = NetworkStateEncoder(input_dim, hidden_dim, dropout)
        self.transformer = TemporalTransformer(hidden_dim, num_heads, num_layers, dropout, max_seq_len)
        self.head_next_state = PhysicalNextStatePredictor(hidden_dim, input_dim, dropout)

        self.head_current_stage = ClassificationHead(hidden_dim, num_stages, dropout)
        self.head_attack_prob = ClassificationHead(hidden_dim, 1, dropout)
        self.head_next_stage = ClassificationHead(hidden_dim, num_stages, dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> WorldModelOutputV2:
        """
        Args:
            x_seq: (B, T, input_dim) — sequence of scaled network state vectors
            mask:  (B, T) bool — True=valid timestep
        Returns:
            WorldModelOutputV2
        """
        h_seq = self.encoder(x_seq)                         # (B, T, hidden_dim)
        _, h_last = self.transformer(h_seq, mask=mask)      # (B, hidden_dim)

        # Physical state transition: h_t -> S_hat_{t+1}
        pred_s_next = self.head_next_state(h_last)          # (B, input_dim)

        # Re-encode predicted state: S_hat_{t+1} -> h_hat_{t+1}
        h_next = self.encoder(pred_s_next)                  # (B, hidden_dim)

        return WorldModelOutputV2(
            logits_current_stage=self.head_current_stage(h_last),
            logits_attack_prob=self.head_attack_prob(h_next).squeeze(-1),
            logits_next_stage=self.head_next_stage(h_next),
            pred_next_state=pred_s_next,
            h_last=h_last,
            h_next=h_next,
        )

    @torch.no_grad()
    def rollout(
        self,
        x_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        k_steps: int = 4,
        temperature: float = 1.0,
        apply_bounds: bool = True,
    ) -> RolloutResult:
        """
        K-step autoregressive rollout in physical state space.

        GUARANTEE: Does NOT consume any future observations.
        At each step k, only the predicted S_hat_{t+k-1} is used as input.

        Args:
            x_seq:       (B, T, input_dim)
            mask:        (B, T) bool
            k_steps:     forecast horizon K in {1, 2, 4, ...}
            temperature: temperature scaling applied to stage logits (default 1.0)
            apply_bounds: apply physical bound enforcement to predicted states
        """
        self.eval()
        B, T, D = x_seq.shape
        assert D == self.input_dim, (
            f"Input dim mismatch: expected {self.input_dim}, got {D}"
        )

        # Encode historical context (consumed only once — no future leakage)
        h_seq = self.encoder(x_seq)
        _, h_curr = self.transformer(h_seq, mask=mask)

        # Last observed state for reference
        current_state = x_seq[:, -1, :].detach().cpu().numpy()

        predicted_states: List[np.ndarray] = []
        attack_probs: List[np.ndarray] = []
        stage_probs: List[np.ndarray] = []
        pred_stages: List[np.ndarray] = []
        confidences: List[np.ndarray] = []
        uncertainties: List[np.ndarray] = []

        nan_detected = False
        collapse_detected = False

        for k in range(1, k_steps + 1):
            # Predict physical next state
            if apply_bounds:
                pred_s = self.head_next_state.forward_bounded(h_curr)
            else:
                pred_s = self.head_next_state(h_curr)

            # Sanitize: catch NaN/Inf from exploding rollout
            if torch.isnan(pred_s).any() or torch.isinf(pred_s).any():
                nan_detected = True
                pred_s = torch.nan_to_num(pred_s, nan=0.0, posinf=1e3, neginf=-1e3)

            # Validate shape: predicted state must match input feature dim
            assert pred_s.shape[-1] == self.input_dim, (
                f"Shape mismatch at rollout step {k}: "
                f"pred_s.shape={pred_s.shape}, expected dim={self.input_dim}"
            )

            # Re-encode and classify
            h_next = self.encoder(pred_s)
            logits_ns = self.head_next_stage(h_next)
            logits_atk = self.head_attack_prob(h_next).squeeze(-1)

            # Apply temperature scaling (calibration)
            T_safe = max(float(temperature), 1e-4)
            p_stage = F.softmax(logits_ns / T_safe, dim=-1).detach().cpu().numpy()
            p_atk = torch.sigmoid(logits_atk).detach().cpu().numpy()
            pred_stage_idx = np.argmax(p_stage, axis=1)

            # Forecast confidence = max probability
            conf = np.max(p_stage, axis=1)

            # Normalized Shannon entropy (uncertainty measure)
            eps = 1e-12
            max_ent = np.log(float(p_stage.shape[1]))
            raw_ent = -np.sum(p_stage * np.log(p_stage + eps), axis=1)
            norm_entropy = raw_ent / max(max_ent, eps)

            predicted_states.append(pred_s.detach().cpu().numpy())
            attack_probs.append(p_atk)
            stage_probs.append(p_stage)
            pred_stages.append(pred_stage_idx)
            confidences.append(conf)
            uncertainties.append(norm_entropy)

            # Mode collapse detection: mean normalized entropy < 0.05
            if float(np.mean(norm_entropy)) < 0.05:
                collapse_detected = True

            # Step forward (autoregressive: no new observations)
            h_curr = h_next

        return RolloutResult(
            current_state=current_state,
            predicted_states=predicted_states,
            attack_probabilities=attack_probs,
            stage_probabilities=stage_probs,
            predicted_stages=pred_stages,
            confidence=confidences,
            uncertainty=uncertainties,
            horizon=k_steps,
            safety_flags={
                "nan_detected": nan_detected,
                "collapse_detected": collapse_detected,
                "feature_dim_valid": bool(predicted_states[0].shape[-1] == self.input_dim),
            },
        )


# Alias for backwards compat
WorldModelV2 = CyberWorldModelV2


# ---------------------------------------------------------------------------
# V2 Multi-task Loss
# ---------------------------------------------------------------------------

class CyberWorldModelLossV2(nn.Module):
    """
    Multi-task loss for CyberWorldModelV2:
        L = lam_stage * L_stage
          + lam_attack * L_attack
          + lam_state  * L_state     (MSE against true S_{t+1} in scaled space)
          + lam_next_stage * L_next_stage

    Unlike V1, L_state is MSE against the physical feature target (not latent MSE).
    This grounds the transition operator in observable network behaviour.

    Default lambda_next_stage=2.0 (transition accuracy critical for forecasting).
    label_smoothing=0.0 to avoid smearing probability mass during fine-grained
    multi-stage evaluation (use 0.1 only if overfitting observed).
    """

    def __init__(
        self,
        lambda_stage: float = 1.0,
        lambda_attack: float = 1.0,
        lambda_state: float = 1.0,
        lambda_next_stage: float = 2.0,
        label_smoothing: float = 0.0,
        stage_class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.lam = (lambda_stage, lambda_attack, lambda_state, lambda_next_stage)
        self.ce_stage = nn.CrossEntropyLoss(
            weight=stage_class_weights, label_smoothing=label_smoothing
        )
        self.ce_next = nn.CrossEntropyLoss(
            weight=stage_class_weights, label_smoothing=label_smoothing
        )
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        out: WorldModelOutputV2,
        y_stage: torch.Tensor,            # (B,) long
        y_attack: torch.Tensor,           # (B,) float
        y_next_stage: torch.Tensor,       # (B,) long
        s_true_next: Optional[torch.Tensor] = None,   # (B, input_dim)
        sample_weights: Optional[torch.Tensor] = None, # (B,) per-sample weight
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_s = self.ce_stage(out.logits_current_stage, y_stage)
        l_a = self.bce(out.logits_attack_prob, y_attack)

        if sample_weights is not None:
            ce_unred = nn.CrossEntropyLoss(reduction="none")
            l_ns = (ce_unred(out.logits_next_stage, y_next_stage) * sample_weights).mean()
        else:
            l_ns = self.ce_next(out.logits_next_stage, y_next_stage)

        l_state = (
            self.mse(out.pred_next_state, s_true_next)
            if s_true_next is not None
            else out.pred_next_state.new_tensor(0.0)
        )

        total = (
            self.lam[0] * l_s
            + self.lam[1] * l_a
            + self.lam[2] * l_state
            + self.lam[3] * l_ns
        )
        return total, {
            "loss_stage": float(l_s.item()),
            "loss_attack": float(l_a.item()),
            "loss_state": float(l_state.item()),
            "loss_next_stage": float(l_ns.item()),
            "loss_total": float(total.item()),
        }


# ---------------------------------------------------------------------------
# V2 Trainer
# ---------------------------------------------------------------------------

class CyberWorldModelTrainerV2:
    """
    Training, validation, and inference wrapper for CyberWorldModelV2.

    Key differences from V1 Trainer:
    - Uses CyberWorldModelV2 (physical state forward simulation).
    - Loss uses physical MSE (L_state) not latent MSE (L_transition).
    - Stores and applies temperature scaling for calibrated inference.
    - rollout() returns structured RolloutResult instead of List[Dict].
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_stages: int = 10,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 60,
        patience: int = 10,
        lambda_stage: float = 1.0,
        lambda_attack: float = 1.0,
        lambda_state: float = 1.0,
        lambda_next_stage: float = 2.0,
        label_smoothing: float = 0.0,
        random_seed: int = 42,
        device: Optional[str] = None,
        temperature: float = 1.0,
    ) -> None:
        self.config: Dict[str, Any] = dict(locals())
        self.config.pop("self")
        self.config.pop("device")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self._set_seed(random_seed)

        self.model = CyberWorldModelV2(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_stages=num_stages,
            dropout=dropout,
        ).to(self.device)

        self.loss_fn: Optional[CyberWorldModelLossV2] = None
        self.is_fitted: bool = False
        self.training_history: List[Dict[str, Any]] = []

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _make_loader(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        y_stage: torch.Tensor,
        y_next_stage: torch.Tensor,
        y_attack: torch.Tensor,
        y_next_state: Optional[torch.Tensor],
        sample_weights: Optional[torch.Tensor],
        shuffle: bool,
    ) -> DataLoader:
        ns = y_next_state if y_next_state is not None else torch.zeros(len(x), self.config["input_dim"])
        sw = sample_weights if sample_weights is not None else torch.ones(len(x))
        return DataLoader(
            TensorDataset(x, mask, y_stage, y_next_stage, y_attack, ns, sw),
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
            drop_last=False,
        )

    def _run_epoch(
        self,
        loader: Optional[DataLoader],
        optimizer: Optional[optim.Optimizer],
    ) -> Dict[str, float]:
        if loader is None:
            return {k: 0.0 for k in ["loss_total", "loss_stage", "loss_attack", "loss_state", "loss_next_stage"]}
        is_train = optimizer is not None
        self.model.train() if is_train else self.model.eval()
        accum = {k: 0.0 for k in ["loss_total", "loss_stage", "loss_attack", "loss_state", "loss_next_stage"]}
        n = 0

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for batch in loader:
                bx, bmask, bstage, bnext_stage, batk, bnext_state, bw = batch
                bx = bx.to(self.device)
                bmask = bmask.to(self.device)
                bstage = bstage.to(self.device)
                bnext_stage = bnext_stage.to(self.device)
                batk = batk.float().to(self.device)
                bnext_state = bnext_state.to(self.device)
                bw = bw.to(self.device)

                if is_train:
                    optimizer.zero_grad()

                out = self.model(bx, bmask)
                loss, comps = self.loss_fn(out, bstage, batk, bnext_stage, bnext_state, bw)

                if is_train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                bs = len(bx)
                n += bs
                for k, v in comps.items():
                    accum[k] += v * bs

        return {k: v / max(n, 1) for k, v in accum.items()}

    def fit(
        self,
        x_train: torch.Tensor,
        mask_train: torch.Tensor,
        y_current_stage_train: torch.Tensor,
        y_next_stage_train: torch.Tensor,
        y_attack_train: torch.Tensor,
        y_next_state_train: Optional[torch.Tensor] = None,
        sample_weights_train: Optional[torch.Tensor] = None,
        x_val: Optional[torch.Tensor] = None,
        mask_val: Optional[torch.Tensor] = None,
        y_current_stage_val: Optional[torch.Tensor] = None,
        y_next_stage_val: Optional[torch.Tensor] = None,
        y_attack_val: Optional[torch.Tensor] = None,
        y_next_state_val: Optional[torch.Tensor] = None,
        stage_class_weights: Optional[torch.Tensor] = None,
    ) -> "CyberWorldModelTrainerV2":
        self._set_seed(self.config["random_seed"])
        if stage_class_weights is not None:
            stage_class_weights = stage_class_weights.to(self.device)

        self.loss_fn = CyberWorldModelLossV2(
            lambda_stage=self.config["lambda_stage"],
            lambda_attack=self.config["lambda_attack"],
            lambda_state=self.config["lambda_state"],
            lambda_next_stage=self.config["lambda_next_stage"],
            label_smoothing=self.config["label_smoothing"],
            stage_class_weights=stage_class_weights,
        )

        has_val = x_val is not None and len(x_val) > 0
        train_loader = self._make_loader(
            x_train, mask_train, y_current_stage_train, y_next_stage_train,
            y_attack_train, y_next_state_train, sample_weights_train, shuffle=True,
        )
        val_loader = (
            self._make_loader(
                x_val, mask_val, y_current_stage_val, y_next_stage_val,
                y_attack_val, y_next_state_val, None, shuffle=False,
            )
            if has_val
            else None
        )

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config["epochs"], eta_min=1e-5
        )

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(self.model.state_dict())
        patience_counter = 0
        self.training_history = []

        for epoch in range(self.config["epochs"]):
            train_losses = self._run_epoch(train_loader, optimizer)
            val_losses = self._run_epoch(val_loader, None)
            scheduler.step()

            self.training_history.append({
                "epoch": epoch + 1,
                "lr": scheduler.get_last_lr()[0],
                "train": train_losses,
                "val": val_losses,
            })

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "V2 Epoch %3d/%d | train=%.4f (s=%.3f a=%.3f st=%.3f ns=%.3f) | val=%.4f",
                    epoch + 1, self.config["epochs"],
                    train_losses["loss_total"], train_losses["loss_stage"],
                    train_losses["loss_attack"], train_losses["loss_state"],
                    train_losses["loss_next_stage"], val_losses["loss_total"],
                )

            ref_loss = val_losses["loss_total"] if has_val else train_losses["loss_total"]
            if ref_loss < best_val_loss - 1e-6:
                best_val_loss = ref_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config["patience"]:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        self.model.load_state_dict(best_weights)
        self.is_fitted = True
        logger.info("CyberWorldModelV2 training complete. Best val_loss=%.4f", best_val_loss)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward(self, x_seq: torch.Tensor, mask: Optional[torch.Tensor]) -> WorldModelOutputV2:
        self.model.eval()
        return self.model(x_seq.to(self.device), mask.to(self.device) if mask is not None else None)

    def predict_current_stage_proba(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """(N, num_stages) softmax distribution for the current stage."""
        return F.softmax(self._forward(x, mask).logits_current_stage, dim=-1).cpu().numpy()

    def predict_next_stage_proba(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_calibration: bool = True,
    ) -> np.ndarray:
        """(N, num_stages) calibrated probability distribution for the next stage."""
        logits = self._forward(x, mask).logits_next_stage
        T = self.temperature if use_calibration else 1.0
        return F.softmax(logits / max(float(T), 1e-4), dim=-1).cpu().numpy()

    def predict_next_stage(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """(N,) integer index of the predicted next stage."""
        return np.argmax(self.predict_next_stage_proba(x, mask, use_calibration=True), axis=1)

    def predict_attack_proba(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """(N,) probability that the next state involves an attack action."""
        return torch.sigmoid(self._forward(x, mask).logits_attack_prob).cpu().numpy()

    def predict_next_state(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """(N, input_dim) predicted next physical state vector in scaled feature space."""
        return self._forward(x, mask).pred_next_state.cpu().numpy()

    def rollout(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, k_steps: int = 4
    ) -> RolloutResult:
        """K-step autoregressive rollout with no future observation leakage."""
        return self.model.rollout(
            x.to(self.device),
            mask.to(self.device) if mask is not None else None,
            k_steps=k_steps,
            temperature=self.temperature,
            apply_bounds=True,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @classmethod
    def load_temperature_from_artifact(cls, artifact_path: Union[str, Path]) -> float:
        """Load validated temperature from calibration artifact JSON."""
        p = Path(artifact_path)
        if not p.exists():
            logger.warning("Calibration artifact not found at %s; using T=1.0", p)
            return 1.0
        with open(p) as f:
            data = json.load(f)
        T = float(data.get("optimal_temperature", 1.0))
        logger.info("Loaded calibration temperature T=%.4f from %s", T, p)
        return T

    def apply_calibration_from_artifact(self, artifact_path: Union[str, Path]) -> None:
        """Set temperature from the validated calibration artifact."""
        self.temperature = self.load_temperature_from_artifact(artifact_path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": self.config,
            "temperature": self.temperature,
            "state_dict": self.model.state_dict(),
            "training_history": self.training_history,
            "model_version": CyberWorldModelV2.MODEL_VERSION,
        }, path)
        logger.info("CyberWorldModelV2 saved to %s", path)

    @classmethod
    def load(
        cls,
        file_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> "CyberWorldModelTrainerV2":
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(**payload["config"], device=device)
        instance.temperature = payload.get("temperature", 1.0)
        instance.model.load_state_dict(payload["state_dict"])
        instance.training_history = payload.get("training_history", [])
        instance.is_fitted = True
        logger.info("CyberWorldModelV2 loaded from %s (model_version=%s)", path, payload.get("model_version"))
        return instance
