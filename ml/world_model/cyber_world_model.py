"""
CyberSentinel AI - Cyber World Model (Phase 8).

Core innovation: a Temporal Cyber World Model that learns how network states
evolve over time and performs K-step forward simulation to forecast the
attacker's likely next stages.

Mathematical formulation:
    S_t -> P(S_t+1 | S_t)            (one-step transition)
    S_t -> S_t+1 -> ... -> S_t+K     (K-step rollout)

Architecture:
    NetworkStateEncoder    : MLP  S_t -> h_t
    TemporalTransformer    : Causal self-attention [h_t-7, ..., h_t] -> ctx_t
    TransitionHead         : h_t -> h_hat_t+1   (world model core - enables rollout)
    CurrentStageHead       : h_t -> P(stage_t)
    AttackProbabilityHead  : h_t -> P(attack_t)
    NextStageHead          : h_next -> P(stage_t+1)
    NextStateHead          : h_next -> S_hat_t+1  (raw feature reconstruction)

Multi-task loss:
    L = lam1*L_stage + lam2*L_attack + lam3*L_transition + lam4*L_next_stage

The TransitionHead is what fundamentally distinguishes this from a classifier:
it predicts a latent representation of the next state, enabling the system to
answer "What is likely to happen next?" across multiple steps without new input.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ml.state.state_builder import FEATURE_NAMES

logger = logging.getLogger(__name__)

INPUT_DIM: int = len(FEATURE_NAMES)  # 24


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

class WorldModelOutput(NamedTuple):
    """All predictions from a single forward pass."""
    logits_current_stage: torch.Tensor   # (B, num_stages)
    logits_attack_prob: torch.Tensor     # (B,)
    logits_next_stage: torch.Tensor      # (B, num_stages)
    pred_next_state: torch.Tensor        # (B, input_dim)
    h_last: torch.Tensor                 # (B, hidden_dim) - current latent
    h_next: torch.Tensor                 # (B, hidden_dim) - predicted next latent


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class NetworkStateEncoder(nn.Module):
    """MLP encoder: S_t -> h_t.  Two-layer GELU MLP with LayerNorm."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., input_dim)  ->  (..., hidden_dim)"""
        return self.layer_norm(self.net(x))


class TemporalTransformer(nn.Module):
    """
    Causal multi-head self-attention over the encoded state sequence.
    Pre-LN (norm_first=True) for training stability.
    Positional embeddings are learned.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        h_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_seq: (B, T, hidden_dim)
            mask:  (B, T) bool - True=valid, False=padding
        Returns:
            context_seq: (B, T, hidden_dim)
            h_last:      (B, hidden_dim) - last valid token
        """
        B, T, _ = h_seq.shape
        positions = torch.arange(T, device=h_seq.device).unsqueeze(0)
        h_seq = h_seq + self.pos_embedding(positions)

        # Base causal mask: (T, T), upper triangular = -inf
        causal = torch.triu(torch.full((T, T), float("-inf"), device=h_seq.device), diagonal=1)

        if mask is not None:
            # Build combined (B, T, T) mask that accounts for both causal ordering and padding
            combined = causal.unsqueeze(0).repeat(B, 1, 1)
            # Mask out key positions that are padding for each sequence
            pad_mask = ~mask  # True = padding
            for b in range(B):
                combined[b, :, pad_mask[b]] = float("-inf")
                # For padding query rows where all positions become -inf, allow self-attention
                # to prevent NaN in softmax (softmax over all -inf produces NaN)
                for i in range(T):
                    if combined[b, i].isneginf().all():
                        combined[b, i, i] = 0.0

            # Expand for multi-head attention: (B * num_heads, T, T)
            attn_mask = combined.repeat_interleave(self.num_heads, dim=0)
            ctx = self.transformer(h_seq, mask=attn_mask)
        else:
            ctx = self.transformer(h_seq, mask=causal)

        # In SequenceBuilder, short sequences are left-padded with zeros (right-aligned).
        # Therefore, the last valid timestep S_t is ALWAYS at the final sequence position index -1.
        h_last = ctx[:, -1, :]

        return ctx, h_last


class TransitionHead(nn.Module):
    """
    World model core: h_t -> h_hat_t+1

    Residual MLP. Enables K-step rollout:
        h_hat_t+1 = TransitionHead(h_t)
        h_hat_t+2 = TransitionHead(h_hat_t+1)
        ...
    Supervised by MSE against true h_t+1 = Encoder(S_t+1).
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.norm(h + self.net(h))


class NextStateHead(nn.Module):
    """Decodes latent back to raw feature space: h -> S_hat."""

    def __init__(self, hidden_dim: int = 128, output_dim: int = INPUT_DIM, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class ClassificationHead(nn.Module):
    """Generic MLP classification head: h -> logits."""

    def __init__(self, hidden_dim: int = 128, num_classes: int = 10, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CyberWorldModel(nn.Module):
    """
    CyberSentinel AI Temporal Cyber World Model.

    Answers the four product questions:
        1. What is happening now?         -> head_current_stage(h_t)
        2. Is this an attack?             -> head_attack_prob(h_t)
        3. What will likely happen next?  -> head_next_stage(h_next)
        4. Why does the model believe it? -> head_next_state(h_next) decodes features

    K-step rollout via iterative TransitionHead application:
        h_t -> h_t+1 -> h_t+2 -> ... -> h_t+K   (no new sensor input needed)
    """

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
        self.transition_head = TransitionHead(hidden_dim, dropout=0.2)

        self.head_current_stage = ClassificationHead(hidden_dim, num_stages, dropout)
        self.head_attack_prob = ClassificationHead(hidden_dim, 1, dropout)
        self.head_next_stage = ClassificationHead(hidden_dim, num_stages, dropout)
        self.head_next_state = NextStateHead(hidden_dim, input_dim, dropout)

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
    ) -> WorldModelOutput:
        """
        Args:
            x_seq: (B, T, input_dim)
            mask:  (B, T) bool - True=valid
        Returns:
            WorldModelOutput
        """
        h_seq = self.encoder(x_seq)                        # (B, T, hidden_dim)
        _, h_last = self.transformer(h_seq, mask=mask)     # (B, hidden_dim)
        h_next = self.transition_head(h_last)              # (B, hidden_dim)

        return WorldModelOutput(
            logits_current_stage=self.head_current_stage(h_last),
            logits_attack_prob=self.head_attack_prob(h_last).squeeze(-1),
            logits_next_stage=self.head_next_stage(h_next),
            pred_next_state=self.head_next_state(h_next),
            h_last=h_last,
            h_next=h_next,
        )

    @torch.no_grad()
    def rollout(
        self,
        x_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        k_steps: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        K-step forward simulation from the last observed state.

        h_t+1 = Transition(h_t)
        h_t+2 = Transition(h_t+1)
        ...
        Each h_t+k is decoded to stage distribution, attack probability,
        and predicted raw feature state - no new sensor data needed.

        Args:
            x_seq:   (B, T, input_dim)
            mask:    (B, T) bool
            k_steps: forecast horizon

        Returns:
            List of k_steps dicts with keys:
                step, stage_probs, attack_prob, next_stage_probs,
                pred_state, stage_pred
        """
        self.eval()
        h_seq = self.encoder(x_seq)
        _, h_current = self.transformer(h_seq, mask=mask)

        steps = []
        for step in range(1, k_steps + 1):
            h_pred = self.transition_head(h_current)
            steps.append({
                "step": step,
                "stage_probs": F.softmax(self.head_current_stage(h_pred), dim=-1).cpu().numpy(),
                "attack_prob": torch.sigmoid(self.head_attack_prob(h_pred).squeeze(-1)).cpu().numpy(),
                "next_stage_probs": F.softmax(self.head_next_stage(h_pred), dim=-1).cpu().numpy(),
                "pred_state": self.head_next_state(h_pred).cpu().numpy(),
                "stage_pred": torch.argmax(self.head_current_stage(h_pred), dim=-1).cpu().numpy(),
            })
            h_current = h_pred

        return steps


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------

class CyberWorldModelLoss(nn.Module):
    """
    L = lam1*L_stage + lam2*L_attack + lam3*L_transition + lam4*L_next_stage

    L_stage      = CrossEntropy(current_stage_logits, y_current_stage)
    L_attack     = BCEWithLogits(attack_logits,        y_attack)
    L_transition = MSE(h_next, Encoder(S_t+1))
    L_next_stage = CrossEntropy(next_stage_logits,     y_next_stage)
    """

    def __init__(
        self,
        lambda_stage: float = 1.0,
        lambda_attack: float = 1.0,
        lambda_transition: float = 1.0,
        lambda_next_stage: float = 1.0,
        label_smoothing: float = 0.1,
        stage_class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.lam = (lambda_stage, lambda_attack, lambda_transition, lambda_next_stage)
        self.ce_stage = nn.CrossEntropyLoss(weight=stage_class_weights, label_smoothing=label_smoothing)
        self.ce_next = nn.CrossEntropyLoss(weight=stage_class_weights, label_smoothing=label_smoothing)
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        out: WorldModelOutput,
        y_stage: torch.Tensor,           # (B,) long
        y_attack: torch.Tensor,          # (B,) float
        y_next_stage: torch.Tensor,      # (B,) long
        h_true_next: Optional[torch.Tensor] = None,  # (B, hidden_dim)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_s = self.ce_stage(out.logits_current_stage, y_stage)
        l_a = self.bce(out.logits_attack_prob, y_attack)
        l_ns = self.ce_next(out.logits_next_stage, y_next_stage)
        l_t = self.mse(out.h_next, h_true_next.detach()) if h_true_next is not None               else out.h_next.new_tensor(0.0)

        total = self.lam[0]*l_s + self.lam[1]*l_a + self.lam[2]*l_t + self.lam[3]*l_ns
        return total, {
            "loss_stage": l_s.item(), "loss_attack": l_a.item(),
            "loss_transition": l_t.item(), "loss_next_stage": l_ns.item(),
            "loss_total": total.item(),
        }


# ---------------------------------------------------------------------------
# Training wrapper
# ---------------------------------------------------------------------------

class CyberWorldModelTrainer:
    """
    Full training/validation/inference wrapper for CyberWorldModel.

    - Multi-task loss with TransitionHead supervision
    - AdamW + CosineAnnealingLR
    - Early stopping (patience configurable)
    - Gradient clipping (max_norm=1.0)
    - Save/load full checkpoint
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
        lambda_transition: float = 1.0,
        lambda_next_stage: float = 1.0,
        label_smoothing: float = 0.1,
        random_seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        self.config: Dict[str, Any] = dict(locals())
        self.config.pop("self")
        self.config.pop("device")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed(random_seed)

        self.model = CyberWorldModel(
            input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads,
            num_layers=num_layers, num_stages=num_stages, dropout=dropout,
        ).to(self.device)

        self.loss_fn: Optional[CyberWorldModelLoss] = None
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
        shuffle: bool,
    ) -> DataLoader:
        tensors = [x, mask, y_stage, y_next_stage, y_attack]
        if y_next_state is not None:
            tensors.append(y_next_state)
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
            drop_last=False,
        )

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: Optional[optim.Optimizer],
        has_next_state: bool,
    ) -> Dict[str, float]:
        is_train = optimizer is not None
        self.model.train() if is_train else self.model.eval()
        accum = {k: 0.0 for k in ["loss_total", "loss_stage", "loss_attack", "loss_transition", "loss_next_stage"]}
        n = 0

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for batch in loader:
                bx, bmask, bstage, bnext_stage, batk = batch[:5]
                bnext_state = batch[5] if has_next_state else None

                bx = bx.to(self.device)
                bmask = bmask.to(self.device)
                bstage = bstage.to(self.device)
                bnext_stage = bnext_stage.to(self.device)
                batk = batk.float().to(self.device)

                h_true_next = None
                if bnext_state is not None:
                    with torch.no_grad():
                        h_true_next = self.model.encoder(bnext_state.to(self.device))

                if is_train:
                    optimizer.zero_grad()

                out = self.model(bx, bmask)
                loss, components = self.loss_fn(out, bstage, batk, bnext_stage, h_true_next)

                if is_train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                bs = len(bx)
                n += bs
                for k, v in components.items():
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
        x_val: Optional[torch.Tensor] = None,
        mask_val: Optional[torch.Tensor] = None,
        y_current_stage_val: Optional[torch.Tensor] = None,
        y_next_stage_val: Optional[torch.Tensor] = None,
        y_attack_val: Optional[torch.Tensor] = None,
        y_next_state_val: Optional[torch.Tensor] = None,
        stage_class_weights: Optional[torch.Tensor] = None,
    ) -> "CyberWorldModelTrainer":
        self._set_seed(self.config["random_seed"])

        if stage_class_weights is not None:
            stage_class_weights = stage_class_weights.to(self.device)

        self.loss_fn = CyberWorldModelLoss(
            lambda_stage=self.config["lambda_stage"],
            lambda_attack=self.config["lambda_attack"],
            lambda_transition=self.config["lambda_transition"],
            lambda_next_stage=self.config["lambda_next_stage"],
            label_smoothing=self.config["label_smoothing"],
            stage_class_weights=stage_class_weights,
        )

        has_ns_train = y_next_state_train is not None
        has_ns_val = y_next_state_val is not None
        has_val = x_val is not None and len(x_val) > 0

        train_loader = self._make_loader(
            x_train, mask_train, y_current_stage_train, y_next_stage_train, y_attack_train,
            y_next_state_train, shuffle=True
        )
        val_loader = None
        if has_val:
            val_loader = self._make_loader(
                x_val, mask_val, y_current_stage_val, y_next_stage_val, y_attack_val,
                y_next_state_val, shuffle=False
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
            train_losses = self._run_epoch(train_loader, optimizer, has_ns_train)
            val_losses = self._run_epoch(val_loader, None, has_ns_val) if has_val else train_losses
            scheduler.step()

            self.training_history.append({
                "epoch": epoch + 1,
                "lr": scheduler.get_last_lr()[0],
                "train": train_losses,
                "val": val_losses,
            })

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "Epoch %3d/%d | train=%.4f (s=%.3f a=%.3f t=%.3f ns=%.3f) | val=%.4f",
                    epoch + 1, self.config["epochs"],
                    train_losses["loss_total"], train_losses["loss_stage"],
                    train_losses["loss_attack"], train_losses["loss_transition"],
                    train_losses["loss_next_stage"], val_losses["loss_total"],
                )

            if val_losses["loss_total"] < best_val_loss - 1e-6:
                best_val_loss = val_losses["loss_total"]
                best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config["patience"]:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        self.model.load_state_dict(best_weights)
        self.is_fitted = True
        logger.info("Training complete. Best val_loss=%.4f", best_val_loss)
        return self

    # ---------- Inference ----------

    @torch.no_grad()
    def _forward(self, x_seq: torch.Tensor, mask: Optional[torch.Tensor]) -> WorldModelOutput:
        self.model.eval()
        x_seq = x_seq.to(self.device)
        mask = mask.to(self.device) if mask is not None else None
        return self.model(x_seq, mask)

    def predict_current_stage_proba(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """(N, num_stages) softmax distribution for the current stage."""
        return F.softmax(self._forward(x, mask).logits_current_stage, dim=-1).cpu().numpy()

    def predict_next_stage_proba(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """(N, num_stages) softmax distribution for the next stage."""
        return F.softmax(self._forward(x, mask).logits_next_stage, dim=-1).cpu().numpy()

    def predict_attack_proba(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """(N,) sigmoid probability that the next state is an attack."""
        return torch.sigmoid(self._forward(x, mask).logits_attack_prob).cpu().numpy()

    def rollout(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, k_steps: int = 4) -> List[Dict[str, Any]]:
        """K-step forward simulation. Returns list of step dicts."""
        x = x.to(self.device)
        mask = mask.to(self.device) if mask is not None else None
        return self.model.rollout(x, mask, k_steps=k_steps)

    # ---------- Persistence ----------

    def save(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": self.config,
            "state_dict": self.model.state_dict(),
            "training_history": self.training_history,
        }, path)
        logger.info("CyberWorldModel saved to %s", path)

    @classmethod
    def load(cls, file_path: Union[str, Path], device: Optional[str] = None) -> "CyberWorldModelTrainer":
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(**payload["config"], device=device)
        instance.model.load_state_dict(payload["state_dict"])
        instance.training_history = payload.get("training_history", [])
        instance.is_fitted = True
        logger.info("CyberWorldModel loaded from %s", path)
        return instance
