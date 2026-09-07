"""
CyberSentinel AI - Temporal GRU Sequential Baseline.

Consumes a temporal history of network states [S_{t-7}, ..., S_t]
and forecasts the attacker's next stage (y_{t+1}) and attack probability.

Serves as the empirical temporal benchmark to verify whether sequence
history improves forecasting over static point-in-time classification.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ml.state.state_builder import FEATURE_NAMES

logger = logging.getLogger(__name__)


class TemporalGRUModel(nn.Module):
    """
    PyTorch GRU sequence model with dual forecasting heads:
    1. Next attack stage distribution (Multi-class CrossEntropy)
    2. Next attack binary probability (Binary CrossEntropy)
    """

    def __init__(
        self,
        input_dim: int = 24,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Input feature projection
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # Causal GRU (unidirectional to prevent future leakage)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Forecasting heads
        self.head_next_stage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.head_next_attack = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x_seq: (batch_size, seq_len, input_dim)
            mask: Optional (batch_size, seq_len) boolean mask: True=valid, False=pad
            
        Returns:
            Tuple[logits_next_stage, logits_next_attack]
        """
        batch_size, seq_len, _ = x_seq.shape

        # Linear projection
        h = self.input_fc(x_seq)

        # Apply mask if provided by zeroing out padded input states
        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()
            h = h * mask_float

        # GRU forward
        gru_out, _ = self.gru(h) # (batch_size, seq_len, hidden_dim)

        # Extract representation from the last valid timestep for each sample
        if mask is not None:
            # Find the index of the last True value in each row
            # If all are True, last index is seq_len - 1
            lengths = mask.sum(dim=1).clamp(min=1) - 1 # (batch_size,)
            idx = lengths.view(-1, 1, 1).expand(-1, 1, self.hidden_dim)
            last_h = gru_out.gather(1, idx).squeeze(1) # (batch_size, hidden_dim)
        else:
            last_h = gru_out[:, -1, :]

        norm_h = self.dropout(self.layer_norm(last_h))

        logits_stage = self.head_next_stage(norm_h)
        logits_attack = self.head_next_attack(norm_h).squeeze(-1)

        return logits_stage, logits_attack


class TemporalGRUBaseline:
    """
    Wrapper handling configuration, training, validation, early stopping,
    and inference for TemporalGRUModel.
    """

    def __init__(
        self,
        input_dim: int = 24,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 40,
        patience: int = 8,
        random_seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_classes": num_classes,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "random_seed": random_seed,
        }
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed(random_seed)

        self.model = TemporalGRUModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        ).to(self.device)

        self.is_fitted = False
        self.training_history: List[Dict[str, float]] = []

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def fit(
        self,
        x_train: torch.Tensor,
        mask_train: torch.Tensor,
        y_next_stage_train: torch.Tensor,
        y_attack_train: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        mask_val: Optional[torch.Tensor] = None,
        y_next_stage_val: Optional[torch.Tensor] = None,
        y_attack_val: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> "TemporalGRUBaseline":
        """
        Trains the GRU model with validation monitoring and early stopping.
        """
        self._set_seed(self.config["random_seed"])

        train_ds = TensorDataset(x_train, mask_train, y_next_stage_train, y_attack_train)
        train_loader = DataLoader(train_ds, batch_size=self.config["batch_size"], shuffle=True)

        has_val = (x_val is not None and len(x_val) > 0)
        if has_val:
            val_ds = TensorDataset(x_val, mask_val, y_next_stage_val, y_attack_val)
            val_loader = DataLoader(val_ds, batch_size=self.config["batch_size"], shuffle=False)

        # Loss functions
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        criterion_stage = nn.CrossEntropyLoss(weight=class_weights)
        criterion_attack = nn.BCEWithLogitsLoss()

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        self.training_history = []

        for epoch in range(self.config["epochs"]):
            self.model.train()
            train_loss_total = 0.0
            n_train_samples = 0

            for b_x, b_mask, b_stage, b_atk in train_loader:
                b_x = b_x.to(self.device)
                b_mask = b_mask.to(self.device)
                b_stage = b_stage.to(self.device)
                b_atk = b_atk.to(self.device)

                optimizer.zero_grad()
                logits_stage, logits_atk = self.model(b_x, b_mask)

                loss_stage = criterion_stage(logits_stage, b_stage)
                loss_atk = criterion_attack(logits_atk, b_atk)
                loss = loss_stage + 0.5 * loss_atk

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_total += loss.item() * len(b_x)
                n_train_samples += len(b_x)

            epoch_train_loss = train_loss_total / max(n_train_samples, 1)

            # Validation
            epoch_val_loss = epoch_train_loss
            if has_val:
                self.model.eval()
                val_loss_total = 0.0
                n_val_samples = 0
                with torch.no_grad():
                    for b_x, b_mask, b_stage, b_atk in val_loader:
                        b_x = b_x.to(self.device)
                        b_mask = b_mask.to(self.device)
                        b_stage = b_stage.to(self.device)
                        b_atk = b_atk.to(self.device)

                        l_stage, l_atk = self.model(b_x, b_mask)
                        loss_s = criterion_stage(l_stage, b_stage)
                        loss_a = criterion_attack(l_atk, b_atk)
                        v_loss = loss_s + 0.5 * loss_a

                        val_loss_total += v_loss.item() * len(b_x)
                        n_val_samples += len(b_x)
                epoch_val_loss = val_loss_total / max(n_val_samples, 1)

            self.training_history.append({
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
            })

            # Early stopping check
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config["patience"]:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Load best model weights
        self.model.load_state_dict(best_weights)
        self.is_fitted = True
        return self

    def predict_next_stage_proba(
        self,
        x_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Forecasts next-stage probabilities (N, num_classes)."""
        self.model.eval()
        with torch.no_grad():
            x_seq = x_seq.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)
            logits_stage, _ = self.model(x_seq, mask)
            probs = torch.softmax(logits_stage, dim=-1).cpu().numpy()
        return probs

    def predict_next_stage(
        self,
        x_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Forecasts most likely next-stage ID (N,)."""
        probs = self.predict_next_stage_proba(x_seq, mask)
        return np.argmax(probs, axis=1)

    def predict_attack_proba(
        self,
        x_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Forecasts future attack probability (N,)."""
        self.model.eval()
        with torch.no_grad():
            x_seq = x_seq.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)
            _, logits_atk = self.model(x_seq, mask)
            probs = torch.sigmoid(logits_atk).cpu().numpy()
        return probs

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves model checkpoint and configuration to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "state_dict": self.model.state_dict(),
            "training_history": self.training_history,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, file_path: Union[str, Path], device: Optional[str] = None) -> "TemporalGRUBaseline":
        """Loads model checkpoint from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"GRU checkpoint file not found: {path}")

        payload = torch.load(path, map_location="cpu")
        instance = cls(**payload["config"], device=device)
        instance.model.load_state_dict(payload["state_dict"])
        instance.training_history = payload.get("training_history", [])
        instance.is_fitted = True
        return instance
