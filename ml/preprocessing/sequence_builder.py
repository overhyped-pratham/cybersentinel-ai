"""
CyberSentinel AI - Scenario-Based Temporal Sequence Builder.

Constructs sequence inputs X = [S_{t-T+1}, ..., S_t] and targets:
- Next network state: S_{t+1}
- Current attack stage: y_t
- Next attack stage: y_{t+1}
- Attack probability / indicator: y_{attack, t+1}

CRITICAL RULES:
1. No sequences cross scenario/trace boundaries.
2. Padding is properly handled with binary attention masks (1 for real, 0 for padded).
3. Target S_{t+1} is strictly excluded from input historical context.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from ml.state.state_builder import FEATURE_NAMES


@dataclass
class TemporalBatch:
    """Container for batched sequence data."""
    x_seq: torch.Tensor          # (batch_size, seq_len, feature_dim)
    mask: torch.Tensor           # (batch_size, seq_len) boolean mask: True=valid, False=padding
    y_current_stage: torch.Tensor # (batch_size,) integer class id
    y_next_stage: torch.Tensor    # (batch_size,) integer class id
    y_attack: torch.Tensor        # (batch_size,) float 0.0 or 1.0
    y_next_state: torch.Tensor    # (batch_size, feature_dim) target S_{t+1}
    scenario_ids: List[str]
    sequence_ids: List[str]


class SequenceBuilder:
    """
    Builds temporal sequence samples from state DataFrames,
    strictly grouped by scenario.
    """

    def __init__(
        self,
        sequence_length: int = 8,
        feature_names: Optional[List[str]] = None,
        pad_short_sequences: bool = True,
    ) -> None:
        self.sequence_length = sequence_length
        self.feature_names = list(feature_names or FEATURE_NAMES)
        self.pad_short_sequences = pad_short_sequences

    def build_sequences(
        self,
        df_states: pd.DataFrame,
        scaled_features: Optional[np.ndarray] = None,
    ) -> TemporalBatch:
        """
        Builds sequence arrays from state data.
        
        Args:
            df_states: DataFrame with columns: scenario_id, stage_id, is_attack, and features.
            scaled_features: Pre-scaled numpy array of shape (N, feature_dim).
                             If None, extracts from df_states[self.feature_names].
                             
        Returns:
            TemporalBatch with PyTorch tensors.
        """
        if df_states.empty:
            feature_dim = len(self.feature_names)
            return TemporalBatch(
                x_seq=torch.zeros((0, self.sequence_length, feature_dim), dtype=torch.float32),
                mask=torch.zeros((0, self.sequence_length), dtype=torch.bool),
                y_current_stage=torch.zeros((0,), dtype=torch.long),
                y_next_stage=torch.zeros((0,), dtype=torch.long),
                y_attack=torch.zeros((0,), dtype=torch.float32),
                y_next_state=torch.zeros((0, feature_dim), dtype=torch.float32),
                scenario_ids=[],
                sequence_ids=[],
            )

        if scaled_features is None:
            features = df_states[self.feature_names].to_numpy(dtype=np.float32)
        else:
            features = np.asarray(scaled_features, dtype=np.float32)

        df_work = df_states.copy().reset_index(drop=True)
        df_work["_orig_idx"] = np.arange(len(df_work))

        feature_dim = features.shape[1]
        x_list = []
        mask_list = []
        current_stage_list = []
        next_stage_list = []
        attack_list = []
        next_state_list = []
        sc_id_list = []
        seq_id_list = []

        # Process each scenario strictly in isolation
        for sc_id, group in df_work.groupby("scenario_id", sort=False):
            # Sort group chronologically
            if "timestamp_start" in group.columns:
                group = group.sort_values("timestamp_start").reset_index(drop=True)
            else:
                group = group.reset_index(drop=True)

            indices = group["_orig_idx"].to_numpy()
            n_states = len(indices)

            # Need at least 2 states in scenario to have an S_t and a target S_{t+1}
            if n_states < 2:
                continue

            stage_ids = group["stage_id"].to_numpy(dtype=np.int64) if "stage_id" in group.columns else np.zeros(n_states, dtype=np.int64)
            is_attacks = group["is_attack"].to_numpy(dtype=np.float32) if "is_attack" in group.columns else np.zeros(n_states, dtype=np.float32)

            # Iterate through current time index t: from 0 to n_states - 2
            # Target is at t + 1
            for t in range(n_states - 1):
                # Historical context indices ending at t
                start_hist = t - self.sequence_length + 1

                if start_hist < 0:
                    if not self.pad_short_sequences:
                        continue
                    # Pad left with zeros
                    pad_len = abs(start_hist)
                    valid_len = t + 1
                    
                    hist_indices = indices[0 : t + 1]
                    valid_feats = features[hist_indices] # shape (valid_len, feature_dim)

                    padded_feats = np.zeros((self.sequence_length, feature_dim), dtype=np.float32)
                    padded_feats[pad_len:] = valid_feats

                    mask = np.zeros(self.sequence_length, dtype=bool)
                    mask[pad_len:] = True
                else:
                    hist_indices = indices[start_hist : t + 1]
                    padded_feats = features[hist_indices]
                    mask = np.ones(self.sequence_length, dtype=bool)

                # Target at t + 1
                next_idx = indices[t + 1]
                target_state = features[next_idx]

                x_list.append(padded_feats)
                mask_list.append(mask)
                current_stage_list.append(stage_ids[t])
                next_stage_list.append(stage_ids[t + 1])
                attack_list.append(is_attacks[t + 1])
                next_state_list.append(target_state)
                sc_id_list.append(str(sc_id))
                seq_id_list.append(f"{sc_id}_seq_{t}")

        if not x_list:
            return TemporalBatch(
                x_seq=torch.zeros((0, self.sequence_length, feature_dim), dtype=torch.float32),
                mask=torch.zeros((0, self.sequence_length), dtype=torch.bool),
                y_current_stage=torch.zeros((0,), dtype=torch.long),
                y_next_stage=torch.zeros((0,), dtype=torch.long),
                y_attack=torch.zeros((0,), dtype=torch.float32),
                y_next_state=torch.zeros((0, feature_dim), dtype=torch.float32),
                scenario_ids=[],
                sequence_ids=[],
            )

        return TemporalBatch(
            x_seq=torch.tensor(np.array(x_list), dtype=torch.float32),
            mask=torch.tensor(np.array(mask_list), dtype=torch.bool),
            y_current_stage=torch.tensor(np.array(current_stage_list), dtype=torch.long),
            y_next_stage=torch.tensor(np.array(next_stage_list), dtype=torch.long),
            y_attack=torch.tensor(np.array(attack_list), dtype=torch.float32),
            y_next_state=torch.tensor(np.array(next_state_list), dtype=torch.float32),
            scenario_ids=sc_id_list,
            sequence_ids=seq_id_list,
        )
