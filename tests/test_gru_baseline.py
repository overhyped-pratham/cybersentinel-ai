"""
Unit tests for ml.temporal.gru_baseline.TemporalGRUBaseline.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from ml.temporal.gru_baseline import TemporalGRUModel, TemporalGRUBaseline


def test_gru_model_forward():
    batch_size = 4
    seq_len = 8
    input_dim = 24
    num_classes = 10

    model = TemporalGRUModel(
        input_dim=input_dim,
        hidden_dim=32,
        num_layers=1,
        num_classes=num_classes,
        dropout=0.0,
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, :3] = False # first sample is padded for first 3 timesteps

    logits_stage, logits_atk = model(x, mask)

    assert logits_stage.shape == (batch_size, num_classes)
    assert logits_atk.shape == (batch_size,)


def test_gru_baseline_fit_predict(tmp_path: Path):
    batch_size = 20
    seq_len = 6
    input_dim = 24
    num_classes = 5

    x_train = torch.randn(batch_size, seq_len, input_dim)
    mask_train = torch.ones(batch_size, seq_len, dtype=torch.bool)
    y_next_stage_train = torch.randint(0, num_classes, (batch_size,))
    y_attack_train = torch.randint(0, 2, (batch_size,)).float()

    gru_baseline = TemporalGRUBaseline(
        input_dim=input_dim,
        hidden_dim=32,
        num_layers=1,
        num_classes=num_classes,
        epochs=3,
        batch_size=8,
        random_seed=42,
    )

    assert not gru_baseline.is_fitted
    gru_baseline.fit(x_train, mask_train, y_next_stage_train, y_attack_train)
    assert gru_baseline.is_fitted

    # Predictions
    stage_probs = gru_baseline.predict_next_stage_proba(x_train, mask_train)
    assert stage_probs.shape == (batch_size, num_classes)
    assert np.allclose(stage_probs.sum(axis=1), 1.0, atol=1e-5)

    pred_stages = gru_baseline.predict_next_stage(x_train, mask_train)
    assert pred_stages.shape == (batch_size,)

    atk_probs = gru_baseline.predict_attack_proba(x_train, mask_train)
    assert atk_probs.shape == (batch_size,)
    assert (atk_probs >= 0.0).all() and (atk_probs <= 1.0).all()

    # Checkpoint save & load
    ckpt_path = tmp_path / "gru_baseline.pt"
    gru_baseline.save(ckpt_path)
    assert ckpt_path.exists()

    loaded = TemporalGRUBaseline.load(ckpt_path)
    assert loaded.is_fitted
    loaded_probs = loaded.predict_next_stage_proba(x_train, mask_train)
    assert np.allclose(stage_probs, loaded_probs, atol=1e-5)
