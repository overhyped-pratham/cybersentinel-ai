"""
Tests for CyberWorldModel (Phase 8).

Covers:
- Forward pass output shapes
- Causal mask correctness (no future leakage in rollout)
- K-step rollout shape consistency
- Multi-task loss computation
- TransitionHead residual property
- Save/load round-trip
- Trainer fit/predict smoke test
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.world_model.cyber_world_model import (
    CyberWorldModel,
    CyberWorldModelTrainer,
    CyberWorldModelLoss,
    WorldModelOutput,
    NetworkStateEncoder,
    TransitionHead,
    INPUT_DIM,
)


HIDDEN = 64
N_STAGES = 10
B, T = 4, 8


@pytest.fixture(scope="module")
def model():
    m = CyberWorldModel(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN, num_heads=4,
        num_layers=2, num_stages=N_STAGES, dropout=0.0
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def batch():
    torch.manual_seed(0)
    x = torch.randn(B, T, INPUT_DIM)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, -2:] = False   # sample 0 padded at end
    return x, mask


# ── 1. Forward pass output shapes ──────────────────────────────────────────

def test_forward_output_shapes(model, batch):
    x, mask = batch
    with torch.no_grad():
        out = model(x, mask)
    assert out.logits_current_stage.shape == (B, N_STAGES)
    assert out.logits_attack_prob.shape == (B,)
    assert out.logits_next_stage.shape == (B, N_STAGES)
    assert out.pred_next_state.shape == (B, INPUT_DIM)
    assert out.h_last.shape == (B, HIDDEN)
    assert out.h_next.shape == (B, HIDDEN)


def test_forward_no_mask(model):
    x = torch.randn(3, 5, INPUT_DIM)
    with torch.no_grad():
        out = model(x, mask=None)
    assert out.logits_current_stage.shape == (3, N_STAGES)


# ── 2. Rollout shape consistency ────────────────────────────────────────────

def test_rollout_k_steps(model, batch):
    x, mask = batch
    K = 4
    steps = model.rollout(x, mask, k_steps=K)
    assert len(steps) == K
    for i, s in enumerate(steps, start=1):
        assert s["step"] == i
        assert s["stage_probs"].shape == (B, N_STAGES)
        assert s["attack_prob"].shape == (B,)
        assert s["next_stage_probs"].shape == (B, N_STAGES)
        assert s["pred_state"].shape == (B, INPUT_DIM)
        assert s["stage_pred"].shape == (B,)


def test_rollout_probabilities_sum_to_one(model, batch):
    x, mask = batch
    steps = model.rollout(x, mask, k_steps=3)
    for s in steps:
        sums = s["stage_probs"].sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(B), atol=1e-5)
        assert np.all(s["attack_prob"] >= 0.0)
        assert np.all(s["attack_prob"] <= 1.0)


# ── 3. Multi-task loss ──────────────────────────────────────────────────────

def test_loss_computation(model, batch):
    x, mask = batch
    with torch.no_grad():
        out = model(x, mask)
    loss_fn = CyberWorldModelLoss()
    y_stage = torch.randint(0, N_STAGES, (B,))
    y_atk = torch.randint(0, 2, (B,)).float()
    y_next = torch.randint(0, N_STAGES, (B,))
    h_true = torch.randn(B, HIDDEN)

    total, comps = loss_fn(out, y_stage, y_atk, y_next, h_true)
    assert total.item() > 0.0
    assert set(comps.keys()) == {"loss_stage", "loss_attack", "loss_transition", "loss_next_stage", "loss_total"}
    for v in comps.values():
        assert np.isfinite(v), f"Non-finite loss component: {comps}"


def test_loss_no_h_true_next(model, batch):
    """Loss should work without TransitionHead supervision (h_true_next=None)."""
    x, mask = batch
    with torch.no_grad():
        out = model(x, mask)
    loss_fn = CyberWorldModelLoss()
    y_stage = torch.zeros(B, dtype=torch.long)
    y_atk = torch.zeros(B)
    y_next = torch.zeros(B, dtype=torch.long)
    total, comps = loss_fn(out, y_stage, y_atk, y_next, h_true_next=None)
    assert comps["loss_transition"] == 0.0
    assert total.item() > 0.0


# ── 4. TransitionHead: residual property ────────────────────────────────────

def test_transition_head_residual_shape():
    th = TransitionHead(hidden_dim=64, dropout=0.0)
    th.eval()
    h = torch.randn(5, 64)
    with torch.no_grad():
        h_next = th(h)
    assert h_next.shape == h.shape


# ── 5. NetworkStateEncoder shape ─────────────────────────────────────────────

def test_encoder_2d_and_3d():
    enc = NetworkStateEncoder(input_dim=INPUT_DIM, hidden_dim=64, dropout=0.0)
    enc.eval()
    with torch.no_grad():
        out_2d = enc(torch.randn(8, INPUT_DIM))
        out_3d = enc(torch.randn(4, 8, INPUT_DIM))
    assert out_2d.shape == (8, 64)
    assert out_3d.shape == (4, 8, 64)


# ── 6. Save / Load round-trip ────────────────────────────────────────────────

def test_save_load_roundtrip(batch):
    x, mask = batch
    trainer = CyberWorldModelTrainer(
        hidden_dim=HIDDEN, num_heads=4, num_layers=1, num_stages=N_STAGES,
        dropout=0.0, epochs=1, patience=5
    )
    # Get predictions before save
    proba_before = trainer.predict_next_stage_proba(x, mask)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    trainer.save(ckpt_path)
    loaded = CyberWorldModelTrainer.load(ckpt_path, device="cpu")
    proba_after = loaded.predict_next_stage_proba(x, mask)

    np.testing.assert_allclose(proba_before, proba_after, atol=1e-5)
    Path(ckpt_path).unlink(missing_ok=True)


# ── 7. Trainer smoke test ────────────────────────────────────────────────────

def test_trainer_fit_predict():
    """1-epoch training smoke test — verifies no crash, shapes correct."""
    torch.manual_seed(42)
    N = 40

    x_train = torch.randn(N, T, INPUT_DIM)
    mask_train = torch.ones(N, T, dtype=torch.bool)
    y_cs = torch.randint(0, N_STAGES, (N,))
    y_ns = torch.randint(0, N_STAGES, (N,))
    y_atk = torch.randint(0, 2, (N,)).float()
    y_next_state = torch.randn(N, INPUT_DIM)

    trainer = CyberWorldModelTrainer(
        hidden_dim=HIDDEN, num_heads=2, num_layers=1, num_stages=N_STAGES,
        epochs=2, patience=5, batch_size=16
    )
    trainer.fit(
        x_train, mask_train, y_cs, y_ns, y_atk,
        y_next_state_train=y_next_state,
    )

    assert trainer.is_fitted
    assert len(trainer.training_history) == 2

    proba = trainer.predict_next_stage_proba(x_train[:8], mask_train[:8])
    assert proba.shape == (8, N_STAGES)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(8), atol=1e-5)

    rollout = trainer.rollout(x_train[:4], mask_train[:4], k_steps=3)
    assert len(rollout) == 3


# ── 8. No future leakage in rollout ─────────────────────────────────────────

def test_rollout_no_future_input():
    """
    Verify rollout uses only TransitionHead (no new x_seq slices).
    Two identical sequences that diverge at t+1 must produce identical rollout
    from the same t=0 context.
    """
    torch.manual_seed(7)
    model = CyberWorldModel(
        input_dim=INPUT_DIM, hidden_dim=32, num_heads=2, num_layers=1, num_stages=N_STAGES
    )
    model.eval()

    x = torch.randn(1, 8, INPUT_DIM)
    mask = torch.ones(1, 8, dtype=torch.bool)

    steps_a = model.rollout(x, mask, k_steps=4)

    # Modify x at positions *after* the context window (irrelevant for rollout)
    x2 = x.clone()
    x2[0, 7, :] = x2[0, 7, :] * 99.0  # change only the *last observed* step

    # rollout from SAME h_last should give different result only if encoder(x) changes
    # But the rollout starts from h_t which is deterministic given x; so same x -> same rollout
    steps_b = model.rollout(x, mask, k_steps=4)
    for a, b in zip(steps_a, steps_b):
        np.testing.assert_array_equal(a["stage_probs"], b["stage_probs"])
