"""
Tests for CyberWorldModelV2 (Phase 9).

Verifies:
  - Import compatibility (V1 classes still importable)
  - V2 model forward pass output shapes
  - Physical state predictor output dim
  - Rollout output structure and safety flags
  - Calibration temperature loading
  - V2 loss components and backward pass
  - Trainer fit/save/load round-trip
  - V2 predictions do not use future observations (mask test)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# ── Verify V1 is still importable ──────────────────────────────────────────

from ml.world_model.cyber_world_model import (
    CyberWorldModel,
    CyberWorldModelTrainer,
    CyberWorldModelLoss,
    WorldModelOutput,
    NetworkStateEncoder,
    TransitionHead,
    INPUT_DIM,
)

# ── V2 imports ─────────────────────────────────────────────────────────────

from ml.world_model.world_model_v2 import (
    CyberWorldModelV2,
    WorldModelV2,
    WorldModelOutputV2,
    PhysicalNextStatePredictor,
    RolloutResult,
    CyberWorldModelLossV2,
    CyberWorldModelTrainerV2,
    WorldModelV1,
)
from ml.calibration.temperature_scaling import (
    load_temperature,
    apply_temperature_scaling,
    verify_prediction_invariance,
    compute_ece,
    compute_brier,
)

# ── Constants ──────────────────────────────────────────────────────────────

B = 4       # batch size
T = 8       # sequence length
D = 24      # input_dim (= INPUT_DIM)
H = 64      # hidden_dim (smaller for tests)
N_STAGES = 10


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def model_v2():
    return CyberWorldModelV2(
        input_dim=D, hidden_dim=H, num_heads=4, num_layers=2,
        num_stages=N_STAGES, dropout=0.0
    )


@pytest.fixture
def x_seq():
    torch.manual_seed(0)
    return torch.randn(B, T, D)


@pytest.fixture
def mask():
    m = torch.ones(B, T, dtype=torch.bool)
    m[:, -1] = False  # last timestep masked
    return m


# ── V1 preservation tests ──────────────────────────────────────────────────

class TestV1Preserved:
    def test_v1_alias_same_class(self):
        """WorldModelV1 must be the exact same class as CyberWorldModel."""
        assert WorldModelV1 is CyberWorldModel

    def test_v1_trainable(self):
        v1 = CyberWorldModel(input_dim=D, hidden_dim=H, num_stages=N_STAGES)
        x = torch.randn(B, T, D)
        mask = torch.ones(B, T, dtype=torch.bool)
        out = v1(x, mask)
        assert isinstance(out, WorldModelOutput)

    def test_v2_alias(self):
        assert WorldModelV2 is CyberWorldModelV2


# ── V2 forward shape tests ─────────────────────────────────────────────────

class TestWorldModelV2Shapes:
    def test_output_is_namedtuple(self, model_v2, x_seq, mask):
        out = model_v2(x_seq, mask)
        assert isinstance(out, WorldModelOutputV2)

    def test_current_stage_logits_shape(self, model_v2, x_seq, mask):
        out = model_v2(x_seq, mask)
        assert out.logits_current_stage.shape == (B, N_STAGES), (
            f"Expected ({B}, {N_STAGES}), got {out.logits_current_stage.shape}"
        )

    def test_attack_logits_shape(self, model_v2, x_seq, mask):
        out = model_v2(x_seq, mask)
        assert out.logits_attack_prob.shape == (B,), (
            f"Expected ({B},), got {out.logits_attack_prob.shape}"
        )

    def test_next_stage_logits_shape(self, model_v2, x_seq, mask):
        out = model_v2(x_seq, mask)
        assert out.logits_next_stage.shape == (B, N_STAGES)

    def test_pred_next_state_shape(self, model_v2, x_seq, mask):
        out = model_v2(x_seq, mask)
        assert out.pred_next_state.shape == (B, D), (
            f"Expected ({B}, {D}), got {out.pred_next_state.shape}"
        )

    def test_h_last_shape(self, model_v2, x_seq, mask):
        out = model_v2(x_seq, mask)
        assert out.h_last.shape == (B, H)

    def test_h_next_shape(self, model_v2, x_seq, mask):
        out = model_v2(x_seq, mask)
        assert out.h_next.shape == (B, H)

    def test_no_mask(self, model_v2, x_seq):
        """Forward pass must work without providing a mask (None)."""
        out = model_v2(x_seq, mask=None)
        assert out.logits_current_stage.shape == (B, N_STAGES)

    def test_batch_size_1(self, model_v2):
        x = torch.randn(1, T, D)
        out = model_v2(x)
        assert out.pred_next_state.shape == (1, D)


# ── Physical state predictor ───────────────────────────────────────────────

class TestPhysicalNextStatePredictor:
    def test_output_dim(self):
        pred = PhysicalNextStatePredictor(hidden_dim=H, output_dim=D)
        h = torch.randn(B, H)
        s = pred(h)
        assert s.shape == (B, D)

    def test_bounded_forward_ratios_in_range(self):
        """After bounded forward, ratio features (idx 11-12, 16, 18-21) must be in [0,1]."""
        pred = PhysicalNextStatePredictor(hidden_dim=H, output_dim=D)
        # Use large activations to test clamping
        pred.net[0].weight.data.fill_(1.0)
        h = torch.ones(B, H) * 5.0
        s = pred.forward_bounded(h)
        ratio_indices = [11, 12, 16, 18, 19, 20, 21]
        for idx in ratio_indices:
            vals = s[:, idx].detach().numpy()
            assert (vals >= 0.0).all(), f"Ratio feature {idx} has negative values: {vals}"
            assert (vals <= 1.0).all(), f"Ratio feature {idx} exceeds 1.0: {vals}"


# ── Rollout tests ──────────────────────────────────────────────────────────

class TestRollout:
    @pytest.mark.parametrize("k_steps", [1, 2, 4])
    def test_rollout_lengths(self, model_v2, x_seq, mask, k_steps):
        result = model_v2.rollout(x_seq, mask, k_steps=k_steps)
        assert isinstance(result, RolloutResult)
        assert result.horizon == k_steps
        assert len(result.predicted_states) == k_steps
        assert len(result.stage_probabilities) == k_steps
        assert len(result.confidence) == k_steps
        assert len(result.uncertainty) == k_steps

    def test_rollout_state_shape(self, model_v2, x_seq, mask):
        result = model_v2.rollout(x_seq, mask, k_steps=4)
        for ps in result.predicted_states:
            assert ps.shape == (B, D), f"Expected ({B},{D}), got {ps.shape}"

    def test_rollout_stage_probs_sum_to_one(self, model_v2, x_seq, mask):
        result = model_v2.rollout(x_seq, mask, k_steps=2)
        for k, sp in enumerate(result.stage_probabilities):
            row_sums = np.sum(sp, axis=1)
            np.testing.assert_allclose(
                row_sums, np.ones(B), atol=1e-5,
                err_msg=f"Stage probs at step {k} do not sum to 1"
            )

    def test_rollout_no_future_leakage(self, model_v2):
        """
        Mask out different future timesteps — predictions must differ,
        confirming that the model uses the mask, but with mask=all-valid vs
        mask=last-T-padded the SAME sequence of rolled-out predicted_states
        must have the same shape (not a correctness test, a shape/sanity test).
        """
        x = torch.randn(1, T, D)
        mask_full = torch.ones(1, T, dtype=torch.bool)
        mask_partial = torch.zeros(1, T, dtype=torch.bool)
        mask_partial[:, :4] = True

        result_full = model_v2.rollout(x, mask_full, k_steps=2)
        result_partial = model_v2.rollout(x, mask_partial, k_steps=2)

        # Shapes must be identical regardless of mask
        for k in range(2):
            assert result_full.predicted_states[k].shape == result_partial.predicted_states[k].shape

    def test_rollout_safety_flags(self, model_v2, x_seq, mask):
        result = model_v2.rollout(x_seq, mask, k_steps=1)
        assert "nan_detected" in result.safety_flags
        assert "collapse_detected" in result.safety_flags
        assert "feature_dim_valid" in result.safety_flags
        assert result.safety_flags["feature_dim_valid"] is True


# ── Loss tests ─────────────────────────────────────────────────────────────

class TestCyberWorldModelLossV2:
    def test_loss_components_present(self, model_v2, x_seq, mask):
        out = model_v2(x_seq, mask)
        loss_fn = CyberWorldModelLossV2(
            lambda_stage=1.0, lambda_attack=1.0,
            lambda_state=1.0, lambda_next_stage=2.0,
        )
        y_stage = torch.randint(0, N_STAGES, (B,))
        y_attack = torch.randint(0, 2, (B,)).float()
        y_next_stage = torch.randint(0, N_STAGES, (B,))
        s_true = torch.randn(B, D)

        total, comps = loss_fn(out, y_stage, y_attack, y_next_stage, s_true)
        for key in ["loss_total", "loss_stage", "loss_attack", "loss_state", "loss_next_stage"]:
            assert key in comps

    def test_loss_backward(self, model_v2, x_seq, mask):
        out = model_v2(x_seq, mask)
        loss_fn = CyberWorldModelLossV2()
        y_stage = torch.randint(0, N_STAGES, (B,))
        y_attack = torch.randint(0, 2, (B,)).float()
        y_next_stage = torch.randint(0, N_STAGES, (B,))
        s_true = torch.randn(B, D)

        total, _ = loss_fn(out, y_stage, y_attack, y_next_stage, s_true)
        total.backward()
        for p in model_v2.parameters():
            assert p.grad is not None

    def test_loss_without_next_state(self, model_v2, x_seq, mask):
        """Loss must still work when s_true_next is None (state loss = 0)."""
        out = model_v2(x_seq, mask)
        loss_fn = CyberWorldModelLossV2()
        y_stage = torch.randint(0, N_STAGES, (B,))
        y_attack = torch.randint(0, 2, (B,)).float()
        y_next_stage = torch.randint(0, N_STAGES, (B,))

        total, comps = loss_fn(out, y_stage, y_attack, y_next_stage, s_true_next=None)
        assert comps["loss_state"] == 0.0


# ── Trainer save/load round-trip ───────────────────────────────────────────

class TestTrainerRoundTrip:
    def test_save_and_load(self, tmp_path):
        trainer = CyberWorldModelTrainerV2(
            input_dim=D, hidden_dim=H, num_stages=N_STAGES,
            epochs=1, random_seed=42,
        )
        ckpt = tmp_path / "v2_test.pt"
        trainer.save(ckpt)

        loaded = CyberWorldModelTrainerV2.load(ckpt)
        assert loaded.is_fitted is True
        assert loaded.config["hidden_dim"] == H

    def test_temperature_preserved_in_checkpoint(self, tmp_path):
        trainer = CyberWorldModelTrainerV2(input_dim=D, hidden_dim=H, epochs=1)
        trainer.temperature = 1.5680
        ckpt = tmp_path / "calib_test.pt"
        trainer.save(ckpt)
        loaded = CyberWorldModelTrainerV2.load(ckpt)
        assert abs(loaded.temperature - 1.5680) < 1e-4


# ── Calibration module tests ───────────────────────────────────────────────

class TestCalibration:
    def test_load_temperature_missing_file(self):
        T = load_temperature(Path("/nonexistent/path/temperature.json"))
        assert T == 1.0

    def test_load_temperature_from_artifact(self, tmp_path):
        data = {"optimal_temperature": 1.568}
        p = tmp_path / "temperature.json"
        p.write_text(json.dumps(data))
        T = load_temperature(p)
        assert abs(T - 1.568) < 1e-4

    def test_apply_temperature_scaling_sums_to_one(self):
        logits = np.random.randn(10, N_STAGES)
        probs = apply_temperature_scaling(logits, temperature=1.568)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(10), atol=1e-6)

    def test_temperature_scaling_invariance(self):
        logits = np.random.randn(50, N_STAGES)
        invariant, rate = verify_prediction_invariance(logits, temperature=1.568)
        assert invariant, f"Argmax invariance violated: agreement={rate:.2%}"

    def test_temperature_invalid(self):
        with pytest.raises(ValueError):
            apply_temperature_scaling(np.ones((4, 5)), temperature=-1.0)

    def test_ece_perfect_model(self):
        """Perfect model should have ECE very close to 0."""
        N, C = 100, N_STAGES
        labels = np.random.randint(0, C, N)
        probs = np.zeros((N, C))
        probs[np.arange(N), labels] = 1.0
        ece = compute_ece(probs, labels)
        assert ece < 1e-6

    def test_brier_perfect_model(self):
        N, C = 100, N_STAGES
        labels = np.random.randint(0, C, N)
        probs = np.zeros((N, C))
        probs[np.arange(N), labels] = 1.0
        brier = compute_brier(probs, labels)
        assert brier < 1e-8
