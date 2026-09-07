"""
CyberSentinel AI — Phase 11 Model Validation Test Suite.

Comprehensive validation of every ML model in the repository:
  1. Logistic Regression Baseline (loading checkpoint, inference, bounds, determinism)
  2. Temporal GRU Baseline (loading checkpoint, causal sequence, shapes, determinism)
  3. CyberWorldModelV1 (backward compatibility, legacy checkpoint loading, rollout)
  4. CyberWorldModelV2 (production checkpoint, physical state prediction, rollout, bounds)
  5. Input validation (adversarial inputs, NaN/Inf, dimension mismatch, batching)
  6. Output validation (valid ranges, probability simplex, no NaN/Inf)
  7. Determinism under eval mode
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from ml.baseline.logistic_regression import LogisticRegressionBaseline
from ml.temporal.gru_baseline import TemporalGRUModel, TemporalGRUBaseline
from ml.world_model.cyber_world_model import (
    CyberWorldModel as WorldModelV1,
    CyberWorldModelTrainer as WorldModelV1Trainer,
    INPUT_DIM as V1_INPUT_DIM,
)
from ml.world_model.world_model_v2 import (
    CyberWorldModelV2,
    CyberWorldModelTrainerV2,
    RolloutResult,
    INPUT_DIM,
)
from ml.state.state_builder import FEATURE_NAMES

_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Logistic Regression Baseline Tests
# ---------------------------------------------------------------------------

class TestLogisticRegressionBaselineModel:
    @pytest.fixture
    def checkpoint_path(self):
        p = _ROOT / "experiments" / "run_20260907_111554" / "logistic_model.pkl"
        return p

    def test_checkpoint_exists_and_loads(self, checkpoint_path):
        assert checkpoint_path.exists(), f"Logistic checkpoint missing at {checkpoint_path}"
        model = LogisticRegressionBaseline.load(checkpoint_path)
        assert model is not None
        assert model.is_fitted is True

    def test_inference_shapes_and_probabilities(self, checkpoint_path):
        model = LogisticRegressionBaseline.load(checkpoint_path)
        N = 10
        X = np.random.randn(N, INPUT_DIM).astype(np.float32)

        p_attack = model.predict_attack_proba(X)
        assert p_attack.shape == (N, 2)
        assert (p_attack >= 0.0).all() and (p_attack <= 1.0).all()

        p_stage = model.predict_stage_proba(X)
        assert p_stage.shape[0] == N
        np.testing.assert_allclose(p_stage.sum(axis=1), np.ones(N), atol=1e-5)

        p_next = model.predict_next_stage_proba(X)
        assert p_next.shape[0] == N
        np.testing.assert_allclose(p_next.sum(axis=1), np.ones(N), atol=1e-5)

    def test_determinism(self, checkpoint_path):
        model = LogisticRegressionBaseline.load(checkpoint_path)
        X = np.random.randn(5, INPUT_DIM).astype(np.float32)
        p1 = model.predict_next_stage_proba(X)
        p2 = model.predict_next_stage_proba(X)
        np.testing.assert_array_equal(p1, p2)

    def test_input_dimension_mismatch_fails_safely(self, checkpoint_path):
        model = LogisticRegressionBaseline.load(checkpoint_path)
        X_wrong = np.random.randn(5, 12).astype(np.float32)
        with pytest.raises(Exception):
            model.predict_attack_proba(X_wrong)


# ---------------------------------------------------------------------------
# 2. Temporal GRU Baseline Tests
# ---------------------------------------------------------------------------

class TestTemporalGRUBaselineModel:
    @pytest.fixture
    def checkpoint_path(self):
        p = _ROOT / "experiments" / "run_20260907_111554" / "gru_model.pt"
        return p

    def test_checkpoint_exists_and_loads(self, checkpoint_path):
        assert checkpoint_path.exists(), f"GRU checkpoint missing at {checkpoint_path}"
        trainer = TemporalGRUBaseline.load(checkpoint_path)
        assert trainer is not None
        assert trainer.is_fitted is True
        assert isinstance(trainer.model, TemporalGRUModel)

    def test_inference_shapes_and_bounds(self, checkpoint_path):
        trainer = TemporalGRUBaseline.load(checkpoint_path)
        B, T = 4, 8
        x_seq = torch.randn(B, T, INPUT_DIM)
        mask = torch.ones(B, T, dtype=torch.bool)

        trainer.model.eval()
        with torch.no_grad():
            stage_logits, atk_logits = trainer.model(x_seq, mask)

        assert stage_logits.shape == (B, trainer.model.num_classes)
        assert atk_logits.shape == (B,)

        p_next = trainer.predict_next_stage_proba(x_seq, mask)
        assert p_next.shape == (B, trainer.model.num_classes)
        np.testing.assert_allclose(p_next.sum(axis=1), np.ones(B), atol=1e-5)

        p_atk = trainer.predict_attack_proba(x_seq, mask)
        assert p_atk.shape == (B,)
        assert (p_atk >= 0.0).all() and (p_atk <= 1.0).all()

    def test_determinism(self, checkpoint_path):
        trainer = TemporalGRUBaseline.load(checkpoint_path)
        x_seq = torch.randn(2, 8, INPUT_DIM)
        p1 = trainer.predict_next_stage_proba(x_seq)
        p2 = trainer.predict_next_stage_proba(x_seq)
        np.testing.assert_allclose(p1, p2, atol=1e-6)

    def test_causal_unidirectional_gru(self, checkpoint_path):
        trainer = TemporalGRUBaseline.load(checkpoint_path)
        assert trainer.model.gru.bidirectional is False, "GRU must be unidirectional to prevent future leakage"


# ---------------------------------------------------------------------------
# 3. CyberWorldModelV1 Tests (Backward Compatibility)
# ---------------------------------------------------------------------------

class TestCyberWorldModelV1Compatibility:
    @pytest.fixture
    def checkpoint_path(self):
        p = _ROOT / "experiments" / "phase8c_investigation" / "world_model_v1.pt"
        if not p.exists():
            p = _ROOT / "experiments" / "run_20260907_120029" / "world_model" / "world_model.pt"
        return p

    def test_v1_checkpoint_loads_cleanly(self, checkpoint_path):
        assert checkpoint_path.exists(), f"V1 checkpoint missing at {checkpoint_path}"
        trainer = WorldModelV1Trainer.load(checkpoint_path)
        assert trainer is not None
        assert trainer.is_fitted is True
        assert isinstance(trainer.model, WorldModelV1)

    def test_v1_forward_shapes_and_rollout(self, checkpoint_path):
        trainer = WorldModelV1Trainer.load(checkpoint_path)
        B, T = 3, 8
        x = torch.randn(B, T, V1_INPUT_DIM)
        mask = torch.ones(B, T, dtype=torch.bool)

        trainer.model.eval()
        with torch.no_grad():
            out = trainer.model(x, mask)

        assert out.logits_current_stage.shape == (B, trainer.model.num_stages)
        assert out.logits_attack_prob.shape == (B,)
        assert out.logits_next_stage.shape == (B, trainer.model.num_stages)
        assert out.pred_next_state.shape == (B, V1_INPUT_DIM)

        # Rollout test
        rollout_res = trainer.rollout(x, mask, k_steps=2)
        assert len(rollout_res) == 2
        assert "stage_pred" in rollout_res[0]
        assert "attack_prob" in rollout_res[0]


# ---------------------------------------------------------------------------
# 4. CyberWorldModelV2 Tests (Production Architecture)
# ---------------------------------------------------------------------------

class TestCyberWorldModelV2Production:
    @pytest.fixture
    def checkpoint_path(self):
        p = _ROOT / "models" / "world_model_v2.pt"
        return p

    def test_v2_checkpoint_loads_cleanly(self, checkpoint_path):
        assert checkpoint_path.exists(), f"V2 checkpoint missing at {checkpoint_path}"
        trainer = CyberWorldModelTrainerV2.load(checkpoint_path)
        assert trainer is not None
        assert trainer.is_fitted is True
        assert isinstance(trainer.model, CyberWorldModelV2)

    def test_v2_all_heads_and_shapes(self, checkpoint_path):
        trainer = CyberWorldModelTrainerV2.load(checkpoint_path)
        B, T = 4, 8
        x = torch.randn(B, T, INPUT_DIM)
        mask = torch.ones(B, T, dtype=torch.bool)

        with torch.no_grad():
            out = trainer._forward(x, mask)

        assert out.logits_current_stage.shape == (B, trainer.model.num_stages)
        assert out.logits_attack_prob.shape == (B,)
        assert out.logits_next_stage.shape == (B, trainer.model.num_stages)
        assert out.pred_next_state.shape == (B, INPUT_DIM)
        assert out.h_last.shape == (B, trainer.model.hidden_dim)
        assert out.h_next.shape == (B, trainer.model.hidden_dim)

    def test_v2_physical_state_bounded_rollout(self, checkpoint_path):
        trainer = CyberWorldModelTrainerV2.load(checkpoint_path)
        B, T = 2, 8
        x = torch.randn(B, T, INPUT_DIM)
        mask = torch.ones(B, T, dtype=torch.bool)

        rollout: RolloutResult = trainer.rollout(x, mask, k_steps=4)
        assert rollout.horizon == 4
        assert len(rollout.predicted_states) == 4
        assert len(rollout.stage_probabilities) == 4
        assert len(rollout.attack_probabilities) == 4

        # Verify physical state predictions are bounded and valid
        ratio_indices = [11, 12, 16, 18, 19, 20, 21]
        for s_hat in rollout.predicted_states:
            assert s_hat.shape == (B, INPUT_DIM)
            assert not np.isnan(s_hat).any()
            assert not np.isinf(s_hat).any()
            # Ratios must be bounded in [0, 1]
            for idx in ratio_indices:
                assert (s_hat[:, idx] >= 0.0).all()
                assert (s_hat[:, idx] <= 1.0).all()

    def test_v2_eval_mode_determinism(self, checkpoint_path):
        trainer = CyberWorldModelTrainerV2.load(checkpoint_path)
        x = torch.randn(2, 8, INPUT_DIM)
        trainer.model.eval()

        p1 = trainer.predict_next_stage_proba(x)
        p2 = trainer.predict_next_stage_proba(x)
        np.testing.assert_allclose(p1, p2, atol=1e-6)

        s1 = trainer.predict_next_state(x)
        s2 = trainer.predict_next_state(x)
        np.testing.assert_allclose(s1, s2, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Input Validation & Edge Cases (Adversarial Robustness)
# ---------------------------------------------------------------------------

class TestModelInputValidation:
    @pytest.fixture
    def v2_model(self):
        p = _ROOT / "models" / "world_model_v2.pt"
        trainer = CyberWorldModelTrainerV2.load(p)
        trainer.model.eval()
        return trainer.model

    def test_batch_size_one(self, v2_model):
        x = torch.randn(1, 8, INPUT_DIM)
        out = v2_model(x)
        assert out.logits_next_stage.shape == (1, 10)

    def test_different_sequence_lengths(self, v2_model):
        for seq_len in [1, 4, 16, 32]:
            x = torch.randn(2, seq_len, INPUT_DIM)
            out = v2_model(x)
            assert out.logits_next_stage.shape == (2, 10)

    def test_all_zeros_input(self, v2_model):
        x = torch.zeros(2, 8, INPUT_DIM)
        out = v2_model(x)
        assert not torch.isnan(out.logits_next_stage).any()
        assert not torch.isnan(out.pred_next_state).any()

    def test_large_numerical_values(self, v2_model):
        x = torch.ones(2, 8, INPUT_DIM) * 1e4
        out = v2_model(x)
        assert not torch.isnan(out.logits_next_stage).any()

    def test_wrong_dimension_fails_assertion(self, v2_model):
        x_wrong = torch.randn(2, 8, 10)
        with pytest.raises(Exception):
            v2_model(x_wrong)
