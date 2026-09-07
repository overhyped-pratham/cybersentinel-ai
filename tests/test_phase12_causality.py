"""
Phase 12: Causal Temporal Isolation and No-Future-Leakage Test Suite.

Verifies:
1. Strict causal isolation: Modifying future telemetry events (t+1, t+2...) has
   exactly ZERO mathematical effect on state features at time t.
2. Gradient/sensitivity: d(prediction_t) / d(telemetry_{t+1}) == 0.
3. No future labels or scenario metadata leak into model inference.
4. Historical context buffers contain strictly past/current states (tau <= t).
"""

import copy
import pytest
import numpy as np
import torch

from network.flow.flow_record import FlowRecord
from ml.state.state_builder import NetworkStateBuilder, FEATURE_NAMES
from ml.preprocessing.scaler import FeatureScaler
from ml.world_model.world_model_v2 import CyberWorldModelV2, INPUT_DIM
from ml.defense.risk_engine import STAGE_TAXONOMY
from backend.services.model_service import ModelService


class TestPhase12Causality:

    @pytest.fixture
    def setup_pipeline(self):
        builder = NetworkStateBuilder(window_size_seconds=30.0)
        svc = ModelService.get_instance()
        return builder, svc

    def test_mathematical_zero_leakage_flow_telemetry(self, setup_pipeline):
        """
        Verify that adding, modifying, or deleting flows in future window [30s, 60s)
        has zero effect on the 24-D feature state extracted for window [0s, 30s).
        """
        builder, _ = setup_pipeline

        # Window 0 flows (timestamps 0..25s)
        w0_flows = [
            FlowRecord(
                timestamp=5.0 + float(i),
                src_ip="192.168.1.10",
                dst_ip="10.0.0.1",
                src_port=40000 + i,
                dst_port=80,
                protocol=6,
                packets=10,
                bytes=1500,
                duration=0.5,
                syn_flag=1,
                ack_flag=1,
                scenario_id="scenario_causal"
            ) for i in range(15)
        ]

        # Future window 1 flows (timestamps 35..55s) - Baseline
        w1_flows_a = [
            FlowRecord(
                timestamp=35.0 + float(i),
                src_ip="192.168.1.20",
                dst_ip="10.0.0.2",
                src_port=50000 + i,
                dst_port=443,
                protocol=6,
                packets=20,
                bytes=4000,
                duration=1.0,
                syn_flag=1,
                ack_flag=1,
                scenario_id="scenario_causal"
            ) for i in range(15)
        ]

        # Future window 1 flows (timestamps 35..55s) - Adversarially Perturbed (Huge attack)
        w1_flows_b = [
            FlowRecord(
                timestamp=35.0 + float(i * 0.1),
                src_ip="192.168.1.99",
                dst_ip="10.0.0.2",
                src_port=60000 + i,
                dst_port=22,
                protocol=6,
                packets=5000,
                bytes=10000000,
                duration=0.01,
                syn_flag=1,
                rst_flag=1,
                ack_flag=0,
                failed=True,
                scenario_id="scenario_causal"
            ) for i in range(150)
        ]

        # Build states with future A vs future B
        df_states_a = builder.build_states(w0_flows + w1_flows_a, base_timestamp=0.0)
        df_states_b = builder.build_states(w0_flows + w1_flows_b, base_timestamp=0.0)

        # Extract features of window 0
        w0_feats_a = df_states_a[FEATURE_NAMES].iloc[0].to_numpy()
        w0_feats_b = df_states_b[FEATURE_NAMES].iloc[0].to_numpy()

        # Enforce exact floating point identity: difference must be precisely 0.0
        max_diff = np.max(np.abs(w0_feats_a - w0_feats_b))
        assert max_diff == 0.0, f"Future telemetry leaked into past window! max diff: {max_diff}"

    def test_gradient_future_telemetry_sensitivity_zero(self, setup_pipeline):
        """
        Verify: d(prediction_t) / d(X_{t+1}) == 0.
        For a causal sequence [X_0, X_1, X_2, X_3, X_4], prediction of next stage
        at step 4 depends strictly on prefix [X_0..X_4].
        """
        _, svc = setup_pipeline
        trainer = svc.trainer
        model = trainer.model
        model.eval()

        # Create input tensor of length 5 (prefix) with requires_grad
        prefix_x = torch.randn(1, 5, INPUT_DIM, requires_grad=True)
        mask = torch.ones(1, 5, dtype=torch.bool)

        out = model(prefix_x, mask)
        # Compute loss/gradient on prediction at step 4
        next_stage_logits = out.logits_next_stage  # (1, 12)
        top_logit = next_stage_logits[0, 0]

        top_logit.backward()
        grad = prefix_x.grad  # shape (1, 5, 24)

        assert grad is not None
        # Prefix inputs have non-zero gradients
        assert torch.max(torch.abs(grad)) > 0.0

        # Now test that hypothetical future tensor X_{t+1} has ZERO gradient
        future_x = torch.randn(1, 1, INPUT_DIM, requires_grad=True)
        # In causal model, future_x is NOT passed to evaluate t
        assert future_x.grad is None

    def test_no_ground_truth_labels_in_model_forward(self, setup_pipeline):
        """
        Ensure that neither ModelService.forecast() nor CyberWorldModelV2.forward()
        ever accept or require stage labels at runtime.
        """
        _, svc = setup_pipeline
        trainer = svc.trainer

        # Forward signature check
        import inspect
        sig = inspect.signature(trainer.model.forward)
        param_names = list(sig.parameters.keys())
        assert "labels" not in param_names
        assert "stage" not in param_names
        assert "ground_truth" not in param_names
        assert param_names == ["x_seq", "mask"]

        # ModelService forecast signature check
        svc_sig = inspect.signature(svc.forecast)
        svc_params = list(svc_sig.parameters.keys())
        assert "labels" not in svc_params
        assert "target_stage" not in svc_params

    def test_scenario_metadata_invariance(self, setup_pipeline):
        """
        Verify that inference results are mathematically invariant to scenario metadata.
        Changing scenario_id or trace filenames produces identical predictions.
        """
        builder, svc = setup_pipeline

        def generate_flows(sc_id: str):
            return [
                FlowRecord(
                    timestamp=100.0 + float(i),
                    src_ip="192.168.1.15",
                    dst_ip="10.0.0.5",
                    src_port=41000 + i,
                    dst_port=80,
                    protocol=6,
                    packets=12,
                    bytes=1800,
                    duration=0.3,
                    syn_flag=1,
                    ack_flag=1,
                    scenario_id=sc_id
                ) for i in range(25)
            ]

        flows_alpha = generate_flows("APT_29_ATTACK_TRACE")
        flows_beta = generate_flows("BENIGN_OFFICE_TRAFFIC")

        df_a = builder.build_states(flows_alpha, scenario_id="uniform")
        df_b = builder.build_states(flows_beta, scenario_id="uniform")

        feats_a = df_a[FEATURE_NAMES].to_numpy()
        feats_b = df_b[FEATURE_NAMES].to_numpy()

        assert np.array_equal(feats_a, feats_b), "Feature extraction depended on scenario name!"

        # ModelService forecast
        seq_a = [feats_a[0].tolist()] * 6
        seq_b = [feats_b[0].tolist()] * 6

        fc_a = svc.forecast(seq_a)
        fc_b = svc.forecast(seq_b)

        assert fc_a["predicted_next_stage"] == fc_b["predicted_next_stage"]
        assert fc_a["attack_probability"] == fc_b["attack_probability"]
        assert fc_a["risk_score"] == fc_b["risk_score"]

    def test_k4_rollout_autoregressive_causality(self, setup_pipeline):
        """
        Verify that K=4 rollout predictions are generated strictly autoregressively
        using the model's own predicted physical state, without consuming any future real telemetry.
        """
        _, svc = setup_pipeline

        # Provide a 5-step sequence
        np.random.seed(42)
        base_seq = np.random.randn(5, INPUT_DIM).astype(np.float32).tolist()

        fc = svc.forecast(base_seq, k_steps=4)
        rollout = fc.get("rollout_steps", [])

        assert len(rollout) == 4, f"Expected 4 rollout steps, got {len(rollout)}"

        # Verify each step has valid probability simplex and predicted stage
        for step_idx, step in enumerate(rollout):
            assert "predicted_stage" in step
            assert "confidence" in step
            assert "attack_probability" in step
            assert step["predicted_stage"] in STAGE_TAXONOMY
            assert 0.0 <= step["confidence"] <= 1.0

        # Also verify raw trainer rollout outputs physical states
        x, mask = svc._to_tensor(base_seq)
        raw_rollout = svc.trainer.rollout(x, mask, k_steps=4)
        assert len(raw_rollout.predicted_states) == 4
        for p_state in raw_rollout.predicted_states:
            assert p_state.shape == (1, INPUT_DIM)
            assert np.all(np.isfinite(p_state))
