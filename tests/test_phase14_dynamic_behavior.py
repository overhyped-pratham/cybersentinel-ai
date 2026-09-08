"""
Phase 14: Dynamic Behavior Test Suite.

Verifies:
1. Multi-host traffic patterns generate distinct, non-identical 24-D physical feature states.
2. Pairwise distances between pattern states are strictly positive.
3. ModelService / CyberWorldModelV2 inference outputs vary dynamically based on physical telemetry.
4. Physical feature deltas and explainability attribute the shifts dynamically without hardcoding.
5. K-step rollouts diverge across patterns based on their distinct initial trajectories.
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.multi_host_traffic_generator import (
    generate_pattern_flows,
    TRAFFIC_PATTERNS,
    SECONDARY_LAPTOP_IP,
    CYBERSENTINEL_HOST_IP,
)
from ml.state.state_builder import NetworkStateBuilder, FEATURE_NAMES
from ml.preprocessing.scaler import FeatureScaler
from backend.services.model_service import ModelService

_ROOT = Path(__file__).resolve().parent.parent


class TestPhase14DynamicBehavior:

    @pytest.fixture(scope="class")
    def model_service(self):
        svc = ModelService.get_instance()
        assert svc.is_loaded, "ModelService failed to load CyberWorldModelV2"
        return svc

    @pytest.fixture(scope="class")
    def scaler(self):
        paths = [
            _ROOT / "models" / "scaler.pkl",
            _ROOT / "experiments" / "run_20260907_111554" / "scaler.pkl",
        ]
        for p in paths:
            if p.exists():
                return FeatureScaler.load(p)
        pytest.fail("FeatureScaler artifact not found in models/ or experiments/")

    def test_24d_physical_feature_divergence(self):
        """Verify that all 5 traffic patterns produce distinct 24-D physical states."""
        builder = NetworkStateBuilder()
        states = {}

        base_ts = 2000.0
        for pattern in TRAFFIC_PATTERNS:
            flows = generate_pattern_flows(
                pattern,
                base_timestamp=base_ts,
                duration_seconds=30.0,
                source_ip=SECONDARY_LAPTOP_IP,
                target_ip=CYBERSENTINEL_HOST_IP,
                scenario_id=f"dyn_test_{pattern}",
            )
            df = builder.build_states(flows, scenario_id=f"dyn_test_{pattern}", base_timestamp=base_ts)
            assert not df.empty, f"State builder produced empty DataFrame for {pattern}"
            vec = [float(df[col].iloc[-1]) for col in FEATURE_NAMES]
            assert len(vec) == 24
            assert not any(math.isnan(x) or math.isinf(x) for x in vec), f"NaN/Inf in state for {pattern}"
            states[pattern] = np.array(vec, dtype=np.float32)

        # 1. Check pairwise Euclidean distance between all pattern pairs
        pattern_list = list(TRAFFIC_PATTERNS)
        for i in range(len(pattern_list)):
            for j in range(i + 1, len(pattern_list)):
                p1 = pattern_list[i]
                p2 = pattern_list[j]
                dist = float(np.linalg.norm(states[p1] - states[p2]))
                assert dist > 1.0, f"Pair ({p1}, {p2}) feature states are suspiciously close: L2 dist = {dist}"

        # 2. Pattern-specific physical validations
        tot_bytes_idx = FEATURE_NAMES.index("tot_bytes_sec") if "tot_bytes_sec" in FEATURE_NAMES else 1
        exfil_bytes = states["large_data_transfer"][tot_bytes_idx]
        normal_bytes = states["normal_background"][tot_bytes_idx]
        assert exfil_bytes > normal_bytes * 5, (
            f"large_data_transfer bytes ({exfil_bytes}) not significantly greater than normal ({normal_bytes})"
        )

        # Connection burst must have elevated flow count / rate
        count_idx = FEATURE_NAMES.index("flow_count") if "flow_count" in FEATURE_NAMES else 0
        burst_flows = states["connection_burst"][count_idx]
        normal_flows = states["normal_background"][count_idx]
        assert burst_flows > normal_flows, (
            f"connection_burst flows ({burst_flows}) not greater than normal ({normal_flows})"
        )

    def test_model_predictions_dynamic_across_patterns(self, model_service, scaler):
        """Verify model output probabilities and predictions dynamically respond to pattern states."""
        builder = NetworkStateBuilder()
        predictions = {}

        base_ts = 3000.0
        for pattern in TRAFFIC_PATTERNS:
            # Build sequence of 4 windows for this pattern and scale
            seq = []
            for w in range(4):
                w_start = base_ts + (w * 30.0)
                flows = generate_pattern_flows(
                    pattern,
                    base_timestamp=w_start,
                    duration_seconds=30.0,
                    source_ip=SECONDARY_LAPTOP_IP,
                    target_ip=CYBERSENTINEL_HOST_IP,
                    scenario_id=f"seq_{pattern}",
                )
                df = builder.build_states(flows, scenario_id=f"seq_{pattern}", base_timestamp=w_start)
                scaled_df = scaler.transform(df)
                seq.append(scaled_df[-1].tolist())

            fc = model_service.forecast(seq, k_steps=3)
            assert "predicted_next_stage" in fc
            assert "stage_probabilities" in fc
            assert "attack_probability" in fc
            predictions[pattern] = fc

        # Check that stage probability vectors are not all identical
        prob_vectors = []
        for pattern, fc in predictions.items():
            probs = list(fc["stage_probabilities"].values())
            prob_vectors.append(probs)

        prob_matrix = np.array(prob_vectors)
        # Standard deviation across patterns should be positive for probability distribution
        stds = np.std(prob_matrix, axis=0)
        assert np.max(stds) > 1e-3, "Predictions across different patterns are static/identical!"

        # Specific pattern checks
        # large_data_transfer has high attack probability and predicts EXFILTRATION
        assert predictions["large_data_transfer"]["predicted_next_stage"] == "EXFILTRATION"
        assert predictions["large_data_transfer"]["attack_probability"] > 0.90

    def test_physical_feature_deltas_explainability(self, model_service, scaler):
        """Verify feature deltas attribute physical differences dynamically."""
        builder = NetworkStateBuilder()

        # Build sequence for large_data_transfer
        exfil_seq = []
        for w in range(4):
            w_start = 4000.0 + (w * 30.0)
            flows = generate_pattern_flows(
                "large_data_transfer",
                base_timestamp=w_start,
                duration_seconds=30.0,
            )
            df = builder.build_states(flows, scenario_id="exfil_expl", base_timestamp=w_start)
            scaled = scaler.transform(df)
            exfil_seq.append(scaled[-1].tolist())

        fc = model_service.forecast(exfil_seq, k_steps=2)
        assert "top_features" in fc
        top_features = fc["top_features"]
        assert len(top_features) > 0

        # Physical feature deltas must be structured and finite
        for feat in top_features:
            assert "feature" in feat
            assert "current" in feat
            assert "predicted" in feat
            assert "abs_change" in feat
            assert "direction" in feat
            assert not math.isnan(feat["abs_change"])
            assert not math.isinf(feat["abs_change"])

        # Top feature for large data transfer should highlight volume/rate
        top_feat_names = [f["feature"] for f in top_features]
        assert any("byte" in name or "pkt" in name or "total" in name for name in top_feat_names)

    def test_k_step_rollout_trajectory_divergence(self, model_service, scaler):
        """Verify K-step rollout trajectories diverge across different traffic patterns."""
        builder = NetworkStateBuilder()
        rollouts = {}

        for pattern in ["normal_background", "large_data_transfer"]:
            seq = []
            for w in range(4):
                w_start = 5000.0 + (w * 30.0)
                flows = generate_pattern_flows(
                    pattern,
                    base_timestamp=w_start,
                    duration_seconds=30.0,
                )
                df = builder.build_states(flows, scenario_id=f"roll_{pattern}", base_timestamp=w_start)
                scaled = scaler.transform(df)
                seq.append(scaled[-1].tolist())

            fc = model_service.forecast(seq, k_steps=3)
            rollouts[pattern] = fc.get("rollout_steps", [])

        assert len(rollouts["normal_background"]) == 3
        assert len(rollouts["large_data_transfer"]) == 3

        # Trajectory stage probabilities or attack probabilities must differ across patterns
        norm_atk = [step["attack_probability"] for step in rollouts["normal_background"]]
        exfil_atk = [step["attack_probability"] for step in rollouts["large_data_transfer"]]
        assert norm_atk != exfil_atk

    def test_input_perturbation_causal_response(self, model_service):
        """Verify that modifying a physical feature in input causes a dynamic change in output."""
        base_seq = [[0.1 * i + 0.01 * j for j in range(24)] for i in range(4)]

        fc_base = model_service.forecast(base_seq, k_steps=1)
        base_event = fc_base.get("_event")
        assert base_event is not None
        base_pred_state = base_event.predicted_next_state

        # Perturb one physical feature dramatically (e.g. feature index 0 by 10.0)
        perturbed_seq = [row[:] for row in base_seq]
        perturbed_seq[-1][0] += 10.0

        fc_perturbed = model_service.forecast(perturbed_seq, k_steps=1)
        perturbed_event = fc_perturbed.get("_event")
        assert perturbed_event is not None
        perturbed_pred_state = perturbed_event.predicted_next_state

        # The predicted state must dynamically respond to the perturbation
        diff = np.abs(np.array(base_pred_state) - np.array(perturbed_pred_state))
        assert np.max(diff) > 1e-4, "Model output was unaffected by input feature perturbation!"
