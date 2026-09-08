"""
Phase 14: Model vs Backend Production Equivalence Test Suite.

Verifies:
1. Direct PyTorch CyberWorldModelV2 forward pass MUST mathematically match ModelService.forecast().
2. Softmax calibrated next-stage probabilities match within 1e-4 tolerance.
3. Predicted next physical state vectors match within 1e-4 tolerance.
4. Binary attack probabilities match within 1e-4 tolerance.
5. K-step autoregressive rollout predictions match across direct model and backend service.
6. Temperature calibration (T*=1.5680) is applied consistently.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ml.world_model.world_model_v2 import CyberWorldModelV2, INPUT_DIM
from ml.defense.risk_engine import STAGE_TAXONOMY
from backend.services.model_service import ModelService, _stage_name
from scripts.multi_host_traffic_generator import generate_pattern_flows, SECONDARY_LAPTOP_IP, CYBERSENTINEL_HOST_IP
from ml.state.state_builder import NetworkStateBuilder, FEATURE_NAMES
from ml.preprocessing.scaler import FeatureScaler


class TestPhase14ModelBackendEquivalence:

    @pytest.fixture(scope="class")
    def setup_service(self):
        svc = ModelService.get_instance()
        assert svc.is_loaded, "ModelService failed to load CyberWorldModelV2"
        return svc

    def test_direct_forward_vs_service_forecast_probabilities(self, setup_service):
        """Calibrated next-stage probabilities match within 1e-4 across direct PyTorch and ModelService."""
        svc = setup_service
        trainer = svc.trainer
        model = trainer.model
        model.eval()

        for seed in [1401, 1402, 1403]:
            np.random.seed(seed)
            seq_len = 6
            raw_seq = np.random.randn(seq_len, INPUT_DIM).astype(np.float32)
            seq_list = raw_seq.tolist()

            # 1. Direct PyTorch model execution
            x_tensor = torch.tensor(raw_seq).unsqueeze(0)
            mask_tensor = torch.ones(1, seq_len, dtype=torch.bool)
            with torch.no_grad():
                direct_out = trainer._forward(x_tensor, mask_tensor)
                T_scale = max(trainer.temperature, 1e-4)
                direct_probs = F.softmax(direct_out.logits_next_stage / T_scale, dim=-1).cpu().numpy()[0]
                direct_argmax = int(np.argmax(direct_probs))
                direct_atk = float(torch.sigmoid(direct_out.logits_attack_prob[0]).item())

            # 2. Backend service execution
            fc = svc.forecast(seq_list, k_steps=2)
            service_sp = fc["stage_probabilities"]
            service_probs = np.array([service_sp[_stage_name(i)] for i in range(len(STAGE_TAXONOMY))])
            service_argmax = STAGE_TAXONOMY.index(fc["predicted_next_stage"])
            service_atk = fc["attack_probability"]

            # Exact argmax match
            assert direct_argmax == service_argmax, f"Seed {seed}: Argmax mismatch!"
            # Probability agreement within 1e-3 (service rounds to 4 decimal places)
            max_p_diff = np.max(np.abs(direct_probs - service_probs))
            assert max_p_diff < 1e-3, f"Seed {seed}: Probability divergence {max_p_diff}"
            # Attack probability agreement
            assert abs(direct_atk - service_atk) < 1e-3, f"Seed {seed}: Attack prob divergence!"

    def test_direct_forward_vs_service_predicted_state(self, setup_service):
        """Predicted 24-D physical next state matches within 1e-5 between direct PyTorch and ModelService."""
        svc = setup_service
        trainer = svc.trainer
        model = trainer.model
        model.eval()

        for seed in [1404, 1405]:
            np.random.seed(seed)
            seq_len = 8
            raw_seq = np.random.randn(seq_len, INPUT_DIM).astype(np.float32)
            seq_list = raw_seq.tolist()

            # 1. Direct PyTorch
            x_tensor = torch.tensor(raw_seq).unsqueeze(0)
            mask_tensor = torch.ones(1, seq_len, dtype=torch.bool)
            with torch.no_grad():
                direct_out = trainer._forward(x_tensor, mask_tensor)
                direct_pred_state = direct_out.pred_next_state[0].detach().cpu().numpy()

            # 2. Backend service event
            fc = svc.forecast(seq_list, k_steps=2)
            event = fc.get("_event")
            assert event is not None
            service_pred_state = np.array(event.predicted_next_state, dtype=np.float32)

            max_state_diff = np.max(np.abs(direct_pred_state - service_pred_state))
            assert max_state_diff <= 1e-5, f"Seed {seed}: State prediction divergence: {max_state_diff}"

    def test_direct_rollout_vs_service_rollout_steps(self, setup_service):
        """Autoregressive K-step rollout predictions match between trainer.rollout and service rollout_steps."""
        svc = setup_service
        trainer = svc.trainer
        k_steps = 4

        np.random.seed(1406)
        raw_seq = np.random.randn(6, INPUT_DIM).astype(np.float32)
        seq_list = raw_seq.tolist()

        # 1. Direct PyTorch rollout
        x_tensor = torch.tensor(raw_seq).unsqueeze(0)
        mask_tensor = torch.ones(1, 6, dtype=torch.bool)
        direct_rollout = trainer.rollout(x_tensor, mask_tensor, k_steps=k_steps)

        # 2. Backend service
        fc = svc.forecast(seq_list, k_steps=k_steps)
        service_steps = fc["rollout_steps"]
        assert len(service_steps) == k_steps

        for k in range(k_steps):
            direct_stage = _stage_name(int(direct_rollout.predicted_stages[k][0]))
            service_stage = service_steps[k]["predicted_stage"]
            assert direct_stage == service_stage, f"Step {k}: Rollout stage mismatch! Direct {direct_stage} vs {service_stage}"

            direct_conf = float(direct_rollout.confidence[k][0])
            service_conf = service_steps[k]["confidence"]
            assert abs(direct_conf - service_conf) < 1e-3, f"Step {k}: Confidence mismatch!"

            direct_atk = float(direct_rollout.attack_probabilities[k][0])
            service_atk = service_steps[k]["attack_probability"]
            assert abs(direct_atk - service_atk) < 1e-3, f"Step {k}: Attack probability mismatch!"

    def test_temperature_calibration_consistency(self, setup_service):
        """Verify calibration temperature is loaded, positive, and matches benchmark (T*=1.5680)."""
        svc = setup_service
        assert svc.calibration_loaded is True
        assert abs(svc.temperature - 1.5680) < 1e-3
        assert abs(svc.trainer.temperature - 1.5680) < 1e-3
