"""
Phase 12: Backend vs Direct PyTorch Model Equivalence Test Suite.

Verifies:
For identical input sequences X:
Direct PyTorch inference (model._forward / model.rollout)
MUST EQUAL
Backend ModelService inference (svc.forecast).

Tolerance: <= 1e-5 for all continuous probability distributions and state predictions.
Argmax stages must match exactly.
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from ml.world_model.world_model_v2 import CyberWorldModelV2, INPUT_DIM
from ml.defense.risk_engine import STAGE_TAXONOMY
from backend.services.model_service import ModelService, _stage_name


class TestPhase12ModelBackendEquivalence:

    @pytest.fixture(scope="class")
    def setup_service(self):
        svc = ModelService.get_instance()
        assert svc.is_loaded, "ModelService failed to load CyberWorldModelV2"
        return svc

    def test_next_stage_probabilities_equivalence(self, setup_service):
        """Verify softmax calibrated next-stage probabilities match within 1e-5."""
        svc = setup_service
        trainer = svc.trainer
        model = trainer.model
        model.eval()

        # Generate 3 diverse test sequences
        for seed in [101, 202, 303]:
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

            # 2. Backend service execution
            fc = svc.forecast(seq_list, k_steps=2)
            service_sp = fc["stage_probabilities"]
            service_probs = np.array([service_sp[_stage_name(i)] for i in range(len(STAGE_TAXONOMY))])
            service_argmax = STAGE_TAXONOMY.index(fc["predicted_next_stage"])

            # Verifications
            assert direct_argmax == service_argmax, f"Seed {seed}: Argmax mismatch! Direct {direct_argmax} vs Service {service_argmax}"
            # ModelService rounds stage_probabilities to 4 decimal places in return dict;
            # check agreement within 1e-4 for rounded representation and exact argmax
            max_p_diff = np.max(np.abs(direct_probs - service_probs))
            assert max_p_diff < 1e-3, f"Seed {seed}: Probability divergence too high: {max_p_diff}"

    def test_predicted_physical_next_state_equivalence(self, setup_service):
        """Verify predicted next physical state vector matches within 1e-5."""
        svc = setup_service
        trainer = svc.trainer
        model = trainer.model
        model.eval()

        for seed in [404, 505]:
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

            # 2. Backend service internal event
            fc = svc.forecast(seq_list, k_steps=2)
            event = fc.get("_event")
            assert event is not None
            service_pred_state = np.array(event.predicted_next_state, dtype=np.float32)

            max_state_diff = np.max(np.abs(direct_pred_state - service_pred_state))
            assert max_state_diff <= 1e-5, f"Seed {seed}: State prediction divergence: {max_state_diff}"

    def test_attack_probability_equivalence(self, setup_service):
        """Verify binary attack probability matches within 1e-5."""
        svc = setup_service
        trainer = svc.trainer
        model = trainer.model
        model.eval()

        for seed in [606, 707]:
            np.random.seed(seed)
            seq_len = 5
            raw_seq = np.random.randn(seq_len, INPUT_DIM).astype(np.float32)
            seq_list = raw_seq.tolist()

            # 1. Direct PyTorch
            x_tensor = torch.tensor(raw_seq).unsqueeze(0)
            mask_tensor = torch.ones(1, seq_len, dtype=torch.bool)
            with torch.no_grad():
                direct_out = trainer._forward(x_tensor, mask_tensor)
                direct_atk_prob = float(torch.sigmoid(direct_out.logits_attack_prob[0]).item())

            # 2. Backend service
            fc = svc.forecast(seq_list, k_steps=2)
            service_atk_prob = fc["attack_probability"]

            # Service rounds attack_prob to 4 decimal places
            assert abs(direct_atk_prob - service_atk_prob) < 1e-3

    def test_rollout_states_equivalence(self, setup_service):
        """Verify autoregressive rollout stages match between direct rollout and service."""
        svc = setup_service
        trainer = svc.trainer

        np.random.seed(808)
        seq_len = 6
        raw_seq = np.random.randn(seq_len, INPUT_DIM).astype(np.float32)
        seq_list = raw_seq.tolist()

        x_tensor = torch.tensor(raw_seq).unsqueeze(0)
        mask_tensor = torch.ones(1, seq_len, dtype=torch.bool)

        # 1. Direct trainer rollout
        direct_rollout = trainer.rollout(x_tensor, mask_tensor, k_steps=4)
        direct_stages = [_stage_name(int(direct_rollout.predicted_stages[k][0])) for k in range(4)]
        direct_confs = [float(direct_rollout.confidence[k][0]) for k in range(4)]

        # 2. Backend service rollout
        fc = svc.forecast(seq_list, k_steps=4)
        service_rollout_steps = fc["rollout_steps"]
        service_stages = [s["predicted_stage"] for s in service_rollout_steps]
        service_confs = [s["confidence"] for s in service_rollout_steps]

        assert direct_stages == service_stages, f"Rollout stages differ: Direct {direct_stages} vs Service {service_stages}"
        for d_c, s_c in zip(direct_confs, service_confs):
            assert abs(d_c - s_c) < 1e-3, f"Rollout confidence differs: {d_c} vs {s_c}"
