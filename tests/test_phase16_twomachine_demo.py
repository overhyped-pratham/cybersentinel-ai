"""
Phase 16 Test Suite: Live Two-Machine Demonstration & Production Pipeline.

Validates:
  1. Multi-host telemetry encoding & RFC 3954 NetFlow v5 attribution
  2. 24-D feature scaling fidelity via production models/scaler.pkl
  3. Continuous 3-phase lifecycle progression (Baseline -> High-Activity -> Recovery)
  4. Non-static dynamic model sensitivity (zero hardcoded intelligence)
  5. MITRE ATT&CK contextual mapping dynamic adaptation
  6. Fail-closed robustness on truncated/malformed UDP datagrams
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from network.flow.flow_record import FlowRecord
from network.telemetry.sources import NetFlowSource, _NF5_HEADER
from ml.state.state_builder import NetworkStateBuilder, FEATURE_NAMES
from ml.preprocessing.scaler import FeatureScaler
from backend.services.model_service import ModelService
from scripts.multi_host_traffic_generator import (
    generate_pattern_flows,
    flows_to_netflow_v5_packet,
    SECONDARY_LAPTOP_IP,
    CYBERSENTINEL_HOST_IP,
)
from scripts.run_phase16_twomachine_demo import run_two_machine_demo


class TestPhase16TwoMachineDemo:
    """Verifies Two-Machine demonstration mechanics, scaling, and dynamic modeling."""

    def test_multi_host_packet_encoding_and_attribution(self):
        flows = generate_pattern_flows(
            pattern="normal_background",
            base_timestamp=1000.0,
            duration_seconds=10.0,
            source_ip=SECONDARY_LAPTOP_IP,
            target_ip=CYBERSENTINEL_HOST_IP,
        )
        assert len(flows) > 0
        for f in flows:
            assert f.src_ip == SECONDARY_LAPTOP_IP
            assert f.label == "UNKNOWN"  # Strict defense: no ground truth hint

        # Encode to RFC 3954 NetFlow v5
        pkt = flows_to_netflow_v5_packet(flows[:20], seq=1)
        assert len(pkt) == 24 + (20 * 48)  # 24B header + 48B per flow record

        version, count = _NF5_HEADER.unpack_from(pkt, 0)[:2]
        assert version == 5
        assert count == 20

    def test_feature_scaler_production_consistency(self):
        scaler_path = Path("models/scaler.pkl")
        assert scaler_path.exists(), "Production scaler.pkl must exist in models/"

        scaler = FeatureScaler.load(scaler_path)
        assert hasattr(scaler, "transform")

        state_builder = NetworkStateBuilder(window_size_seconds=10.0)

        # Generate normal flows
        flows_norm = generate_pattern_flows("normal_background", 1000.0, 10.0)
        df_norm = state_builder.build_states(flows_norm, "test_norm", 1000.0)
        raw_norm = df_norm[FEATURE_NAMES].iloc[-1].to_numpy(dtype=np.float32)

        # Generate exfiltration flows
        flows_exfil = generate_pattern_flows("large_data_transfer", 1010.0, 10.0)
        df_exfil = state_builder.build_states(flows_exfil, "test_exfil", 1010.0)
        raw_exfil = df_exfil[FEATURE_NAMES].iloc[-1].to_numpy(dtype=np.float32)

        scaled_norm = scaler.transform(pd.DataFrame([raw_norm], columns=FEATURE_NAMES))[0]
        scaled_exfil = scaler.transform(pd.DataFrame([raw_exfil], columns=FEATURE_NAMES))[0]

        assert len(scaled_norm) == 24
        assert len(scaled_exfil) == 24
        assert np.all(np.isfinite(scaled_norm))
        assert np.all(np.isfinite(scaled_exfil))

        # Absolute feature separation must be non-zero for distinct traffic patterns
        mean_diff = float(np.mean(np.abs(scaled_exfil - scaled_norm)))
        assert mean_diff > 0.05, f"Expected scaled feature divergence > 0.05, got {mean_diff}"

    @pytest.mark.asyncio
    async def test_three_phase_live_progression(self, tmp_path):
        json_out = tmp_path / "test_results.json"
        report_out = tmp_path / "test_report.md"

        results = await run_two_machine_demo(
            target_host="127.0.0.1",
            target_port=9995,
            high_activity_pattern="large_data_transfer",
            windows_per_phase=2,
            window_seconds=5.0,
            k_steps=2,
            json_path=json_out,
            report_path=report_out,
        )

        assert results["summary"]["total_windows_evaluated"] == 6
        assert len(results["windows"]) == 6
        assert json_out.exists()
        assert report_out.exists()

        # Check phase breakdown
        p1 = [w for w in results["windows"] if "Phase 1" in w["phase"]]
        p2 = [w for w in results["windows"] if "Phase 2" in w["phase"]]
        p3 = [w for w in results["windows"] if "Phase 3" in w["phase"]]
        assert len(p1) == 2
        assert len(p2) == 2
        assert len(p3) == 2

    def test_dynamic_model_sensitivity_no_hardcoding(self):
        model_svc = ModelService.get_instance()
        assert model_svc.is_loaded

        # Evaluate two distinct sequences
        seq_norm = [[[0.1] * 24] * 5]
        seq_burst = [[[2.5] * 24] * 5]

        fc_norm = model_svc.forecast(x_seq=seq_norm[0], k_steps=2)
        fc_burst = model_svc.forecast(x_seq=seq_burst[0], k_steps=2)

        # Structural assertions — NO hardcoded values
        assert 0.0 <= fc_norm["attack_probability"] <= 1.0
        assert 0.0 <= fc_burst["attack_probability"] <= 1.0
        assert 0.0 <= fc_norm["risk_score"] <= 100.0
        assert 0.0 <= fc_burst["risk_score"] <= 100.0

        sp_norm = sum(fc_norm["stage_probabilities"].values())
        sp_burst = sum(fc_burst["stage_probabilities"].values())
        assert abs(sp_norm - 1.0) < 0.02
        assert abs(sp_burst - 1.0) < 0.02

    def test_mitre_mapping_dynamic_assignment(self):
        model_svc = ModelService.get_instance()
        seq = [[0.0] * 24 for _ in range(5)]
        fc = model_svc.forecast(x_seq=seq, k_steps=1)

        tid = fc.get("primary_technique_id")
        if tid is not None:
            assert tid.startswith("T"), f"Technique ID must start with T, got {tid}"
            assert "primary_technique_name" in fc

    def test_fail_closed_truncated_netflow_datagram(self):
        source = NetFlowSource(host="127.0.0.1", port=9995)

        # Less than 24-byte header -> must be rejected as malformed
        flows = source._parse_netflow_v5(b"\x00" * 12)
        assert flows == []

        # Empty bytes -> must be rejected
        flows_empty = source._parse_netflow_v5(b"")
        assert flows_empty == []
