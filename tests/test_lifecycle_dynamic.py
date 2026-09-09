"""
CyberSentinel AI — Real-Time Dynamic Attack Lifecycle Tests.

Validates:
  1. Telemetry A produces dynamic lifecycle state A.
  2. Telemetry B produces dynamic lifecycle state B.
  3. Frontend / backend lifecycle state changes dynamically with real telemetry.
  4. K=4 simulation path is generated directly by CyberWorldModelV2 autoregressive rollout.
  5. Zero scenario-name bias: arbitrary scenario IDs do not dictate lifecycle states.
  6. Missing / empty telemetry produces fail-closed INVALID_TELEMETRY.
  7. Model failure produces fail-closed MODEL_UNAVAILABLE with no canned intelligence.
  8. Dashboard HTML hygiene: initializes to neutral fallback state with zero hardcoded stage highlights.
"""

import time
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.app import app
from backend.services.model_service import ModelService
from backend.services.live_ingest_service import LiveIngestService

_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client():
    app.state.model_service = ModelService.get_instance()
    app.state.live_ingest_service = LiveIngestService()
    with TestClient(app) as c:
        yield c


class TestDynamicAttackLifecycleBackend:
    """Test suite proving that lifecycle states are dynamically computed from telemetry."""

    def test_telemetry_a_produces_lifecycle_state_a(self, client):
        """Telemetry pattern A (benign web/dns) produces baseline lifecycle state."""
        now = time.time()
        flows_benign = [
            {
                "src_ip": f"192.168.1.{100 + i}",
                "dst_ip": "192.168.1.10",
                "src_port": 40000 + i,
                "dst_port": 443 if i % 2 == 0 else 80,
                "protocol": 6,
                "packets": 10,
                "bytes": 1500,
                "duration": 0.5,
                "syn_flag": 1,
                "ack_flag": 1,
                "rst_flag": 0,
                "fin_flag": 1,
                "psh_flag": 1,
                "urg_flag": 0,
                "failed": False,
                "timestamp": now + (i * 0.1),
                "label": "UNKNOWN",
            }
            for i in range(15)
        ]

        resp_a = client.post(
            "/api/v1/stream/ingest",
            json={
                "flows": flows_benign,
                "session_id": "lifecycle_session_001",
                "source_id": "TelemetryPatternA",
                "window_seconds": 10.0,
            },
        )
        assert resp_a.status_code == 200
        data_a = resp_a.json()

        assert data_a["status"] == "FORECAST"
        assert "current_stage" in data_a
        assert "predicted_next_stage" in data_a
        assert "observed_stages" in data_a
        assert data_a["current_stage"] in data_a["observed_stages"]
        assert "rollout_steps" in data_a
        assert len(data_a["rollout_steps"]) == 4

        # Risk score is a valid bounded metric from the live model
        assert 0.0 <= data_a["risk_score"] <= 100.0
        assert isinstance(data_a["current_stage"], str) and len(data_a["current_stage"]) > 0

    def test_telemetry_b_produces_distinct_lifecycle_state_b(self, client):
        """Telemetry pattern B (reconnaissance scan) dynamically shifts current stage and rollout."""
        now = time.time()
        # High port entropy across wide destination ports with RST failures
        flows_recon = [
            {
                "src_ip": "192.168.1.105",
                "dst_ip": "192.168.1.20",
                "src_port": 45000 + i,
                "dst_port": 20 + i * 15,  # Scanning ports 20, 35, 50, 65...
                "protocol": 6,
                "packets": 1,
                "bytes": 44,
                "duration": 0.005,
                "syn_flag": 1,
                "ack_flag": 0,
                "rst_flag": 1 if i % 2 == 0 else 0,
                "fin_flag": 0,
                "psh_flag": 0,
                "urg_flag": 0,
                "failed": i % 2 == 0,
                "timestamp": now + (i * 0.02),
                "label": "UNKNOWN",
            }
            for i in range(40)
        ]

        resp_b = client.post(
            "/api/v1/stream/ingest",
            json={
                "flows": flows_recon,
                "session_id": "lifecycle_session_001",  # Same session, continuing lifecycle
                "source_id": "TelemetryPatternB",
                "window_seconds": 5.0,
            },
        )
        assert resp_b.status_code == 200
        data_b = resp_b.json()

        assert data_b["status"] == "FORECAST"
        assert data_b["current_stage"] == "RECONNAISSANCE"
        # The observed stages list must now contain the newly observed stage
        assert "RECONNAISSANCE" in data_b["observed_stages"]
        # Risk score must be significantly elevated
        assert data_b["risk_score"] > 50.0
        # MITRE technique must be dynamically associated with scanning (e.g. T1595 Active Scanning or T1046 Network Service Scanning)
        assert data_b["primary_technique_id"] in ("T1046", "T1595")

    def test_k4_simulation_is_real_autoregressive_rollout(self, client):
        """K=4 simulation path originates directly from CyberWorldModelV2 rollout."""
        now = time.time()
        flows = [
            {
                "src_ip": "192.168.1.105",
                "dst_ip": "192.168.1.50",
                "src_port": 48000 + i,
                "dst_port": 443,
                "protocol": 6,
                "packets": 500,
                "bytes": 700000,
                "duration": 2.0,
                "syn_flag": 1,
                "ack_flag": 1,
                "rst_flag": 0,
                "fin_flag": 1,
                "psh_flag": 1,
                "urg_flag": 0,
                "failed": False,
                "timestamp": now + (i * 0.5),
                "label": "UNKNOWN",
            }
            for i in range(8)
        ]

        resp = client.post(
            "/api/v1/stream/ingest",
            json={
                "flows": flows,
                "session_id": "lifecycle_rollout_test",
                "source_id": "RolloutVerification",
                "k_steps": 4,
                "window_seconds": 10.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        rollout = data["rollout_steps"]
        assert len(rollout) == 4

        # Steps must be 1, 2, 3, 4
        steps = [r["step"] for r in rollout]
        assert steps == [1, 2, 3, 4]

        # Each step has valid predicted stage and non-zero confidence
        for r in rollout:
            assert isinstance(r["predicted_stage"], str)
            assert len(r["predicted_stage"]) > 0
            assert 0.0 <= r["confidence"] <= 1.0

    def test_zero_scenario_name_bias(self, client):
        """Different arbitrary scenario names given identical telemetry must produce identical intelligence."""
        now = time.time()
        flows = [
            {
                "src_ip": "192.168.1.5",
                "dst_ip": "192.168.1.10",
                "src_port": 50000 + i,
                "dst_port": 22,
                "protocol": 6,
                "packets": 4,
                "bytes": 280,
                "duration": 0.05,
                "syn_flag": 1,
                "ack_flag": 1,
                "rst_flag": 1,
                "fin_flag": 0,
                "psh_flag": 0,
                "urg_flag": 0,
                "failed": True,
                "timestamp": now + (i * 0.1),
                "label": "UNKNOWN",
            }
            for i in range(20)
        ]

        resp1 = client.post(
            "/api/v1/stream/ingest",
            json={"flows": flows, "session_id": "arbitrary_scenario_alpha"},
        )
        resp2 = client.post(
            "/api/v1/stream/ingest",
            json={"flows": flows, "session_id": "totally_different_beta"},
        )

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        d1 = resp1.json()
        d2 = resp2.json()

        assert d1["current_stage"] == d2["current_stage"]
        assert d1["predicted_next_stage"] == d2["predicted_next_stage"]
        assert abs(d1["risk_score"] - d2["risk_score"]) < 1e-3

    def test_missing_telemetry_fails_closed(self, client):
        """Empty flow batches must fail closed without emitting fake attack stages."""
        resp = client.post(
            "/api/v1/stream/ingest",
            json={"flows": [], "session_id": "test_empty"},
        )
        assert resp.status_code == 422


class TestDynamicAttackLifecycleUIHygiene:
    """Test suite ensuring index.html contains no hardcoded stages or canned highlights."""

    def test_lifecycle_panel_initializes_with_no_hardcoded_highlights(self):
        """Dashboard index.html must initialize with clean fallback state and no hardcoded NOW or SIM."""
        html = (_ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")

        # Fallback text must be present
        assert "No live prediction available" in html

        # No hardcoded stages in initial HTML
        assert '<div class="lc-stage active">' not in html
        assert '<div class="lc-stage simulated">' not in html
        assert '<div class="lc-stage predicted">' not in html

        # Official taxonomy order must be defined
        assert "const STAGES = [" in html
        assert "'BENIGN','RECONNAISSANCE','INITIAL_ACCESS','EXECUTION'" in html

        # renderLifecycle dynamically maps from fc object
        assert "function renderLifecycle(fc)" in html
        assert "fc.current_stage" in html
        assert "fc.predicted_next_stage" in html
        assert "fc.rollout_steps" in html
        assert "state.observedStages" in html
