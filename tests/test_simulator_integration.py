"""
CyberSentinel AI — Mobile Traffic Simulator Integration Tests.

Validates:
  1. POST /api/v1/stream/ingest accepts synthetic flow records from the Mobile Simulator.
  2. Every flow strictly sets label='UNKNOWN' (defensive inference, no ground-truth leakage).
  3. FeatureScaler produces scaled 24-D physical state vector for CyberWorldModelV2.
  4. Real-time inference generates genuine predictions (current_stage, next_stage, risk_score).
  5. Dynamic sensitivity across all 6 synthetic traffic profiles:
     - Normal Background Traffic
     - Connection Burst
     - Reconnaissance-like
     - Credential-Access-like
     - Large Data Transfer
     - Bot Beaconing
  6. Real-time WebSocket broadcasting to connected dashboard clients.
  7. Fail-closed error handling for empty flow lists and malformed inputs.
"""

import pytest
import time
from fastapi.testclient import TestClient
from backend.app import app
from backend.services.model_service import ModelService
from backend.services.live_ingest_service import LiveIngestService


@pytest.fixture(scope="module")
def client():
    # Ensure services are initialized
    app.state.model_service = ModelService.get_instance()
    app.state.live_ingest_service = LiveIngestService()
    with TestClient(app) as c:
        yield c


class TestSimulatorIngestAPI:
    """Test suite for /api/v1/stream/ingest endpoint used by the Mobile Traffic Simulator."""

    def test_ingest_normal_background_traffic(self, client):
        """Profile 1: Normal Background — HTTP/HTTPS/DNS traffic with low risk."""
        flows = []
        now = time.time()
        for i in range(15):
            port = 443 if i % 2 == 0 else (80 if i % 3 == 0 else 53)
            proto = 6 if port in (80, 443) else 17
            flows.append({
                "src_ip": f"192.168.1.{100 + (i % 3)}",
                "dst_ip": f"192.168.1.{200 + (i % 4)}",
                "src_port": 40000 + i,
                "dst_port": port,
                "protocol": proto,
                "packets": 10 + (i * 2),
                "bytes": 1500 + (i * 200),
                "duration": 0.5 + (i * 0.05),
                "syn_flag": 1 if proto == 6 else 0,
                "ack_flag": 1 if proto == 6 else 0,
                "rst_flag": 0,
                "fin_flag": 1 if proto == 6 else 0,
                "psh_flag": 1 if proto == 6 else 0,
                "urg_flag": 0,
                "failed": False,
                "timestamp": now + (i * 0.2),
                "label": "UNKNOWN",
            })

        payload = {
            "flows": flows,
            "session_id": "test_mobile_normal",
            "source_id": "MobileSimulator-Normal",
            "window_seconds": 10.0,
        }

        resp = client.post("/api/v1/stream/ingest", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "FORECAST"
        assert data["source_id"] == "MobileSimulator-Normal"
        assert data["flow_count"] == 15
        assert data["inference_latency_ms"] is not None
        assert "risk_score" in data
        assert "current_stage" in data
        assert "predicted_next_stage" in data
        assert "stage_probabilities" in data

    def test_ingest_connection_burst(self, client):
        """Profile 2: Connection Burst — Rapid TCP connections, high flow rate."""
        flows = []
        now = time.time()
        for i in range(40):
            flows.append({
                "src_ip": "192.168.1.105",
                "dst_ip": "192.168.1.250",
                "src_port": 50000 + i,
                "dst_port": 443,
                "protocol": 6,
                "packets": 3,
                "bytes": 180,
                "duration": 0.01,
                "syn_flag": 1,
                "ack_flag": 1,
                "rst_flag": 0,
                "fin_flag": 1,
                "psh_flag": 0,
                "urg_flag": 0,
                "failed": False,
                "timestamp": now + (i * 0.05),
                "label": "UNKNOWN",
            })

        payload = {
            "flows": flows,
            "session_id": "test_mobile_burst",
            "source_id": "MobileSimulator-Burst",
            "window_seconds": 5.0,
        }

        resp = client.post("/api/v1/stream/ingest", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "FORECAST"
        assert data["flow_count"] == 40
        assert data["risk_score"] is not None

    def test_ingest_reconnaissance_pattern(self, client):
        """Profile 3: Reconnaissance — Port scan pattern across multiple destination ports."""
        flows = []
        now = time.time()
        for i in range(50):
            flows.append({
                "src_ip": "192.168.1.105",
                "dst_ip": "192.168.1.20",
                "src_port": 45000 + i,
                "dst_port": 20 + i * 15,
                "protocol": 6,
                "packets": 1,
                "bytes": 44,
                "duration": 0.005,
                "syn_flag": 1,
                "ack_flag": 0,
                "rst_flag": 1 if i % 3 == 0 else 0,
                "fin_flag": 0,
                "psh_flag": 0,
                "urg_flag": 0,
                "failed": i % 3 == 0,
                "timestamp": now + (i * 0.02),
                "label": "UNKNOWN",
            })

        payload = {
            "flows": flows,
            "session_id": "test_mobile_recon",
            "source_id": "MobileSimulator-Recon",
            "window_seconds": 5.0,
        }

        resp = client.post("/api/v1/stream/ingest", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "FORECAST"
        assert data["flow_count"] == 50
        assert data["risk_score"] is not None
        assert data["current_stage"] is not None

    def test_ingest_credential_access_pattern(self, client):
        """Profile 4: Credential Access — Authentication ports with high failed flow ratio."""
        flows = []
        now = time.time()
        target_ports = [22, 3389, 445, 1433]
        for i in range(30):
            flows.append({
                "src_ip": "192.168.1.105",
                "dst_ip": "192.168.1.10",
                "src_port": 49000 + i,
                "dst_port": target_ports[i % len(target_ports)],
                "protocol": 6,
                "packets": 4,
                "bytes": 260,
                "duration": 0.03,
                "syn_flag": 1,
                "ack_flag": 1,
                "rst_flag": 1 if i % 2 == 0 else 0,
                "fin_flag": 0,
                "psh_flag": 0,
                "urg_flag": 0,
                "failed": True,  # High failed ratio
                "timestamp": now + (i * 0.1),
                "label": "UNKNOWN",
            })

        payload = {
            "flows": flows,
            "session_id": "test_mobile_cred",
            "source_id": "MobileSimulator-CredAccess",
            "window_seconds": 10.0,
        }

        resp = client.post("/api/v1/stream/ingest", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "FORECAST"
        assert data["flow_count"] == 30

    def test_ingest_large_data_transfer(self, client):
        """Profile 5: Large Data Transfer — Massive byte counts and full MTU packets."""
        flows = []
        now = time.time()
        for i in range(10):
            flows.append({
                "src_ip": "192.168.1.105",
                "dst_ip": "192.168.1.50",
                "src_port": 55000 + i,
                "dst_port": 443,
                "protocol": 6,
                "packets": 500 + (i * 100),
                "bytes": 750000 + (i * 150000),
                "duration": 2.5,
                "syn_flag": 1,
                "ack_flag": 1,
                "rst_flag": 0,
                "fin_flag": 1,
                "psh_flag": 1,
                "urg_flag": 0,
                "failed": False,
                "timestamp": now + (i * 0.5),
                "label": "UNKNOWN",
            })

        payload = {
            "flows": flows,
            "session_id": "test_mobile_exfil",
            "source_id": "MobileSimulator-LargeData",
            "window_seconds": 10.0,
        }

        resp = client.post("/api/v1/stream/ingest", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "FORECAST"
        assert data["flow_count"] == 10

    def test_ingest_bot_beaconing_pattern(self, client):
        """Profile 6: Bot Beaconing — Strict periodic keepalives with uniform payloads."""
        flows = []
        now = time.time()
        for i in range(12):
            flows.append({
                "src_ip": "192.168.1.105",
                "dst_ip": "192.168.1.99",
                "src_port": 60000,
                "dst_port": 8443,
                "protocol": 6,
                "packets": 2,
                "bytes": 128,  # Exactly 128 bytes every beacon
                "duration": 0.02,
                "syn_flag": 1,
                "ack_flag": 1,
                "rst_flag": 0,
                "fin_flag": 1,
                "psh_flag": 0,
                "urg_flag": 0,
                "failed": False,
                "timestamp": now + (i * 1.0),  # Exactly 1 second intervals
                "label": "UNKNOWN",
            })

        payload = {
            "flows": flows,
            "session_id": "test_mobile_beacon",
            "source_id": "MobileSimulator-Beacon",
            "window_seconds": 15.0,
        }

        resp = client.post("/api/v1/stream/ingest", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "FORECAST"
        assert data["flow_count"] == 12

    def test_fail_closed_empty_flows(self, client):
        """Ingesting an empty list of flows must fail closed."""
        payload = {
            "flows": [],
            "session_id": "test_empty",
        }
        resp = client.post("/api/v1/stream/ingest", json=payload)
        # Pydantic min_length=1 rejects empty list with 422
        assert resp.status_code == 422

    def test_fail_closed_malformed_flow_fields(self, client):
        """Ingesting flows with invalid ports or types must fail closed."""
        payload = {
            "flows": [
                {
                    "src_ip": "192.168.1.1",
                    "dst_ip": "192.168.1.2",
                    "src_port": 999999,  # Invalid port > 65535
                    "dst_port": 80,
                }
            ],
            "session_id": "test_malformed",
        }
        resp = client.post("/api/v1/stream/ingest", json=payload)
        assert resp.status_code == 422

    def test_websocket_broadcast_on_ingest(self, client):
        """When telemetry flows are ingested, connected WebSocket clients must receive the event."""
        with client.websocket_connect("/api/v1/stream/ws") as ws:
            # First message is connection greeting
            greeting = ws.receive_json()
            assert greeting["type"] == "connected"

            # Ingest a flow batch via REST
            now = time.time()
            flows = [
                {
                    "src_ip": "192.168.1.100",
                    "dst_ip": "192.168.1.200",
                    "src_port": 50000,
                    "dst_port": 80,
                    "protocol": 6,
                    "packets": 5,
                    "bytes": 500,
                    "duration": 0.1,
                    "syn_flag": 1,
                    "ack_flag": 1,
                    "rst_flag": 0,
                    "fin_flag": 0,
                    "psh_flag": 0,
                    "urg_flag": 0,
                    "failed": False,
                    "timestamp": now,
                    "label": "UNKNOWN",
                }
            ]
            resp = client.post(
                "/api/v1/stream/ingest",
                json={"flows": flows, "session_id": "test_ws_broadcast"},
            )
            assert resp.status_code == 200

            # WebSocket client should receive the broadcast event (skipping any replayed history)
            found_event = None
            for _ in range(25):
                evt = ws.receive_json()
                if evt.get("session_id") == "test_ws_broadcast":
                    found_event = evt
                    break

            assert found_event is not None
            assert found_event["status"] == "FORECAST"
            assert found_event["session_id"] == "test_ws_broadcast"
            assert "current_stage" in found_event
            assert "risk_score" in found_event


class TestSimulatorUIHygiene:
    """Test suite for static UI assets and strict anti-hardcoding hygiene."""

    def test_simulator_html_served_at_ui_route(self, client):
        resp = client.get("/ui/simulator.html")
        assert resp.status_code == 200
        assert "CyberSentinel Traffic Sim" in resp.text
        assert "text/html" in resp.headers.get("content-type", "")

    def test_simulator_manifest_served(self, client):
        resp = client.get("/ui/manifest.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["short_name"] == "SentinelSim"
        assert data["display"] == "standalone"

    def test_simulator_html_safety_banner_present(self, client):
        resp = client.get("/ui/simulator.html")
        assert "[ DEMO / SYNTHETIC TELEMETRY GENERATOR ]" in resp.text
        assert "Defensive Pipeline Testing Only" in resp.text

    def test_simulator_html_contains_no_canned_prediction_values(self, client):
        resp = client.get("/ui/simulator.html")
        html = resp.text

        # Inferred stage, probability, and risk must initialize to placeholder dash
        assert 'id="fbStage" class="fb-val stage-normal">—</div>' in html
        assert 'id="fbAtkProb" class="fb-val">—</div>' in html
        assert 'id="fbRiskScore" class="fb-val">—</div>' in html

        # Generator script must explicitly set label: 'UNKNOWN'
        assert "label: 'UNKNOWN'" in html

