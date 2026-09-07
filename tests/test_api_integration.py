"""
CyberSentinel AI — Integration Test Suite (Phase 10).

Tests the full FastAPI Command Center backend:
  - /health and /model/info endpoints
  - /forecast canonical unified schema validation
  - /rollout autoregressive execution
  - /explain feature attribution
  - /mitre deterministic ATT&CK mapping
  - /risk engine evaluation
  - /agent/query routing, template fallback, and grounded responses
  - Replay session lifecycle (start -> step -> status -> scenarios)
  - Provenance validation across all endpoints
  - Error handling and edge cases
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.services.model_service import ModelService
from ml.state.state_builder import FEATURE_NAMES
from ml.world_model.world_model_v2 import INPUT_DIM


@pytest.fixture(scope="module")
def client():
    """Initializes TestClient and triggers startup event."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_seq():
    """Generates realistic sequence (T=8, D=24)."""
    import numpy as np
    np.random.seed(42)
    return np.random.randn(8, INPUT_DIM).astype(float).tolist()


# ---------------------------------------------------------------------------
# Health & Model Info
# ---------------------------------------------------------------------------

class TestHealthAndInfo:
    def test_health_endpoint(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "calibration_loaded" in data
        assert "timestamp" in data
        assert data["model_loaded"] is True
        assert data["status"] == "ok"

    def test_model_info_endpoint(self, client):
        resp = client.get("/api/v1/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_version"] == "CyberWorldModelV2"
        assert data["input_dim"] == INPUT_DIM
        assert data["num_stages"] == 10
        assert "benchmark" in data
        assert "WorldModelV2_DirectTransition" in data["benchmark"]
        assert data["benchmark"]["WorldModelV2_DirectTransition"]["next_stage_top1"] == 0.9773

    def test_root_redirect(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (302, 307)
        assert "/ui/index.html" in resp.headers["location"]


# ---------------------------------------------------------------------------
# Forecast & Unified Schema
# ---------------------------------------------------------------------------

class TestForecastEndpoints:
    def test_forecast_valid(self, client, sample_seq):
        resp = client.post("/api/v1/forecast", json={"x_seq": sample_seq, "k_steps": 4})
        assert resp.status_code == 200
        data = resp.json()

        # Check required fields
        required_fields = [
            "timestamp", "model_version", "forecast_horizon", "current_stage",
            "attack_probability", "predicted_next_stage", "next_stage_probability",
            "confidence", "transition_detected", "transition_confidence",
            "uncertainty_entropy", "stage_probabilities", "rollout_steps",
            "top_features", "mitre_techniques", "risk_score", "risk_level",
            "recommended_priority", "time_to_transition_hint", "safety_flags",
            "provenance"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

        # Value bounds
        assert 0.0 <= data["attack_probability"] <= 1.0
        assert 0.0 <= data["confidence"] <= 1.0
        assert 0.0 <= data["uncertainty_entropy"] <= 1.0
        assert 0.0 <= data["risk_score"] <= 100.0
        assert len(data["rollout_steps"]) == 4

        # Provenance verification
        prov = data["provenance"]
        assert prov["prediction"] == "CyberWorldModelV2"
        assert prov["explanation"] == "physical_state_delta"
        assert "MITRE" in prov["mitre"]
        assert "RiskEngine" in prov["risk"]

    def test_forecast_invalid_dimension(self, client):
        # 10 features instead of 24
        invalid_seq = [[0.0] * 10] * 8
        resp = client.post("/api/v1/forecast", json={"x_seq": invalid_seq})
        assert resp.status_code in (400, 500)

    def test_forecast_empty_seq(self, client):
        resp = client.post("/api/v1/forecast", json={"x_seq": []})
        assert resp.status_code == 422  # Pydantic validation error


# ---------------------------------------------------------------------------
# Specialized Endpoints (Rollout, Explain, MITRE, Risk)
# ---------------------------------------------------------------------------

class TestSpecializedEndpoints:
    def test_rollout_endpoint(self, client, sample_seq):
        resp = client.post("/api/v1/rollout", json={"x_seq": sample_seq, "k_steps": 4})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["rollout_steps"]) == 4
        assert "safety_flags" in data
        assert "provenance" in data

    def test_explain_endpoint(self, client, sample_seq):
        resp = client.post("/api/v1/explain", json={"x_seq": sample_seq})
        assert resp.status_code == 200
        data = resp.json()
        assert "top_features" in data
        assert len(data["top_features"]) <= 5
        assert "explanation_narrative" in data
        assert len(data["explanation_narrative"]) > 0

    def test_mitre_endpoint(self, client, sample_seq):
        resp = client.post("/api/v1/mitre", json={"x_seq": sample_seq})
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_next_stage" in data
        assert "mitre_techniques" in data
        assert "not LLM-generated" in data["note"]

    def test_risk_endpoint(self, client, sample_seq):
        resp = client.post("/api/v1/risk", json={"x_seq": sample_seq})
        assert resp.status_code == 200
        data = resp.json()
        assert "risk_score" in data
        assert "risk_level" in data
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


# ---------------------------------------------------------------------------
# Defensive Agent Query Mode
# ---------------------------------------------------------------------------

class TestDefensiveAgentEndpoint:
    def test_agent_query_current_state(self, client):
        resp = client.post("/api/v1/agent/query", json={
            "query": "What is happening right now?",
            "current_forecast": {
                "current_stage": "RECONNAISSANCE",
                "predicted_next_stage": "CREDENTIAL_ACCESS",
                "confidence": 0.88,
                "attack_probability": 0.92,
                "transition_detected": True,
                "risk_level": "HIGH",
                "risk_score": 78.5,
            }
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "RECONNAISSANCE" in data["answer"]
        assert "CREDENTIAL_ACCESS" in data["answer"]
        assert data["grounded_in_model_output"] is True
        assert len(data["tool_calls"]) > 0

    def test_agent_query_investigate(self, client):
        resp = client.post("/api/v1/agent/query", json={
            "query": "What should the analyst investigate?",
            "current_forecast": {
                "current_stage": "RECONNAISSANCE",
                "predicted_next_stage": "CREDENTIAL_ACCESS",
                "confidence": 0.91,
                "attack_probability": 0.85,
                "risk_level": "HIGH",
                "risk_score": 75.0,
                "primary_technique_id": "T1110",
                "primary_technique_name": "Brute Force",
                "top_features": [
                    {"feature": "failed_flow_count", "current": 2.0, "predicted": 15.0, "rel_change_pct": 650.0}
                ]
            }
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "T1110" in data["answer"]
        assert "failed_flow_count" in data["answer"]

    def test_agent_query_without_forecast(self, client):
        resp = client.post("/api/v1/agent/query", json={
            "query": "Compare models benchmark."
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "CyberWorldModelV2" in data["answer"]
        assert "83.33%" in data["answer"]


# ---------------------------------------------------------------------------
# Replay Mode Lifecycle
# ---------------------------------------------------------------------------

class TestReplayLifecycle:
    def test_replay_scenarios_list(self, client):
        resp = client.get("/api/v1/replay/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        assert "scenarios" in data
        assert "trace_multistage_03" in data["scenarios"]

    def test_full_replay_session(self, client):
        # 1. Start
        start_resp = client.post("/api/v1/replay/start", json={
            "scenario_id": "trace_multistage_03",
            "k_steps": 4
        })
        assert start_resp.status_code == 200
        s_data = start_resp.json()
        session_id = s_data["session_id"]
        total_wins = s_data["total_windows"]
        assert total_wins > 0

        # 2. Status
        status_resp = client.get(f"/api/v1/replay/status?session_id={session_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["current_window"] == 0

        # 3. Step 1
        step_resp = client.post(f"/api/v1/replay/step?session_id={session_id}")
        assert step_resp.status_code == 200
        step_data = step_resp.json()
        assert step_data["window_index"] == 0
        assert step_data["forecast"] is not None
        assert "ground_truth_stage" in step_data["forecast"]

        # 4. Step 2
        step2_resp = client.post(f"/api/v1/replay/step?session_id={session_id}")
        assert step2_resp.status_code == 200
        assert step2_resp.json()["window_index"] == 1

    def test_replay_invalid_scenario(self, client):
        resp = client.post("/api/v1/replay/start", json={"scenario_id": "nonexistent_trace"})
        assert resp.status_code == 404
