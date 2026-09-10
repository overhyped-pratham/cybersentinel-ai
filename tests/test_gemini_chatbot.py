"""
CyberSentinel AI — Comprehensive Gemini AI Security Analyst Test Suite.

Verifies:
  1. Operational status endpoint (/api/v1/agent/status).
  2. Grounded chat endpoint (/api/v1/agent/chat) with explicit forecast.
  3. Grounding integrity: Responses are anchored in verified model output.
  4. Insufficient telemetry handling: Explicit notice when no data is available.
  5. Deterministic fallback when Gemini is offline, errored, or unconfigured.
  6. Input validation: Rejection of empty or invalid messages.
  7. Secret isolation: GEMINI_API_KEY is never leaked in responses.
  8. Rate limiting and large query truncation.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.services.evidence_service import EvidenceService
from backend.services.gemini_service import GeminiService


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_forecast():
    return {
        "status": "FORECAST",
        "current_stage": "Exploitation",
        "predicted_next_stage": "Installation",
        "attack_probability": 0.885,
        "confidence": 0.924,
        "risk_score": 78.4,
        "risk_level": "HIGH",
        "recommended_priority": "P1 — Critical",
        "transition_detected": True,
        "time_to_transition_hint": "Imminent (1-2 windows)",
        "primary_technique_id": "T1059",
        "primary_technique_name": "Command and Scripting Interpreter",
        "mitre_techniques": [
            {"technique_id": "T1059", "name": "Command and Scripting Interpreter", "tactic": "Execution"},
            {"technique_id": "T1068", "name": "Exploitation for Privilege Escalation", "tactic": "Privilege Escalation"},
        ],
        "top_features": [
            {"feature": "syn_ratio", "current": 0.72, "predicted": 0.89, "rel_change_pct": 23.6},
            {"feature": "pkt_rate", "current": 1420.0, "predicted": 2850.0, "rel_change_pct": 100.7},
        ],
        "rollout_steps": [
            {"step": 1, "predicted_stage": "Installation", "confidence": 0.89},
            {"step": 2, "predicted_stage": "Command_and_Control", "confidence": 0.78},
        ],
        "telemetry_features": {
            "flow_count": 340,
            "flows_per_second": 56.6,
            "pkt_rate": 1420.0,
            "syn_ratio": 0.72,
        },
        "explanation_narrative": "Sharp increase in SYN packets accompanied by rapid process execution.",
    }


# ===========================================================================
# 1. Status & Health Tests
# ===========================================================================

def test_agent_status_endpoint(client):
    """Ensure /api/v1/agent/status responds with operational telemetry."""
    resp = client.get("/api/v1/agent/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["status"] in ("online", "fallback", "offline")
    assert "model" in data
    assert "rate_limit_rpm" in data
    assert data["rate_limit_rpm"] > 0


# ===========================================================================
# 2. Validation Tests
# ===========================================================================

def test_agent_chat_empty_message_rejected(client):
    """Empty messages should fail validation (HTTP 422)."""
    resp = client.post("/api/v1/agent/chat", json={"message": ""})
    assert resp.status_code == 422


def test_agent_chat_excessive_message_rejected(client):
    """Messages exceeding 2000 characters should fail schema validation."""
    resp = client.post("/api/v1/agent/chat", json={"message": "A" * 2001})
    assert resp.status_code == 422


# ===========================================================================
# 3. Grounded Evidence Extraction Tests
# ===========================================================================

def test_evidence_service_extraction(mock_forecast):
    """Test that EvidenceService correctly standardizes raw forecast payload."""
    svc = EvidenceService()
    evidence = svc.get_latest_evidence(current_forecast=mock_forecast)

    assert evidence["has_sufficient_evidence"] is True
    assert evidence["current_stage"] == "Exploitation"
    assert evidence["predicted_next_stage"] == "Installation"
    assert evidence["risk_score"] == 78.4
    assert evidence["attack_probability"] == 0.885
    assert evidence["primary_technique_id"] == "T1059"
    assert len(evidence["features"]) == 2
    assert len(evidence["mitre"]) == 2
    assert len(evidence["rollout"]) == 2


def test_evidence_service_insufficient_data():
    """Test handling when no forecast or telemetry is available."""
    svc = EvidenceService()
    evidence = svc.get_latest_evidence(current_forecast=None)

    assert evidence["has_sufficient_evidence"] is False
    assert evidence["status"] == "NO_TELEMETRY_AVAILABLE"
    assert "Insufficient telemetry" in evidence["detail"]

    prompt = svc.format_grounding_prompt(evidence, "What is the current threat?")
    assert "STATUS: INSUFFICIENT_TELEMETRY" in prompt


# ===========================================================================
# 4. Fallback & Grounding Tests
# ===========================================================================

def test_chat_fallback_when_gemini_disabled(client, mock_forecast):
    """When Gemini client is None, system uses deterministic fallback without 500."""
    orig_client = getattr(app.state.gemini_service, "_client", None)
    app.state.gemini_service._client = None
    try:
        resp = client.post(
            "/api/v1/agent/chat",
            json={
                "message": "Explain the current attack and recommend actions.",
                "current_forecast": mock_forecast,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["fallback"] is True
        assert data["llm_backend"] == "deterministic_fallback"
        # Must contain grounding message
        assert "Gemini analyst unavailable. CyberSentinel deterministic intelligence remains operational." in data["answer"]
        # Must contain grounded facts
        assert "Exploitation" in data["answer"]
        assert "Installation" in data["answer"]
        assert "78.4/100" in data["answer"]
        assert "T1059" in data["answer"]
    finally:
        app.state.gemini_service._client = orig_client


def test_chat_insufficient_telemetry_response(client):
    """When no telemetry is passed or recorded, answer states insufficient evidence."""
    orig_client = getattr(app.state.gemini_service, "_client", None)
    app.state.gemini_service._client = None
    try:
        resp = client.post(
            "/api/v1/agent/chat",
            json={
                "message": "What is the threat score?",
                "current_forecast": None,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Insufficient telemetry/model evidence is available to determine this." in data["answer"]
        assert data["evidence"]["has_sufficient_evidence"] is False
    finally:
        app.state.gemini_service._client = orig_client


# ===========================================================================
# 5. Mocked Gemini Success Test
# ===========================================================================

def test_chat_with_mocked_gemini(client, mock_forecast):
    """Verify standard grounded generation when Gemini returns a valid analysis."""
    mock_text = (
        "### **Current Assessment**\n"
        "CyberSentinel classified the current stage as **Exploitation** with 88.5% attack probability "
        "and 78.4/100 risk score. Next stage projected is Installation.\n\n"
        "### **Empirical Indicators**\n"
        "SYN ratio increased by 23.6% and packet rate spiked by 100.7%.\n\n"
        "### **Recommended SOC Actions**\n"
        "1. Isolate target host from corporate network.\n"
        "2. Kill unauthorized command interpreter processes.\n"
    )

    mock_resp_obj = MagicMock()
    mock_resp_obj.text = mock_text

    mock_genai_client = MagicMock()
    mock_genai_client.models.generate_content.return_value = mock_resp_obj

    with patch("backend.services.gemini_service.GeminiService._init_client"):
        gemini_svc = GeminiService(api_key="mock_key", model="gemini-2.5-flash")
        gemini_svc._client = mock_genai_client

        app.state.gemini_service = gemini_svc

        resp = client.post(
            "/api/v1/agent/chat",
            json={
                "message": "What is the threat level and what should we do?",
                "current_forecast": mock_forecast,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["fallback"] is False
        assert data["llm_backend"] == "gemini"
        assert "Exploitation" in data["answer"]
        assert "88.5%" in data["answer"]
        assert data["evidence"]["current_stage"] == "Exploitation"


# ===========================================================================
# 6. Security & Secret Isolation Tests
# ===========================================================================

def test_api_key_not_leaked_in_response(client, mock_forecast):
    """Confirm the API key is NEVER exposed in the JSON response."""
    test_key = "AIzaSyTEST_SECRET_KEY_12345"
    with patch.dict(os.environ, {"GEMINI_API_KEY": test_key}):
        resp = client.post(
            "/api/v1/agent/chat",
            json={"message": "System info dump", "current_forecast": mock_forecast},
        )
        assert resp.status_code == 200
        raw_text = resp.text
        assert test_key not in raw_text
        assert "api_key" not in raw_text.lower() or '"api_key_configured"' in raw_text


def test_rate_limiting_enforcement():
    """Verify that rapid successive queries trigger rate limit fallback."""
    gemini_svc = GeminiService(api_key="mock_key")
    gemini_svc.rate_limit_per_minute = 3  # Set artificially low for testing

    evidence = {"has_sufficient_evidence": True, "current_stage": "Reconnaissance"}

    # Run 3 queries (within limit)
    for _ in range(3):
        assert gemini_svc._check_rate_limit() is True

    # 4th query should hit limit
    assert gemini_svc._check_rate_limit() is False
    res = gemini_svc.generate_response("test", evidence, "test query")
    assert res["fallback"] is True
    assert "Rate limit reached" in res["answer"]
