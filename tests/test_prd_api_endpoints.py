"""
CyberSentinel AI — Test Suite for Canonical PRD API Endpoints.

Tests:
  POST /api/traffic
  POST /api/predict
  GET  /api/alerts
  GET  /api/alerts/{id}
  GET  /api/dashboard
  GET  /api/statistics
  GET  /api/attack-story/{id}
  POST /api/world-model/rollout
  POST /api/feedback
  GET  /api/adaptation/history
  WS   /ws/live
"""

from pathlib import Path
import json
import pytest
from fastapi.testclient import TestClient

from backend.app import app

_WORKSPACE = Path(__file__).resolve().parent.parent


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Traffic & Predict
# ---------------------------------------------------------------------------

def test_post_traffic_benign(client):
    res = client.post(
        "/api/traffic",
        json={
            "src_ip": "192.168.1.50",
            "dst_ip": "10.0.0.1",
            "features": [0.0] * 24,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ingested"
    assert "risk_score" in data
    assert "is_attack" in data
    assert "threat_classification" in data


def test_post_traffic_invalid_dim(client):
    res = client.post(
        "/api/traffic",
        json={"features": [1.0, 2.0]},  # Only 2 features instead of 24
    )
    assert res.status_code == 422


def test_post_predict_valid(client):
    res = client.post(
        "/api/predict",
        json={"state": [0.0] * 24, "k_steps": 2},
    )
    assert res.status_code == 200
    data = res.json()
    assert "current_stage" in data
    assert "predicted_next_stage" in data
    assert "risk_score" in data
    assert "known_classifier" in data
    assert "novelty_detector" in data
    assert "world_model_rollout" in data


# ---------------------------------------------------------------------------
# 2. Alerts & Dashboard
# ---------------------------------------------------------------------------

def test_get_dashboard(client):
    res = client.get("/api/dashboard")
    assert res.status_code == 200
    data = res.json()
    assert "summary" in data
    assert "total_traffic" in data["summary"]
    assert "attack_distribution" in data
    assert "top_suspicious_sources" in data
    assert "live_traffic_series" in data


def test_get_statistics(client):
    res = client.get("/api/statistics")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "operational"
    assert "uptime_seconds" in data
    assert "model_versions" in data
    assert "world_model" in data["model_versions"]


def test_get_alerts_list(client):
    # Ingest suspicious state to ensure at least one alert exists
    client.post(
        "/api/traffic",
        json={
            "src_ip": "10.0.0.99",
            "dst_ip": "10.0.0.1",
            "features": [15.0] * 24,
        },
    )
    res = client.get("/api/alerts")
    assert res.status_code == 200
    data = res.json()
    assert "alerts" in data
    assert len(data["alerts"]) >= 1

    alert_id = data["alerts"][0]["alert_id"]
    detail = client.get(f"/api/alerts/{alert_id}")
    assert detail.status_code == 200
    assert detail.json()["alert_id"] == alert_id


# ---------------------------------------------------------------------------
# 3. Attack Story
# ---------------------------------------------------------------------------

def test_get_attack_story_live(client):
    res = client.get("/api/attack-story/live")
    assert res.status_code == 200
    data = res.json()
    assert "story_id" in data
    assert "title" in data
    assert "steps" in data
    assert "containment_recommendations" in data


# ---------------------------------------------------------------------------
# 4. World Model Rollout
# ---------------------------------------------------------------------------

def test_post_world_model_rollout(client):
    seq = [[0.0] * 24 for _ in range(8)]
    res = client.post(
        "/api/world-model/rollout",
        json={"state_seq": seq, "k_steps": 3},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["horizon_steps"] == 3
    assert "rollout" in data
    assert len(data["rollout"]["rollout_steps"]) == 3


# ---------------------------------------------------------------------------
# 5. Feedback & Adaptive Learning
# ---------------------------------------------------------------------------

def test_post_feedback_and_history(client):
    res = client.post(
        "/api/feedback",
        json={
            "feature_vector": [1.0] * 24,
            "validated_label": "EXFILTRATION",
            "is_malicious": True,
            "analyst_notes": "Confirmed high volume egress flow.",
            "trigger_update": False,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "recorded"
    assert data["sample"]["validated_label"] == "EXFILTRATION"

    hist = client.get("/api/adaptation/history")
    assert hist.status_code == 200
    assert "active_version" in hist.json()


# ---------------------------------------------------------------------------
# 6. WebSocket WS /ws/live
# ---------------------------------------------------------------------------

def test_websocket_live(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"type": "ping"}))
        data = ws.receive_json()
        assert data["type"] == "pong"
        assert "timestamp" in data
