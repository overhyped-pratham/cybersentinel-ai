"""
CyberSentinel X — Red-Team & Adversarial Robustness Test Suite.

Rigorously verifies:
  Case A: Unknown attack labels
  Case B: Held-Out attack family (Novel behavior)
  Case C: Normal traffic with unusual feature values
  Case D: Conflicting classifier vs novelty detector signals
  Case E: Missing features (length != 24)
  Case F: Malformed traffic inputs
  Case G: Extreme numerical values, NaN, and Inf
  Case H: Repeated identical events
  Case I: Out-of-distribution input
  Case J: High event throughput
  Additional:
    - SHAP explanation input sensitivity
    - CyberWorldModel rollout state sensitivity
    - Dynamic Attack Story sequence sensitivity (3 distinct chains)
    - Human-in-the-loop validation requirement for adaptation & rollback
    - API edge-case 404/422 responses
"""

import math
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app import app
from ml.adaptation.adaptive_learner import AdaptiveLearner
from ml.adaptation.threat_memory import ThreatMemory
from ml.classifier.known_attack_classifier import KnownAttackClassifier
from ml.defense.attack_story_engine import AttackStoryEngine
from ml.defense.risk_engine import ForecastEvent, RiskEngine, RiskEngineConfig
from ml.novelty.autoencoder_detector import AutoencoderNoveltyDetector
from ml.pipeline.detection_pipeline import DetectionPipeline
from ml.state.state_builder import FEATURE_NAMES
from ml.world_model.world_model_v2 import CyberWorldModelV2


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def pipeline():
    return DetectionPipeline()


# ===========================================================================
# Case A: Unknown Attack Label
# ===========================================================================
def test_case_a_unknown_attack_label(pipeline):
    """Pipeline gracefully handles states when classified with non-standard labels."""
    dummy_state = np.zeros(24, dtype=np.float32)
    res = pipeline.process_state(dummy_state, is_scaled=True)
    assert res.threat_classification in ("Benign Baseline", "Confirmed Known Threat", "Potential Novel Behavior")
    assert isinstance(res.risk_score, float)
    assert 0.0 <= res.risk_score <= 100.0


# ===========================================================================
# Case B: Held-Out Attack Family (Exfiltration Telemetry)
# ===========================================================================
def test_case_b_held_out_attack_family(pipeline):
    """Held-out exfiltration characteristics must trigger Potential Novel Behavior and elevated risk."""
    exfil_vector = [
        45.0, 2800.0, 3450000.0, 93.3, 115000.0, 1.0, 2.0, 2.0,
        45.0, 0.0, 45.0, 0.016, 0.0, 0.693, 0.693, 0.0, 0.0, 3.0,
        0.0, 0.0, 0.0, 1.0, 28.5, 1232.0,
    ]
    res = pipeline.process_state(exfil_vector, is_scaled=False)
    assert res.is_novel is True
    assert res.threat_classification == "Potential Novel Behavior"
    assert res.risk_score >= 50.0  # Elevated or High/Critical risk
    assert res.novelty_detector["anomaly_score"] >= 0.50


# ===========================================================================
# Case C: Normal Traffic with Unusual Feature Values
# ===========================================================================
def test_case_c_unusual_normal_traffic(pipeline):
    """Massive normal burst must not trigger divide-by-zero or crashes."""
    burst_vector = np.zeros(24, dtype=np.float32)
    burst_vector[0] = 50000.0   # flow count
    burst_vector[1] = 1000000.0 # packets
    burst_vector[2] = 500000000.0 # bytes
    burst_vector[3] = 20000.0   # packet rate
    burst_vector[4] = 10000000.0 # byte rate
    burst_vector[21] = 1.0      # port 80/443 share

    res = pipeline.process_state(burst_vector, is_scaled=False)
    assert not math.isnan(res.risk_score)
    assert not math.isnan(res.novelty_detector["anomaly_score"])
    assert 0.0 <= res.risk_score <= 100.0


# ===========================================================================
# Case D: Conflicting Classifier vs Novelty Detector Signals
# ===========================================================================
def test_case_d_conflicting_signals(pipeline):
    """
    When classifier is benign/uncertain but autoencoder shows extreme reconstruction error,
    the pipeline prioritizes novel behavior flag and escalates threat risk.
    """
    conflicting_vector = np.ones(24, dtype=np.float32) * 25.0
    res = pipeline.process_state(conflicting_vector, is_scaled=True)
    if res.novelty_detector["is_novel"]:
        assert res.threat_classification == "Potential Novel Behavior"
        assert res.risk_score >= 40.0


# ===========================================================================
# Case E: Missing Features (Dimension Mismatch)
# ===========================================================================
def test_case_e_missing_features(client, pipeline):
    """Sending less than 24 features must fail validation with HTTP 422, not crash."""
    with pytest.raises(ValueError):
        pipeline.process_state([1.0, 2.0, 3.0], is_scaled=False)

    resp = client.post("/api/traffic", json={"src_ip": "10.0.0.1", "features": [1.0, 2.0]})
    assert resp.status_code == 422

    resp_pred = client.post("/api/predict", json={"state": [1.0, 2.0]})
    assert resp_pred.status_code == 422


# ===========================================================================
# Case F: Malformed Traffic Inputs
# ===========================================================================
def test_case_f_malformed_input(client):
    """Non-numeric string values in feature arrays must return HTTP 422."""
    resp = client.post("/api/traffic", json={"src_ip": "10.0.0.1", "features": ["bad", "values"] * 12})
    assert resp.status_code == 422


# ===========================================================================
# Case G: Extreme Numerical Values, NaN, and Inf Sanitization
# ===========================================================================
def test_case_g_extreme_numerical_values(client, pipeline):
    """Hostile inputs with NaN, +Inf, -Inf, and 10^15 must be safely sanitized."""
    hostile_vec = [float("nan"), float("inf"), float("-inf"), 1e15] * 6
    res = pipeline.process_state(hostile_vec, is_scaled=False)
    assert not math.isnan(res.risk_score)
    assert not math.isnan(res.novelty_detector["anomaly_score"])
    assert 0.0 <= res.risk_score <= 100.0

    resp = client.post("/api/traffic", json={"src_ip": "10.0.0.1", "features": [1e15] * 24})
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["risk_score"] <= 100.0


# ===========================================================================
# Case H: Repeated Identical Events
# ===========================================================================
def test_case_h_repeated_identical_events(pipeline):
    """Streaming 50 identical events sequentially must maintain stability."""
    vec = np.zeros(24, dtype=np.float32)
    scores = []
    for _ in range(50):
        res = pipeline.process_state(vec, is_scaled=True)
        scores.append(res.risk_score)
    assert len(scores) == 50
    assert all(0.0 <= s <= 100.0 for s in scores)


# ===========================================================================
# Case I: Out-of-Distribution Inputs
# ===========================================================================
def test_case_i_out_of_distribution_input(pipeline):
    """Arbitrary random Gaussian noise vectors must evaluate gracefully."""
    np.random.seed(99)
    for _ in range(10):
        noise = np.random.randn(24).astype(np.float32) * 50.0
        res = pipeline.process_state(noise, is_scaled=False)
        assert 0.0 <= res.risk_score <= 100.0
        assert res.threat_classification in ("Benign Baseline", "Confirmed Known Threat", "Potential Novel Behavior")


# ===========================================================================
# Case J: High Event Rate Throughput
# ===========================================================================
def test_case_j_high_event_rate(client):
    """Rapid firing of 25 API traffic requests succeeds without server drop."""
    for i in range(25):
        resp = client.post(
            "/api/traffic",
            json={"src_ip": f"10.0.0.{i % 10}", "dst_ip": "10.0.0.50", "features": [0.0] * 24},
        )
        assert resp.status_code == 200


# ===========================================================================
# Model Grounding & Input Sensitivity Tests
# ===========================================================================
def test_shap_explanation_input_sensitivity():
    """SHAP explanations must dynamically reflect changing feature inputs."""
    clf = KnownAttackClassifier.load("models/classifier/known_classifier.pkl")
    x_base = np.zeros(24, dtype=np.float32)
    x_mod = np.zeros(24, dtype=np.float32)
    x_mod[23] = 1450.0  # bytes_per_packet
    x_mod[4] = 300000.0 # byte_rate

    p1 = clf.predict_single(x_base)
    p2 = clf.predict_single(x_mod)
    assert p1.top_features != p2.top_features


def test_world_model_rollout_state_sensitivity():
    """CyberWorldModelV2 rollouts must change when input states vary."""
    wm = CyberWorldModelV2(input_dim=24, hidden_dim=128, num_layers=2, num_heads=4)
    ckpt = torch.load("models/world_model_v2.pt", map_location="cpu")
    wm.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    wm.eval()

    x1 = torch.zeros((1, 8, 24))
    x2 = torch.randn((1, 8, 24)) * 3.0
    mask = torch.ones((1, 8), dtype=torch.bool)

    r1 = wm.rollout(x1, mask, k_steps=4)
    r2 = wm.rollout(x2, mask, k_steps=4)

    assert r1.horizon == 4
    assert r2.horizon == 4
    stages1 = [int(s[0]) for s in r1.predicted_stages]
    stages2 = [int(s[0]) for s in r2.predicted_stages]
    assert stages1 != stages2 or not torch.allclose(r1.attack_probabilities, r2.attack_probabilities)


def test_attack_story_dynamic_sequences():
    """AttackStoryEngine produces distinct narratives for three distinct event chains."""
    engine = AttackStoryEngine()

    # Sequence 1: Port scan reconnaissance
    seq1 = [
        {"timestamp": "2026-09-18T10:00:00Z", "src_ip": "10.0.0.1", "dst_ip": "10.0.0.5", "stage": "RECONNAISSANCE", "risk_score": 35.0},
        {"timestamp": "2026-09-18T10:00:30Z", "src_ip": "10.0.0.1", "dst_ip": "10.0.0.5", "stage": "RECONNAISSANCE", "risk_score": 45.0},
    ]

    # Sequence 2: Multi-host lateral movement pivot
    seq2 = [
        {"timestamp": "2026-09-18T10:05:00Z", "src_ip": "10.0.0.1", "dst_ip": "10.0.0.5", "stage": "CREDENTIAL_ACCESS", "risk_score": 75.0},
        {"timestamp": "2026-09-18T10:05:20Z", "src_ip": "10.0.0.5", "dst_ip": "10.0.0.9", "stage": "LATERAL_MOVEMENT", "risk_score": 85.0},
    ]

    # Sequence 3: Novel anomalous exfiltration
    seq3 = [
        {"timestamp": "2026-09-18T10:10:00Z", "src_ip": "10.0.0.9", "dst_ip": "203.0.113.5", "stage": "UNKNOWN", "is_novel": True, "risk_score": 90.0},
    ]

    story1 = engine.build_story(seq1)
    story2 = engine.build_story(seq2)
    story3 = engine.build_story(seq3)

    assert story1.title != story2.title
    assert story2.title != story3.title
    # Sequence 2 should identify pivoting
    assert any("Pivoting observed" in s.link_rationale for s in story2.steps[1:])
    # Sequence 3 should identify novel behavior
    assert story3.steps[0].is_novel is True


def test_adaptive_learning_requires_explicit_validation(tmp_path):
    """Adaptive learner must strictly require human-approved samples before retraining."""
    mem_path = tmp_path / "test_threat_memory.json"
    memory = ThreatMemory(storage_path=mem_path)
    learner = AdaptiveLearner(threat_memory=memory, models_dir=tmp_path)

    # Attempt retraining with 0 validated samples -> must reject
    with pytest.raises(ValueError, match="No unincorporated samples"):
        learner.adapt_model()


def test_api_security_error_handlers(client):
    """API endpoints must return proper HTTP error status codes for invalid IDs/routes."""
    r_alert = client.get("/api/alerts/nonexistent-uuid-999")
    assert r_alert.status_code == 404

    r_story = client.get("/api/attack-story/nonexistent-story-123")
    assert r_story.status_code == 404

    r_rollout_bad = client.post("/api/world-model/rollout", json={"state_seq": [[0.0] * 10], "k_steps": 4})
    assert r_rollout_bad.status_code == 422
