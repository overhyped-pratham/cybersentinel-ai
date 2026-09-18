"""
CyberSentinel AI — End-to-End DetectionPipeline Tests.
"""

from pathlib import Path
import numpy as np
import pytest

from ml.pipeline.detection_pipeline import DetectionPipeline, PipelineDetectionResult

_WORKSPACE = Path(__file__).resolve().parent.parent


@pytest.fixture
def pipeline():
    return DetectionPipeline()


def test_pipeline_instantiation(pipeline):
    assert pipeline is not None
    assert pipeline.classifier is not None
    assert pipeline.novelty_detector is not None
    assert pipeline.risk_engine is not None


def test_process_state_benign(pipeline):
    benign_state = np.zeros(24, dtype=np.float32)
    result = pipeline.process_state(benign_state, is_scaled=True)

    assert isinstance(result, PipelineDetectionResult)
    assert 0.0 <= result.risk_score <= 100.0
    assert result.risk_severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert result.threat_classification in ["Confirmed Known Threat", "Potential Novel Behavior", "Benign Baseline"]
    assert "predicted_category" in result.known_classifier
    assert "anomaly_score" in result.novelty_detector
    assert "components" in result.risk_assessment
    assert "rollout_steps" in result.world_model_rollout
    assert len(result.current_state) == 24


def test_process_state_novel_deviation(pipeline):
    # Large anomalous state vector
    extreme_state = np.ones(24, dtype=np.float32) * 20.0
    result = pipeline.process_state(extreme_state, is_scaled=True)

    assert result.is_novel is True
    assert result.novelty_detector["anomaly_score"] > 0.50
    assert result.threat_classification == "Potential Novel Behavior"
    assert result.risk_score >= 30.0  # Elevated risk on novel behavior


def test_to_dict_serialization(pipeline):
    state = np.zeros(24, dtype=np.float32)
    result = pipeline.process_state(state, is_scaled=True)
    d = result.to_dict()

    assert isinstance(d, dict)
    for key in [
        "timestamp", "current_stage", "predicted_next_stage", "is_attack",
        "is_novel", "threat_classification", "risk_score", "risk_severity",
        "known_classifier", "novelty_detector", "risk_assessment", "world_model_rollout"
    ]:
        assert key in d, f"Missing key {key} in pipeline output dict"
