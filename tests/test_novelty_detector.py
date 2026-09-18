"""
CyberSentinel AI — Layer 2 Autoencoder Novelty Detector Tests.
"""

from pathlib import Path
import numpy as np
import pytest

from ml.novelty.autoencoder_detector import AutoencoderNoveltyDetector, NoveltyDetectionResult
from ml.state.state_builder import FEATURE_NAMES

_WORKSPACE = Path(__file__).resolve().parent.parent


@pytest.fixture
def novelty_detector():
    model_path = _WORKSPACE / "models" / "novelty" / "autoencoder.pt"
    if model_path.exists():
        return AutoencoderNoveltyDetector.load(model_path)
    det = AutoencoderNoveltyDetector(input_dim=24, latent_dim=8)
    X = np.random.randn(40, 24).astype(np.float32)
    det.fit(X, epochs=5, batch_size=8)
    det.calibrate(X[:10])
    return det


def test_detector_structure(novelty_detector):
    assert novelty_detector.is_trained is True
    assert novelty_detector.is_calibrated is True
    assert novelty_detector.threshold > 0.0
    assert len(novelty_detector.feature_names) == 24


def test_detect_single_format(novelty_detector):
    sample = np.zeros(24, dtype=np.float32)
    result = novelty_detector.detect_single(sample)
    assert isinstance(result, NoveltyDetectionResult)
    assert result.reconstruction_error >= 0.0
    assert 0.0 <= result.anomaly_score <= 1.0
    assert isinstance(result.is_novel, bool)
    assert result.label in ["Potential Novel Behavior", "Normal Baseline"]
    assert len(result.feature_deviations) > 0


def test_anomaly_score_increases_with_extreme_deviation(novelty_detector):
    normal_sample = np.zeros(24, dtype=np.float32)
    extreme_sample = np.ones(24, dtype=np.float32) * 50.0  # extreme synthetic deviation
    normal_res = novelty_detector.detect_single(normal_sample)
    extreme_res = novelty_detector.detect_single(extreme_sample)

    assert extreme_res.reconstruction_error > normal_res.reconstruction_error
    assert extreme_res.anomaly_score > normal_res.anomaly_score
    assert extreme_res.is_novel is True
    assert extreme_res.label == "Potential Novel Behavior"
