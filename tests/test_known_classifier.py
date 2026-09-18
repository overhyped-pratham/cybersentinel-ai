"""
CyberSentinel AI — Layer 1 Known Attack Classifier Tests.
"""

from pathlib import Path
import numpy as np
import pytest

from ml.classifier.known_attack_classifier import KnownAttackClassifier, ClassifierPrediction
from ml.state.state_builder import FEATURE_NAMES

_WORKSPACE = Path(__file__).resolve().parent.parent


@pytest.fixture
def trained_classifier():
    model_path = _WORKSPACE / "models" / "classifier" / "known_classifier.pkl"
    if model_path.exists():
        return KnownAttackClassifier.load(model_path)
    # If not saved, train a small one
    clf = KnownAttackClassifier(n_estimators=10, max_depth=3)
    X = np.random.randn(50, 24).astype(np.float32)
    y = ["BENIGN"] * 25 + ["RECONNAISSANCE"] * 25
    clf.fit(X, y)
    return clf


def test_classifier_structure(trained_classifier):
    assert trained_classifier.is_trained is True
    assert len(trained_classifier.feature_names) == 24
    assert len(trained_classifier.class_names) >= 2


def test_predict_single_benign_shape(trained_classifier):
    sample = np.zeros(24, dtype=np.float32)
    pred = trained_classifier.predict_single(sample)
    assert isinstance(pred, ClassifierPrediction)
    assert 0.0 <= pred.confidence <= 1.0
    assert 0.0 <= pred.attack_probability <= 1.0
    assert isinstance(pred.is_attack, bool)
    assert pred.predicted_category in trained_classifier.class_names
    assert len(pred.top_features) > 0


def test_probability_distribution_sums_to_one(trained_classifier):
    sample = np.ones(24, dtype=np.float32) * 0.5
    probs = trained_classifier.predict_proba(sample)[0]
    assert np.isclose(np.sum(probs), 1.0, atol=1e-4)


def test_shap_explanation(trained_classifier):
    sample = np.ones(24, dtype=np.float32) * 2.0
    explanations = trained_classifier.explain_instance(sample, top_k=5)
    assert len(explanations) <= 5
    for item in explanations:
        assert "feature" in item
        assert item["feature"] in FEATURE_NAMES
        assert "attribution" in item
