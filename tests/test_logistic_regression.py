"""
Unit tests for ml.baseline.logistic_regression.LogisticRegressionBaseline.
"""

import pytest
import numpy as np
from pathlib import Path

from ml.baseline.logistic_regression import LogisticRegressionBaseline


def test_logistic_regression_fit_predict(tmp_path: Path):
    rng = np.random.default_rng(42)
    n_samples = 40
    n_features = 24

    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    y_attack = (rng.uniform(size=n_samples) > 0.5).astype(int)
    y_stage = rng.integers(0, 4, size=n_samples)
    y_next_stage = rng.integers(0, 4, size=n_samples)

    model = LogisticRegressionBaseline(C=1.0, max_iter=200, random_state=42)
    assert not model.is_fitted

    model.fit(X, y_attack, y_stage, y_next_stage)
    assert model.is_fitted

    # Attack prediction
    pred_atk = model.predict_attack(X)
    assert pred_atk.shape == (n_samples,)
    proba_atk = model.predict_attack_proba(X)
    assert proba_atk.shape == (n_samples, 2)
    assert np.allclose(proba_atk.sum(axis=1), 1.0)

    # Stage prediction
    pred_stage = model.predict_stage(X)
    assert pred_stage.shape == (n_samples,)
    proba_stage = model.predict_stage_proba(X)
    assert proba_stage.shape[0] == n_samples
    assert np.allclose(proba_stage.sum(axis=1), 1.0)

    # Next stage prediction
    proba_next = model.predict_next_stage_proba(X, num_total_classes=10)
    assert proba_next.shape == (n_samples, 10)
    assert np.allclose(proba_next.sum(axis=1), 1.0)

    # Persistence
    save_path = tmp_path / "baseline_lr.pkl"
    model.save(save_path)
    assert save_path.exists()

    loaded = LogisticRegressionBaseline.load(save_path)
    assert loaded.is_fitted
    assert np.allclose(loaded.predict_attack_proba(X), proba_atk)


def test_unfitted_raises():
    model = LogisticRegressionBaseline()
    X = np.zeros((5, 24))
    with pytest.raises(RuntimeError):
        model.predict_attack(X)
