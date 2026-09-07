"""
Unit tests for ml.evaluation.metrics.
"""

import pytest
import numpy as np
from pathlib import Path

from ml.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_forecasting_metrics,
    save_confusion_matrix_plot,
)


def test_binary_classification_metrics():
    # 4 samples: TP, TN, FP, FN
    y_true = np.array([1, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 0])
    y_prob = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])

    metrics = calculate_classification_metrics(y_true, y_pred, y_prob, is_binary=True)

    assert metrics["accuracy"] == 0.5
    # Confusion matrix: [[TN=1, FP=1], [FN=1, TP=1]]
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["fpr"] == 0.5
    assert metrics["roc_auc"] is not None
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_forecasting_metrics():
    # True next stages: [1, 2, 3]
    # Forecast probs: shape (3, 5)
    y_true_next = np.array([1, 2, 3])
    y_prob_next = np.array([
        [0.05, 0.80, 0.05, 0.05, 0.05], # top-1 correct (1)
        [0.10, 0.40, 0.35, 0.10, 0.05], # top-1 wrong (1), but top-2 correct (2)
        [0.70, 0.10, 0.05, 0.10, 0.05], # top-1 wrong (0), top-3 wrong
    ])

    f_metrics = calculate_forecasting_metrics(y_true_next, y_prob_next, k=2)

    # 1 of 3 is top-1 correct -> acc = 1/3
    assert np.isclose(f_metrics["next_stage_accuracy"], 1.0 / 3.0)
    # 2 of 3 is top-2 correct -> acc = 2/3
    assert np.isclose(f_metrics["top_2_accuracy"], 2.0 / 3.0)
    # Brier score must be non-negative
    assert f_metrics["brier_score"] >= 0.0


def test_save_confusion_matrix_plot(tmp_path: Path):
    cm = [[5, 1], [0, 8]]
    class_names = ["BENIGN", "ATTACK"]
    out_file = tmp_path / "cm.png"

    save_confusion_matrix_plot(cm, class_names, out_file, title="Test Confusion Matrix")
    assert out_file.exists()
    assert out_file.stat().st_size > 500
