"""
CyberSentinel AI — Calibration Module (Phase 9).

Loads the validation-fitted temperature scaling parameter and
applies it to raw model logits to produce calibrated probabilities.

Guarantee: temperature scaling does NOT change the argmax of the distribution
(prediction invariance). It only adjusts probability magnitudes for better
calibration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_ARTIFACT = Path(__file__).resolve().parent.parent.parent / "artifacts" / "calibration" / "temperature.json"


def load_temperature(artifact_path: Optional[Union[str, Path]] = None) -> float:
    """
    Load optimal temperature T from the calibration artifact.

    Args:
        artifact_path: Path to temperature.json. Defaults to artifacts/calibration/temperature.json.

    Returns:
        T (float): optimal temperature value; 1.0 if artifact not found.
    """
    path = Path(artifact_path) if artifact_path else _DEFAULT_ARTIFACT
    if not path.exists():
        logger.warning("Calibration artifact not found at %s. Using T=1.0 (uncalibrated).", path)
        return 1.0
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    T = float(data.get("optimal_temperature", 1.0))
    logger.info("Calibration temperature T=%.4f loaded from %s", T, path)
    return T


def apply_temperature_scaling(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Apply temperature scaling to raw logits and return calibrated probabilities.

    P_calibrated(y) = softmax(logits / T)

    Args:
        logits:      (N, num_classes) — raw pre-softmax logits
        temperature: positive float; T > 1 softens, T < 1 sharpens.

    Returns:
        probs: (N, num_classes) — calibrated probabilities summing to 1.

    Raises:
        ValueError: if temperature <= 0.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}")

    T = max(float(temperature), 1e-6)
    scaled = logits / T
    # Numerically stable softmax
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def verify_prediction_invariance(
    logits: np.ndarray,
    temperature: float,
) -> Tuple[bool, float]:
    """
    Verify that temperature scaling does not change predicted class (argmax invariance).

    Args:
        logits:      (N, num_classes) raw logits
        temperature: scaling temperature

    Returns:
        (invariant: bool, agreement_rate: float)
    """
    pred_raw = np.argmax(logits, axis=1)
    probs_cal = apply_temperature_scaling(logits, temperature)
    pred_cal = np.argmax(probs_cal, axis=1)
    agreement = float(np.mean(pred_raw == pred_cal))
    return agreement == 1.0, agreement


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) via equal-width confidence binning.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (confidences > bins[b]) & (confidences <= bins[b + 1])
        if np.any(mask):
            ece += np.sum(mask) / len(labels) * abs(np.mean(correct[mask]) - np.mean(confidences[mask]))
    return float(ece)


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Brier score for multi-class probabilistic forecast.
    Brier = mean over N of: sum_c (p(c) - 1[y==c])^2
    """
    N, C = probs.shape
    one_hot = np.zeros((N, C))
    for i, lbl in enumerate(labels):
        one_hot[i, int(lbl)] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
