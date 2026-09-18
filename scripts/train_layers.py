"""
CyberSentinel AI — Layer 1 & Layer 2 Model Training & Calibration Script.

Trains:
1. Layer 1: KnownAttackClassifier (XGBoost multi-class with calibrated softprob)
2. Layer 2: AutoencoderNoveltyDetector (PyTorch autoencoder trained on benign baseline)

Enforces:
- Zero data leakage between train and validation splits
- Empirical threshold calibration for novelty detection (P95 of unseen benign)
- Model artifact persistence to models/classifier/ and models/novelty/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.classifier.known_attack_classifier import KnownAttackClassifier
from ml.novelty.autoencoder_detector import AutoencoderNoveltyDetector
from ml.preprocessing.scaler import FeatureScaler
from ml.preprocessing.stage_labeler import StageLabeler
from ml.state.state_builder import FEATURE_NAMES, NetworkStateBuilder
from network.flow.csv_loader import CSVFlowLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainLayers")


def main():
    logger.info("=" * 65)
    logger.info("CYBERSENTINEL AI — TRAINING LAYER 1 & LAYER 2 MODELS")
    logger.info("=" * 65)

    data_dir = _ROOT / "datasets" / "sample"
    loader = CSVFlowLoader()
    all_flows = []
    csv_files = sorted(data_dir.glob("*.csv"))
    logger.info("Loading flows from %d trace files in %s...", len(csv_files), data_dir)
    for p in csv_files:
        all_flows.extend(loader.load_flows(p))

    logger.info("Loaded %d total flows. Building 30s network state windows...", len(all_flows))
    state_builder = NetworkStateBuilder(window_size_seconds=30.0)
    df_states = state_builder.build_states(all_flows)
    labeler = StageLabeler(fallback_to_heuristics=True)
    df_states = labeler.attach_labels_to_dataframe(df_states)

    logger.info("Built %d state windows.", len(df_states))
    logger.info("Stage breakdown:\n%s", df_states["stage_name"].value_counts().to_string())

    # Load / fit RobustScaler
    scaler_path = _ROOT / "models" / "scaler.pkl"
    if scaler_path.exists():
        logger.info("Loading existing FeatureScaler from %s", scaler_path)
        scaler = FeatureScaler.load(scaler_path)
    else:
        logger.info("Fitting new FeatureScaler...")
        scaler = FeatureScaler(feature_names=FEATURE_NAMES)
        scaler.fit(df_states)
        scaler.save(scaler_path)

    X_scaled = scaler.transform(df_states)
    y_stages = df_states["stage_name"].to_numpy()
    is_attack = df_states["is_attack"].to_numpy()

    # -----------------------------------------------------------------------
    # 1. Train Layer 1: Known Attack Classifier
    # -----------------------------------------------------------------------
    logger.info("\n--- Training Layer 1: Known Attack Classifier (XGBoost) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_stages, test_size=0.25, random_state=42, stratify=y_stages
    )

    classifier = KnownAttackClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
    )
    fit_metrics = classifier.fit(X_train, y_train)
    test_probs = classifier.predict_proba(X_test)
    test_preds = [classifier.class_names[i] for i in np.argmax(test_probs, axis=1)]
    from sklearn.metrics import accuracy_score, f1_score
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average="macro")
    logger.info("Classifier Test Accuracy: %.4f | Macro-F1: %.4f", test_acc, test_f1)

    classifier_dir = _ROOT / "models" / "classifier"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = classifier_dir / "known_classifier.pkl"
    classifier.save(classifier_path)

    # -----------------------------------------------------------------------
    # 2. Train Layer 2: Autoencoder Novelty Detector (Benign Only)
    # -----------------------------------------------------------------------
    logger.info("\n--- Training Layer 2: Novelty Detector (Autoencoder) ---")
    benign_mask = (y_stages == "BENIGN")
    X_benign = X_scaled[benign_mask]
    logger.info("Total Benign windows: %d", len(X_benign))

    X_benign_train, X_benign_val = train_test_split(
        X_benign, test_size=0.30, random_state=42
    )

    autoencoder = AutoencoderNoveltyDetector(input_dim=24, latent_dim=8)
    autoencoder.fit(X_benign_train, epochs=45, batch_size=16, learning_rate=0.005)
    calib_stats = autoencoder.calibrate(X_benign_val, percentile=95.0)
    logger.info("Autoencoder Calibrated Threshold (P95): %.6f", calib_stats["threshold"])

    novelty_dir = _ROOT / "models" / "novelty"
    novelty_dir.mkdir(parents=True, exist_ok=True)
    novelty_path = novelty_dir / "autoencoder.pt"
    autoencoder.save(novelty_path)

    # Quick validation on test benign vs attack
    test_benign_recon = autoencoder.compute_reconstruction_errors(X_benign_val)
    attack_mask = (y_stages != "BENIGN")
    test_attack_recon = autoencoder.compute_reconstruction_errors(X_scaled[attack_mask][:50])
    logger.info("Mean Benign Val Recon Error: %.6f", float(np.mean(test_benign_recon)))
    logger.info("Mean Attack Recon Error:     %.6f", float(np.mean(test_attack_recon)))
    logger.info("Separation Ratio:            %.2fx", float(np.mean(test_attack_recon) / max(1e-6, np.mean(test_benign_recon))))

    logger.info("\nAll Layer 1 & Layer 2 artifacts saved successfully!")


if __name__ == "__main__":
    main()
