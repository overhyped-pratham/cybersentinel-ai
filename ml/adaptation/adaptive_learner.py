"""
CyberSentinel AI — Controlled Adaptive Learning Pipeline.

PRD Requirement (Section 12, Non-Negotiable Principle 4 & Priority 5):
- Incorporates human-validated samples into a controlled model update pipeline.
- Generates semantic versioning (e.g., 2.0.0 -> 2.1.0).
- Produces rigorous BEFORE vs AFTER evaluation metrics on the newly learned threats.
- Provides immediate rollback capability to previous model checkpoint.
- Preserves full audit ledger in artifacts/adaptation/adaptation_ledger.json.
"""

from __future__ import annotations

import datetime
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ml.adaptation.threat_memory import ThreatMemory, ValidatedSample
from ml.classifier.known_attack_classifier import KnownAttackClassifier
from ml.novelty.autoencoder_detector import AutoencoderNoveltyDetector
from ml.preprocessing.scaler import FeatureScaler
from ml.state.state_builder import FEATURE_NAMES

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
_LEDGER_PATH = _ROOT / "artifacts" / "adaptation" / "adaptation_ledger.json"


@dataclass
class AdaptationRecord:
    """Audit record for a single controlled model adaptation step."""
    update_id: str
    from_version: str
    to_version: str
    timestamp: str
    sample_count: int
    sample_ids: List[str]
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    metrics_delta: Dict[str, Any]
    model_snapshot_path: str
    rollback_available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AdaptationRecord":
        return cls(**d)


class AdaptiveLearner:
    """
    Orchestrates controlled retraining and model adaptation using human-validated Threat Memory.
    """

    def __init__(
        self,
        classifier: Optional[KnownAttackClassifier] = None,
        threat_memory: Optional[ThreatMemory] = None,
        models_dir: Optional[Path] = None,
        ledger_path: Optional[Path] = None,
    ) -> None:
        self.models_dir = models_dir if models_dir is not None else (_ROOT / "models")
        self.ledger_path = ledger_path if ledger_path is not None else _LEDGER_PATH
        self.threat_memory = threat_memory if threat_memory is not None else ThreatMemory()
        self.classifier = classifier if classifier is not None else self._load_active_classifier()
        self.active_version = "2.0.0"
        self._ledger: List[AdaptationRecord] = []
        self._load_ledger()

    def _load_active_classifier(self) -> KnownAttackClassifier:
        clf_path = self.models_dir / "classifier" / "known_classifier.pkl"
        if clf_path.exists():
            return KnownAttackClassifier.load(clf_path)
        return KnownAttackClassifier()

    def _load_ledger(self) -> None:
        if not self.ledger_path.exists():
            return
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._ledger = [AdaptationRecord.from_dict(d) for d in data.get("updates", [])]
            if self._ledger:
                self.active_version = self._ledger[-1].to_version
        except Exception as exc:
            logger.error("Failed to load adaptation ledger: %s", exc)

    def _save_ledger(self) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "active_version": self.active_version,
            "total_updates": len(self._ledger),
            "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "updates": [rec.to_dict() for rec in self._ledger],
        }
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def evaluate_on_samples(
        self,
        classifier: KnownAttackClassifier,
        samples: List[ValidatedSample],
    ) -> Dict[str, Any]:
        """Evaluates classifier performance specifically on a list of validated samples."""
        if not samples:
            return {"accuracy": 1.0, "mean_confidence": 1.0, "correct_count": 0, "total": 0}

        X = np.array([s.feature_vector for s in samples], dtype=np.float32)
        y_true = [s.validated_label for s in samples]

        probs = classifier.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        pred_labels = [classifier.class_names[idx] for idx in pred_indices]
        confidences = [float(probs[i, pred_indices[i]]) for i in range(len(samples))]

        # Check match against validated label or attack identification
        correct = sum(1 for i in range(len(samples)) if pred_labels[i] == y_true[i] or (samples[i].is_malicious and pred_labels[i] != "BENIGN"))
        acc = float(correct / len(samples))
        mean_conf = float(np.mean(confidences))

        return {
            "accuracy": round(acc, 4),
            "mean_confidence": round(mean_conf, 4),
            "correct_count": correct,
            "total": len(samples),
            "predicted_labels": pred_labels,
        }

    def adapt_model(
        self,
        new_version: Optional[str] = None,
        base_dataset_path: Optional[Path] = None,
    ) -> AdaptationRecord:
        """
        Executes controlled model adaptation:
          1. Collects unincorporated validated samples from ThreatMemory.
          2. Evaluates current model on these samples (BEFORE metrics).
          3. Backs up current model for rollback capability.
          4. Fine-tunes/retrains classifier with base training data + new validated samples.
          5. Evaluates updated model on these samples (AFTER metrics).
          6. Records audit entry to adaptation ledger.
        """
        pending_samples = self.threat_memory.get_unincorporated_samples()
        if not pending_samples:
            logger.info("No unincorporated validated samples found in Threat Memory. Skipping adaptation.")
            raise ValueError("No unincorporated samples in Threat Memory to adapt on.")

        # Compute next version
        curr_ver = self.active_version
        if new_version is None:
            parts = curr_ver.split(".")
            minor = int(parts[1]) + 1
            to_ver = f"{parts[0]}.{minor}.0"
        else:
            to_ver = new_version

        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        update_id = f"adapt-{int(datetime.datetime.now(datetime.timezone.utc).timestamp())}"

        # 1. Evaluate BEFORE Adaptation
        logger.info("[AdaptiveLearning] Evaluating model %s BEFORE adaptation...", curr_ver)
        before_metrics = self.evaluate_on_samples(self.classifier, pending_samples)

        # 2. Snapshot current model for Rollback
        clf_dir = self.models_dir / "classifier"
        active_clf_path = clf_dir / "known_classifier.pkl"
        snapshot_path = clf_dir / f"known_classifier_{curr_ver}.pkl"
        if active_clf_path.exists():
            shutil.copy2(active_clf_path, snapshot_path)
            logger.info("Created rollback snapshot at %s", snapshot_path)

        # 3. Form combined training dataset
        # Load sample dataset features
        from network.flow.csv_loader import CSVFlowLoader
        from ml.state.state_builder import NetworkStateBuilder
        from ml.preprocessing.stage_labeler import StageLabeler
        
        data_dir = base_dataset_path or (_ROOT / "datasets" / "sample")
        loader = CSVFlowLoader()
        all_flows = []
        for csv_file in sorted(data_dir.glob("*.csv"))[:10]: # Representative traces
            all_flows.extend(loader.load_flows(csv_file))

        state_builder = NetworkStateBuilder(window_size_seconds=30.0)
        df_states = state_builder.build_states(all_flows)
        labeler = StageLabeler(fallback_to_heuristics=True)
        df_states = labeler.attach_labels_to_dataframe(df_states)

        scaler = FeatureScaler.load(_ROOT / "models" / "scaler.pkl")
        X_base = scaler.transform(df_states)
        y_base = df_states["stage_name"].to_numpy().tolist()

        # Add validated samples with weight/repetition
        X_new = [s.feature_vector for s in pending_samples]
        y_new = [s.validated_label for s in pending_samples]

        # Duplicate new samples so they are strongly learned
        X_combined = np.vstack([X_base] + [np.array(X_new, dtype=np.float32)] * 5)
        y_combined = y_base + y_new * 5

        # 4. Train Updated Classifier
        logger.info("[AdaptiveLearning] Retraining classifier on %d total samples...", len(X_combined))
        updated_classifier = KnownAttackClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
        )
        updated_classifier.fit(X_combined, y_combined)

        # 5. Evaluate AFTER Adaptation
        logger.info("[AdaptiveLearning] Evaluating model %s AFTER adaptation...", to_ver)
        after_metrics = self.evaluate_on_samples(updated_classifier, pending_samples)

        # Save new model as active
        updated_classifier.save(active_clf_path)
        self.classifier = updated_classifier
        self.active_version = to_ver

        # Mark samples in ThreatMemory
        sample_ids = [s.sample_id for s in pending_samples]
        self.threat_memory.mark_incorporated(sample_ids, to_ver)

        # Metrics delta
        metrics_delta = {
            "accuracy_delta": round(after_metrics["accuracy"] - before_metrics["accuracy"], 4),
            "confidence_delta": round(after_metrics["mean_confidence"] - before_metrics["mean_confidence"], 4),
        }

        record = AdaptationRecord(
            update_id=update_id,
            from_version=curr_ver,
            to_version=to_ver,
            timestamp=now_str,
            sample_count=len(pending_samples),
            sample_ids=sample_ids,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            metrics_delta=metrics_delta,
            model_snapshot_path=str(snapshot_path),
            rollback_available=True,
        )

        self._ledger.append(record)
        self._save_ledger()
        logger.info("Adaptation completed successfully: %s -> %s (delta: %s)", curr_ver, to_ver, metrics_delta)
        return record

    def rollback(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Rolls back the active model to the specified version (or the most recent snapshot).
        """
        clf_dir = self.models_dir / "classifier"
        active_clf_path = clf_dir / "known_classifier.pkl"

        if target_version:
            target_snapshot = clf_dir / f"known_classifier_{target_version}.pkl"
            if not target_snapshot.exists():
                raise FileNotFoundError(f"Snapshot for version {target_version} not found at {target_snapshot}")
            restored_ver = target_version
        else:
            if not self._ledger:
                raise RuntimeError("No adaptation history found to rollback.")
            last_rec = self._ledger[-1]
            target_snapshot = Path(last_rec.model_snapshot_path)
            restored_ver = last_rec.from_version

        if not target_snapshot.exists():
            raise FileNotFoundError(f"Rollback file missing: {target_snapshot}")

        shutil.copy2(target_snapshot, active_clf_path)
        self.classifier = KnownAttackClassifier.load(active_clf_path)
        previous_ver = self.active_version
        self.active_version = restored_ver

        logger.info("Rolled back model from %s to %s", previous_ver, restored_ver)
        return {
            "status": "rolled_back",
            "previous_version": previous_ver,
            "active_version": restored_ver,
            "restored_snapshot": str(target_snapshot),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._ledger]
