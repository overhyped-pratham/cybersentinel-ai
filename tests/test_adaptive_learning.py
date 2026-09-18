"""
CyberSentinel AI — Adaptive Learning & Threat Memory Tests.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from ml.adaptation.threat_memory import ThreatMemory, ValidatedSample
from ml.adaptation.adaptive_learner import AdaptiveLearner, AdaptationRecord
from ml.classifier.known_attack_classifier import KnownAttackClassifier


@pytest.fixture
def temp_dirs():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        yield Path(d1), Path(d2)


def test_threat_memory_lifecycle(temp_dirs):
    storage_dir, _ = temp_dirs
    store_file = storage_dir / "threat_memory.json"
    tm = ThreatMemory(storage_path=store_file)

    assert len(tm.get_all_samples()) == 0

    sample = tm.add_validation(
        feature_vector=np.ones(24),
        validated_label="EXFILTRATION",
        is_malicious=True,
        original_stage="UNKNOWN",
        original_risk_score=75.0,
        original_anomaly_score=0.88,
        analyst_notes="Confirmed novel exfiltration behavior across port 443.",
        analyst_id="analyst_01",
    )

    assert sample.validated_label == "EXFILTRATION"
    assert sample.is_malicious is True
    assert len(tm.get_unincorporated_samples()) == 1

    # Reload from disk
    tm2 = ThreatMemory(storage_path=store_file)
    assert len(tm2.get_all_samples()) == 1
    loaded = tm2.get_sample(sample.sample_id)
    assert loaded is not None
    assert loaded.analyst_id == "analyst_01"

    # Mark incorporated
    tm2.mark_incorporated([sample.sample_id], version="2.1.0")
    assert len(tm2.get_unincorporated_samples()) == 0


def test_adaptive_learner_workflow(temp_dirs):
    models_dir, ledger_dir = temp_dirs
    clf_dir = models_dir / "classifier"
    clf_dir.mkdir(parents=True)
    ledger_file = ledger_dir / "adaptation_ledger.json"

    # Initial base classifier
    base_clf = KnownAttackClassifier(n_estimators=10, max_depth=3)
    X = np.random.randn(30, 24).astype(np.float32)
    y = ["BENIGN"] * 15 + ["RECONNAISSANCE"] * 15
    base_clf.fit(X, y)
    base_clf.save(clf_dir / "known_classifier.pkl")

    tm = ThreatMemory(storage_path=ledger_dir / "tm.json")
    # Add a novel sample that looks like a new attack
    novel_vec = np.ones(24) * 3.0
    tm.add_validation(
        feature_vector=novel_vec,
        validated_label="CREDENTIAL_ACCESS",
        is_malicious=True,
    )

    learner = AdaptiveLearner(
        classifier=base_clf,
        threat_memory=tm,
        models_dir=models_dir,
        ledger_path=ledger_file,
    )

    record = learner.adapt_model(new_version="2.1.0")
    assert isinstance(record, AdaptationRecord)
    assert record.from_version == "2.0.0"
    assert record.to_version == "2.1.0"
    assert learner.active_version == "2.1.0"
    assert "accuracy" in record.before_metrics
    assert "accuracy" in record.after_metrics
    assert Path(record.model_snapshot_path).exists()

    # Test Rollback
    rollback_res = learner.rollback()
    assert rollback_res["status"] == "rolled_back"
    assert learner.active_version == "2.0.0"
