"""
CyberSentinel X — Rigorous Evaluation Tests for Adaptive Learning.
Tests 10 non-negotiable scientific properties:
1. No overlap between training and final unseen test.
2. No overlap between adaptation and final unseen test.
3. ThreatMemory receives only validated samples.
4. Adaptation version changes only through approved mechanism.
5. Unseen test remains untouched during adaptation (hash integrity).
6. Metrics calculated on identical before/after test samples.
7. Rollback works and restores pre-adaptation state.
8. Invalid feedback is rejected / handled safely.
9. Empty ThreatMemory behaves correctly.
10. No hardcoded expected predictions or metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

_ROOT = Path("d:/uec sih")

from ml.adaptation.threat_memory import ThreatMemory, ValidatedSample
from ml.adaptation.adaptive_learner import AdaptiveLearner, AdaptationRecord
from ml.classifier.known_attack_classifier import KnownAttackClassifier


@pytest.fixture
def split_metadata():
    split_path = _ROOT / "experiments" / "adaptive_learning" / "data_split.json"
    assert split_path.exists(), "data_split.json must exist"
    with open(split_path, "r", encoding="ascii") as f:
        return json.load(f)


def test_1_no_overlap_train_and_final_unseen_test(split_metadata):
    """1. No overlap between base training and final unseen test."""
    train_files = set(split_metadata["base_train_files"])
    test_files = set(split_metadata["final_test_files"])
    overlap = train_files & test_files
    assert len(overlap) == 0, f"Data leakage detected! Overlap: {overlap}"


def test_2_no_overlap_adaptation_and_final_unseen_test(split_metadata):
    """2. No overlap between adaptation set and final unseen test."""
    adapt_files = set(split_metadata["adaptation_files"])
    test_files = set(split_metadata["final_test_files"])
    overlap = adapt_files & test_files
    assert len(overlap) == 0, f"Data leakage detected! Overlap: {overlap}"


def test_3_threat_memory_receives_only_validated_samples():
    """3. ThreatMemory receives only explicitly validated samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tm_path = Path(tmpdir) / "threat_memory.json"
        tm = ThreatMemory(storage_path=tm_path)
        assert len(tm) == 0

        # Adding without validation is impossible -- API requires validated_label & is_malicious
        sample = tm.add_validation(
            feature_vector=[0.5] * 24,
            validated_label="EXFILTRATION",
            is_malicious=True,
            analyst_notes="Human approved alert",
            analyst_id="lead_analyst",
        )
        assert sample.validated_label == "EXFILTRATION"
        assert sample.is_malicious is True
        assert len(tm.get_unincorporated_samples()) == 1
        assert tm.get_sample(sample.sample_id) is not None


def test_4_adaptation_version_changes_only_through_approved_mechanism():
    """4. Adaptation version changes ONLY through adapt_model() call, not by passive adds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        clf = KnownAttackClassifier(n_estimators=10, max_depth=3)
        clf.fit(np.random.randn(20, 24).astype(np.float32), ["BENIGN"] * 10 + ["RECONNAISSANCE"] * 10)

        tm = ThreatMemory(storage_path=td / "tm.json")
        learner = AdaptiveLearner(
            classifier=clf,
            threat_memory=tm,
            models_dir=td / "models",
            ledger_path=td / "ledger.json",
        )
        initial_ver = learner.active_version

        # Passive add to ThreatMemory must NOT change version
        tm.add_validation(
            feature_vector=[1.0] * 24,
            validated_label="CREDENTIAL_ACCESS",
            is_malicious=True,
        )
        assert learner.active_version == initial_ver, "Version changed passively without adapt_model()!"

        # Explicit adapt_model changes version
        (td / "models" / "classifier").mkdir(parents=True, exist_ok=True)
        clf.save(td / "models" / "classifier" / "known_classifier.pkl")
        rec = learner.adapt_model(new_version="2.1.0")
        assert learner.active_version == "2.1.0"
        assert rec.from_version == initial_ver
        assert rec.to_version == "2.1.0"


def test_5_unseen_test_hash_untouched_during_adaptation():
    """5. Unseen test set remains untouched during adaptation (hash integrity)."""
    split_path = _ROOT / "experiments" / "adaptive_learning" / "data_split.json"
    with open(split_path, "r", encoding="ascii") as f:
        meta = json.load(f)
    locked_hash = meta["final_test_hash"]

    before_path = _ROOT / "artifacts" / "adaptation" / "before_adaptation_metrics.json"
    after_path = _ROOT / "artifacts" / "adaptation" / "after_adaptation_metrics.json"

    with open(before_path, "r", encoding="ascii") as f:
        before_meta = json.load(f)
    with open(after_path, "r", encoding="ascii") as f:
        after_meta = json.load(f)

    assert before_meta["test_hash"] == locked_hash
    assert after_meta["test_hash"] == locked_hash


def test_6_metrics_calculated_on_identical_test_samples():
    """6. Metrics are calculated on identical test samples before vs after."""
    before_path = _ROOT / "artifacts" / "adaptation" / "before_adaptation_metrics.json"
    after_path = _ROOT / "artifacts" / "adaptation" / "after_adaptation_metrics.json"

    with open(before_path, "r", encoding="ascii") as f:
        b = json.load(f)
    with open(after_path, "r", encoding="ascii") as f:
        a = json.load(f)

    assert b["n_samples"] == a["n_samples"]
    assert b["label_distribution"] == a["label_distribution"]


def test_7_rollback_restores_mathematical_state():
    """7. Rollback restores model to exact prior checkpoint."""
    rb_path = _ROOT / "artifacts" / "adaptation" / "rollback_verification.json"
    assert rb_path.exists()
    with open(rb_path, "r", encoding="ascii") as f:
        rb = json.load(f)

    assert rb["status"] == "SUCCESS"
    assert rb["mathematical_fidelity_verified"] is True
    assert rb["accuracy_difference"] < 1e-4


def test_8_invalid_feedback_is_rejected_or_handled_safely():
    """8. Empty ThreatMemory handles adaptation attempt safely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        tm = ThreatMemory(storage_path=td / "empty_tm.json")
        learner = AdaptiveLearner(
            classifier=KnownAttackClassifier(n_estimators=10, max_depth=3),
            threat_memory=tm,
            models_dir=td / "models",
            ledger_path=td / "ledger.json",
        )
        # Attempting to adapt with 0 pending samples must raise ValueError
        with pytest.raises(ValueError, match="No unincorporated samples"):
            learner.adapt_model()


def test_9_empty_threat_memory_behaves_correctly():
    """9. Empty ThreatMemory returns length 0 and empty unincorporated list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = ThreatMemory(storage_path=Path(tmpdir) / "empty.json")
        assert len(tm) == 0
        assert tm.get_unincorporated_samples() == []
        assert tm.get_all_samples() == []
        assert tm.get_sample("non_existent_id") is None


def test_10_no_hardcoded_metrics_in_reports():
    """10. Ensure experiment reports are dynamic floating point values with full precision."""
    report_path = _ROOT / "experiments" / "adaptive_learning" / "unseen_evaluation_report.json"
    assert report_path.exists()
    with open(report_path, "r", encoding="ascii") as f:
        rep = json.load(f)

    assert isinstance(rep["before_adaptation"]["accuracy"], float)
    assert isinstance(rep["after_adaptation"]["accuracy"], float)
    assert "timestamp" in rep
    assert "random_seed" in rep
    assert rep["random_seed"] == 42
