"""
CyberSentinel AI — Unseen Attack Experiment Automated Validation Test.
"""

from pathlib import Path
import json
import pytest

_WORKSPACE = Path(__file__).resolve().parent.parent


def test_unseen_experiment_artifacts_exist():
    report_json = _WORKSPACE / "experiments" / "unseen_attack" / "unseen_experiment_report.json"
    report_md = _WORKSPACE / "experiments" / "unseen_attack" / "README.md"
    assert report_json.exists(), "Experiment report JSON missing"
    assert report_md.exists(), "Experiment README.md missing"


def test_unseen_experiment_metrics_pass_criteria():
    report_json = _WORKSPACE / "experiments" / "unseen_attack" / "unseen_experiment_report.json"
    with open(report_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Known attack classifier performance
    known = data["known_attack_performance"]
    assert known["accuracy"] >= 0.90, f"Known accuracy {known['accuracy']} below target 0.90"
    assert known["f1_macro"] >= 0.85, f"Known macro-F1 {known['f1_macro']} below target 0.85"
    assert known["false_positive_rate"] <= 0.05, f"Known FPR {known['false_positive_rate']} above target 0.05"

    # 2. Held-out novel behavior performance
    held_out = data["held_out_novel_behavior_performance"]
    assert held_out["novelty_detection_rate"] >= 0.80, f"Novelty detection rate {held_out['novelty_detection_rate']} below 0.80"
    assert held_out["auroc"] >= 0.85, f"Novelty AUROC {held_out['auroc']} below 0.85"
    assert held_out["novel_behavior_rate"] >= 0.80, f"Novel behavior classification rate below 0.80"
    assert held_out["mean_risk_score"] >= 50.0, f"Held-out risk score {held_out['mean_risk_score']} below 50.0"

    # 3. Leakage controls verified
    assert len(data["leakage_controls"]) >= 3
