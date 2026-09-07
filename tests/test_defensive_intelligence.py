"""
Tests for Phase 9 Defensive Intelligence Layer:
  - ForecastEvent construction and serialization
  - RiskEngine scoring formula correctness
  - Priority threshold mapping
  - MITRE mapper deterministic lookups
  - CyberSentinelAgentTools interface contracts
  - Explainability: feature delta computation and stage relevance
"""

from __future__ import annotations

import datetime
from typing import Dict

import numpy as np
import pytest

from ml.defense.risk_engine import (
    ForecastEvent,
    RiskEngine,
    RiskEngineConfig,
    RiskAssessment,
    STAGE_TAXONOMY,
    DEFAULT_SEVERITY,
)
from mitre.mappings.mitre_mapper import (
    get_techniques_for_stage,
    get_primary_technique,
    get_mitre_summary,
    list_all_stages,
    STAGE_TO_MITRE,
)
from ml.world_model.explainability import (
    compute_feature_deltas,
    get_top_features,
    get_stage_relevant_features,
    build_explanation,
    STAGE_FEATURE_RELEVANCE,
)
from agent.tools.forecasting_tools import CyberSentinelAgentTools
from ml.state.state_builder import FEATURE_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_STAGES = len(STAGE_TAXONOMY)
D = 24  # feature dimension


def _make_event(
    current_stage: str = "RECONNAISSANCE",
    predicted_stage: str = "CREDENTIAL_ACCESS",
    attack_probability: float = 0.85,
    confidence: float = 0.75,
    uncertainty: float = 0.15,
) -> ForecastEvent:
    """Create a minimal ForecastEvent for testing."""
    probs = {s: 0.0 for s in STAGE_TAXONOMY}
    probs[predicted_stage] = confidence
    leftover = (1.0 - confidence) / max(1, N_STAGES - 1)
    for s in STAGE_TAXONOMY:
        if s != predicted_stage:
            probs[s] = leftover

    return ForecastEvent(
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        model_version="2.0",
        horizon_seconds=30,
        current_stage=current_stage,
        current_state=list(np.zeros(D)),
        predicted_stage=predicted_stage,
        predicted_next_state=list(np.zeros(D)),
        attack_probability=attack_probability,
        stage_probabilities=probs,
        confidence=confidence,
        uncertainty_entropy=uncertainty,
        transition_detected=(predicted_stage != current_stage),
        top_features=[],
        scenario_id="test_scenario",
    )


# ---------------------------------------------------------------------------
# ForecastEvent Tests
# ---------------------------------------------------------------------------

class TestForecastEvent:
    def test_construction(self):
        event = _make_event()
        assert event.current_stage == "RECONNAISSANCE"
        assert event.predicted_stage == "CREDENTIAL_ACCESS"
        assert event.transition_detected is True

    def test_benign_no_transition(self):
        event = _make_event(current_stage="BENIGN", predicted_stage="BENIGN")
        assert event.transition_detected is False

    def test_to_dict_keys(self):
        event = _make_event()
        d = event.to_dict()
        for key in ["timestamp", "current_stage", "predicted_stage",
                    "attack_probability", "confidence", "transition_detected",
                    "stage_probabilities", "model_version"]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_probabilities_rounded(self):
        event = _make_event()
        d = event.to_dict()
        for stage, prob in d["stage_probabilities"].items():
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0

    def test_stage_probs_sum_approx_one(self):
        event = _make_event(confidence=0.75)
        total = sum(event.stage_probabilities.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total:.4f}"


# ---------------------------------------------------------------------------
# RiskEngine Tests
# ---------------------------------------------------------------------------

class TestRiskEngine:
    def test_returns_risk_assessment(self):
        engine = RiskEngine()
        event = _make_event()
        result = engine.evaluate(event, horizon_steps=1)
        assert isinstance(result, RiskAssessment)

    def test_score_bounded_0_100(self):
        engine = RiskEngine()
        for stage in STAGE_TAXONOMY:
            event = _make_event(predicted_stage=stage, attack_probability=1.0, confidence=1.0)
            result = engine.evaluate(event)
            assert 0.0 <= result.risk_score <= 100.0, (
                f"Score {result.risk_score} out of range for stage {stage}"
            )

    def test_critical_for_exfiltration(self):
        """High-confidence exfiltration with attack=1.0 should trigger CRITICAL."""
        engine = RiskEngine()
        event = _make_event(
            predicted_stage="EXFILTRATION",
            attack_probability=1.0,
            confidence=0.95,
        )
        result = engine.evaluate(event, horizon_steps=1)
        assert result.severity in ("HIGH", "CRITICAL"), (
            f"Expected HIGH or CRITICAL for exfiltration, got {result.severity} "
            f"(score={result.risk_score:.1f})"
        )

    def test_low_for_benign(self):
        """Low attack probability on BENIGN stage should give LOW priority."""
        engine = RiskEngine()
        event = _make_event(
            current_stage="BENIGN",
            predicted_stage="BENIGN",
            attack_probability=0.02,
            confidence=0.95,
        )
        result = engine.evaluate(event, horizon_steps=4)
        assert result.severity in ("LOW", "MEDIUM"), (
            f"Expected LOW for benign, got {result.severity} (score={result.risk_score:.1f})"
        )

    def test_score_formula_components_sum_to_total(self):
        """Verify that component parts add up consistently to the raw score."""
        engine = RiskEngine()
        event = _make_event(attack_probability=0.7, confidence=0.8)
        result = engine.evaluate(event, horizon_steps=1)
        reconstructed = (
            result.component_attack
            + result.component_severity
            + result.component_confidence
            + result.component_urgency
        )
        assert abs(result.risk_score - reconstructed * 100) < 0.1

    def test_custom_config(self):
        config = RiskEngineConfig(
            weight_attack=0.60,
            weight_severity=0.20,
            weight_forecast_confidence=0.10,
            weight_urgency=0.10,
        )
        engine = RiskEngine(config=config)
        event = _make_event(attack_probability=1.0, confidence=1.0)
        result = engine.evaluate(event)
        assert 0 <= result.risk_score <= 100

    def test_to_dict_keys(self):
        engine = RiskEngine()
        result = engine.evaluate(_make_event())
        d = result.to_dict()
        for key in ["risk_score", "severity", "recommended_priority",
                    "time_to_transition_hint", "formula", "components"]:
            assert key in d

    def test_urgency_decreases_with_horizon(self):
        """Risk score should decrease as horizon_steps increases (further forecast = less urgent)."""
        engine = RiskEngine()
        event = _make_event(attack_probability=0.7, confidence=0.7)
        score_h1 = engine.evaluate(event, horizon_steps=1).risk_score
        score_h4 = engine.evaluate(event, horizon_steps=4).risk_score
        assert score_h1 >= score_h4, (
            f"Expected score(h=1)={score_h1:.1f} >= score(h=4)={score_h4:.1f}"
        )

    def test_transition_detected_in_hint(self):
        engine = RiskEngine()
        event = _make_event(current_stage="RECONNAISSANCE", predicted_stage="CREDENTIAL_ACCESS")
        result = engine.evaluate(event)
        assert "RECONNAISSANCE" in result.time_to_transition_hint
        assert "CREDENTIAL_ACCESS" in result.time_to_transition_hint


# ---------------------------------------------------------------------------
# MITRE Mapper Tests
# ---------------------------------------------------------------------------

class TestMitreMapper:
    def test_all_attack_stages_have_techniques(self):
        """Every non-BENIGN stage must have at least one technique."""
        for stage in STAGE_TAXONOMY:
            if stage == "BENIGN":
                continue
            techniques = get_techniques_for_stage(stage)
            assert len(techniques) >= 1, f"No techniques for stage: {stage}"

    def test_benign_has_no_techniques(self):
        assert get_techniques_for_stage("BENIGN") == []

    def test_technique_fields_present(self):
        """Every technique must have the required fields."""
        required_fields = {"technique_id", "name", "tactic", "rationale",
                           "sub_techniques", "mapping_provenance"}
        for stage, techniques in STAGE_TO_MITRE.items():
            for tech in techniques:
                missing = required_fields - set(tech.keys())
                assert not missing, f"Stage {stage} technique missing fields: {missing}"

    def test_technique_ids_format(self):
        """All MITRE technique IDs must start with 'T' followed by 4 digits."""
        import re
        pattern = re.compile(r'^T\d{4}(\.\d{3})?$')
        for stage, techniques in STAGE_TO_MITRE.items():
            for tech in techniques:
                tid = tech["technique_id"]
                if tid == "UNKNOWN":
                    continue
                assert pattern.match(tid), (
                    f"Invalid technique ID format: '{tid}' in stage '{stage}'"
                )

    def test_no_duplicate_technique_ids_per_stage(self):
        for stage, techniques in STAGE_TO_MITRE.items():
            ids = [t["technique_id"] for t in techniques]
            assert len(ids) == len(set(ids)), (
                f"Duplicate technique IDs in stage '{stage}': {ids}"
            )

    def test_mapping_provenance_is_mitre_v14(self):
        for stage, techniques in STAGE_TO_MITRE.items():
            for tech in techniques:
                prov = tech["mapping_provenance"]
                assert "MITRE ATT&CK" in prov and "v14" in prov, (
                    f"Unexpected provenance in stage {stage}: '{prov}'"
                )

    def test_get_primary_technique(self):
        primary = get_primary_technique("RECONNAISSANCE")
        assert primary is not None
        assert primary["technique_id"].startswith("T")

    def test_get_primary_technique_benign_is_none(self):
        assert get_primary_technique("BENIGN") is None

    def test_get_mitre_summary_structure(self):
        summary = get_mitre_summary("CREDENTIAL_ACCESS")
        assert "stage" in summary
        assert "technique_count" in summary
        assert "techniques" in summary
        assert summary["technique_count"] >= 1

    def test_list_all_stages(self):
        stages = list_all_stages()
        assert "RECONNAISSANCE" in stages
        assert "EXFILTRATION" in stages
        assert "BENIGN" in stages


# ---------------------------------------------------------------------------
# Explainability Tests
# ---------------------------------------------------------------------------

class TestExplainability:
    @pytest.fixture
    def cur_state(self):
        np.random.seed(42)
        return np.random.randn(D).astype(np.float32)

    @pytest.fixture
    def pred_state(self, cur_state):
        return cur_state + np.random.randn(D).astype(np.float32) * 0.5

    def test_compute_feature_deltas_length(self, cur_state, pred_state):
        deltas = compute_feature_deltas(cur_state, pred_state)
        assert len(deltas) == D

    def test_compute_feature_deltas_sorted_desc(self, cur_state, pred_state):
        deltas = compute_feature_deltas(cur_state, pred_state)
        abs_changes = [d["abs_change"] for d in deltas]
        assert abs_changes == sorted(abs_changes, reverse=True)

    def test_delta_fields_present(self, cur_state, pred_state):
        deltas = compute_feature_deltas(cur_state, pred_state)
        for d in deltas:
            for field in ["feature", "current", "predicted", "abs_change", "rel_change_pct", "direction"]:
                assert field in d, f"Missing field: {field}"

    def test_direction_correct(self, cur_state, pred_state):
        deltas = compute_feature_deltas(cur_state, pred_state)
        for d in deltas:
            if d["predicted"] > d["current"]:
                assert d["direction"] == "increase"
            elif d["predicted"] < d["current"]:
                assert d["direction"] == "decrease"
            else:
                assert d["direction"] == "stable"

    def test_feature_names_match_feature_names(self, cur_state, pred_state):
        deltas = compute_feature_deltas(cur_state, pred_state, feature_names=FEATURE_NAMES)
        delta_names = {d["feature"] for d in deltas}
        assert delta_names == set(FEATURE_NAMES)

    def test_get_top_features(self, cur_state, pred_state):
        deltas = compute_feature_deltas(cur_state, pred_state)
        top5 = get_top_features(deltas, top_k=5)
        assert len(top5) == 5
        # Top-5 should all have abs_change >= the 6th largest
        if len(deltas) > 5:
            assert top5[-1]["abs_change"] >= deltas[5]["abs_change"]

    def test_stage_relevant_features_subset(self, cur_state, pred_state):
        deltas = compute_feature_deltas(cur_state, pred_state)
        relevant = get_stage_relevant_features(deltas, "RECONNAISSANCE")
        relevant_names = {d["feature"] for d in relevant}
        assert relevant_names.issubset(set(STAGE_FEATURE_RELEVANCE["RECONNAISSANCE"]))

    def test_build_explanation_structure(self, cur_state, pred_state):
        exp = build_explanation(cur_state, pred_state,
                                "RECONNAISSANCE", "CREDENTIAL_ACCESS", top_k=5)
        for key in ["current_stage", "predicted_stage", "top_k_changed_features",
                    "stage_relevant_features", "narrative", "provenance"]:
            assert key in exp, f"Missing key: {key}"

    def test_explanation_provenance_no_llm(self, cur_state, pred_state):
        exp = build_explanation(cur_state, pred_state, "BENIGN", "RECONNAISSANCE")
        prov = exp["provenance"].lower()
        assert "no llm" in prov

    def test_explanation_dim_mismatch_raises(self):
        with pytest.raises(AssertionError):
            compute_feature_deltas(np.zeros(10), np.zeros(10), feature_names=FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Agent Tools Interface Tests
# ---------------------------------------------------------------------------

class TestCyberSentinelAgentTools:
    @pytest.fixture
    def tools(self):
        return CyberSentinelAgentTools()

    @pytest.fixture
    def event(self):
        return _make_event(
            current_stage="RECONNAISSANCE",
            predicted_stage="CREDENTIAL_ACCESS",
            attack_probability=0.82,
            confidence=0.77,
        )

    def test_get_current_state_keys(self, tools):
        result = tools.get_current_state(
            current_stage="RECONNAISSANCE",
            attack_probability=0.82,
        )
        assert result["tool"] == "get_current_state"
        assert "current_stage" in result
        assert "attack_probability" in result

    def test_get_attack_forecast_keys(self, tools, event):
        result = tools.get_attack_forecast(event)
        assert result["tool"] == "get_attack_forecast"
        assert "predicted_stage" in result
        assert "forecast_confidence" in result
        assert "transition_detected" in result

    def test_get_transition_analysis_keys(self, tools, event):
        result = tools.get_transition_analysis(event)
        assert result["tool"] == "get_transition_analysis"
        assert result["transition_detected"] is True

    def test_get_feature_importance_provenance(self, tools, event):
        result = tools.get_feature_importance(event)
        assert "provenance" in result
        prov = result["provenance"].lower()
        assert "no llm" in prov

    def test_get_mitre_mapping_no_invented_ids(self, tools, event):
        result = tools.get_mitre_mapping(event)
        assert result["tool"] == "get_mitre_mapping"
        # Must include note about static mapping
        assert "static" in result["note"].lower()

    def test_get_risk_assessment_keys(self, tools, event):
        result = tools.get_risk_assessment(event)
        assert result["tool"] == "get_risk_assessment"
        assert "risk_score" in result
        assert "severity" in result

    def test_get_model_metrics_immutable_keys(self, tools):
        result = tools.get_model_metrics()
        assert result["tool"] == "get_model_metrics"
        assert "WorldModelV2_DirectTransition" in result["models"]
        assert "TemporalGRU_Baseline" in result["models"]

    def test_get_model_metrics_v2_top1(self, tools):
        """Phase 8C benchmark: V2 next-stage Top-1 must be 97.73% (immutable)."""
        metrics = tools.get_model_metrics()
        v2 = metrics["models"]["WorldModelV2_DirectTransition"]
        assert abs(v2["next_stage_top1"] - 0.9773) < 1e-4

    def test_get_rollout_keys(self, tools):
        rollout_results = [
            {"step": 1, "predicted_stage": "CREDENTIAL_ACCESS", "confidence": 0.81},
            {"step": 2, "predicted_stage": "LATERAL_MOVEMENT", "confidence": 0.60},
        ]
        result = tools.get_rollout(rollout_results)
        assert result["tool"] == "get_rollout"
        assert result["horizon_steps"] == 2
