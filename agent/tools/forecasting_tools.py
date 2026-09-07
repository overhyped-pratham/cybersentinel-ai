"""
CyberSentinel AI — Structured Agent Tool Interface (Phase 9).

Exposes deterministic, JSON-returning tools for the future LLM/agent layer.

CRITICAL DESIGN PRINCIPLE:
  The LLM must NEVER inspect model internals or generate detections independently.
  All conclusions must be derived from the structured tool outputs below.
  The LLM's role is to EXPLAIN tool results, not to generate them.

Available tools:
  get_current_state()      — current network state summary
  get_attack_forecast()    — next-stage prediction + confidence + attack prob
  get_rollout()            — K-step autoregressive forecast
  get_transition_analysis()— transition detection and confidence
  get_feature_importance() — top changed features (explainability)
  get_mitre_mapping()      — deterministic MITRE technique mapping
  get_risk_assessment()    — risk score and defensive priority
  get_model_metrics()      — benchmark metrics from Phase 8C (immutable reference)
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from ml.defense.risk_engine import (
    ForecastEvent,
    RiskEngine,
    RiskEngineConfig,
    STAGE_TAXONOMY,
)
from ml.world_model.explainability import build_explanation
from mitre.mappings.mitre_mapper import get_mitre_summary


# Immutable benchmark metrics from Phase 8C empirical investigation.
# These come from experiments/phase8c_investigation/phase8c_summary.json.
# Do NOT regenerate or modify these numbers.
_PHASE9_BENCHMARK = {
    "WorldModelV2_DirectTransition": {
        "next_stage_top1": 0.9773,
        "next_stage_top3": 1.0000,
        "true_transition_accuracy": 0.8333,
        "brier_score_uncalibrated": 0.0452,
        "attack_fpr": 0.0,
        "k4_path_accuracy": 0.25,
        "test_split": "hard_multistage_holdout",
        "test_n": 44,
        "genuine_transitions": 6,
        "source": "experiments/phase8c_investigation/phase8c_summary.json",
    },
    "TemporalGRU_Baseline": {
        "next_stage_top1": 0.8182,
        "next_stage_top3": 0.9773,
        "true_transition_accuracy": 0.6667,
        "brier_score_uncalibrated": 0.2913,
        "attack_fpr": 0.5333,
        "test_split": "hard_multistage_holdout",
        "test_n": 44,
        "genuine_transitions": 6,
        "source": "experiments/phase8c_investigation/phase8c_summary.json",
    },
    "LogisticRegression_Baseline": {
        "next_stage_top1": 0.5000,
        "next_stage_top3": 0.7273,
        "true_transition_accuracy": 0.0,
        "brier_score_uncalibrated": 0.7206,
        "attack_fpr": 0.0,
        "test_split": "hard_multistage_holdout",
        "test_n": 44,
        "genuine_transitions": 6,
        "source": "experiments/phase8c_investigation/phase8c_summary.json",
    },
}


# ---------------------------------------------------------------------------
# CyberSentinelAgentTools
# ---------------------------------------------------------------------------

class CyberSentinelAgentTools:
    """
    Structured tool interface for the CyberSentinel agent.

    All methods return JSON-serializable dicts.
    No method makes probabilistic claims without grounding in model outputs.
    """

    def __init__(
        self,
        risk_engine: Optional[RiskEngine] = None,
        stage_taxonomy: Optional[List[str]] = None,
    ) -> None:
        self.risk_engine = risk_engine or RiskEngine()
        self.stage_taxonomy = stage_taxonomy or STAGE_TAXONOMY

    def get_current_state(
        self,
        current_stage: str,
        attack_probability: float,
        feature_vector: Optional[List[float]] = None,
        feature_names: Optional[List[str]] = None,
        scenario_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Summarise the current network observation.

        Returns:
            dict with current_stage, attack_probability, key features, timestamp.
        """
        result: Dict[str, Any] = {
            "tool": "get_current_state",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "current_stage": current_stage,
            "attack_probability": round(float(attack_probability), 4),
            "scenario_id": scenario_id,
        }
        if feature_vector is not None and feature_names is not None:
            result["feature_snapshot"] = {
                name: round(float(val), 4)
                for name, val in zip(feature_names, feature_vector)
            }
        return result

    def get_attack_forecast(self, event: ForecastEvent) -> Dict[str, Any]:
        """
        Return the one-step attack forecast from a ForecastEvent.

        Returns:
            dict with predicted_stage, attack_probability, confidence, uncertainty,
            transition_detected, top_3_stage_probabilities.
        """
        top3 = sorted(event.stage_probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        return {
            "tool": "get_attack_forecast",
            "timestamp": event.timestamp,
            "current_stage": event.current_stage,
            "predicted_stage": event.predicted_stage,
            "attack_probability": round(event.attack_probability, 4),
            "forecast_confidence": round(event.confidence, 4),
            "forecast_uncertainty_entropy": round(event.uncertainty_entropy, 4),
            "transition_detected": event.transition_detected,
            "top_3_stage_probabilities": [
                {"stage": s, "probability": round(p, 4)} for s, p in top3
            ],
            "horizon_seconds": event.horizon_seconds,
            "model_version": event.model_version,
            "note": (
                "forecast_confidence is the calibrated max probability. "
                "forecast_uncertainty_entropy=0 means very certain; =1 means maximally uncertain."
            ),
        }

    def get_rollout(
        self,
        rollout_results: List[Dict[str, Any]],
        stage_taxonomy: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Summarise K-step rollout results for the agent.

        Args:
            rollout_results: output from RolloutResult or a pre-processed list of
              {step, predicted_stage_name, confidence, uncertainty, attack_prob}
        """
        taxonomy = stage_taxonomy or self.stage_taxonomy
        return {
            "tool": "get_rollout",
            "horizon_steps": len(rollout_results),
            "forecast_path": rollout_results,
            "note": (
                "Each step is an autoregressive prediction: no future observations consumed. "
                f"K={len(rollout_results)} step path accuracy on hard holdout: 25.0% "
                "(limited by small number of genuine transitions in test dataset)."
            ),
        }

    def get_transition_analysis(self, event: ForecastEvent) -> Dict[str, Any]:
        """
        Analyse the predicted stage transition.

        Returns transition type, probability mass on new stage, and urgency hint.
        """
        same_stage_prob = event.stage_probabilities.get(event.current_stage, 0.0)
        new_stage_prob = event.stage_probabilities.get(event.predicted_stage, 0.0)
        return {
            "tool": "get_transition_analysis",
            "current_stage": event.current_stage,
            "predicted_stage": event.predicted_stage,
            "transition_detected": event.transition_detected,
            "probability_current_stage": round(same_stage_prob, 4),
            "probability_predicted_stage": round(new_stage_prob, 4),
            "confidence": round(event.confidence, 4),
            "note": (
                "true_transition_accuracy of 83.33% on hard holdout "
                "(5/6 genuine transitions correctly predicted at window t)."
            ),
        }

    def get_feature_importance(self, event: ForecastEvent) -> Dict[str, Any]:
        """
        Return feature importance explanation from the ForecastEvent.

        Provenance: grounded in predicted physical state delta S_hat_{t+1} - S_t.
        No LLM-generated conclusions.
        """
        return {
            "tool": "get_feature_importance",
            "predicted_stage": event.predicted_stage,
            "top_features": event.top_features,
            "provenance": (
                "Features ranked by |S_hat_{t+1}[i] - S_t[i]| from "
                "CyberWorldModelV2 physical state predictor. "
                "No LLM-generated conclusions."
            ),
        }

    def get_mitre_mapping(self, event: ForecastEvent) -> Dict[str, Any]:
        """
        Return the deterministic MITRE ATT&CK technique mapping for the predicted stage.
        """
        summary = get_mitre_summary(event.predicted_stage)
        return {
            "tool": "get_mitre_mapping",
            "predicted_stage": event.predicted_stage,
            **summary,
            "note": (
                "Mapping is deterministic (static lookup table, MITRE ATT&CK Enterprise v14). "
                "Technique IDs are NOT ML-generated."
            ),
        }

    def get_risk_assessment(
        self, event: ForecastEvent, horizon_steps: int = 1
    ) -> Dict[str, Any]:
        """
        Run the deterministic risk engine and return a structured assessment.
        """
        assessment = self.risk_engine.evaluate(event, horizon_steps=horizon_steps)
        return {
            "tool": "get_risk_assessment",
            **assessment.to_dict(),
        }

    def get_model_metrics(self) -> Dict[str, Any]:
        """
        Return the immutable Phase 8C benchmark metrics.

        IMPORTANT: These numbers are read-only and must not be regenerated
        merely to improve presentation.
        """
        return {
            "tool": "get_model_metrics",
            "source": "Phase 8C Hard Multi-Stage Holdout empirical investigation",
            "note": (
                "Metrics are immutable. From experiments/phase8c_investigation/phase8c_summary.json. "
                "Test split: N=44 sequences, 6 genuine stage transitions."
            ),
            "models": _PHASE9_BENCHMARK,
        }
