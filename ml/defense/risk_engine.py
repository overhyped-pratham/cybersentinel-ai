"""
CyberSentinel AI — Structured Forecast Event & Defensive Risk Engine (Phase 9).

ForecastEvent: Standard interface object produced at every inference step.
RiskEngine: Deterministic, configurable risk scoring from a ForecastEvent.

Risk score formula:
    risk_score = 100 * (
        w_attack    * P(attack)
      + w_severity  * severity(predicted_next_stage)
      + w_forecast  * P(next_stage)        [calibrated confidence]
      + w_urgency   * urgency(horizon_k)
    )
    where urgency(1) = 1.0, urgency(2) = 0.75, urgency(4) = 0.50.

Defensive priority thresholds:
    >= 75 → CRITICAL
    >= 50 → HIGH
    >= 30 → MEDIUM
    <  30 → LOW

All scoring parameters are configurable via RiskEngineConfig.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Stage taxonomy constants
# ---------------------------------------------------------------------------

STAGE_TAXONOMY = [
    "BENIGN",
    "RECONNAISSANCE",
    "INITIAL_ACCESS",
    "EXECUTION",
    "CREDENTIAL_ACCESS",
    "DISCOVERY",
    "LATERAL_MOVEMENT",
    "COMMAND_AND_CONTROL",
    "EXFILTRATION",
    "UNKNOWN",
]

# Default severity scores: 0.0 (benign) to 1.0 (critical threat)
DEFAULT_SEVERITY: Dict[str, float] = {
    "BENIGN": 0.0,
    "RECONNAISSANCE": 0.25,
    "INITIAL_ACCESS": 0.50,
    "EXECUTION": 0.65,
    "CREDENTIAL_ACCESS": 0.75,
    "DISCOVERY": 0.35,
    "LATERAL_MOVEMENT": 0.85,
    "COMMAND_AND_CONTROL": 0.90,
    "EXFILTRATION": 1.00,
    "UNKNOWN": 0.20,
}


# ---------------------------------------------------------------------------
# ForecastEvent
# ---------------------------------------------------------------------------

@dataclass
class ForecastEvent:
    """
    Standard interface object produced at every CyberWorldModelV2 inference step.

    This is the contract between the ML layer and all downstream components
    (risk engine, MITRE mapper, agent tools, replay mode, dashboard).

    IMPORTANT: The agent must NEVER inspect model internals directly.
    All conclusions must be drawn from this structured object.
    """

    # Core identification
    timestamp: str                         # ISO 8601 timestamp at inference time
    model_version: str                     # e.g. "2.0"
    horizon_seconds: int                   # forecast window size in seconds

    # Current observation
    current_stage: str                     # e.g. "RECONNAISSANCE"
    current_state: Optional[List[float]]   # scaled feature vector S_t

    # Forecast
    predicted_stage: str                   # argmax of calibrated next-stage distribution
    predicted_next_state: Optional[List[float]]   # S_hat_{t+1} feature vector

    # Probabilities
    attack_probability: float              # P(attack) at predicted next state
    stage_probabilities: Dict[str, float]  # full distribution over all stages
    confidence: float                      # max probability in calibrated distribution

    # Uncertainty
    uncertainty_entropy: float             # normalized Shannon entropy (0=certain, 1=uniform)

    # Transition detection
    transition_detected: bool              # True if predicted_stage != current_stage

    # Explainability
    top_features: List[Dict[str, Any]]     # top-K changed features from explainability module

    # Novelty & Anomaly signals (Layer 2 & Layer 1 integration)
    anomaly_score: float = 0.0             # normalized [0, 1] from AutoencoderNoveltyDetector
    reconstruction_error: float = 0.0      # raw MSE from Autoencoder
    is_novel: bool = False                 # True if flagged as Potential Novel Behavior
    novelty_label: str = "Normal Baseline"
    known_category: Optional[str] = None   # category from KnownAttackClassifier
    known_confidence: float = 0.0          # confidence from KnownAttackClassifier

    # Context
    scenario_id: Optional[str] = None     # trace/scenario identifier for replay mode

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict for agent tools."""
        return {
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "horizon_seconds": self.horizon_seconds,
            "current_stage": self.current_stage,
            "predicted_stage": self.predicted_stage,
            "attack_probability": round(self.attack_probability, 4),
            "confidence": round(self.confidence, 4),
            "uncertainty_entropy": round(self.uncertainty_entropy, 4),
            "transition_detected": self.transition_detected,
            "stage_probabilities": {k: round(v, 4) for k, v in self.stage_probabilities.items()},
            "top_features": self.top_features[:5],
            "anomaly_score": round(self.anomaly_score, 4),
            "reconstruction_error": round(self.reconstruction_error, 6),
            "is_novel": bool(self.is_novel),
            "novelty_label": self.novelty_label,
            "known_category": self.known_category,
            "known_confidence": round(self.known_confidence, 4),
            "scenario_id": self.scenario_id,
        }


# ---------------------------------------------------------------------------
# Risk Engine
# ---------------------------------------------------------------------------

@dataclass
class RiskEngineConfig:
    """Configurable parameters for the risk scoring formula."""
    weight_attack: float = 0.40
    weight_severity: float = 0.35
    weight_forecast_confidence: float = 0.15
    weight_urgency: float = 0.10
    severity_scores: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SEVERITY))

    # Horizon urgency decay: horizon_steps -> urgency factor
    urgency_map: Dict[int, float] = field(default_factory=lambda: {
        1: 1.00,   # 30 seconds lead time — maximum urgency
        2: 0.75,   # 60 seconds
        3: 0.60,
        4: 0.50,   # 120 seconds
    })

    # Priority thresholds
    threshold_critical: float = 75.0
    threshold_high: float = 50.0
    threshold_medium: float = 30.0

    @classmethod
    def from_yaml(cls, config_path: Optional[str | Path] = None) -> "RiskEngineConfig":
        """Load configuration dynamically from YAML file."""
        import yaml
        from pathlib import Path
        path = Path(config_path) if config_path else Path(__file__).resolve().parent.parent.parent / "configs" / "default_config.yaml"
        if not path.exists():
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            re_cfg = raw.get("risk_engine", {})
            w = re_cfg.get("weights", {})
            sev = re_cfg.get("stage_severity_scores", {})
            return cls(
                weight_attack=w.get("attack_probability", 0.40),
                weight_severity=w.get("stage_severity", 0.35),
                weight_forecast_confidence=w.get("confidence_weight", 0.15),
                weight_urgency=w.get("forecast_probability", 0.10),
                severity_scores=sev if sev else dict(DEFAULT_SEVERITY),
            )
        except Exception:
            return cls()


@dataclass
class RiskAssessment:
    """Output of the RiskEngine evaluation."""
    risk_score: float               # [0, 100]
    severity: str                   # LOW / MEDIUM / HIGH / CRITICAL
    component_attack: float         # contribution from attack probability
    component_severity: float       # contribution from stage severity
    component_confidence: float     # contribution from forecast confidence
    component_urgency: float        # contribution from horizon urgency
    recommended_priority: str       # defensive action priority
    time_to_transition_hint: str    # human hint about transition timing
    formula_description: str        # formula provenance
    component_anomaly: float = 0.0  # contribution from anomaly deviation
    anomaly_score: float = 0.0      # normalized anomaly score [0, 1]
    is_novel: bool = False          # novelty flag
    threat_classification: str = "Known Threat"  # "Potential Novel Behavior" | "Known Threat" | "Benign Baseline"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 2),
            "severity": self.severity,
            "recommended_priority": self.recommended_priority,
            "time_to_transition_hint": self.time_to_transition_hint,
            "is_novel": self.is_novel,
            "threat_classification": self.threat_classification,
            "components": {
                "attack_probability": round(self.component_attack, 4),
                "stage_severity": round(self.component_severity, 4),
                "forecast_confidence": round(self.component_confidence, 4),
                "horizon_urgency": round(self.component_urgency, 4),
                "anomaly_deviation": round(self.component_anomaly, 4),
            },
            "formula": self.formula_description,
        }


class RiskEngine:
    """
    Deterministic defensive risk scoring engine.

    Input:  ForecastEvent
    Output: RiskAssessment

    Formula:
        risk_score = 100 * (
            w_attack   * P(attack)
          + w_severity  * severity(predicted_next_stage)
          + w_confidence * P(predicted_stage)
          + w_urgency   * urgency(horizon_k)
          + w_anomaly   * anomaly_score
        )
    """

    def __init__(self, config: Optional[RiskEngineConfig] = None) -> None:
        self.config = config if config is not None else RiskEngineConfig.from_yaml()

    def evaluate(self, event: ForecastEvent, horizon_steps: int = 1) -> RiskAssessment:
        """
        Compute a deterministic risk assessment from a ForecastEvent.

        Args:
            event:          ForecastEvent from inference
            horizon_steps:  forecast step (1 = next window, 2 = window+2, etc.)

        Returns:
            RiskAssessment
        """
        cfg = self.config

        # Component 1: Attack probability
        c_attack = cfg.weight_attack * float(event.attack_probability)

        # Component 2: Predicted stage severity
        sev = cfg.severity_scores.get(event.predicted_stage, cfg.severity_scores.get("UNKNOWN", 0.2))
        is_novel = getattr(event, "is_novel", False)
        if is_novel and sev < 0.60:
            sev = 0.60
        c_severity = cfg.weight_severity * sev

        # Component 3: Forecast confidence
        c_conf = cfg.weight_forecast_confidence * float(event.confidence)

        # Component 4: Horizon urgency
        urgency = cfg.urgency_map.get(horizon_steps, 0.40)
        c_urgency = cfg.weight_urgency * urgency

        # Component 5: Anomaly signal (if configured)
        anomaly_val = getattr(event, "anomaly_score", 0.0)
        weight_anom = getattr(cfg, "weight_anomaly", 0.0)
        c_anomaly = weight_anom * float(anomaly_val)

        raw_score = c_attack + c_severity + c_conf + c_urgency + c_anomaly
        risk_score = min(100.0, max(0.0, raw_score * 100.0))

        # Threat classification adhering to PRD Principle 5
        is_novel = getattr(event, "is_novel", False)
        if is_novel:
            threat_cls = "Potential Novel Behavior"
        elif event.predicted_stage != "BENIGN" and event.attack_probability >= 0.5:
            threat_cls = "Known Threat"
        else:
            threat_cls = "Benign Baseline"

        # Priority thresholds
        if risk_score >= cfg.threshold_critical:
            severity_label = "CRITICAL"
            priority = "IMMEDIATE — isolate affected hosts, escalate to incident response"
        elif risk_score >= cfg.threshold_high:
            severity_label = "HIGH"
            priority = "URGENT — activate monitoring, prepare containment actions"
        elif risk_score >= cfg.threshold_medium:
            severity_label = "MEDIUM"
            priority = "ELEVATED — increase log collection, alert SOC analyst"
        else:
            severity_label = "LOW"
            priority = "ROUTINE — continue baseline monitoring"

        # Transition timing hint
        if event.transition_detected:
            hint = (
                f"Transition {event.current_stage} → {event.predicted_stage} "
                f"predicted within the next {event.horizon_seconds}s window."
            )
        else:
            hint = (
                f"Stage {event.predicted_stage} continuation predicted for "
                f"the next {event.horizon_seconds}s window."
            )

        formula = (
            "risk = 100 * (w_attack * P(attack) + w_severity * severity(stage) "
            "+ w_confidence * confidence + w_urgency * urgency(horizon)). "
            f"Weights: attack={cfg.weight_attack}, severity={cfg.weight_severity}, "
            f"confidence={cfg.weight_forecast_confidence}, urgency={cfg.weight_urgency}."
        )

        return RiskAssessment(
            risk_score=risk_score,
            severity=severity_label,
            component_attack=c_attack,
            component_severity=c_severity,
            component_confidence=c_conf,
            component_urgency=c_urgency,
            recommended_priority=priority,
            time_to_transition_hint=hint,
            formula_description=formula,
            component_anomaly=c_anomaly,
            anomaly_score=float(anomaly_val),
            is_novel=is_novel,
            threat_classification=threat_cls,
        )
