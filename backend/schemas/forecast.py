"""
CyberSentinel AI — Phase 10 Pydantic Schemas.

Defines the canonical unified response schema CyberSentinelForecast
and all request/response models for the FastAPI backend.

Every field has a deterministic source documented in `provenance`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Provenance model
# ---------------------------------------------------------------------------

class Provenance(BaseModel):
    prediction: str = "CyberWorldModelV2"
    explanation: str = "physical_state_delta"
    mitre: str = "MITRE_ATT&CK_v14_static_mapping"
    risk: str = "RiskEngine_deterministic"
    narrative: str = "CyberSentinel_defensive_agent"


# ---------------------------------------------------------------------------
# Feature delta (explainability)
# ---------------------------------------------------------------------------

class FeatureDelta(BaseModel):
    feature: str
    current: float
    predicted: float
    abs_change: float
    rel_change_pct: float
    direction: str   # "increase" | "decrease" | "stable"


# ---------------------------------------------------------------------------
# MITRE technique
# ---------------------------------------------------------------------------

class MitreTechnique(BaseModel):
    technique_id: str
    name: str
    tactic: str
    rationale: str
    sub_techniques: List[str] = Field(default_factory=list)
    mapping_provenance: str = "MITRE ATT&CK Enterprise v14 static mapping"


# ---------------------------------------------------------------------------
# Safety flags from rollout
# ---------------------------------------------------------------------------

class SafetyFlags(BaseModel):
    nan_detected: bool = False
    collapse_detected: bool = False
    feature_dim_valid: bool = True


# ---------------------------------------------------------------------------
# One rollout step
# ---------------------------------------------------------------------------

class RolloutStep(BaseModel):
    step: int
    predicted_stage: str
    confidence: float
    uncertainty: float
    attack_probability: float


# ---------------------------------------------------------------------------
# CANONICAL UNIFIED FORECAST RESPONSE
# ---------------------------------------------------------------------------

class CyberSentinelForecast(BaseModel):
    """
    Canonical response schema for every CyberSentinel forecast.

    Every field has a deterministic source documented in `provenance`.
    No field may be populated by LLM inference.
    """

    # Identity
    timestamp: str
    model_version: str = "CyberWorldModelV2"
    forecast_horizon: int = 30   # seconds

    # Current observation (from model)
    current_stage: str
    attack_probability: float = Field(ge=0.0, le=1.0)

    # Forecast
    predicted_next_stage: str
    next_stage_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    transition_detected: bool
    transition_confidence: float = Field(ge=0.0, le=1.0)

    # Uncertainty
    uncertainty_entropy: float = Field(ge=0.0, le=1.0)
    calibrated_temperature: Optional[float] = None

    # Full stage probability distribution
    stage_probabilities: Dict[str, float] = Field(default_factory=dict)

    # Rollout
    rollout_steps: List[RolloutStep] = Field(default_factory=list)

    # Explainability
    top_features: List[FeatureDelta] = Field(default_factory=list)
    stage_relevant_features: List[FeatureDelta] = Field(default_factory=list)
    explanation_narrative: str = ""

    # MITRE
    mitre_techniques: List[MitreTechnique] = Field(default_factory=list)
    primary_technique_id: Optional[str] = None
    primary_technique_name: Optional[str] = None

    # Risk
    risk_score: float = Field(ge=0.0, le=100.0)
    risk_level: str   # LOW | MEDIUM | HIGH | CRITICAL
    recommended_priority: str = ""
    time_to_transition_hint: str = ""

    # Safety
    safety_flags: SafetyFlags = Field(default_factory=SafetyFlags)

    # Provenance — every downstream consumer MUST read this
    provenance: Provenance = Field(default_factory=Provenance)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    """
    Request body for POST /forecast.

    x_seq: (T, D) — sequence of scaled feature vectors.
    mask:  (T,)   — True = valid timestep.
    """
    x_seq: List[List[float]] = Field(
        description="Sequence of scaled feature vectors shape (T, D)"
    )
    mask: Optional[List[bool]] = Field(
        default=None,
        description="Boolean mask, True=valid. Length must equal T."
    )
    k_steps: int = Field(default=4, ge=1, le=16)

    @field_validator("x_seq")
    @classmethod
    def validate_x_seq(cls, v):
        if len(v) == 0:
            raise ValueError("INVALID_TELEMETRY: x_seq must not be empty")
        d = len(v[0])
        if d != 24:
            raise ValueError(f"INVALID_TELEMETRY: Feature vector dimension must be 24, got {d}")
        for row in v:
            if len(row) != d:
                raise ValueError("INVALID_TELEMETRY: All feature vectors must have the same length")
        return v


class AgentQueryRequest(BaseModel):
    query: str = Field(description="Natural language analyst question")
    session_id: Optional[str] = Field(default=None, description="Replay session ID for context")
    current_forecast: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Last CyberSentinelForecast dict for context"
    )


class AgentQueryResponse(BaseModel):
    answer: str
    tool_calls: List[str] = Field(default_factory=list, description="Tools consulted to build this answer")
    provenance: str = "CyberSentinel_defensive_agent"
    llm_backend: str   # "ollama" | "template_fallback"
    grounded_in_model_output: bool = True


class ReplayStartRequest(BaseModel):
    scenario_id: str = Field(default="trace_multistage_03")
    k_steps: int = Field(default=4, ge=1, le=16)


class ReplayStartResponse(BaseModel):
    session_id: str
    scenario_id: str
    total_windows: int
    stage_sequence: List[str]
    message: str


class ReplayStepResponse(BaseModel):
    session_id: str
    window_index: int
    total_windows: int
    is_complete: bool
    forecast: Optional[CyberSentinelForecast] = None


class ReplayStatusResponse(BaseModel):
    session_id: str
    scenario_id: str
    current_window: int
    total_windows: int
    is_complete: bool
    elapsed_seconds: Optional[float] = None


class ModelInfoResponse(BaseModel):
    model_version: str
    model_class: str
    input_dim: int
    hidden_dim: int
    num_stages: int
    num_heads: int
    num_layers: int
    temperature: float
    is_calibrated: bool
    checkpoint_path: Optional[str]
    benchmark: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str   # "ok" | "degraded"
    model_loaded: bool
    calibration_loaded: bool
    dataset_available: bool
    ollama_available: bool
    timestamp: str
