"""
CyberSentinel AI — Streaming API Schemas (Phase 13).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    source_kind: str = Field("replay", description="'pcap', 'netflow', or 'replay'")
    source_path: Optional[str] = Field(None, description="Path to PCAP or CSV file (replay/pcap sources)")
    host: Optional[str] = Field("0.0.0.0", description="UDP host for NetFlow source")
    port: Optional[int] = Field(9995, description="UDP port for NetFlow source")
    window_seconds: float = Field(30.0, ge=5.0, le=300.0, description="Telemetry window size in seconds")
    stride_seconds: Optional[float] = Field(None, ge=5.0, description="Window stride (default = window_seconds)")
    k_steps: int = Field(4, ge=1, le=10, description="K-step rollout depth")
    session_id: Optional[str] = Field(None, description="Optional custom session ID")
    realtime_factor: float = Field(0.0, ge=0.0, description="Replay speed: 0=fast, 1=wall-clock")


class StopSessionRequest(BaseModel):
    session_id: str


class StreamEvent(BaseModel):
    status: str
    event_id: Optional[str] = None
    window_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: str
    window_start: Optional[float] = None
    window_end: Optional[float] = None
    flow_count: Optional[int] = None
    flows_per_second: Optional[float] = None
    dropped_malformed: Optional[int] = None
    source_id: Optional[str] = None
    inference_latency_ms: Optional[float] = None
    model_version: Optional[str] = None
    provenance: Optional[str] = None

    current_stage: Optional[str] = None
    predicted_next_stage: Optional[str] = None
    attack_probability: Optional[float] = None
    confidence: Optional[float] = None
    transition_detected: Optional[bool] = None
    transition_probability: Optional[float] = None
    stage_probabilities: Optional[Dict[str, float]] = None
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    recommended_priority: Optional[str] = None
    time_to_transition_hint: Optional[str] = None
    top_features: Optional[List[Dict[str, Any]]] = None
    explanation_narrative: Optional[str] = None
    stage_relevant_features: Optional[List[Dict[str, Any]]] = None
    mitre_techniques: Optional[List[str]] = None
    primary_technique_id: Optional[str] = None
    primary_technique_name: Optional[str] = None
    rollout_steps: Optional[List[Dict[str, Any]]] = None
    safety_flags: Optional[List[str]] = None
