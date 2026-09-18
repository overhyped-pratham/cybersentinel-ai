"""
CyberSentinel AI — PRD Canonical API Endpoints (Section 19).

Provides:
  POST /api/traffic             — Ingest network flow / telemetry
  POST /api/predict             — 4-layer ML inference
  GET  /api/alerts              — Recent alerts stream
  GET  /api/alerts/{id}         — Alert detail with SHAP & risk breakdown
  GET  /api/dashboard           — Aggregated SOC metrics & live traffic series
  GET  /api/statistics          — System throughput, latency, model versions
  GET  /api/attack-story/{id}   — Dynamically correlated attack story
  POST /api/world-model/rollout — K-step autoregressive rollout
  POST /api/feedback            — Human analyst validation (Threat Memory & Adaptive Update)
  GET  /api/adaptation/history  — Adaptation audit ledger
  POST /api/adaptation/rollback — Rollback model version
"""

from __future__ import annotations

import datetime
import logging
import math
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import numpy as np

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ml.adaptation.adaptive_learner import AdaptiveLearner
from ml.adaptation.threat_memory import ThreatMemory
from ml.defense.attack_story_engine import AttackStory, AttackStoryEngine
from ml.pipeline.detection_pipeline import DetectionPipeline, PipelineDetectionResult
from ml.state.state_builder import FEATURE_NAMES

logger = logging.getLogger(__name__)

prd_router = APIRouter(tags=["CyberSentinel X Core API"])


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class TrafficIngestRequest(BaseModel):
    src_ip: str = Field(default="192.168.1.105", description="Source IP address")
    dst_ip: str = Field(default="10.0.0.15", description="Destination IP address")
    protocol: str = Field(default="TCP", description="Transport protocol")
    features: Optional[List[float]] = Field(
        default=None,
        description="24-dimensional continuous feature vector S_t. If omitted, default benign/flow vector is used.",
    )
    feature_dict: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional dictionary mapping feature names to numerical values.",
    )


class PredictRequest(BaseModel):
    state: List[float] = Field(..., description="24-dimensional scaled or unscaled network state vector S_t")
    is_scaled: bool = Field(default=False, description="True if state is already scaled by FeatureScaler")
    k_steps: int = Field(default=4, ge=1, le=10, description="K-step rollout horizon")
    historical_seq: Optional[List[List[float]]] = Field(
        default=None, description="Optional historical sequence for CyberWorldModel context"
    )


class FeedbackRequest(BaseModel):
    sample_id: Optional[str] = Field(default=None, description="Identifier of alert or sample")
    feature_vector: List[float] = Field(..., description="24-dimensional feature vector of validated sample")
    validated_label: str = Field(..., description="Analyst validated ground truth (e.g. EXFILTRATION, BENIGN)")
    is_malicious: bool = Field(..., description="True if verified attack, False if benign/false-positive")
    analyst_notes: str = Field(default="", description="Investigation notes / rationale")
    analyst_id: str = Field(default="soc_lead", description="Identifier of human analyst")
    trigger_update: bool = Field(default=False, description="If True, immediately triggers controlled model adaptation")


class RolloutRequest(BaseModel):
    state_seq: List[List[float]] = Field(..., description="Sequence of historical state vectors (T, 24)")
    k_steps: int = Field(default=4, ge=1, le=10, description="Number of steps to roll forward")


# ---------------------------------------------------------------------------
# Global In-Memory Store for Live Alerts & Telemetry
# ---------------------------------------------------------------------------

class AlertManager:
    """Manages recent telemetry events, alerts, and attack stories in-memory."""
    def __init__(self, max_history: int = 500) -> None:
        self.max_history = max_history
        self.telemetry_events: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        self.pipeline: Optional[DetectionPipeline] = None
        self.story_engine = AttackStoryEngine()
        self.threat_memory = ThreatMemory()
        self.learner = AdaptiveLearner(threat_memory=self.threat_memory)
        self.start_time = time.time()
        self.stats = {
            "total_traffic": 0,
            "normal_traffic": 0,
            "attacks": 0,
            "anomalies": 0,
            "critical_alerts": 0,
        }

    def get_pipeline(self) -> DetectionPipeline:
        if self.pipeline is None:
            self.pipeline = DetectionPipeline()
        return self.pipeline

    def record_event(
        self,
        features: List[float],
        src_ip: str,
        dst_ip: str,
    ) -> PipelineDetectionResult:
        pipeline = self.get_pipeline()
        result = pipeline.process_state(features, is_scaled=False)
        self.stats["total_traffic"] += 1

        event_id = f"evt-{uuid.uuid4().hex[:8]}"
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

        event_record = {
            "event_id": event_id,
            "timestamp": now_str,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "stage": result.current_stage,
            "predicted_next_stage": result.predicted_next_stage,
            "is_attack": result.is_attack,
            "is_novel": result.is_novel,
            "risk_score": result.risk_score,
            "anomaly_score": result.novelty_detector["anomaly_score"],
            "attack_probability": result.known_classifier["attack_probability"],
            "threat_classification": result.threat_classification,
            "top_features": result.top_contributing_features,
            "result": result.to_dict(),
        }

        self.telemetry_events.append(event_record)
        if len(self.telemetry_events) > self.max_history:
            self.telemetry_events.pop(0)

        if result.is_attack:
            self.stats["attacks"] += 1
        else:
            self.stats["normal_traffic"] += 1

        if result.is_novel:
            self.stats["anomalies"] += 1

        if result.risk_severity == "CRITICAL":
            self.stats["critical_alerts"] += 1

        # Record alert if high risk, novel, or known attack
        if result.is_attack or result.is_novel or result.risk_score >= 30.0:
            alert_id = f"alt-{uuid.uuid4().hex[:8]}"
            alert_record = {
                "alert_id": alert_id,
                "event_id": event_id,
                "timestamp": now_str,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "category": result.known_classifier["predicted_category"],
                "confidence": result.known_classifier["confidence"],
                "is_novel": result.is_novel,
                "threat_classification": result.threat_classification,
                "anomaly_score": result.novelty_detector["anomaly_score"],
                "risk_score": result.risk_score,
                "severity": result.risk_severity,
                "recommended_priority": result.recommended_priority,
                "top_features": result.top_contributing_features,
                "future_rollout": result.world_model_rollout,
                "raw_state": result.current_state,
            }
            self.alerts.append(alert_record)
            if len(self.alerts) > self.max_history:
                self.alerts.pop(0)

        return result


# Global singleton instance
_alert_manager = AlertManager()


# ---------------------------------------------------------------------------
# API Route Handlers
# ---------------------------------------------------------------------------

@prd_router.post("/traffic")
def ingest_traffic(body: TrafficIngestRequest):
    """
    POST /api/traffic — Ingest network flow telemetry.
    Passes through the 4-layer detection pipeline, updates live statistics,
    and returns immediate detection verdict.
    """
    if body.features is not None:
        raw_vec = body.features
    elif body.feature_dict is not None:
        raw_vec = [body.feature_dict.get(f, 0.0) for f in FEATURE_NAMES]
    else:
        # Default benign synthetic state
        raw_vec = [0.0] * len(FEATURE_NAMES)

    if len(raw_vec) != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=422,
            detail=f"Expected 24 features matching schema, got {len(raw_vec)}",
        )

    # Sanitize NaN/Inf in incoming telemetry
    clean_vec = [0.0 if (x is None or math.isnan(x) or math.isinf(x)) else float(x) for x in raw_vec]

    res: PipelineDetectionResult = _alert_manager.record_event(
        features=clean_vec,
        src_ip=body.src_ip,
        dst_ip=body.dst_ip,
    )

    return JSONResponse(content={
        "status": "ingested",
        "timestamp": res.timestamp,
        "is_attack": res.is_attack,
        "is_novel": res.is_novel,
        "threat_classification": res.threat_classification,
        "risk_score": res.risk_score,
        "risk_severity": res.risk_severity,
        "current_stage": res.current_stage,
        "predicted_next_stage": res.predicted_next_stage,
        "recommended_priority": res.recommended_priority,
        "known_classifier": res.known_classifier,
        "novelty_detector": res.novelty_detector,
        "risk_assessment": res.risk_assessment,
    })


@prd_router.post("/predict")
def predict_state(body: PredictRequest):
    """
    POST /api/predict — Run 4-layer ML inference on a given cyber state vector S_t.
    """
    if len(body.state) != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=422,
            detail=f"State vector must contain exactly {len(FEATURE_NAMES)} features",
        )

    clean_state = [0.0 if (x is None or math.isnan(x) or math.isinf(x)) else float(x) for x in body.state]

    pipeline = _alert_manager.get_pipeline()
    res = pipeline.process_state(
        raw_or_scaled_state=clean_state,
        is_scaled=body.is_scaled,
        historical_seq=body.historical_seq,
        k_steps=body.k_steps,
    )
    return JSONResponse(content=res.to_dict())


@prd_router.get("/alerts")
def list_alerts(
    limit: int = Query(default=50, ge=1, le=200),
    severity: Optional[str] = Query(default=None),
    novel_only: bool = Query(default=False),
):
    """
    GET /api/alerts — Returns live alerts stream with optional filters.
    """
    alerts = _alert_manager.alerts
    if severity:
        alerts = [a for a in alerts if a["severity"].upper() == severity.upper()]
    if novel_only:
        alerts = [a for a in alerts if a["is_novel"]]

    return JSONResponse(content={
        "total": len(alerts),
        "alerts": alerts[-limit:][::-1],
    })


@prd_router.get("/alerts/{alert_id}")
def get_alert_detail(alert_id: str):
    """
    GET /api/alerts/{id} — Returns detailed alert investigation package:
    classification, confidence, anomaly score, risk score, feature explanation,
    and world-model rollout.
    """
    for a in _alert_manager.alerts:
        if a["alert_id"] == alert_id:
            return JSONResponse(content=a)
    raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")


@prd_router.get("/dashboard")
def get_dashboard():
    """
    GET /api/dashboard — Returns main SOC command center overview:
    total traffic, normal traffic, attacks, anomalies, critical alerts,
    live traffic graph series, attack distribution, and top suspicious sources.
    """
    stats = _alert_manager.stats
    recent_events = _alert_manager.telemetry_events[-30:]

    # Attack distribution
    dist: Dict[str, int] = {}
    for a in _alert_manager.alerts:
        cat = a["category"]
        dist[cat] = dist.get(cat, 0) + 1

    # Top suspicious sources
    sources: Dict[str, int] = {}
    for a in _alert_manager.alerts:
        src = a["src_ip"]
        sources[src] = sources.get(src, 0) + 1
    top_sources = sorted(
        [{"src_ip": k, "alert_count": v} for k, v in sources.items()],
        key=lambda x: x["alert_count"],
        reverse=True,
    )[:5]

    # Live traffic series
    series = [
        {
            "time": e["timestamp"][-8:],
            "risk_score": e["risk_score"],
            "anomaly_score": round(e["anomaly_score"] * 100, 1),
            "stage": e["stage"],
        }
        for e in recent_events
    ]

    return JSONResponse(content={
        "summary": {
            "total_traffic": stats["total_traffic"],
            "normal_traffic": stats["normal_traffic"],
            "attacks": stats["attacks"],
            "anomalies": stats["anomalies"],
            "critical_alerts": stats["critical_alerts"],
        },
        "attack_distribution": dist,
        "top_suspicious_sources": top_sources,
        "live_traffic_series": series,
        "recent_alerts": _alert_manager.alerts[-10:][::-1],
    })


@prd_router.get("/statistics")
def get_statistics():
    """
    GET /api/statistics — Returns system health, model versions, throughput, and uptime.
    """
    uptime_sec = time.time() - _alert_manager.start_time
    total = _alert_manager.stats["total_traffic"]
    throughput_eps = round(total / max(1.0, uptime_sec), 2)

    return JSONResponse(content={
        "status": "operational",
        "uptime_seconds": round(uptime_sec, 1),
        "total_events_processed": total,
        "throughput_events_per_sec": throughput_eps,
        "model_versions": {
            "world_model": "2.0.0",
            "known_classifier": _alert_manager.learner.active_version,
            "novelty_detector": "autoencoder-v1.0",
            "risk_engine": "v2.0-dynamic",
        },
        "stats": _alert_manager.stats,
    })


@prd_router.get("/attack-story/{story_id}")
def get_attack_story(story_id: str):
    """
    GET /api/attack-story/{id} — Returns a dynamically correlated attack story.
    If story_id is 'current' or 'live', dynamically correlates recent alert events.
    """
    if story_id in ("current", "live", "latest"):
        recent = _alert_manager.alerts[-15:]
        story = _alert_manager.story_engine.build_story(recent, story_id="story-live")
        return JSONResponse(content=story.to_dict())

    story = _alert_manager.story_engine.get_story(story_id)
    if story is None:
        raise HTTPException(status_code=404, detail=f"Attack story '{story_id}' not found")
    return JSONResponse(content=story.to_dict())


@prd_router.post("/world-model/rollout")
def run_world_model_rollout(body: RolloutRequest):
    """
    POST /api/world-model/rollout — K-step autoregressive future state simulation.
    """
    pipeline = _alert_manager.get_pipeline()
    if pipeline.world_model is None:
        raise HTTPException(status_code=503, detail="CyberWorldModelV2 not loaded")

    seq_arr = np.asarray(body.state_seq, dtype=np.float32)
    if seq_arr.ndim != 2 or seq_arr.shape[1] != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=422,
            detail=f"Expected state_seq shape (T, {len(FEATURE_NAMES)}), got {seq_arr.shape}",
        )

    # Scale sequence if scaler available
    if pipeline.scaler is not None:
        import pandas as pd
        df_tmp = pd.DataFrame(seq_arr, columns=FEATURE_NAMES)
        scaled_seq = pipeline.scaler.transform(df_tmp)
    else:
        scaled_seq = seq_arr

    res = pipeline.process_state(
        raw_or_scaled_state=scaled_seq[-1],
        is_scaled=True,
        historical_seq=scaled_seq,
        k_steps=body.k_steps,
    )

    return JSONResponse(content={
        "horizon_steps": body.k_steps,
        "rollout": res.world_model_rollout,
        "current_stage": res.current_stage,
        "predicted_next_stage": res.predicted_next_stage,
        "risk_score": res.risk_score,
        "provenance": "CyberWorldModelV2 autoregressive physical transition",
    })


@prd_router.post("/feedback")
def submit_feedback(body: FeedbackRequest):
    """
    POST /api/feedback — Human analyst feedback endpoint for Human-in-the-Loop Adaptive Learning.
    Validates novel events, records them in Threat Memory, and optionally triggers a controlled model update.
    """
    sample = _alert_manager.threat_memory.add_validation(
        feature_vector=body.feature_vector,
        validated_label=body.validated_label,
        is_malicious=body.is_malicious,
        analyst_notes=body.analyst_notes,
        analyst_id=body.analyst_id,
        sample_id=body.sample_id,
    )

    adaptation_record = None
    if body.trigger_update:
        try:
            record = _alert_manager.learner.adapt_model()
            adaptation_record = record.to_dict()
        except Exception as exc:
            logger.warning("Adaptive update could not run: %s", exc)

    return JSONResponse(content={
        "status": "recorded",
        "sample": sample.to_dict(),
        "threat_memory_total": len(_alert_manager.threat_memory.get_all_samples()),
        "adaptation_record": adaptation_record,
    })


@prd_router.get("/adaptation/history")
def get_adaptation_history():
    """Returns audit log of all model adaptations, before/after metrics, and versions."""
    return JSONResponse(content={
        "active_version": _alert_manager.learner.active_version,
        "history": _alert_manager.learner.get_adaptation_history(),
    })


@prd_router.post("/adaptation/rollback")
def rollback_model_version(target_version: Optional[str] = None):
    """Rolls back the active model to previous checkpoint."""
    try:
        res = _alert_manager.learner.rollback(target_version=target_version)
        return JSONResponse(content=res)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
