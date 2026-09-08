"""
CyberSentinel AI — FastAPI Route Handlers (Phase 10).
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.schemas.forecast import (
    ForecastRequest,
    AgentQueryRequest,
    AgentQueryResponse,
    ReplayStartRequest,
    ReplayStartResponse,
    ReplayStepResponse,
    ReplayStatusResponse,
    ModelInfoResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_svc(request: Request):
    return request.app.state.model_service


def _get_replay(request: Request):
    return request.app.state.replay_service


def _get_agent(request: Request):
    return request.app.state.agent


def _clean_fc(fc: Dict[str, Any]) -> Dict[str, Any]:
    """Remove internal fields before serializing to API response."""
    clean = {k: v for k, v in fc.items() if not k.startswith("_")}
    return clean


# ---------------------------------------------------------------------------
# Health & Info
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    svc = _get_svc(request)
    agent = _get_agent(request)
    ollama_ok = False
    if agent is not None:
        agent._ensure_ollama_checked()
        ollama_ok = bool(agent._ollama_available)

    from pathlib import Path
    dataset_ok = (Path(__file__).resolve().parent.parent.parent / "datasets" / "sample").exists()

    status = "ok" if svc.is_loaded else "degraded"
    return HealthResponse(
        status=status,
        model_loaded=svc.is_loaded,
        calibration_loaded=svc.calibration_loaded,
        dataset_available=dataset_ok,
        ollama_available=ollama_ok,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(request: Request):
    svc = _get_svc(request)
    info = svc.get_model_info()
    if "error" in info:
        raise HTTPException(status_code=503, detail=info["error"])
    return ModelInfoResponse(**info)


# ---------------------------------------------------------------------------
# Forecast endpoints
# ---------------------------------------------------------------------------

@router.post("/forecast")
def forecast(body: ForecastRequest, request: Request):
    svc = _get_svc(request)
    if not svc.is_loaded:
        raise HTTPException(status_code=503, detail="MODEL_UNAVAILABLE")
    try:
        fc = svc.forecast(
            x_seq=body.x_seq,
            mask_list=body.mask,
            k_steps=body.k_steps,
        )
        return JSONResponse(content=_clean_fc(fc))
    except Exception as exc:
        logger.error("/forecast error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/rollout")
def rollout(body: ForecastRequest, request: Request):
    """Return only the K-step rollout portion of the forecast."""
    svc = _get_svc(request)
    if not svc.is_loaded:
        raise HTTPException(status_code=503, detail="MODEL_UNAVAILABLE")
    try:
        fc = svc.forecast(x_seq=body.x_seq, mask_list=body.mask, k_steps=body.k_steps)
        return JSONResponse(content={
            "rollout_steps": fc["rollout_steps"],
            "safety_flags": fc["safety_flags"],
            "current_stage": fc["current_stage"],
            "horizon_steps": body.k_steps,
            "provenance": fc["provenance"],
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/explain")
def explain(body: ForecastRequest, request: Request):
    """Return explainability output for the given sequence."""
    svc = _get_svc(request)
    if not svc.is_loaded:
        raise HTTPException(status_code=503, detail="MODEL_UNAVAILABLE")
    try:
        fc = svc.forecast(x_seq=body.x_seq, mask_list=body.mask, k_steps=1)
        return JSONResponse(content={
            "current_stage": fc["current_stage"],
            "predicted_next_stage": fc["predicted_next_stage"],
            "top_features": fc["top_features"],
            "stage_relevant_features": fc["stage_relevant_features"],
            "explanation_narrative": fc["explanation_narrative"],
            "provenance": fc["provenance"],
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/mitre")
def mitre(body: ForecastRequest, request: Request):
    """Return MITRE ATT&CK mapping for predicted next stage."""
    svc = _get_svc(request)
    if not svc.is_loaded:
        raise HTTPException(status_code=503, detail="MODEL_UNAVAILABLE")
    try:
        fc = svc.forecast(x_seq=body.x_seq, mask_list=body.mask, k_steps=1)
        return JSONResponse(content={
            "predicted_next_stage": fc["predicted_next_stage"],
            "primary_technique_id": fc["primary_technique_id"],
            "primary_technique_name": fc["primary_technique_name"],
            "mitre_techniques": fc["mitre_techniques"],
            "note": "MITRE ATT&CK Enterprise v14 static mapping — not LLM-generated.",
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/risk")
def risk(body: ForecastRequest, request: Request):
    """Return risk assessment for the given sequence."""
    svc = _get_svc(request)
    if not svc.is_loaded:
        raise HTTPException(status_code=503, detail="MODEL_UNAVAILABLE")
    try:
        fc = svc.forecast(x_seq=body.x_seq, mask_list=body.mask, k_steps=1)
        return JSONResponse(content={
            "risk_score": fc["risk_score"],
            "risk_level": fc["risk_level"],
            "recommended_priority": fc["recommended_priority"],
            "time_to_transition_hint": fc["time_to_transition_hint"],
            "attack_probability": fc["attack_probability"],
            "predicted_next_stage": fc["predicted_next_stage"],
            "provenance": fc["provenance"],
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@router.post("/agent/query", response_model=AgentQueryResponse)
def agent_query(body: AgentQueryRequest, request: Request):
    agent = _get_agent(request)
    try:
        result = agent.answer(
            query=body.query,
            current_forecast=body.current_forecast,
            session_id=body.session_id,
        )
        return AgentQueryResponse(**result)
    except Exception as exc:
        logger.error("/agent/query error: %s", exc, exc_info=True)
        # Graceful degradation — return template answer even on error
        return AgentQueryResponse(
            answer=f"I encountered an error processing your query. "
                   f"Please check the /health endpoint for system status.",
            tool_calls=[],
            provenance="error_fallback",
            llm_backend="error_fallback",
            grounded_in_model_output=False,
        )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

@router.post("/replay/start", response_model=ReplayStartResponse)
def replay_start(body: ReplayStartRequest, request: Request):
    svc = _get_replay(request)
    try:
        session_id, info = svc.start_session(body.scenario_id, k_steps=body.k_steps)
        return ReplayStartResponse(**info)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("/replay/start error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/replay/step")
def replay_step(session_id: str, request: Request):
    svc = _get_replay(request)
    model_svc = _get_svc(request)
    if not model_svc.is_loaded:
        raise HTTPException(status_code=503, detail="MODEL_UNAVAILABLE")
    try:
        result = svc.step(session_id)
        # Remove internal tensor fields before serialization
        if result.get("forecast"):
            result["forecast"] = _clean_fc(result["forecast"])
        return JSONResponse(content=result)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("/replay/step error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/replay/status")
def replay_status(session_id: str, request: Request):
    svc = _get_replay(request)
    try:
        return JSONResponse(content=svc.get_status(session_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/replay/scenarios")
def replay_scenarios(request: Request):
    """List all available scenario IDs for replay."""
    svc = _get_replay(request)
    try:
        return JSONResponse(content={"scenarios": svc.list_available_scenarios()})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
