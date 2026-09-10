"""
CyberSentinel AI — AI Security Analyst API Endpoints.

Provides:
  - POST /api/v1/agent/chat: Grounded natural-language threat analysis and playbooks.
  - GET  /api/v1/agent/status: Real-time connectivity and operational readiness.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.services.evidence_service import EvidenceService
from backend.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

agent_router = APIRouter(tags=["AI Security Analyst"])


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Inquiry for AI Security Analyst")
    session_id: Optional[str] = Field(None, description="Active telemetry or replay session ID")
    current_forecast: Optional[Dict[str, Any]] = Field(None, description="Current model forecast object")


class AgentChatResponse(BaseModel):
    answer: str
    timestamp: str
    model: str
    evidence: Dict[str, Any]
    llm_backend: str
    fallback: bool = False


class AgentStatusResponse(BaseModel):
    status: str
    model: str
    available: bool
    api_key_configured: bool
    rate_limit_rpm: int


def _get_services(request: Request) -> tuple[EvidenceService, GeminiService]:
    """Retrieve or lazily create EvidenceService and GeminiService instances."""
    app = request.app

    evidence_svc = getattr(app.state, "evidence_service", None)
    if evidence_svc is None:
        live_svc = getattr(app.state, "live_ingest_service", None)
        replay_svc = getattr(app.state, "replay_service", None)
        evidence_svc = EvidenceService(live_ingest_service=live_svc, replay_service=replay_svc)
        app.state.evidence_service = evidence_svc

    gemini_svc = getattr(app.state, "gemini_service", None)
    if gemini_svc is None:
        gemini_svc = GeminiService()
        app.state.gemini_service = gemini_svc

    return evidence_svc, gemini_svc


@agent_router.post("/agent/chat", response_model=AgentChatResponse)
def agent_chat(body: AgentChatRequest, request: Request):
    """
    Grounded AI Security Analyst chat endpoint.
    Reasoning is strictly grounded in CyberSentinel world-model predictions and telemetry.
    """
    evidence_svc, gemini_svc = _get_services(request)

    try:
        evidence = evidence_svc.get_latest_evidence(
            current_forecast=body.current_forecast,
            session_id=body.session_id,
        )
        grounded_prompt = evidence_svc.format_grounding_prompt(evidence, body.message)
        result = gemini_svc.generate_response(grounded_prompt, evidence, body.message)

        return AgentChatResponse(
            answer=result["answer"],
            timestamp=_utcnow(),
            model=result["model"],
            evidence=result["evidence"],
            llm_backend=result["llm_backend"],
            fallback=result.get("fallback", False),
        )
    except Exception as exc:
        logger.error("[agent_chat] Error processing analyst query: %s", exc, exc_info=True)
        # Always return safe fallback without crashing or returning 500
        fallback_evidence = evidence_svc.get_latest_evidence(body.current_forecast, body.session_id)
        fallback_answer = (
            "Gemini analyst unavailable. CyberSentinel deterministic intelligence remains operational.\n\n"
            + gemini_svc._deterministic_fallback(fallback_evidence, body.message)
        )
        return AgentChatResponse(
            answer=fallback_answer,
            timestamp=_utcnow(),
            model="cybersentinel_rule_engine",
            evidence=fallback_evidence,
            llm_backend="error_fallback",
            fallback=True,
        )


@agent_router.get("/agent/status", response_model=AgentStatusResponse)
def agent_status(request: Request):
    """Return AI Security Analyst operational status and Gemini availability."""
    _, gemini_svc = _get_services(request)
    status_data = gemini_svc.get_status()
    return AgentStatusResponse(**status_data)
