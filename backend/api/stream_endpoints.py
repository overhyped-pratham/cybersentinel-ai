"""
CyberSentinel AI — Live Streaming API Endpoints (Phase 13).

Provides:
  GET  /api/v1/stream/ws        — WebSocket endpoint (bidirectional)
  GET  /api/v1/stream/events    — Server-Sent Events endpoint (unidirectional)
  POST /api/v1/stream/start     — Start a live telemetry session
  POST /api/v1/stream/stop      — Stop a live telemetry session
  GET  /api/v1/stream/status    — Current session status + stats
  GET  /api/v1/stream/history   — Last N broadcast events (for late-joining clients)

WebSocket protocol:
  Server → Client: JSON-encoded event objects (see LiveIngestService._build_live_event)
  Client → Server: JSON control messages:
    {"type": "ping"}            → {"type": "pong"}
    {"type": "subscribe"}       → acknowledged
    {"type": "unsubscribe"}     → closes connection

Fail-closed:
  All status codes propagate: MODEL_UNAVAILABLE, TELEMETRY_UNAVAILABLE,
  INVALID_TELEMETRY, SCALER_UNAVAILABLE, CALIBRATION_UNAVAILABLE.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

stream_router = APIRouter(prefix="/stream", tags=["streaming"])

# Heartbeat interval in seconds
_HEARTBEAT_INTERVAL = 15.0
# Maximum events to return in history endpoint
_MAX_HISTORY = 100


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

from backend.schemas.stream import StartSessionRequest, StopSessionRequest, IngestTelemetryRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_live_svc(request: Request):
    svc = getattr(request.app.state, "live_ingest_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="LIVE_INGEST_SERVICE_UNAVAILABLE")
    return svc


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# REST control endpoints
# ---------------------------------------------------------------------------

@stream_router.post("/start")
async def start_session(body: StartSessionRequest, request: Request):
    """Start a live telemetry ingestion session."""
    svc = _get_live_svc(request)

    try:
        from network.telemetry.sources import create_source
        kwargs: Dict[str, Any] = {}

        if body.source_kind == "replay":
            if not body.source_path:
                raise HTTPException(status_code=422, detail="source_path required for replay source")
            kwargs = {
                "path": body.source_path,
                "realtime_factor": body.realtime_factor,
                "scenario_id": body.session_id or "live_replay",
            }
        elif body.source_kind == "pcap":
            if not body.source_path:
                raise HTTPException(status_code=422, detail="source_path required for pcap source")
            kwargs = {
                "path": body.source_path,
                "tail": True,
                "scenario_id": body.session_id or "live_pcap",
            }
        elif body.source_kind == "netflow":
            kwargs = {
                "host": body.host or "0.0.0.0",
                "port": body.port or 9995,
                "scenario_id": body.session_id or "live_netflow",
            }
        else:
            raise HTTPException(status_code=422, detail=f"Unknown source_kind: {body.source_kind!r}")

        source = create_source(body.source_kind, **kwargs)
        session_id = await svc.start_session(
            source,
            session_id=body.session_id,
            window_seconds=body.window_seconds,
            stride_seconds=body.stride_seconds,
            k_steps=body.k_steps,
            mode=body.mode or "LIVE",
        )
        return JSONResponse(content={
            "status": "started",
            "session_id": session_id,
            "mode": body.mode or "LIVE",
            "source_kind": body.source_kind,
            "window_seconds": body.window_seconds,
            "stride_seconds": body.stride_seconds or body.window_seconds,
            "k_steps": body.k_steps,
            "timestamp": _utcnow(),
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/stream/start error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@stream_router.post("/ingest")
async def ingest_telemetry(body: IngestTelemetryRequest, request: Request):
    """
    Ingest a batch of raw synthetic network telemetry flow records (Phase 16 / Mobile Simulator).

    Enforces fail-closed validation, converts to 24-D physical network state, transforms
    via production FeatureScaler, runs live CyberWorldModelV2 inference, and broadcasts to WebSocket.
    """
    svc = _get_live_svc(request)
    try:
        event = await svc.ingest_flows(
            flows=body.flows,
            session_id=body.session_id,
            source_id=body.source_id or "MobileSimulator",
            k_steps=body.k_steps or 4,
            window_seconds=body.window_seconds or 10.0,
            immediate_inference=True,
        )
        return JSONResponse(content=event)
    except Exception as exc:
        logger.error("/stream/ingest error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@stream_router.post("/stop")
async def stop_session(body: StopSessionRequest, request: Request):
    """Stop a running telemetry session."""
    svc = _get_live_svc(request)
    await svc.stop_session(body.session_id)
    return JSONResponse(content={"status": "stopped", "session_id": body.session_id})


@stream_router.get("/status")
async def stream_status(request: Request):
    """Return live ingest service stats and active sessions."""
    svc = _get_live_svc(request)
    return JSONResponse(content={
        "status": "ok",
        "stats": svc.stats,
        "active_sessions": list(svc._active_tasks.keys()),
        "subscriber_count": len(svc._subscribers),
        "timestamp": _utcnow(),
    })


@stream_router.get("/health")
async def stream_health(request: Request):
    """Return comprehensive system health and observability metrics."""
    svc = _get_live_svc(request)
    return JSONResponse(content=svc.get_system_health())


@stream_router.get("/history")
async def stream_history(request: Request, limit: int = 50):
    """Return the last N forecast events from the bounded event log."""
    svc = _get_live_svc(request)
    events = svc.event_log
    events = events[-min(limit, _MAX_HISTORY):]
    return JSONResponse(content={"events": events, "count": len(events)})


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@stream_router.websocket("/ws")
async def websocket_stream(websocket: WebSocket, request: Request = None):
    """
    WebSocket streaming endpoint.

    Server continuously emits structured JSON forecast events.
    Client may send control messages (ping/subscribe/unsubscribe).

    Protocol:
      Client connects → Server sends {"type": "connected", "timestamp": ...}
      Server emits forecast events as JSON on each telemetry window
      Client sends {"type": "ping"} → Server responds {"type": "pong"}
      Heartbeat: Server sends {"type": "heartbeat"} every 15 seconds
      Client disconnects → subscription cleaned up automatically
    """
    # Get the live ingest service from app state
    live_svc = getattr(websocket.app.state, "live_ingest_service", None)
    if live_svc is None:
        await websocket.close(code=1011, reason="LIVE_INGEST_SERVICE_UNAVAILABLE")
        return

    from backend.middleware.security import security_manager

    # Enforce connection limits
    if not security_manager.acquire_ws_slot():
        logger.warning("[WS] Connection limit reached (%d max) — rejecting client %s",
                       security_manager.max_ws_connections, websocket.client)
        await websocket.close(code=1013, reason="MAX_CONNECTIONS_REACHED")
        return

    await websocket.accept()
    logger.info("[WS] Client connected: %s", websocket.client)

    # Register subscriber queue
    q = live_svc.subscribe()

    async def _send_json(obj: Any) -> bool:
        """Send JSON message; return False if connection is dead."""
        try:
            await websocket.send_text(json.dumps(obj, default=str))
            return True
        except Exception:
            return False

    # Greeting
    await _send_json({
        "type": "connected",
        "timestamp": _utcnow(),
        "message": "CyberSentinel AI live stream active",
    })

    # Replay history for late-joining clients
    for evt in live_svc.event_log[-10:]:
        if not await _send_json(evt):
            break

    async def _heartbeat():
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            if not await _send_json({"type": "heartbeat", "timestamp": _utcnow()}):
                break

    async def _event_sender():
        """Forward events from subscriber queue to WebSocket."""
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=1.0)
                if not await _send_json(event):
                    break
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _message_receiver():
        """Handle incoming client control messages."""
        while True:
            try:
                text = await websocket.receive_text()
                try:
                    msg = json.loads(text)
                    msg_type = msg.get("type", "")
                    if msg_type == "ping":
                        await _send_json({"type": "pong", "timestamp": _utcnow()})
                    elif msg_type == "unsubscribe":
                        break
                    else:
                        logger.debug("[WS] Unknown control message: %s", msg_type)
                except json.JSONDecodeError:
                    logger.warning("[WS] Malformed control message from client")
            except WebSocketDisconnect:
                break
            except Exception:
                break

    try:
        heartbeat_task = asyncio.create_task(_heartbeat())
        sender_task = asyncio.create_task(_event_sender())
        receiver_task = asyncio.create_task(_message_receiver())

        # Wait for any task to finish (e.g. client disconnects)
        done, pending = await asyncio.wait(
            [heartbeat_task, sender_task, receiver_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    except WebSocketDisconnect:
        logger.info("[WS] Client disconnected: %s", websocket.client)
    except Exception as exc:
        logger.error("[WS] Error: %s", exc, exc_info=True)
    finally:
        from backend.middleware.security import security_manager
        security_manager.release_ws_slot()
        live_svc.unsubscribe(q)
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("[WS] Connection cleaned up: %s", websocket.client)


# ---------------------------------------------------------------------------
# Server-Sent Events endpoint
# ---------------------------------------------------------------------------

@stream_router.get("/events")
async def sse_stream(request: Request):
    """
    Server-Sent Events endpoint for unidirectional streaming.

    Emits forecast events as SSE data frames.
    Suitable for browser EventSource API.
    """
    live_svc = getattr(request.app.state, "live_ingest_service", None)
    if live_svc is None:
        raise HTTPException(status_code=503, detail="LIVE_INGEST_SERVICE_UNAVAILABLE")

    q = live_svc.subscribe()

    async def _generate():
        # Initial connection event
        yield f"event: connected\ndata: {json.dumps({'timestamp': _utcnow()})}\n\n"

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=_HEARTBEAT_INTERVAL)
                    data = json.dumps(event, default=str)
                    event_type = event.get("status", "forecast").lower()
                    yield f"event: {event_type}\ndata: {data}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat
                    yield f"event: heartbeat\ndata: {json.dumps({'timestamp': _utcnow()})}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            live_svc.unsubscribe(q)
            logger.info("[SSE] Client disconnected")

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
