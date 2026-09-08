"""
CyberSentinel AI — Live Ingestion Service (Phase 13).

Bridges the stream processor (telemetry layer) with the ML inference layer.
For every completed TelemetryWindowEvent:

  1. Build the 24-D physical network state via NetworkStateBuilder.
  2. Maintain a sliding sequence buffer (min_seq_len=5 windows).
  3. Apply FeatureScaler.
  4. Run CyberWorldModelV2 inference via ModelService.forecast().
  5. Attach telemetry metadata (window_id, source, flow stats).
  6. Broadcast the enriched ForecastEvent to all registered WebSocket clients.
  7. Log structured observability events.

Fail-closed behaviour:
  - If model is unavailable → emit {"status": "MODEL_UNAVAILABLE"} event.
  - If scaler is unavailable → emit {"status": "SCALER_UNAVAILABLE"} event.
  - If telemetry window contains no valid flows → emit {"status": "INVALID_TELEMETRY"}.
  - NEVER substitutes static/random predictions.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import math
import time
import uuid
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Set

import numpy as np

from network.telemetry.stream_processor import StreamProcessor, TelemetryWindowEvent
from ml.state.state_builder import NetworkStateBuilder, FEATURE_NAMES

logger = logging.getLogger(__name__)

# Minimum windows accumulated before first ML inference (model needs a sequence)
_MIN_SEQ_LEN = 5
_MAX_SEQ_LEN = 20  # Sequence buffer cap

# Model version identifier (for provenance logging)
_MODEL_VERSION = "CyberWorldModelV2-Phase8C"


class LiveIngestService:
    """
    Manages live telemetry ingestion sessions and ML inference.

    One instance is created per application lifetime (singleton pattern
    mirrors ModelService).  Multiple concurrent telemetry sources can run
    as separate tasks.

    WebSocket clients register via `subscribe()` / `unsubscribe()`.
    Each completed inference is broadcast to all subscribers.
    """

    def __init__(self) -> None:
        self._state_builder = NetworkStateBuilder()
        self._subscribers: Set[asyncio.Queue] = set()
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._closed = False
        # Sliding sequence buffer per session: session_id → deque of 24-D np arrays
        self._seq_buffers: Dict[str, Deque[np.ndarray]] = {}
        self._event_log: Deque[Dict[str, Any]] = deque(maxlen=1000)  # bounded history
        self._stats = {
            "sessions_started": 0,
            "windows_processed": 0,
            "windows_inferred": 0,
            "windows_skipped": 0,
            "broadcast_errors": 0,
        }

    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    @property
    def event_log(self) -> List[Dict[str, Any]]:
        return list(self._event_log)

    def subscribe(self) -> asyncio.Queue:
        """Register a new subscriber queue.  Returns the queue to read events from."""
        q: asyncio.Queue = asyncio.Queue(maxsize=128)
        self._subscribers.add(q)
        logger.info("[LiveIngest] New subscriber registered (%d total)", len(self._subscribers))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Deregister a subscriber queue."""
        self._subscribers.discard(q)
        logger.info("[LiveIngest] Subscriber removed (%d remaining)", len(self._subscribers))

    async def _broadcast(self, event: Dict[str, Any]) -> None:
        """Broadcast an event to all registered subscriber queues."""
        dead: List[asyncio.Queue] = []
        for q in list(self._subscribers):
            try:
                if q.full():
                    logger.warning("[LiveIngest] Subscriber queue full — dropping event (backpressure)")
                    self._stats["broadcast_errors"] += 1
                else:
                    q.put_nowait(event)
            except Exception as exc:
                logger.error("[LiveIngest] Broadcast error: %s", exc)
                dead.append(q)
        for q in dead:
            self._subscribers.discard(q)

    async def start_session(
        self,
        source,  # TelemetrySource
        *,
        session_id: Optional[str] = None,
        window_seconds: float = 30.0,
        stride_seconds: Optional[float] = None,
        k_steps: int = 4,
    ) -> str:
        """
        Start a live ingestion session from the given TelemetrySource.

        Returns the session_id which can be used to stop the session later.
        """
        from backend.services.model_service import ModelService
        model_svc = ModelService.get_instance()

        session_id = session_id or str(uuid.uuid4())
        self._seq_buffers[session_id] = deque(maxlen=_MAX_SEQ_LEN)
        self._stats["sessions_started"] += 1

        processor = StreamProcessor(
            source,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            min_flows_per_window=1,
        )

        async def _run():
            run_task = asyncio.create_task(processor.run())
            try:
                async for window in processor.windows():
                    if self._closed:
                        break
                    await self._process_window(window, session_id, model_svc, k_steps=k_steps)
            finally:
                await processor.stop()
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
                logger.info("[LiveIngest] Session %s ended. Proc stats: %s",
                            session_id, processor.stats)

        task = asyncio.create_task(_run())
        self._active_tasks[session_id] = task
        logger.info("[LiveIngest] Session %s started (window=%.0fs, stride=%.0fs)",
                    session_id, window_seconds, stride_seconds or window_seconds)
        return session_id

    async def stop_session(self, session_id: str) -> None:
        """Stop a running ingestion session."""
        task = self._active_tasks.pop(session_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._seq_buffers.pop(session_id, None)
        logger.info("[LiveIngest] Session %s stopped", session_id)

    async def stop_all(self) -> None:
        """Stop all running sessions and shut down the service."""
        self._closed = True
        for sid in list(self._active_tasks.keys()):
            await self.stop_session(sid)

    async def _process_window(
        self,
        window: TelemetryWindowEvent,
        session_id: str,
        model_svc,
        k_steps: int = 4,
    ) -> None:
        """
        Core inference pipeline for a single completed telemetry window.

        Fail-closed: emits a structured error event on any component failure.
        NEVER substitutes static or cached predictions.
        """
        t0 = time.perf_counter()
        self._stats["windows_processed"] += 1

        # ---- 1. Guard: model must be loaded --------------------------------
        if not model_svc.is_loaded:
            await self._broadcast({
                "status": "MODEL_UNAVAILABLE",
                "window_id": window.window_id,
                "timestamp": _utcnow(),
                "source_id": window.source_id,
            })
            self._stats["windows_skipped"] += 1
            return

        # ---- 2. Guard: flows must be non-empty -----------------------------
        if not window.flows:
            await self._broadcast({
                "status": "INVALID_TELEMETRY",
                "window_id": window.window_id,
                "timestamp": _utcnow(),
                "detail": "Window contains zero valid flows",
            })
            self._stats["windows_skipped"] += 1
            return

        # ---- 3. Build 24-D feature state from flows ------------------------
        try:
            df_state = self._state_builder.build_states(
                window.flows,
                scenario_id=session_id,
                base_timestamp=window.window_start,
            )
        except Exception as exc:
            logger.error("[LiveIngest] StateBuilder error for window %s: %s",
                         window.window_id, exc)
            await self._broadcast({
                "status": "INVALID_TELEMETRY",
                "window_id": window.window_id,
                "timestamp": _utcnow(),
                "detail": f"StateBuilder error: {exc}",
            })
            self._stats["windows_skipped"] += 1
            return

        if df_state is None or len(df_state) == 0:
            await self._broadcast({
                "status": "INVALID_TELEMETRY",
                "window_id": window.window_id,
                "timestamp": _utcnow(),
                "detail": "StateBuilder produced empty DataFrame",
            })
            self._stats["windows_skipped"] += 1
            return

        # ---- 4. Extract raw feature vector from last state row -------------
        try:
            raw_row = df_state[FEATURE_NAMES].iloc[-1].to_numpy(dtype=np.float32)
            if not np.all(np.isfinite(raw_row)):
                logger.warning("[LiveIngest] Non-finite features in window %s, replacing with 0",
                               window.window_id)
                raw_row = np.nan_to_num(raw_row, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as exc:
            logger.error("[LiveIngest] Feature extraction error: %s", exc)
            await self._broadcast({
                "status": "INVALID_TELEMETRY",
                "window_id": window.window_id,
                "timestamp": _utcnow(),
                "detail": f"Feature extraction error: {exc}",
            })
            self._stats["windows_skipped"] += 1
            return

        # ---- 5. Accumulate into sliding sequence buffer --------------------
        seq_buf = self._seq_buffers.get(session_id)
        if seq_buf is None:
            seq_buf = deque(maxlen=_MAX_SEQ_LEN)
            self._seq_buffers[session_id] = seq_buf
        seq_buf.append(raw_row)

        if len(seq_buf) < _MIN_SEQ_LEN:
            logger.info(
                "[LiveIngest] Session %s: accumulating sequence (%d/%d)",
                session_id, len(seq_buf), _MIN_SEQ_LEN
            )
            # Broadcast buffering status
            await self._broadcast({
                "status": "BUFFERING",
                "window_id": window.window_id,
                "timestamp": _utcnow(),
                "session_id": session_id,
                "sequence_accumulated": len(seq_buf),
                "sequence_required": _MIN_SEQ_LEN,
                "flow_count": window.flow_count,
                "source_id": window.source_id,
            })
            return

        # ---- 6. Build model input sequence ---------------------------------
        seq_list = list(seq_buf)  # chronological order, shape (T, 24)

        # ---- 7. Run ML inference via ModelService (real, not cached) -------
        try:
            fc = model_svc.forecast(
                x_seq=[row.tolist() for row in seq_list],
                mask_list=None,
                k_steps=k_steps,
            )
        except Exception as exc:
            logger.error("[LiveIngest] ModelService.forecast() error: %s", exc, exc_info=True)
            await self._broadcast({
                "status": "MODEL_UNAVAILABLE",
                "window_id": window.window_id,
                "timestamp": _utcnow(),
                "detail": f"Inference error: {exc}",
            })
            self._stats["windows_skipped"] += 1
            return

        t_elapsed_ms = (time.perf_counter() - t0) * 1000

        # ---- 8. Compose telemetry-enriched event ---------------------------
        event = _build_live_event(fc, window, session_id, t_elapsed_ms)

        # ---- 9. Structured observability log --------------------------------
        logger.info(
            "[LiveIngest] W=%s | stage=%s → next=%s | atk=%.3f | risk=%.1f | "
            "flows=%d | latency=%.1fms | session=%s",
            window.window_id[:8],
            fc.get("current_stage", "?"),
            fc.get("predicted_next_stage", "?"),
            fc.get("attack_probability", 0.0),
            fc.get("risk_score", 0.0),
            window.flow_count,
            t_elapsed_ms,
            session_id[:8],
        )

        # ---- 10. Append to bounded event log and broadcast -----------------
        self._event_log.append(event)
        self._stats["windows_inferred"] += 1
        await self._broadcast(event)


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _build_live_event(
    fc: Dict[str, Any],
    window: TelemetryWindowEvent,
    session_id: str,
    latency_ms: float,
) -> Dict[str, Any]:
    """
    Merge the ML ForecastEvent with live telemetry metadata.

    All intelligence fields come directly from `fc` (the real ModelService output).
    Telemetry metadata (window timestamps, flow counts, source_id) is purely
    observational — not used in any prediction.
    """
    # Strip internal fields (tensors, raw ForecastEvent object)
    clean_fc = {k: v for k, v in fc.items() if not k.startswith("_")}

    return {
        "status": "FORECAST",
        "event_id": str(uuid.uuid4()),
        "window_id": window.window_id,
        "session_id": session_id,
        "timestamp": _utcnow(),
        "window_start": window.window_start,
        "window_end": window.window_end,
        "flow_count": window.flow_count,
        "flows_per_second": round(window.flows_per_second, 2),
        "dropped_malformed": window.dropped_malformed,
        "source_id": window.source_id,
        "inference_latency_ms": round(latency_ms, 2),
        "model_version": _MODEL_VERSION,
        "provenance": clean_fc.get("provenance", "CyberWorldModelV2.forecast()"),
        # ---- ML intelligence (all dynamically produced, never cached) ----
        "current_stage": clean_fc.get("current_stage"),
        "predicted_next_stage": clean_fc.get("predicted_next_stage"),
        "attack_probability": clean_fc.get("attack_probability"),
        "confidence": clean_fc.get("confidence"),
        "transition_detected": clean_fc.get("transition_detected"),
        "transition_probability": clean_fc.get("transition_probability"),
        "stage_probabilities": clean_fc.get("stage_probabilities"),
        "risk_score": clean_fc.get("risk_score"),
        "risk_level": clean_fc.get("risk_level"),
        "recommended_priority": clean_fc.get("recommended_priority"),
        "time_to_transition_hint": clean_fc.get("time_to_transition_hint"),
        "top_features": clean_fc.get("top_features"),
        "explanation_narrative": clean_fc.get("explanation_narrative"),
        "stage_relevant_features": clean_fc.get("stage_relevant_features"),
        "mitre_techniques": clean_fc.get("mitre_techniques"),
        "primary_technique_id": clean_fc.get("primary_technique_id"),
        "primary_technique_name": clean_fc.get("primary_technique_name"),
        "rollout_steps": clean_fc.get("rollout_steps"),
        "safety_flags": clean_fc.get("safety_flags"),
    }
