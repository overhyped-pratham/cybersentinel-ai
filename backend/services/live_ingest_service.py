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
from network.flow.flow_record import FlowRecord
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
        self._scaler = self._load_scaler()
        self._subscribers: Set[asyncio.Queue] = set()
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._closed = False
        # Sliding sequence buffer per session: session_id → deque of 24-D np arrays
        self._seq_buffers: Dict[str, Deque[np.ndarray]] = {}
        self._observed_stages: Dict[str, Set[str]] = {}
        self._event_log: Deque[Dict[str, Any]] = deque(maxlen=1000)  # bounded history
        self._stats = {
            "sessions_started": 0,
            "windows_processed": 0,
            "windows_inferred": 0,
            "windows_skipped": 0,
            "broadcast_errors": 0,
        }
        self._last_health = {
            "telemetry_source": "none",
            "flows_per_second": 0.0,
            "current_window_id": None,
            "inference_latency_ms": 0.0,
            "dropped_malformed_count": 0,
            "backpressure_status": "nominal",
            "last_inference_timestamp": None,
            "mode": "LIVE",
        }

    def _load_scaler(self):
        try:
            from ml.preprocessing.scaler import FeatureScaler
            from pathlib import Path
            workspace = Path(__file__).resolve().parent.parent.parent
            paths = [
                workspace / "models" / "scaler.pkl",
                workspace / "experiments" / "run_20260907_111554" / "scaler.pkl",
                workspace / "experiments" / "run_20260907_120029" / "world_model" / "scaler.pkl",
            ]
            for p in paths:
                if p.exists():
                    logger.info("[LiveIngest] Loaded FeatureScaler from %s", p)
                    return FeatureScaler.load(p)
        except Exception as exc:
            logger.warning("[LiveIngest] Scaler load failed: %s", exc)
        return None

    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    @property
    def event_log(self) -> List[Dict[str, Any]]:
        return list(self._event_log)

    def get_system_health(self) -> Dict[str, Any]:
        from backend.services.model_service import ModelService
        from backend.middleware.security import security_manager

        model_svc = ModelService.get_instance()
        return {
            "status": "healthy" if (model_svc.is_loaded and self._scaler is not None) else "degraded",
            "mode": self._last_health.get("mode", "LIVE"),
            "telemetry_source": self._last_health.get("telemetry_source", "none"),
            "flows_per_second": self._last_health.get("flows_per_second", 0.0),
            "current_window_id": self._last_health.get("current_window_id"),
            "inference_latency_ms": self._last_health.get("inference_latency_ms", 0.0),
            "model_version": _MODEL_VERSION,
            "model_available": model_svc.is_loaded,
            "scaler_available": self._scaler is not None,
            "websocket_subscribers": len(self._subscribers),
            "active_ws_connections": security_manager.active_ws_count,
            "max_ws_connections": security_manager.max_ws_connections,
            "dropped_malformed_count": self._last_health.get("dropped_malformed_count", 0),
            "queue_backpressure_status": self._last_health.get("backpressure_status", "nominal"),
            "last_inference_timestamp": self._last_health.get("last_inference_timestamp"),
            "uptime_windows_inferred": self._stats["windows_inferred"],
            "timestamp": _utcnow(),
        }

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
        mode: str = "LIVE",
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
        self._last_health["mode"] = mode

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
        self._observed_stages.pop(session_id, None)
        logger.info("[LiveIngest] Session %s stopped", session_id)

    async def stop_all(self) -> None:
        """Stop all running sessions and shut down the service."""
        self._closed = True
        for sid in list(self._active_tasks.keys()):
            await self.stop_session(sid)

    async def ingest_flows(
        self,
        flows: List[Any],
        *,
        session_id: Optional[str] = None,
        source_id: str = "MobileSimulator",
        k_steps: int = 4,
        window_seconds: float = 10.0,
        immediate_inference: bool = True,
    ) -> Dict[str, Any]:
        """
        Directly ingest a batch of synthetic flow records from the Mobile Simulator or API client.

        Converts inputs into standardized FlowRecords (strictly setting label='UNKNOWN'),
        packages them into a TelemetryWindowEvent, runs through the 24-D feature scaler and
        CyberWorldModelV2, broadcasts the result over WebSocket, and returns the live event.
        """
        from backend.services.model_service import ModelService
        model_svc = ModelService.get_instance()

        session_id = session_id or f"sim_{uuid.uuid4().hex[:8]}"

        if not flows:
            err_event = {
                "status": "INVALID_TELEMETRY",
                "timestamp": _utcnow(),
                "session_id": session_id,
                "source_id": source_id,
                "detail": "No flow records provided in ingestion batch",
            }
            await self._broadcast(err_event)
            return err_event

        now_ts = time.time()
        start_ts = now_ts - window_seconds
        records: List[FlowRecord] = []
        dropped_count = 0

        for i, item in enumerate(flows):
            try:
                d = item.model_dump() if hasattr(item, "model_dump") else (item.dict() if hasattr(item, "dict") else dict(item))

                ts = d.get("timestamp")
                if ts is None:
                    ts = start_ts + (float(i) / max(len(flows), 1)) * window_seconds

                rec = FlowRecord(
                    timestamp=float(ts),
                    src_ip=str(d["src_ip"]),
                    dst_ip=str(d["dst_ip"]),
                    src_port=int(d["src_port"]),
                    dst_port=int(d["dst_port"]),
                    protocol=int(d.get("protocol", 6)),
                    packets=int(d.get("packets", 1)),
                    bytes=int(d.get("bytes", 60)),
                    duration=float(d.get("duration", 0.01)),
                    syn_flag=int(d.get("syn_flag", 0)),
                    ack_flag=int(d.get("ack_flag", 0)),
                    rst_flag=int(d.get("rst_flag", 0)),
                    fin_flag=int(d.get("fin_flag", 0)),
                    psh_flag=int(d.get("psh_flag", 0)),
                    urg_flag=int(d.get("urg_flag", 0)),
                    failed=bool(d.get("failed", False)),
                    scenario_id=session_id,
                    label="UNKNOWN",  # STRICT DEFENSIVE DISCIPLINE: zero ground truth leakage
                )
                records.append(rec)
            except Exception as exc:
                logger.warning("[LiveIngest] Dropping malformed simulator flow %d: %s", i, exc)
                dropped_count += 1

        if not records:
            err_event = {
                "status": "INVALID_TELEMETRY",
                "timestamp": _utcnow(),
                "session_id": session_id,
                "source_id": source_id,
                "detail": "All flows in batch were malformed or invalid",
                "dropped_malformed": dropped_count,
            }
            await self._broadcast(err_event)
            return err_event

        records.sort(key=lambda r: r.timestamp)
        win_start = records[0].timestamp
        win_end = max(records[-1].timestamp + 0.001, win_start + window_seconds)

        window = TelemetryWindowEvent(
            window_id=str(uuid.uuid4()),
            window_start=win_start,
            window_end=win_end,
            flows=records,
            flow_count=len(records),
            dropped_malformed=dropped_count,
            source_id=source_id,
            wall_time=time.time(),
        )

        return await self._process_window(
            window=window,
            session_id=session_id,
            model_svc=model_svc,
            k_steps=k_steps,
            immediate_inference=immediate_inference,
        )

    async def _process_window(
        self,
        window: TelemetryWindowEvent,
        session_id: str,
        model_svc,
        k_steps: int = 4,
        immediate_inference: bool = False,
    ) -> Dict[str, Any]:
        """
        Core inference pipeline for a single completed telemetry window.

        Fail-closed: emits a structured error event on any component failure.
        NEVER substitutes static or cached predictions.
        """
        t0 = time.perf_counter()
        self._stats["windows_processed"] += 1

        # ---- 1. Guard: model must be loaded --------------------------------
        if not model_svc.is_loaded:
            event = {
                "status": "MODEL_UNAVAILABLE",
                "window_id": window.window_id,
                "timestamp": _utcnow(),
                "source_id": window.source_id,
            }
            await self._broadcast(event)
            self._stats["windows_skipped"] += 1
            return event

        # ---- 2. Guard: flows must be non-empty -----------------------------
        if not window.flows:
            event = {
                "status": "INVALID_TELEMETRY",
                "window_id": window.window_id,
                "timestamp": _utcnow(),
                "detail": "Window contains zero valid flows",
            }
            await self._broadcast(event)
            self._stats["windows_skipped"] += 1
            return event

        # ---- 3. Build 24-D feature state from flows ------------------------
        try:
            win_duration = max(5.0, (window.window_end - window.window_start) + 1.0)
            builder = NetworkStateBuilder(window_size_seconds=win_duration, step_size_seconds=win_duration)
            df_state = builder.build_states(
                window.flows,
                scenario_id=session_id,
                base_timestamp=window.window_start,
            )
        except Exception as exc:
            logger.error("[LiveIngest] StateBuilder error for window %s: %s",
                         window.window_id, exc)
            event = {
                "status": "INVALID_TELEMETRY",
                "window_id": window.window_id,
                "timestamp": _utcnow(),
                "detail": f"StateBuilder error: {exc}",
            }
            await self._broadcast(event)
            self._stats["windows_skipped"] += 1
            return event

        if df_state is None or len(df_state) == 0:
            event = {
                "status": "INVALID_TELEMETRY",
                "window_id": window.window_id,
                "timestamp": _utcnow(),
                "detail": "StateBuilder produced empty DataFrame",
            }
            await self._broadcast(event)
            self._stats["windows_skipped"] += 1
            return event

        # ---- 4. Extract raw feature vector and scale ------------------------
        try:
            raw_row = df_state[FEATURE_NAMES].iloc[-1].to_numpy(dtype=np.float32)
            if not np.all(np.isfinite(raw_row)):
                logger.warning("[LiveIngest] Non-finite features in window %s, replacing with 0",
                               window.window_id)
                raw_row = np.nan_to_num(raw_row, nan=0.0, posinf=0.0, neginf=0.0)

            if self._scaler is not None:
                import pandas as pd
                df_single = pd.DataFrame([raw_row], columns=FEATURE_NAMES)
                scaled_row = self._scaler.transform(df_single)[0]
            else:
                scaled_row = raw_row
        except Exception as exc:
            logger.error("[LiveIngest] Feature extraction error: %s", exc)
            event = {
                "status": "INVALID_TELEMETRY",
                "window_id": window.window_id,
                "timestamp": _utcnow(),
                "detail": f"Feature extraction error: {exc}",
            }
            await self._broadcast(event)
            self._stats["windows_skipped"] += 1
            return event

        # ---- 5. Accumulate into sliding sequence buffer --------------------
        seq_buf = self._seq_buffers.get(session_id)
        if seq_buf is None:
            seq_buf = deque(maxlen=_MAX_SEQ_LEN)
            self._seq_buffers[session_id] = seq_buf
        seq_buf.append(scaled_row)

        if len(seq_buf) < _MIN_SEQ_LEN:
            if immediate_inference:
                while len(seq_buf) < _MIN_SEQ_LEN:
                    seq_buf.appendleft(scaled_row)
            else:
                logger.info(
                    "[LiveIngest] Session %s: accumulating sequence (%d/%d)",
                    session_id, len(seq_buf), _MIN_SEQ_LEN
                )
                # Broadcast buffering status
                event = {
                    "status": "BUFFERING",
                    "window_id": window.window_id,
                    "timestamp": _utcnow(),
                    "session_id": session_id,
                    "sequence_accumulated": len(seq_buf),
                    "sequence_required": _MIN_SEQ_LEN,
                    "flow_count": window.flow_count,
                    "source_id": window.source_id,
                }
                await self._broadcast(event)
                return event

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
            event = {
                "status": "MODEL_UNAVAILABLE",
                "window_id": window.window_id,
                "timestamp": _utcnow(),
                "detail": f"Inference error: {exc}",
            }
            await self._broadcast(event)
            self._stats["windows_skipped"] += 1
            return event

        t_elapsed_ms = (time.perf_counter() - t0) * 1000

        self._last_health.update({
            "telemetry_source": window.source_id,
            "flows_per_second": round(window.flows_per_second, 2),
            "current_window_id": window.window_id,
            "inference_latency_ms": round(t_elapsed_ms, 2),
            "dropped_malformed_count": window.dropped_malformed,
            "last_inference_timestamp": _utcnow(),
            "backpressure_status": "warning" if self._stats["broadcast_errors"] > 0 else "nominal",
        })

        # Track observed stages across session
        cur_st = fc.get("current_stage")
        if cur_st:
            self._observed_stages.setdefault(session_id, set()).add(cur_st)
        session_observed = sorted(list(self._observed_stages.get(session_id, set())))

        # ---- 8. Compose telemetry-enriched event ---------------------------
        event = _build_live_event(
            fc, window, session_id, t_elapsed_ms,
            mode=self._last_health.get("mode", "LIVE"),
            observed_stages=session_observed,
        )

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
        return event


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _build_live_event(
    fc: Dict[str, Any],
    window: TelemetryWindowEvent,
    session_id: str,
    latency_ms: float,
    mode: str = "LIVE",
    observed_stages: Optional[List[str]] = None,
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
        "mode": mode,
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
        "observed_stages": observed_stages if observed_stages is not None else clean_fc.get("observed_stages", []),
    }
