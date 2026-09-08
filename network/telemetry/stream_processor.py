"""
CyberSentinel AI — Async Stream Processor (Phase 13).

Consumes an async TelemetrySource, accumulates flows into configurable
time windows, and emits completed 24-D feature vectors for ML inference.

Design Principles:
  - No prediction caching: every completed window triggers a fresh ML call.
  - Bounded memory: internal deque has a capped maximum flow backlog.
  - Causal: window features are computed only from flows within [t_start, t_end).
  - Backpressure: when the downstream inference queue is full, new windows are
    dropped (not blocking) with a warning log.
  - Graceful shutdown: `stop()` drains ongoing windows cleanly.
  - Malformed flow tolerance: invalid FlowRecord fields cause skip, not crash.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Deque, Dict, List, Optional, Tuple

from network.flow.flow_record import FlowRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Window event (emitted per completed 30-s window)
# ---------------------------------------------------------------------------

@dataclass
class TelemetryWindowEvent:
    """
    Encapsulates a completed telemetry window ready for ML inference.

    Attributes:
        window_id: Unique UUID string identifying this window.
        window_start: Unix timestamp of window start (inclusive).
        window_end: Unix timestamp of window end (exclusive).
        flows: List of FlowRecord objects within the window.
        flow_count: Number of valid flows.
        dropped_malformed: Count of flows discarded due to validation errors.
        source_id: Identifier of the telemetry source that produced this window.
        wall_time: System wall-clock time when the window was sealed.
    """
    window_id: str
    window_start: float
    window_end: float
    flows: List[FlowRecord]
    flow_count: int
    dropped_malformed: int
    source_id: str
    wall_time: float = field(default_factory=time.time)

    @property
    def duration_seconds(self) -> float:
        return self.window_end - self.window_start

    @property
    def flows_per_second(self) -> float:
        d = self.duration_seconds
        return self.flow_count / d if d > 0 else 0.0


# ---------------------------------------------------------------------------
# Stream Processor
# ---------------------------------------------------------------------------

class StreamProcessor:
    """
    Async pipeline: TelemetrySource → windowed FlowRecords → TelemetryWindowEvent.

    Args:
        source: Any TelemetrySource instance (PCAP, NetFlow, Replay).
        window_seconds: Width of each feature-extraction window in seconds.
        stride_seconds: How far to advance between consecutive windows.
                        Defaults to window_seconds (non-overlapping).
        max_flow_backlog: Maximum flows held in memory between windows.
        max_output_queue: Maximum pending TelemetryWindowEvents before backpressure.
        min_flows_per_window: Windows with fewer flows are skipped (not emitted).
    """

    def __init__(
        self,
        source,  # TelemetrySource
        *,
        window_seconds: float = 30.0,
        stride_seconds: Optional[float] = None,
        max_flow_backlog: int = 50_000,
        max_output_queue: int = 64,
        min_flows_per_window: int = 1,
    ) -> None:
        self._source = source
        self._window_seconds = window_seconds
        self._stride_seconds = stride_seconds if stride_seconds is not None else window_seconds
        self._max_backlog = max_flow_backlog
        self._max_out = max_output_queue
        self._min_flows = min_flows_per_window
        self._closed = False
        self._output: asyncio.Queue[TelemetryWindowEvent] = asyncio.Queue(maxsize=max_output_queue)
        self._stats = {
            "total_flows_received": 0,
            "total_flows_dropped_malformed": 0,
            "total_windows_emitted": 0,
            "total_windows_skipped_empty": 0,
            "total_output_dropped_backpressure": 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    async def stop(self) -> None:
        """Signal shutdown to both the source and the processing loop."""
        self._closed = True
        await self._source.close()

    async def run(self) -> None:
        """
        Reads from the source and populates the output queue.
        Call this as a background task: `asyncio.create_task(processor.run())`
        """
        buffer: Deque[FlowRecord] = deque(maxlen=self._max_backlog)
        w_start: Optional[float] = None

        try:
            async for raw_flow in self._source.stream():
                if self._closed:
                    break

                # Validate + sanitize the flow record
                flow = self._validate_flow(raw_flow)
                if flow is None:
                    self._stats["total_flows_dropped_malformed"] += 1
                    continue

                self._stats["total_flows_received"] += 1

                # Initialize window anchor on first valid flow
                if w_start is None:
                    w_start = flow.timestamp

                # Handle out-of-order flow: skip if more than 1 window in the past
                if flow.timestamp < w_start - self._window_seconds:
                    logger.debug("[StreamProcessor] OOO flow dropped: ts=%.3f w_start=%.3f",
                                 flow.timestamp, w_start)
                    continue

                buffer.append(flow)

                # Flush windows while the buffer has passed the stride boundary
                while buffer and w_start is not None:
                    w_end = w_start + self._window_seconds

                    # Check if we have enough flows reaching past w_end to close this window
                    # (or if the source has ended)
                    last_ts = buffer[-1].timestamp
                    if last_ts < w_end:
                        break  # Window not yet complete

                    # Extract flows within [w_start, w_end)
                    window_flows = [f for f in buffer if w_start <= f.timestamp < w_end]

                    # Emit window
                    await self._emit_window(window_flows, w_start, w_end)

                    # Advance window by stride
                    next_start = w_start + self._stride_seconds

                    # Remove flows that are no longer needed by future windows
                    while buffer and buffer[0].timestamp < next_start:
                        buffer.popleft()

                    w_start = next_start

        except asyncio.CancelledError:
            logger.info("[StreamProcessor] Cancelled")
        except Exception as exc:
            logger.error("[StreamProcessor] Fatal error in run(): %s", exc, exc_info=True)
        finally:
            # Flush remaining buffer as a partial window
            if buffer and w_start is not None and not self._closed:
                w_end = w_start + self._window_seconds
                window_flows = list(buffer)
                await self._emit_window(window_flows, w_start, w_end, partial=True)
            logger.info("[StreamProcessor] Stopped. Stats: %s", self._stats)

    async def _emit_window(
        self,
        flows: List[FlowRecord],
        w_start: float,
        w_end: float,
        partial: bool = False,
    ) -> None:
        if len(flows) < self._min_flows:
            self._stats["total_windows_skipped_empty"] += 1
            logger.debug("[StreamProcessor] Skipping empty window [%.1f, %.1f)", w_start, w_end)
            return

        event = TelemetryWindowEvent(
            window_id=str(uuid.uuid4()),
            window_start=w_start,
            window_end=w_end,
            flows=flows,
            flow_count=len(flows),
            dropped_malformed=self._stats["total_flows_dropped_malformed"],
            source_id=self._source.source_id,
        )

        if self._output.full():
            self._stats["total_output_dropped_backpressure"] += 1
            logger.warning(
                "[StreamProcessor] Output queue full — dropping window %s (backpressure)",
                event.window_id
            )
            return

        await self._output.put(event)
        self._stats["total_windows_emitted"] += 1
        logger.info(
            "[StreamProcessor] Window %s: %d flows, %.1f f/s%s",
            event.window_id[:8],
            event.flow_count,
            event.flows_per_second,
            " [PARTIAL]" if partial else "",
        )

    def _validate_flow(self, flow: FlowRecord) -> Optional[FlowRecord]:
        """
        Validates that a FlowRecord has finite, sane values.
        Returns the flow if valid, None if it should be dropped.
        """
        try:
            if not math.isfinite(flow.timestamp) or flow.timestamp < 0:
                return None
            if flow.packets < 0 or flow.bytes < 0:
                return None
            if not (0 <= flow.src_port <= 65535) or not (0 <= flow.dst_port <= 65535):
                return None
            if not flow.src_ip or not flow.dst_ip:
                return None
            return flow
        except Exception:
            return None

    async def windows(self) -> AsyncIterator[TelemetryWindowEvent]:
        """
        Async generator that yields completed TelemetryWindowEvents.
        Must be called concurrently with `run()`.
        """
        while not self._closed:
            try:
                event = await asyncio.wait_for(self._output.get(), timeout=1.0)
                yield event
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
