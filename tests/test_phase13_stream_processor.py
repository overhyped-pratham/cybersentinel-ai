"""
Phase 13 Test Suite: Stream Processor.

Tests window construction, causal time-bounding, malformed flow filtering,
backpressure protection, bounded memory, and graceful shutdown.
"""

import asyncio
import math
import time
from typing import AsyncIterator, List
import pytest

from network.flow.flow_record import FlowRecord
from network.telemetry.sources import TelemetrySource
from network.telemetry.stream_processor import StreamProcessor, TelemetryWindowEvent


class MockSource(TelemetrySource):
    """Controlled in-memory TelemetrySource for testing."""

    def __init__(self, flows: List[FlowRecord], delay: float = 0.0):
        self._flows = flows
        self._delay = delay
        self._closed = False

    @property
    def source_id(self) -> str:
        return "MockSource"

    async def close(self) -> None:
        self._closed = True

    async def stream(self) -> AsyncIterator[FlowRecord]:
        for f in self._flows:
            if self._closed:
                break
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            yield f


def make_flow(ts: float, sport: int = 1024, dport: int = 80, pkts: int = 10, bytes_: int = 500) -> FlowRecord:
    return FlowRecord(
        timestamp=ts,
        src_ip="192.168.1.10",
        dst_ip="10.0.0.1",
        src_port=sport,
        dst_port=dport,
        protocol=6,
        packets=pkts,
        bytes=bytes_,
        duration=1.0,
    )


class TestStreamProcessorWindowing:
    """Verifies that windows are accurately constructed and temporally bounded."""

    @pytest.mark.asyncio
    async def test_window_stride_and_duration(self):
        # 100 seconds of flows, 1 flow every second: ts = 0, 1, 2, ..., 99
        flows = [make_flow(float(t)) for t in range(100)]
        src = MockSource(flows)
        proc = StreamProcessor(src, window_seconds=30.0, stride_seconds=30.0)

        task = asyncio.create_task(proc.run())
        windows: List[TelemetryWindowEvent] = []
        async for w in proc.windows():
            windows.append(w)
            if len(windows) >= 3:
                await proc.stop()
                break
        await task

        assert len(windows) >= 3
        # Check window bounds: first window should start at 0, end at 30
        assert windows[0].window_start == 0.0
        assert windows[0].window_end == 30.0
        assert windows[0].flow_count == 30

        # Second window should start at 30, end at 60
        assert windows[1].window_start == 30.0
        assert windows[1].window_end == 60.0
        assert windows[1].flow_count == 30

        # Third window should start at 60, end at 90
        assert windows[2].window_start == 60.0
        assert windows[2].window_end == 90.0
        assert windows[2].flow_count == 30

    @pytest.mark.asyncio
    async def test_causal_temporal_containment(self):
        """Strictly ensures flows in window are within [w_start, w_end)."""
        flows = [make_flow(float(t)) for t in range(60)]
        src = MockSource(flows)
        proc = StreamProcessor(src, window_seconds=30.0, stride_seconds=30.0)

        task = asyncio.create_task(proc.run())
        windows = []
        async for w in proc.windows():
            windows.append(w)
            if len(windows) >= 2:
                await proc.stop()
                break
        await task

        for w in windows:
            for f in w.flows:
                assert w.window_start <= f.timestamp < w.window_end, (
                    f"Flow ts={f.timestamp} outside window [{w.window_start}, {w.window_end})"
                )


class TestStreamProcessorRobustness:
    """Verifies malformed handling, backpressure, and bounds."""

    @pytest.mark.asyncio
    async def test_malformed_flow_filtering(self):
        # Create flows where some are malformed
        valid_flow_1 = make_flow(1.0)
        valid_flow_2 = make_flow(10.0)
        valid_flow_3 = make_flow(35.0)

        class MalformedFlow:
            # Missing or invalid fields
            timestamp = float("nan")
            src_ip = ""
            dst_ip = "10.0.0.1"
            src_port = 70000  # invalid port > 65535
            dst_port = 80
            protocol = 6
            packets = -5
            bytes = 100

        flows = [valid_flow_1, MalformedFlow(), valid_flow_2, valid_flow_3]
        src = MockSource(flows)
        proc = StreamProcessor(src, window_seconds=30.0, stride_seconds=30.0)

        task = asyncio.create_task(proc.run())
        windows = []
        async for w in proc.windows():
            windows.append(w)
            if len(windows) >= 1:
                await proc.stop()
                break
        await task

        # The malformed flow must have been dropped
        assert proc.stats["total_flows_dropped_malformed"] >= 1
        assert windows[0].flow_count == 2  # valid_flow_1 and valid_flow_2

    @pytest.mark.asyncio
    async def test_backpressure_drops_overflow_windows(self):
        """When output queue is full, excess windows are dropped instead of blocking/exploding."""
        # 300 flows across 300 seconds -> 10 windows
        flows = [make_flow(float(t)) for t in range(300)]
        src = MockSource(flows)
        # Set max_output_queue very small to test backpressure
        proc = StreamProcessor(src, window_seconds=30.0, stride_seconds=30.0, max_output_queue=2)

        # Run processor WITHOUT consuming from proc.windows() immediately
        task = asyncio.create_task(proc.run())
        await asyncio.sleep(0.2)
        await proc.stop()
        await task

        # Verify that output queue did not exceed maxsize and drops were logged
        assert proc.stats["total_output_dropped_backpressure"] > 0
        assert proc._output.qsize() <= 2

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        flows = [make_flow(float(t)) for t in range(1000)]
        src = MockSource(flows, delay=0.01)
        proc = StreamProcessor(src, window_seconds=30.0)

        task = asyncio.create_task(proc.run())
        await asyncio.sleep(0.05)
        # Signal stop
        await proc.stop()
        await task  # should return quickly without hanging
        assert proc._closed is True
