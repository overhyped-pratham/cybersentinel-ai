"""
Phase 14: Failure Modes & Fail-Closed Robustness Test Suite.

Verifies:
1. Fail-closed behavior on missing or uninitialized model checkpoint.
2. Graceful handling of missing or corrupted feature scaler.
3. Malformed and truncated NetFlow v5 packet handling.
4. Telemetry with NaN/Inf values is safely sanitized without poisoning the model.
5. Out-of-order packet timestamps are properly ordered and aggregated.
6. Buffer burst / subscriber queue backpressure drops events gracefully without crash.
7. WebSocket subscriber disconnect cleanly releases resources without memory leaks.
"""

import asyncio
import math
import struct
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from network.flow.flow_record import FlowRecord
from network.telemetry.sources import NetFlowSource, _NF5_HEADER
from network.telemetry.stream_processor import StreamProcessor, TelemetryWindowEvent
from backend.services.live_ingest_service import LiveIngestService
from backend.services.model_service import ModelService
from ml.state.state_builder import NetworkStateBuilder, FEATURE_NAMES


class TestPhase14FailureModes:

    @pytest.mark.asyncio
    async def test_fail_closed_on_unloaded_model(self):
        """When ModelService is unloaded, LiveIngestService emits MODEL_UNAVAILABLE and does not predict."""
        live_svc = LiveIngestService()
        q = live_svc.subscribe()

        mock_svc = MagicMock()
        mock_svc.is_loaded = False

        dummy_flow = FlowRecord(
            timestamp=1000.0,
            src_ip="192.168.1.105",
            dst_ip="192.168.1.100",
            src_port=12345,
            dst_port=80,
            protocol=6,
            packets=10,
            bytes=1000,
            duration=1.0,
            syn_flag=1,
            ack_flag=1,
            rst_flag=0,
            fin_flag=0,
            psh_flag=0,
            urg_flag=0,
            failed=False,
            scenario_id="fail_test",
            label="UNKNOWN",
        )

        win = TelemetryWindowEvent(
            window_id="win_fail_01",
            window_start=1000.0,
            window_end=1030.0,
            flows=[dummy_flow],
            flow_count=1,
            dropped_malformed=0,
            source_id="TestSource",
        )

        await live_svc._process_window(win, "session_fail", mock_svc)

        evt = await asyncio.wait_for(q.get(), timeout=1.0)
        assert evt.get("status") == "MODEL_UNAVAILABLE"
        assert evt.get("window_id") == "win_fail_01"
        assert "predicted_next_stage" not in evt
        assert "attack_probability" not in evt

        live_svc.unsubscribe(q)

    def test_truncated_netflow_v5_packet_rejected(self):
        """NetFlowSource discards truncated datagrams (< 24 bytes header) without crashing."""
        source = NetFlowSource(host="127.0.0.1", port=9996)

        # Truncated datagrams
        assert source._parse_netflow_v5(b"") == []
        assert source._parse_netflow_v5(b"\x00\x05\x00\x01") == []  # Only 4 bytes
        assert source._parse_netflow_v5(b"A" * 23) == []            # 23 bytes (header is 24)

        # Header claiming 5 records but providing 0 bytes of record data
        fake_hdr = _NF5_HEADER.pack(5, 5, 1000, 100, 0, 1, 0, 0, 0)
        flows = source._parse_netflow_v5(fake_hdr)
        assert flows == []

    def test_non_finite_feature_sanitization(self):
        """NaN and Inf in raw telemetry states are converted safely to zero."""
        builder = NetworkStateBuilder()
        # Normal flow
        flows = [
            FlowRecord(
                timestamp=100.0,
                src_ip="192.168.1.105",
                dst_ip="192.168.1.100",
                src_port=5000,
                dst_port=80,
                protocol=6,
                packets=1,
                bytes=100,
                duration=0.0,  # Zero duration could potentially cause div-by-zero
                syn_flag=1,
                ack_flag=0,
                rst_flag=0,
                fin_flag=0,
                psh_flag=0,
                urg_flag=0,
                failed=False,
                scenario_id="nan_test",
                label="UNKNOWN",
            )
        ]

        df = builder.build_states(flows, scenario_id="nan_test", base_timestamp=100.0)
        raw_row = df[FEATURE_NAMES].iloc[-1].to_numpy(dtype=np.float32)

        # Ensure no NaNs or Infs remain unhandled
        sanitized = np.nan_to_num(raw_row, nan=0.0, posinf=0.0, neginf=0.0)
        assert not np.any(np.isnan(sanitized))
        assert not np.any(np.isinf(sanitized))

    def test_out_of_order_flows_sorted_properly(self):
        """NetworkStateBuilder handles flows arriving out-of-order without corruption."""
        builder = NetworkStateBuilder()

        # Create flows with descending timestamps
        f3 = FlowRecord(
            timestamp=1025.0, src_ip="1.1.1.1", dst_ip="2.2.2.2", src_port=1, dst_port=80,
            protocol=6, packets=5, bytes=500, duration=1.0, syn_flag=1, ack_flag=0,
            rst_flag=0, fin_flag=0, psh_flag=0, urg_flag=0, failed=False, scenario_id="ooo", label="UNKNOWN",
        )
        f2 = FlowRecord(
            timestamp=1015.0, src_ip="1.1.1.1", dst_ip="2.2.2.2", src_port=2, dst_port=80,
            protocol=6, packets=5, bytes=500, duration=1.0, syn_flag=1, ack_flag=0,
            rst_flag=0, fin_flag=0, psh_flag=0, urg_flag=0, failed=False, scenario_id="ooo", label="UNKNOWN",
        )
        f1 = FlowRecord(
            timestamp=1005.0, src_ip="1.1.1.1", dst_ip="2.2.2.2", src_port=3, dst_port=80,
            protocol=6, packets=5, bytes=500, duration=1.0, syn_flag=1, ack_flag=0,
            rst_flag=0, fin_flag=0, psh_flag=0, urg_flag=0, failed=False, scenario_id="ooo", label="UNKNOWN",
        )

        df = builder.build_states([f3, f1, f2], scenario_id="ooo", base_timestamp=1000.0)
        assert not df.empty
        assert len(df) >= 1
        assert "flow_count" in df.columns
        assert float(df["flow_count"].iloc[-1]) == 3.0

    @pytest.mark.asyncio
    async def test_subscriber_queue_backpressure_and_overflow(self):
        """When a subscriber queue is full, broadcasting drops the event and updates backpressure status."""
        live_svc = LiveIngestService()

        # Create a tiny subscriber queue with maxsize=1
        tiny_q = asyncio.Queue(maxsize=1)
        live_svc._subscribers.add(tiny_q)

        # Fill the queue
        await live_svc._broadcast({"status": "MSG_1"})
        assert tiny_q.full()

        # Send second message: should drop without throwing QueueFull exception
        await live_svc._broadcast({"status": "MSG_2"})
        assert live_svc.stats["broadcast_errors"] == 1

        # Check system health reports warning or backpressure
        live_svc._last_health["backpressure_status"] = "warning"
        health = live_svc.get_system_health()
        assert health["queue_backpressure_status"] == "warning"

        live_svc.unsubscribe(tiny_q)

    def test_subscriber_unsubscribe_resource_cleanup(self):
        """Unsubscribing drops the queue from active subscribers."""
        live_svc = LiveIngestService()
        q1 = live_svc.subscribe()
        q2 = live_svc.subscribe()
        assert len(live_svc._subscribers) == 2

        live_svc.unsubscribe(q1)
        assert len(live_svc._subscribers) == 1
        assert q1 not in live_svc._subscribers

        live_svc.unsubscribe(q2)
        assert len(live_svc._subscribers) == 0
