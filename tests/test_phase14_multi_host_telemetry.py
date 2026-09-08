"""
Phase 14 Test Suite: Real-World Multi-Host Telemetry.

Verifies:
  - Multi-host topology (host: 192.168.1.100, secondary laptop: 192.168.1.105)
  - Flow generation and NetFlow v5 encoding across all 5 controlled patterns
  - Zero attack labels passed to inference pipeline (label="UNKNOWN")
  - Ingestion through StreamProcessor and LiveIngestService
"""

import asyncio
import pytest

from scripts.multi_host_traffic_generator import (
    generate_pattern_flows,
    flows_to_netflow_v5_packet,
    SECONDARY_LAPTOP_IP,
    CYBERSENTINEL_HOST_IP,
)
from network.telemetry.sources import NetFlowSource
from network.telemetry.stream_processor import StreamProcessor, TelemetryWindowEvent
from backend.services.live_ingest_service import LiveIngestService
from backend.services.model_service import ModelService


class TestMultiHostTelemetryGeneration:
    """Verifies multi-host traffic generator and NetFlow packetization."""

    @pytest.mark.parametrize("pattern", [
        "normal_background",
        "connection_burst",
        "repeated_attempts",
        "port_diversity",
        "large_data_transfer",
    ])
    def test_pattern_flows_structure_and_ip_attribution(self, pattern):
        flows = generate_pattern_flows(pattern, base_timestamp=1000.0, duration_seconds=30.0)
        assert len(flows) > 0

        # All flows must originate from the authorized secondary device
        for f in flows:
            assert f.src_ip == SECONDARY_LAPTOP_IP
            assert f.label == "UNKNOWN"  # Strictly unlabelled for genuine inference
            assert 1000.0 <= f.timestamp < 1030.0
            assert f.packets > 0
            assert f.bytes > 0
            assert 0 <= f.src_port <= 65535
            assert 0 <= f.dst_port <= 65535

    def test_netflow_v5_packet_roundtrip(self):
        flows = generate_pattern_flows("connection_burst", base_timestamp=1000.0, duration_seconds=30.0)
        pkt = flows_to_netflow_v5_packet(flows[:20], seq=1)
        assert len(pkt) > 24

        # Parse through NetFlowSource decoder
        nf_source = NetFlowSource(port=9995)
        parsed = nf_source._parse_netflow_v5(pkt)
        assert len(parsed) == 20
        assert all(f.src_ip == SECONDARY_LAPTOP_IP for f in parsed)
        assert all(f.label is None or f.label == "UNKNOWN" for f in parsed)


class TestMultiHostTelemetryIngestion:
    """Verifies end-to-end multi-host ingestion through the live service."""

    @pytest.mark.asyncio
    async def test_stream_processor_ingests_multihost_flows(self):
        flows = generate_pattern_flows("port_diversity", base_timestamp=0.0, duration_seconds=60.0)
        from tests.test_phase13_stream_processor import MockSource

        src = MockSource(flows)
        proc = StreamProcessor(src, window_seconds=30.0)

        task = asyncio.create_task(proc.run())
        windows = []
        async for w in proc.windows():
            windows.append(w)
            if len(windows) >= 2:
                await proc.stop()
                break
        await task

        assert len(windows) >= 2
        assert all(w.flow_count > 0 for w in windows)
        for w in windows:
            for f in w.flows:
                assert f.src_ip == SECONDARY_LAPTOP_IP

    @pytest.mark.asyncio
    async def test_live_ingest_multihost_session(self):
        model_svc = ModelService.get_instance()
        live_svc = LiveIngestService()
        q = live_svc.subscribe()

        # Feed 5 multi-host windows
        for i in range(5):
            w_start = float(i * 30)
            flows = generate_pattern_flows("normal_background", base_timestamp=w_start, duration_seconds=30.0)
            win = TelemetryWindowEvent(
                window_id=f"mh_win_{i}",
                window_start=w_start,
                window_end=w_start + 30.0,
                flows=flows,
                flow_count=len(flows),
                dropped_malformed=0,
                source_id=f"Host({SECONDARY_LAPTOP_IP})",
            )
            await live_svc._process_window(win, "multihost_session", model_svc)

        events = []
        while not q.empty():
            events.append(q.get_nowait())
        live_svc.unsubscribe(q)

        forecast_events = [e for e in events if e.get("status") == "FORECAST"]
        assert len(forecast_events) >= 1
        last = forecast_events[-1]
        assert last["source_id"] == f"Host({SECONDARY_LAPTOP_IP})"
        assert last["current_stage"] in ["BENIGN", "RECONNAISSANCE", "CREDENTIAL_ACCESS", "LATERAL_MOVEMENT", "EXFILTRATION"]
        assert 0.0 <= last["attack_probability"] <= 1.0
        assert 0.0 <= last["risk_score"] <= 100.0
