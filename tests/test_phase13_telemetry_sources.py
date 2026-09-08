"""
Phase 13 Test Suite: Telemetry Sources.

Tests TelemetrySource ABC, PCAPSource, NetFlowSource, ReplaySource,
and factory create_source().
"""

import asyncio
import io
import os
import struct
import tempfile
from pathlib import Path
import pytest

from network.flow.flow_record import FlowRecord
from network.telemetry.sources import (
    TelemetrySource,
    PCAPSource,
    NetFlowSource,
    ReplaySource,
    create_source,
    _NF5_HEADER,
    _NF5_RECORD,
)

_CSV_SAMPLE = Path(__file__).resolve().parent.parent / "datasets" / "sample" / "trace_multistage_01.csv"


class TestTelemetrySourceInterface:
    """Verifies that TelemetrySource is a strict abstract base class."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            TelemetrySource()

    def test_factory_invalid_kind(self):
        with pytest.raises(ValueError, match="Unknown telemetry source kind"):
            create_source("unsupported_source_xyz")

    def test_factory_create_replay(self):
        source = create_source("replay", path=str(_CSV_SAMPLE))
        assert isinstance(source, ReplaySource)
        assert "trace_multistage_01" in source.source_id


class TestReplaySource:
    """Tests ReplaySource functionality on real CSV telemetry."""

    @pytest.mark.asyncio
    async def test_replay_reads_flows_chronologically(self):
        source = ReplaySource(_CSV_SAMPLE, realtime_factor=0.0)
        flows = []
        async for f in source.stream():
            flows.append(f)
            if len(flows) >= 50:
                await source.close()
                break

        assert len(flows) == 50
        assert all(isinstance(f, FlowRecord) for f in flows)
        # Check timestamps are monotonically non-decreasing
        for i in range(1, len(flows)):
            assert flows[i].timestamp >= flows[i - 1].timestamp

    @pytest.mark.asyncio
    async def test_replay_close_graceful(self):
        source = ReplaySource(_CSV_SAMPLE, realtime_factor=0.0)
        gen = source.stream()
        first_flow = await anext(gen)
        assert isinstance(first_flow, FlowRecord)
        await source.close()
        # Next read should raise StopAsyncIteration or exit
        remaining = []
        async for f in gen:
            remaining.append(f)
        assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_replay_custom_scenario_id(self):
        source = ReplaySource(_CSV_SAMPLE, scenario_id="custom_scenario_13")
        async for f in source.stream():
            assert f.scenario_id == "custom_scenario_13"
            await source.close()
            break


class TestNetFlowSource:
    """Tests NetFlow v5 parser and UDP stream receiver."""

    def _build_mock_netflow_v5_packet(self, count=2):
        """Constructs a valid NetFlow v5 packet header and records."""
        hdr = _NF5_HEADER.pack(
            5,          # version
            count,      # count
            10000,      # sys_uptime
            1700000000, # unix_secs
            0,          # unix_nsecs
            1,          # flow_seq
            0,          # engine_type
            0,          # engine_id
            0,          # sampling
        )
        records = []
        for i in range(count):
            rec = _NF5_RECORD.pack(
                0x0A000001 + i,  # src: 10.0.0.1+i
                0x0A000002,      # dst: 10.0.0.2
                0,               # next_hop
                1, 2,            # in/out iface
                10,              # d_pkts
                1000,            # d_octets
                1000, 2000,      # first, last
                1024 + i, 80,    # src_port, dst_port
                0,               # pad1
                0x02,            # tcp_flags (SYN)
                6,               # prot (TCP)
                0,               # tos
                0, 0,            # src/dst AS
                24, 24,          # masks
                0,               # pad2
            )
            records.append(rec)
        return hdr + b"".join(records)

    def test_parse_valid_netflow_v5(self):
        source = NetFlowSource(port=9996)
        pkt = self._build_mock_netflow_v5_packet(count=2)
        flows = source._parse_netflow_v5(pkt)
        assert len(flows) == 2
        f0 = flows[0]
        assert f0.src_ip == "10.0.0.1"
        assert f0.dst_ip == "10.0.0.2"
        assert f0.dst_port == 80
        assert f0.protocol == 6
        assert f0.packets == 10
        assert f0.bytes == 1000
        assert f0.syn_flag == 1
        assert f0.timestamp == 1700000000.0

    def test_parse_malformed_truncated_datagram(self):
        source = NetFlowSource(port=9996)
        # Truncated header (< 24 bytes)
        flows = source._parse_netflow_v5(b"too_short")
        assert flows == []

        # Header claims 5 records but packet is cut off
        pkt = self._build_mock_netflow_v5_packet(count=1)[:30]
        flows = source._parse_netflow_v5(pkt)
        assert flows == []

    def test_parse_unsupported_version(self):
        source = NetFlowSource(port=9996)
        hdr = _NF5_HEADER.pack(9, 1, 0, 0, 0, 0, 0, 0, 0) # version 9
        flows = source._parse_netflow_v5(hdr + b"\x00" * 48)
        assert flows == []
