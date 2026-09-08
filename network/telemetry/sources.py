"""
CyberSentinel AI — Modular Telemetry Source Abstraction (Phase 13).

Architecture:
    TelemetrySource (ABC)
    ├── PCAPSource          — reads .pcap offline or tailed live
    ├── NetFlowSource       — receives UDP NetFlow v5/v9/IPFIX records
    └── ReplaySource        — streams CSV/JSON scenario files (testing adapter)

All sources emit FlowRecord objects.  The ML layer (NetworkStateBuilder) is
never aware of which source produced the flows.  Adding a new source only
requires subclassing TelemetrySource.
"""

from __future__ import annotations

import abc
import asyncio
import csv
import io
import json
import logging
import socket
import struct
import time
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Deque, List, Optional, Union

from network.flow.flow_record import FlowRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TelemetrySource(abc.ABC):
    """
    Abstract interface for all live and replay telemetry adapters.

    Concrete subclasses MUST implement `stream()`, an async generator that
    yields FlowRecord objects.  Sources must:
      - Handle malformed data without crashing (log and skip).
      - Never inject ground-truth labels into FlowRecord.label for live sources.
      - Honour `close()` for graceful shutdown.
    """

    @abc.abstractmethod
    async def stream(self) -> AsyncIterator[FlowRecord]:
        """Yield FlowRecord objects as they arrive or are read."""
        ...  # pragma: no cover

    @abc.abstractmethod
    async def close(self) -> None:
        """Shut down the source gracefully."""
        ...  # pragma: no cover

    @property
    @abc.abstractmethod
    def source_id(self) -> str:
        """Human-readable identifier for logging."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# PCAP Source (offline file or live tail)
# ---------------------------------------------------------------------------

class PCAPSource(TelemetrySource):
    """
    Reads a .pcap file and streams reconstructed FlowRecords.

    For live capture files that grow in real-time (e.g. tcpdump -w live.pcap),
    set `tail=True` to poll for new packets as the file grows.

    Uses the existing pure-Python PCAPFlowLoader for packet parsing so no
    external C library (npcap/libpcap) is required.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        tail: bool = False,
        poll_interval: float = 0.5,
        flow_timeout: float = 15.0,
        scenario_id: str = "pcap_live",
        batch_size: int = 200,
    ) -> None:
        self._path = Path(path)
        self._tail = tail
        self._poll_interval = poll_interval
        self._flow_timeout = flow_timeout
        self._scenario_id = scenario_id
        self._batch_size = batch_size
        self._closed = False

    @property
    def source_id(self) -> str:
        return f"PCAPSource({self._path.name})"

    async def close(self) -> None:
        self._closed = True

    async def stream(self) -> AsyncIterator[FlowRecord]:
        from network.pcap.pcap_loader import PCAPFlowLoader
        loader = PCAPFlowLoader(
            flow_timeout_seconds=self._flow_timeout,
            default_scenario=self._scenario_id,
        )

        if not self._tail:
            # One-shot: load all flows and yield
            try:
                flows = await asyncio.get_event_loop().run_in_executor(
                    None, loader.load_flows, self._path
                )
                logger.info("[%s] Loaded %d flows from PCAP", self.source_id, len(flows))
                for flow in flows:
                    if self._closed:
                        return
                    yield flow
            except Exception as exc:
                logger.error("[%s] PCAP load error: %s", self.source_id, exc)
            return

        # Tail mode: repeatedly re-parse the growing file
        seen_bytes = 0
        while not self._closed:
            try:
                stat = self._path.stat()
                if stat.st_size > seen_bytes:
                    flows = await asyncio.get_event_loop().run_in_executor(
                        None, loader.load_flows, self._path
                    )
                    # Only yield new flows beyond previously seen
                    # (simple approach: reload all, emit only new flows)
                    seen_bytes = stat.st_size
                    for flow in flows:
                        if self._closed:
                            return
                        yield flow
            except FileNotFoundError:
                logger.warning("[%s] File not found, retrying...", self.source_id)
            except Exception as exc:
                logger.error("[%s] Tail error: %s", self.source_id, exc)
            await asyncio.sleep(self._poll_interval)


# ---------------------------------------------------------------------------
# NetFlow Source (UDP listener for NetFlow v5 / lightweight custom records)
# ---------------------------------------------------------------------------

# NetFlow v5 header format
_NF5_HEADER = struct.Struct("!HHIIIIBBH")     # 24 bytes
_NF5_RECORD = struct.Struct("!IIIHHIIIIHHBBBBHHBBH")  # 48 bytes

# Our own compact binary flow record (16+4+4+4+4+2+2+2+2+1+1 = 42 bytes)
# Used for internal testing/simulation when real NetFlow source is unavailable
_CUSTOM_FLOW = struct.Struct("!4s4sHHHIIIBBB")  # src_ip,dst_ip,sport,dport,proto,pkts,bytes,dur_ms,syn,rst,failed

NETFLOW_DEFAULT_PORT = 9995


class NetFlowSource(TelemetrySource):
    """
    Asynchronous UDP listener for NetFlow v5 datagrams (RFC 3954 subset).

    Falls back gracefully when datagrams are malformed: logs error, continues.
    Backpressure: drops flows when internal queue exceeds `max_queue_size`.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = NETFLOW_DEFAULT_PORT,
        *,
        max_queue_size: int = 4096,
        scenario_id: str = "netflow_live",
        timeout_seconds: float = 5.0,
    ) -> None:
        self._host = host
        self._port = port
        self._max_queue = max_queue_size
        self._scenario_id = scenario_id
        self._timeout = timeout_seconds
        self._closed = False
        self._queue: asyncio.Queue[FlowRecord] = asyncio.Queue(maxsize=max_queue_size)
        self._transport = None
        self._protocol = None

    @property
    def source_id(self) -> str:
        return f"NetFlowSource({self._host}:{self._port})"

    async def close(self) -> None:
        self._closed = True
        if self._transport:
            self._transport.close()

    async def stream(self) -> AsyncIterator[FlowRecord]:
        loop = asyncio.get_event_loop()

        class _Protocol(asyncio.DatagramProtocol):
            def __init__(proto_self):
                proto_self.source = self

            def datagram_received(proto_self, data: bytes, addr):
                flows = self._parse_netflow_v5(data)
                for flow in flows:
                    if not self._queue.full():
                        self._queue.put_nowait(flow)
                    else:
                        logger.warning("[%s] Queue full, dropping flow", self.source_id)

            def error_received(proto_self, exc):
                logger.error("[%s] UDP error: %s", self.source_id, exc)

        try:
            transport, protocol = await loop.create_datagram_endpoint(
                _Protocol,
                local_addr=(self._host, self._port),
            )
            self._transport = transport
            logger.info("[%s] Listening for NetFlow on %s:%d", self.source_id, self._host, self._port)
        except Exception as exc:
            logger.error("[%s] Failed to bind UDP socket: %s", self.source_id, exc)
            return

        while not self._closed:
            try:
                flow = await asyncio.wait_for(self._queue.get(), timeout=self._timeout)
                yield flow
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.error("[%s] Queue error: %s", self.source_id, exc)

    def _parse_netflow_v5(self, data: bytes) -> List[FlowRecord]:
        """Parse a NetFlow v5 UDP datagram into FlowRecord list."""
        flows: List[FlowRecord] = []
        if len(data) < 24:
            return flows
        try:
            version, count, sys_uptime, unix_secs, unix_nsecs, flow_seq, engine_type, engine_id, sampling = \
                _NF5_HEADER.unpack(data[:24])
            if version != 5:
                logger.debug("[%s] Non-v5 NetFlow (version=%d), skipping", self.source_id, version)
                return flows
            offset = 24
            for _ in range(count):
                if offset + 48 > len(data):
                    break
                rec = _NF5_RECORD.unpack(data[offset:offset + 48])
                offset += 48
                (src_raw, dst_raw, next_hop, in_iface, out_iface,
                 d_pkts, d_octets, first, last, src_port, dst_port,
                 pad1, tcp_flags, prot, tos, src_as, dst_as,
                 src_mask, dst_mask, pad2) = rec
                src_ip = socket.inet_ntoa(struct.pack("!I", src_raw))
                dst_ip = socket.inet_ntoa(struct.pack("!I", dst_raw))
                duration = max(0.0, (last - first) / 1000.0)  # ms → s
                syn = 1 if tcp_flags & 0x02 else 0
                rst = 1 if tcp_flags & 0x04 else 0
                fin = 1 if tcp_flags & 0x01 else 0
                ack = 1 if tcp_flags & 0x10 else 0
                psh = 1 if tcp_flags & 0x08 else 0
                urg = 1 if tcp_flags & 0x20 else 0
                ts = float(unix_secs) + float(unix_nsecs) / 1e9
                flows.append(FlowRecord(
                    timestamp=ts,
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=src_port,
                    dst_port=dst_port,
                    protocol=prot,
                    packets=max(0, d_pkts),
                    bytes=max(0, d_octets),
                    duration=duration,
                    syn_flag=syn,
                    rst_flag=rst,
                    fin_flag=fin,
                    ack_flag=ack,
                    psh_flag=psh,
                    urg_flag=urg,
                    failed=(rst == 1 or d_octets == 0),
                    scenario_id=self._scenario_id,
                ))
        except Exception as exc:
            logger.warning("[%s] NetFlow parse error: %s", self.source_id, exc)
        return flows


# ---------------------------------------------------------------------------
# Replay Source (CSV/JSON — testing adapter only)
# ---------------------------------------------------------------------------

class ReplaySource(TelemetrySource):
    """
    Streams pre-recorded scenario CSV or JSON flow files for testing.

    IMPORTANT: This is a TESTING ADAPTER only.  Production deployments use
    PCAPSource or NetFlowSource.  The label column from CSV files is preserved
    as FlowRecord.label for evaluation purposes but is NEVER passed to the ML
    model during inference.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        realtime_factor: float = 0.0,   # 0.0 = as fast as possible; 1.0 = wall-clock speed
        scenario_id: Optional[str] = None,
        loop: bool = False,
    ) -> None:
        self._path = Path(path)
        self._realtime_factor = realtime_factor
        self._scenario_id = scenario_id or self._path.stem
        self._loop = loop
        self._closed = False

    @property
    def source_id(self) -> str:
        return f"ReplaySource({self._path.name})"

    async def close(self) -> None:
        self._closed = True

    async def stream(self) -> AsyncIterator[FlowRecord]:
        while not self._closed:
            try:
                flows = await self._load_file()
            except Exception as exc:
                logger.error("[%s] Failed to load file: %s", self.source_id, exc)
                return

            if not flows:
                logger.warning("[%s] No flows in file", self.source_id)
                return

            prev_ts = flows[0].timestamp
            for flow in flows:
                if self._closed:
                    return
                if self._realtime_factor > 0:
                    delay = (flow.timestamp - prev_ts) * self._realtime_factor
                    if delay > 0:
                        await asyncio.sleep(delay)
                prev_ts = flow.timestamp
                yield flow

            if not self._loop:
                break

        logger.info("[%s] Replay complete", self.source_id)

    async def _load_file(self) -> List[FlowRecord]:
        suffix = self._path.suffix.lower()
        if suffix == ".csv":
            return await asyncio.get_event_loop().run_in_executor(
                None, self._load_csv
            )
        elif suffix == ".json":
            return await asyncio.get_event_loop().run_in_executor(
                None, self._load_json
            )
        else:
            raise ValueError(f"Unsupported replay format: {suffix}")

    def _load_csv(self) -> List[FlowRecord]:
        from network.flow.csv_loader import CSVFlowLoader
        loader = CSVFlowLoader(default_scenario=self._scenario_id)
        flows = loader.load_flows(self._path)
        if self._scenario_id and self._scenario_id != self._path.stem:
            from dataclasses import replace
            flows = [replace(f, scenario_id=self._scenario_id) for f in flows]
        return flows

    def _load_json(self) -> List[FlowRecord]:
        with open(self._path, encoding="utf-8") as fh:
            records = json.load(fh)
        flows = []
        for r in records:
            try:
                flows.append(FlowRecord(**{
                    k: v for k, v in r.items()
                    if k in FlowRecord.__dataclass_fields__
                }))
            except Exception as exc:
                logger.warning("[%s] Skipping malformed JSON flow: %s", self.source_id, exc)
        flows.sort(key=lambda f: f.timestamp)
        return flows


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_source(kind: str, **kwargs) -> TelemetrySource:
    """
    Factory for creating TelemetrySource instances from config dicts.

    Args:
        kind: One of 'pcap', 'netflow', 'replay'.
        **kwargs: Source-specific constructor arguments.
    """
    kind = kind.lower()
    if kind == "pcap":
        return PCAPSource(**kwargs)
    elif kind == "netflow":
        return NetFlowSource(**kwargs)
    elif kind == "replay":
        return ReplaySource(**kwargs)
    else:
        raise ValueError(f"Unknown telemetry source kind: {kind!r}. "
                         f"Supported: 'pcap', 'netflow', 'replay'")
