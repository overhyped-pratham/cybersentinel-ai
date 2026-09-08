"""
CyberSentinel AI — Multi-Host Real Network Traffic Generator (Phase 14).

Simulates or transmits real network telemetry generated from an authorized
secondary laptop/device on the local network (e.g., 192.168.1.105).

Features:
  - 5 Controlled Traffic Patterns (Normal, Burst, Repeated, Port Diversity, Exfil)
  - Multi-host 5-tuples (Host: 192.168.1.100, Second Laptop: 192.168.1.105, Servers)
  - Exports FlowRecords, binary NetFlow v5 UDP datagrams, PCAP, or CSV traces
  - Strict Defense: NO attack labels passed to inference pipeline (label="UNKNOWN")
"""

import argparse
import math
import os
import random
import socket
import struct
import sys
import time
from pathlib import Path
from typing import List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from network.flow.flow_record import FlowRecord
from network.telemetry.sources import _NF5_HEADER, _NF5_RECORD

# Multi-host LAN topology
CYBERSENTINEL_HOST_IP = "192.168.1.100"
SECONDARY_LAPTOP_IP = "192.168.1.105"
GATEWAY_IP = "192.168.1.1"
INTERNAL_FILE_SERVER_IP = "10.0.0.10"
INTERNAL_DB_SERVER_IP = "10.0.0.20"

TRAFFIC_PATTERNS = [
    "normal_background",
    "connection_burst",
    "repeated_attempts",
    "port_diversity",
    "large_data_transfer",
]


def generate_pattern_flows(
    pattern: str,
    base_timestamp: float,
    duration_seconds: float = 30.0,
    source_ip: str = SECONDARY_LAPTOP_IP,
    target_ip: str = CYBERSENTINEL_HOST_IP,
    scenario_id: str = "multihost_live",
) -> List[FlowRecord]:
    """
    Generates a realistic stream of FlowRecords representing a specific
    controlled network traffic pattern originating from the secondary laptop.

    Patterns:
      - 'normal_background': Regular web browsing, DNS queries, NTP keepalives.
      - 'connection_burst': High-frequency TCP SYN connection bursts.
      - 'repeated_attempts': Repeated connection attempts with abnormal RST teardowns on admin ports (22, 3389).
      - 'port_diversity': Traffic distributed across 50+ unique ports (scanning pattern).
      - 'large_data_transfer': Sustained bulk data transfer with high bytes-per-packet (exfil pattern).
    """
    pattern = pattern.lower()
    flows: List[FlowRecord] = []
    rng = random.Random(42 + int(base_timestamp) % 1000)

    if pattern == "normal_background":
        # 25-40 typical browsing & service flows
        n_flows = rng.randint(25, 40)
        common_ports = [80, 443, 53, 8080]
        for i in range(n_flows):
            t_offset = rng.uniform(0.1, duration_seconds - 0.5)
            sport = rng.randint(49152, 65535)
            dport = rng.choice(common_ports)
            proto = 17 if dport == 53 else 6
            pkts = rng.randint(6, 20)
            bytes_ = pkts * rng.randint(64, 500)
            flows.append(FlowRecord(
                timestamp=base_timestamp + t_offset,
                src_ip=source_ip,
                dst_ip=target_ip if dport != 53 else GATEWAY_IP,
                src_port=sport,
                dst_port=dport,
                protocol=proto,
                packets=pkts,
                bytes=bytes_,
                duration=rng.uniform(0.05, 2.5),
                syn_flag=1 if proto == 6 else 0,
                ack_flag=1 if proto == 6 else 0,
                rst_flag=0,
                fin_flag=1 if proto == 6 and rng.random() > 0.3 else 0,
                psh_flag=1 if rng.random() > 0.5 else 0,
                urg_flag=0,
                failed=False,
                scenario_id=scenario_id,
                label="UNKNOWN",
            ))

    elif pattern == "connection_burst":
        # 150-250 rapid connections within window
        n_flows = rng.randint(180, 240)
        for i in range(n_flows):
            t_offset = rng.uniform(0.0, duration_seconds - 0.1)
            sport = rng.randint(1024, 65535)
            dport = rng.choice([80, 443, 8443])
            pkts = rng.randint(2, 6)
            bytes_ = pkts * 60
            flows.append(FlowRecord(
                timestamp=base_timestamp + t_offset,
                src_ip=source_ip,
                dst_ip=target_ip,
                src_port=sport,
                dst_port=dport,
                protocol=6,
                packets=pkts,
                bytes=bytes_,
                duration=rng.uniform(0.01, 0.2),
                syn_flag=1,
                ack_flag=0,
                rst_flag=0,
                fin_flag=0,
                psh_flag=0,
                urg_flag=0,
                failed=False,
                scenario_id=scenario_id,
                label="UNKNOWN",
            ))

    elif pattern == "repeated_attempts":
        # 80-120 repeated connection attempts to auth services with RST/failed flags
        n_flows = rng.randint(80, 120)
        auth_ports = [22, 3389, 445]
        for i in range(n_flows):
            t_offset = rng.uniform(0.0, duration_seconds - 0.1)
            sport = rng.randint(1024, 65535)
            dport = rng.choice(auth_ports)
            pkts = rng.randint(3, 8)
            bytes_ = pkts * 80
            flows.append(FlowRecord(
                timestamp=base_timestamp + t_offset,
                src_ip=source_ip,
                dst_ip=target_ip,
                src_port=sport,
                dst_port=dport,
                protocol=6,
                packets=pkts,
                bytes=bytes_,
                duration=rng.uniform(0.05, 0.8),
                syn_flag=1,
                ack_flag=1,
                rst_flag=1,  # abnormal teardown
                fin_flag=0,
                psh_flag=1,
                urg_flag=0,
                failed=True,  # authentication rejection
                scenario_id=scenario_id,
                label="UNKNOWN",
            ))

    elif pattern == "port_diversity":
        # 120-180 flows targeting 80+ distinct destination ports across servers
        n_flows = rng.randint(120, 180)
        for i in range(n_flows):
            t_offset = rng.uniform(0.0, duration_seconds - 0.1)
            sport = rng.randint(30000, 60000)
            dport = rng.randint(1, 1024)
            flows.append(FlowRecord(
                timestamp=base_timestamp + t_offset,
                src_ip=source_ip,
                dst_ip=target_ip if i % 2 == 0 else INTERNAL_FILE_SERVER_IP,
                src_port=sport,
                dst_port=dport,
                protocol=6,
                packets=rng.randint(1, 3),
                bytes=rng.randint(40, 120),
                duration=rng.uniform(0.001, 0.05),
                syn_flag=1,
                ack_flag=0,
                rst_flag=1 if rng.random() > 0.5 else 0,
                fin_flag=0,
                psh_flag=0,
                urg_flag=0,
                failed=(rng.random() > 0.4),
                scenario_id=scenario_id,
                label="UNKNOWN",
            ))

    elif pattern == "large_data_transfer":
        # 10-20 massive data transfers with high byte-to-packet ratio
        n_flows = rng.randint(15, 25)
        for i in range(n_flows):
            t_offset = rng.uniform(0.0, duration_seconds - 1.0)
            sport = rng.randint(40000, 60000)
            dport = 443 if i % 2 == 0 else 8080
            pkts = rng.randint(200, 1200)
            bytes_ = pkts * 1460  # MTU sized payloads
            flows.append(FlowRecord(
                timestamp=base_timestamp + t_offset,
                src_ip=source_ip,
                dst_ip=INTERNAL_DB_SERVER_IP if i % 3 == 0 else target_ip,
                src_port=sport,
                dst_port=dport,
                protocol=6,
                packets=pkts,
                bytes=bytes_,
                duration=rng.uniform(5.0, 25.0),
                syn_flag=1,
                ack_flag=1,
                rst_flag=0,
                fin_flag=1,
                psh_flag=1,
                urg_flag=0,
                failed=False,
                scenario_id=scenario_id,
                label="UNKNOWN",
            ))
    else:
        raise ValueError(f"Unknown traffic pattern: {pattern}")

    flows.sort(key=lambda f: f.timestamp)
    return flows


def flows_to_netflow_v5_packet(flows: List[FlowRecord], seq: int = 1) -> bytes:
    """Encodes a batch of up to 30 FlowRecords into an RFC 3954 NetFlow v5 UDP packet."""
    batch = flows[:30]
    count = len(batch)
    if count == 0:
        return b""

    first_ts = batch[0].timestamp
    unix_secs = int(first_ts)
    unix_nsecs = int((first_ts - unix_secs) * 1e9)

    hdr = _NF5_HEADER.pack(
        5,              # Version 5
        count,          # Flow count
        int(first_ts * 1000) & 0xFFFFFFFF,  # sys_uptime (ms)
        unix_secs,
        unix_nsecs,
        seq,            # flow_sequence
        0,              # engine_type
        0,              # engine_id
        0,              # sampling_interval
    )

    records = []
    for f in batch:
        src_bytes = socket.inet_aton(f.src_ip)
        dst_bytes = socket.inet_aton(f.dst_ip)
        src_raw = struct.unpack("!I", src_bytes)[0]
        dst_raw = struct.unpack("!I", dst_bytes)[0]

        tcp_flags = 0
        if f.fin_flag: tcp_flags |= 0x01
        if f.syn_flag: tcp_flags |= 0x02
        if f.rst_flag: tcp_flags |= 0x04
        if f.psh_flag: tcp_flags |= 0x08
        if f.ack_flag: tcp_flags |= 0x10
        if f.urg_flag: tcp_flags |= 0x20

        first_ms = int(f.timestamp * 1000) & 0xFFFFFFFF
        last_ms = int((f.timestamp + f.duration) * 1000) & 0xFFFFFFFF

        rec = _NF5_RECORD.pack(
            src_raw,
            dst_raw,
            0,          # next_hop
            1,          # input_iface
            2,          # output_iface
            f.packets,
            f.bytes,
            first_ms,
            last_ms,
            f.src_port,
            f.dst_port,
            0,          # pad1
            tcp_flags,
            f.protocol,
            0,          # tos
            0,          # src_as
            0,          # dst_as
            24,         # src_mask
            24,         # dst_mask
            0,          # pad2
        )
        records.append(rec)

    return hdr + b"".join(records)


def transmit_live_netflow(
    target_host: str = "127.0.0.1",
    target_port: int = 9995,
    pattern: str = "normal_background",
    num_windows: int = 5,
    window_seconds: float = 30.0,
) -> int:
    """Transmits real NetFlow v5 UDP packets over the socket to CyberSentinel."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    total_packets_sent = 0
    seq = 1

    print(f"[*] Transmitting pattern '{pattern}' to {target_host}:{target_port} ({num_windows} windows)...")
    base_ts = time.time()

    for w in range(num_windows):
        w_start = base_ts + (w * window_seconds)
        flows = generate_pattern_flows(pattern, base_timestamp=w_start, duration_seconds=window_seconds)

        # Chunk into NetFlow packets (max 30 flows per packet)
        for chunk_idx in range(0, len(flows), 30):
            chunk = flows[chunk_idx:chunk_idx + 30]
            pkt = flows_to_netflow_v5_packet(chunk, seq=seq)
            sock.sendto(pkt, (target_host, target_port))
            total_packets_sent += 1
            seq += len(chunk)

        print(f"  -> Window {w + 1}/{num_windows}: sent {len(flows)} flows in UDP datagrams")

    sock.close()
    print(f"[OK] Completed. Total NetFlow datagrams transmitted: {total_packets_sent}")
    return total_packets_sent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CyberSentinel Multi-Host Network Traffic Generator")
    parser.add_argument("--pattern", default="normal_background", choices=[
        "normal_background", "connection_burst", "repeated_attempts", "port_diversity", "large_data_transfer"
    ])
    parser.add_argument("--target-host", default="127.0.0.1", help="CyberSentinel host IP")
    parser.add_argument("--target-port", type=int, default=9995, help="NetFlow UDP port")
    parser.add_argument("--windows", type=int, default=5, help="Number of 30-second windows")
    args = parser.parse_args()

    transmit_live_netflow(
        target_host=args.target_host,
        target_port=args.target_port,
        pattern=args.pattern,
        num_windows=args.windows,
    )
