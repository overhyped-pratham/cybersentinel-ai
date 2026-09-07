"""
Unit tests for network.pcap.pcap_loader.PCAPFlowLoader.
"""

import struct
import socket
import pytest
from pathlib import Path

from network.pcap.pcap_loader import PCAPFlowLoader


def create_synthetic_pcap(file_path: Path) -> None:
    """Creates a minimal valid libpcap file with two TCP packets."""
    # Global header (24 bytes)
    magic = 0xA1B2C3D4
    v_maj, v_min = 2, 4
    thiszone, sigfigs = 0, 0
    snaplen, network = 65535, 1 # DLT_EN10MB

    global_hdr = struct.pack("<IHHiIII", magic, v_maj, v_min, thiszone, sigfigs, snaplen, network)

    packets_data = []

    # Packet 1: SYN packet from 192.168.1.100:54321 to 192.168.1.10:445
    # Ethernet header (14 bytes)
    eth_hdr = b"\x00\x11\x22\x33\x44\x55" + b"\x66\x77\x88\x99\xaa\xbb" + struct.pack("!H", 0x0800)
    # IPv4 header (20 bytes)
    # version 4, ihl 5 -> 0x45, dscp 0, total_len 44, id 1, flags/frag 0, ttl 64, proto 6 (TCP), csum 0
    ip_src = socket.inet_aton("192.168.1.100")
    ip_dst = socket.inet_aton("192.168.1.10")
    ip_hdr = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 44, 1, 0, 64, 6, 0, ip_src, ip_dst)
    # TCP header (20 bytes): sport 54321, dport 445, seq 1, ack 0, offset 5 (0x50), flags SYN (0x02), win 8192, csum 0, urg 0
    tcp_hdr = struct.pack("!HHIIBBHHH", 54321, 445, 1, 0, 0x50, 0x02, 8192, 0, 0)
    pkt1_payload = eth_hdr + ip_hdr + tcp_hdr

    pkt1_hdr = struct.pack("<IIII", 1680000000, 100000, len(pkt1_payload), len(pkt1_payload))
    packets_data.append(pkt1_hdr + pkt1_payload)

    # Packet 2: ACK packet in same flow 2 seconds later
    tcp_hdr2 = struct.pack("!HHIIBBHHH", 54321, 445, 2, 1, 0x50, 0x10, 8192, 0, 0)
    pkt2_payload = eth_hdr + ip_hdr + tcp_hdr2
    pkt2_hdr = struct.pack("<IIII", 1680000002, 200000, len(pkt2_payload), len(pkt2_payload))
    packets_data.append(pkt2_hdr + pkt2_payload)

    with open(file_path, "wb") as f:
        f.write(global_hdr)
        for p in packets_data:
            f.write(p)


def test_pcap_loader_parsing(tmp_path: Path):
    pcap_path = tmp_path / "test.pcap"
    create_synthetic_pcap(pcap_path)

    loader = PCAPFlowLoader(flow_timeout_seconds=15.0)
    flows = loader.load_flows(pcap_path)

    assert len(flows) == 1
    f = flows[0]
    assert f.src_ip == "192.168.1.100"
    assert f.dst_ip == "192.168.1.10"
    assert f.src_port == 54321
    assert f.dst_port == 445
    assert f.protocol == 6
    assert f.packets == 2
    assert f.syn_flag == 1
    assert f.ack_flag == 1
    assert f.duration == pytest.approx(2.1, rel=1e-1)
