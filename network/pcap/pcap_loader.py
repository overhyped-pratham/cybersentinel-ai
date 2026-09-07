"""
CyberSentinel AI - Pure-Python Offline PCAP Flow Ingestion Engine.

Parses binary libpcap capture files (.pcap) without requiring external C libraries
or third-party network capture drivers (npcap/winpcap/libpcap).
Reconstructs packet streams into standardized FlowRecord instances.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import struct
import socket
import logging

from network.flow.flow_record import FlowRecord

logger = logging.getLogger(__name__)

PCAP_MAGIC_LITTLE = 0xA1B2C3D4
PCAP_MAGIC_BIG = 0xD4C3B2A1
PCAP_MAGIC_NANO_LITTLE = 0xA1B23C4D
PCAP_MAGIC_NANO_BIG = 0x4D3CB2A1


class PCAPFlowLoader:
    """
    Offline PCAP file parser and flow assembler.
    Extracts IPv4 TCP/UDP packets and groups them into flow records.
    """

    def __init__(self, flow_timeout_seconds: float = 15.0, default_scenario: str = "pcap_trace") -> None:
        self.flow_timeout_seconds = flow_timeout_seconds
        self.default_scenario = default_scenario

    def load_flows(self, pcap_path: Union[str, Path]) -> List[FlowRecord]:
        """Parses a PCAP file and returns reconstructed FlowRecords."""
        path = Path(pcap_path)
        if not path.exists():
            raise FileNotFoundError(f"PCAP file not found: {path}")

        if path.stat().st_size < 24:
            return []

        scenario_id = path.stem or self.default_scenario

        with open(path, "rb") as f:
            header_bytes = f.read(24)
            if len(header_bytes) < 24:
                return []

            magic = struct.unpack("<I", header_bytes[0:4])[0]
            if magic in (PCAP_MAGIC_LITTLE, PCAP_MAGIC_NANO_LITTLE):
                endian = "<"
                is_nano = (magic == PCAP_MAGIC_NANO_LITTLE)
            elif magic in (PCAP_MAGIC_BIG, PCAP_MAGIC_NANO_BIG):
                endian = ">"
                is_nano = (magic == PCAP_MAGIC_NANO_BIG)
            else:
                raise ValueError(f"Unsupported PCAP magic number: {hex(magic)}")

            _, _, _, _, snaplen, link_type = struct.unpack(f"{endian}HHIIII", header_bytes[4:24])

            # Link type 1 = Ethernet (DLT_EN10MB)
            if link_type != 1:
                logger.warning(f"Non-Ethernet link type ({link_type}) in {path}; parsing may be limited.")

            raw_packets = []
            while True:
                pkt_hdr = f.read(16)
                if len(pkt_hdr) < 16:
                    break

                ts_sec, ts_usec, incl_len, _ = struct.unpack(f"{endian}IIII", pkt_hdr)
                pkt_data = f.read(incl_len)
                if len(pkt_data) < incl_len:
                    break

                ts = ts_sec + (ts_usec / 1e9 if is_nano else ts_usec / 1e6)
                packet_info = self._parse_packet(pkt_data, ts)
                if packet_info is not None:
                    raw_packets.append(packet_info)

        return self._assemble_flows(raw_packets, scenario_id=scenario_id)

    def _parse_packet(self, data: bytes, timestamp: float) -> Optional[Dict]:
        """Parses single raw Ethernet frame for IPv4 headers and flags."""
        if len(data) < 14:
            return None

        # Ethernet frame
        eth_proto = struct.unpack("!H", data[12:14])[0]
        ip_offset = 14

        # Handle 802.1Q VLAN tag
        if eth_proto == 0x8100:
            if len(data) < 18:
                return None
            eth_proto = struct.unpack("!H", data[16:18])[0]
            ip_offset = 18

        # Only process IPv4 (0x0800)
        if eth_proto != 0x0800:
            return None

        if len(data) < ip_offset + 20:
            return None

        ip_header = data[ip_offset:ip_offset + 20]
        version_ihl = ip_header[0]
        version = version_ihl >> 4
        if version != 4:
            return None

        ihl = (version_ihl & 0x0F) * 4
        total_len = struct.unpack("!H", ip_header[2:4])[0]
        protocol = ip_header[9]
        src_ip = socket.inet_ntoa(ip_header[12:16])
        dst_ip = socket.inet_ntoa(ip_header[16:20])

        transport_offset = ip_offset + ihl
        src_port = 0
        dst_port = 0
        syn_flag = 0
        rst_flag = 0
        fin_flag = 0
        ack_flag = 0
        psh_flag = 0
        urg_flag = 0

        if protocol == 6:  # TCP
            if len(data) < transport_offset + 20:
                return None
            tcp_hdr = data[transport_offset:transport_offset + 20]
            src_port, dst_port = struct.unpack("!HH", tcp_hdr[0:4])
            flags = tcp_hdr[13]
            fin_flag = (flags & 0x01)
            syn_flag = (flags & 0x02) >> 1
            rst_flag = (flags & 0x04) >> 2
            psh_flag = (flags & 0x08) >> 3
            ack_flag = (flags & 0x10) >> 4
            urg_flag = (flags & 0x20) >> 5
        elif protocol == 17:  # UDP
            if len(data) < transport_offset + 8:
                return None
            udp_hdr = data[transport_offset:transport_offset + 8]
            src_port, dst_port = struct.unpack("!HH", udp_hdr[0:4])
        elif protocol == 1:  # ICMP
            src_port = 0
            dst_port = 0
        else:
            return None

        return {
            "timestamp": timestamp,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "bytes": max(total_len, len(data) - ip_offset),
            "syn_flag": syn_flag,
            "rst_flag": rst_flag,
            "fin_flag": fin_flag,
            "ack_flag": ack_flag,
            "psh_flag": psh_flag,
            "urg_flag": urg_flag,
        }

    def _assemble_flows(self, packets: List[Dict], scenario_id: str) -> List[FlowRecord]:
        """Assembles chronologically sorted packets into discrete 5-tuple flow records."""
        if not packets:
            return []

        packets.sort(key=lambda x: x["timestamp"])
        active_flows: Dict[Tuple[str, str, int, int, int], Dict] = {}
        completed_flows: List[FlowRecord] = []

        for pkt in packets:
            key = (pkt["src_ip"], pkt["dst_ip"], pkt["src_port"], pkt["dst_port"], pkt["protocol"])

            # Check if active flow exists and is within timeout
            if key in active_flows:
                flow = active_flows[key]
                if pkt["timestamp"] - flow["last_time"] > self.flow_timeout_seconds:
                    # Flush expired flow
                    completed_flows.append(self._to_record(flow, scenario_id))
                    active_flows[key] = self._create_flow_state(pkt)
                else:
                    # Update flow
                    flow["last_time"] = pkt["timestamp"]
                    flow["packets"] += 1
                    flow["bytes"] += pkt["bytes"]
                    flow["syn_flag"] = max(flow["syn_flag"], pkt["syn_flag"])
                    flow["rst_flag"] = max(flow["rst_flag"], pkt["rst_flag"])
                    flow["fin_flag"] = max(flow["fin_flag"], pkt["fin_flag"])
                    flow["ack_flag"] = max(flow["ack_flag"], pkt["ack_flag"])
                    flow["psh_flag"] = max(flow["psh_flag"], pkt["psh_flag"])
                    flow["urg_flag"] = max(flow["urg_flag"], pkt["urg_flag"])
            else:
                active_flows[key] = self._create_flow_state(pkt)

        # Flush remaining flows
        for flow in active_flows.values():
            completed_flows.append(self._to_record(flow, scenario_id))

        completed_flows.sort(key=lambda f: f.timestamp)
        return completed_flows

    def _create_flow_state(self, pkt: Dict) -> Dict:
        """Initializes tracking state for a new flow."""
        return {
            "start_time": pkt["timestamp"],
            "last_time": pkt["timestamp"],
            "src_ip": pkt["src_ip"],
            "dst_ip": pkt["dst_ip"],
            "src_port": pkt["src_port"],
            "dst_port": pkt["dst_port"],
            "protocol": pkt["protocol"],
            "packets": 1,
            "bytes": pkt["bytes"],
            "syn_flag": pkt["syn_flag"],
            "rst_flag": pkt["rst_flag"],
            "fin_flag": pkt["fin_flag"],
            "ack_flag": pkt["ack_flag"],
            "psh_flag": pkt["psh_flag"],
            "urg_flag": pkt["urg_flag"],
        }

    def _to_record(self, flow: Dict, scenario_id: str) -> FlowRecord:
        """Converts accumulated flow dictionary into immutable FlowRecord."""
        duration = max(0.0, flow["last_time"] - flow["start_time"])
        failed = (flow["rst_flag"] == 1) or (flow["bytes"] == 0)
        return FlowRecord(
            timestamp=flow["start_time"],
            src_ip=flow["src_ip"],
            dst_ip=flow["dst_ip"],
            src_port=flow["src_port"],
            dst_port=flow["dst_port"],
            protocol=flow["protocol"],
            packets=flow["packets"],
            bytes=flow["bytes"],
            duration=duration,
            syn_flag=flow["syn_flag"],
            rst_flag=flow["rst_flag"],
            fin_flag=flow["fin_flag"],
            ack_flag=flow["ack_flag"],
            psh_flag=flow["psh_flag"],
            urg_flag=flow["urg_flag"],
            failed=failed,
            scenario_id=scenario_id,
            label="UNKNOWN",
        )
