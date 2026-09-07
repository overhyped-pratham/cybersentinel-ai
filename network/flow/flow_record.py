"""
CyberSentinel AI - Flow Record Representation.

Represents a single parsed, normalized network flow record.
All raw inputs (CSV, PCAP) are standardized into this intermediate schema.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class FlowRecord:
    """
    Standardized immutable network flow record.
    
    Attributes:
        timestamp: Start time of the flow in Unix epoch seconds (UTC).
        src_ip: Source IPv4/IPv6 string address.
        dst_ip: Destination IPv4/IPv6 string address.
        src_port: Source port integer (0-65535).
        dst_port: Destination port integer (0-65535).
        protocol: Transport protocol number (6=TCP, 17=UDP, 1=ICMP).
        packets: Total packets transmitted across the flow.
        bytes: Total bytes transmitted across the flow.
        duration: Flow duration in seconds.
        syn_flag: 1 if SYN flag was set, else 0.
        rst_flag: 1 if RST flag was set, else 0.
        fin_flag: 1 if FIN flag was set, else 0.
        ack_flag: 1 if ACK flag was set, else 0.
        psh_flag: 1 if PSH flag was set, else 0.
        urg_flag: 1 if URG flag was set, else 0.
        failed: True if flow ended abnormally (RST, ICMP unreachable, 0 bytes reply).
        scenario_id: Optional scenario/session grouping identifier for leakage-free splitting.
        label: Raw dataset label or attack designation if known (e.g. 'BENIGN', 'PortScan').
    """
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    packets: int
    bytes: int
    duration: float = 0.0
    syn_flag: int = 0
    rst_flag: int = 0
    fin_flag: int = 0
    ack_flag: int = 0
    psh_flag: int = 0
    urg_flag: int = 0
    failed: bool = False
    scenario_id: str = "default"
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError(f"Flow timestamp cannot be negative: {self.timestamp}")
        if not (0 <= self.src_port <= 65535):
            raise ValueError(f"Invalid src_port: {self.src_port}")
        if not (0 <= self.dst_port <= 65535):
            raise ValueError(f"Invalid dst_port: {self.dst_port}")
        if self.packets < 0:
            raise ValueError(f"Packets cannot be negative: {self.packets}")
        if self.bytes < 0:
            raise ValueError(f"Bytes cannot be negative: {self.bytes}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a dictionary."""
        return {
            "timestamp": self.timestamp,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "packets": self.packets,
            "bytes": self.bytes,
            "duration": self.duration,
            "syn_flag": self.syn_flag,
            "rst_flag": self.rst_flag,
            "fin_flag": self.fin_flag,
            "ack_flag": self.ack_flag,
            "psh_flag": self.psh_flag,
            "urg_flag": self.urg_flag,
            "failed": self.failed,
            "scenario_id": self.scenario_id,
            "label": self.label,
        }
