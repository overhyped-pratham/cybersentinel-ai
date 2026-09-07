"""
CyberSentinel AI - Network State Construction Engine.

Aggregates raw flow records into discrete temporal network state vectors S_t
over configurable window sizes (default: 30.0 seconds).
Calculates approximately 24 curated, explainable statistical & behavioral features.
Strictly ensures zero temporal lookahead leakage.
"""

from typing import List, Dict, Any, Optional, Sequence
import math
import logging
from collections import Counter
import numpy as np
import pandas as pd

from network.flow.flow_record import FlowRecord

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "flow_count",
    "total_packets",
    "total_bytes",
    "pkt_rate",
    "byte_rate",
    "unique_src_ips",
    "unique_dst_ips",
    "unique_dst_ports",
    "syn_count",
    "rst_count",
    "fin_count",
    "syn_ratio",
    "rst_ratio",
    "dst_ip_entropy",
    "dst_port_entropy",
    "failed_flow_count",
    "failed_flow_ratio",
    "tcp_flag_diversity",
    "port_445_share",
    "port_3389_share",
    "port_22_share",
    "port_80_443_share",
    "mean_flow_duration",
    "bytes_per_packet",
]


def shannon_entropy(counts: Sequence[int]) -> float:
    """Calculates Shannon entropy in bits for a frequency distribution."""
    total = sum(counts)
    if total <= 1:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return float(entropy)


class NetworkStateBuilder:
    """
    Constructs discrete network states S_t from temporal flow records.
    Each 30-second window yields a standardized feature vector.
    """

    def __init__(
        self,
        window_size_seconds: float = 30.0,
        step_size_seconds: Optional[float] = None,
        min_flows_per_window: int = 1,
    ) -> None:
        self.window_size = float(window_size_seconds)
        self.step_size = float(step_size_seconds) if step_size_seconds is not None else self.window_size
        self.min_flows_per_window = min_flows_per_window
        self.feature_names = list(FEATURE_NAMES)

    def build_states(
        self,
        flows: Sequence[FlowRecord],
        scenario_id: Optional[str] = None,
        base_timestamp: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Groups flows into temporal windows and computes state vectors S_t.
        
        Args:
            flows: Chronologically sorted sequence of FlowRecord instances.
            scenario_id: Optional trace/scenario override.
            base_timestamp: Optional start time for the first window. If None, uses first flow's timestamp.
            
        Returns:
            pd.DataFrame: DataFrame where each row is state S_t with metadata.
        """
        if not flows:
            return pd.DataFrame(columns=["window_idx", "timestamp_start", "timestamp_end", "scenario_id"] + self.feature_names)

        # Separate flows by scenario if multiple scenarios are present
        flows_by_scenario: Dict[str, List[FlowRecord]] = {}
        for f in flows:
            sc_id = scenario_id if scenario_id is not None else f.scenario_id
            if sc_id not in flows_by_scenario:
                flows_by_scenario[sc_id] = []
            flows_by_scenario[sc_id].append(f)

        all_state_rows = []

        for sc_id, sc_flows in flows_by_scenario.items():
            sc_flows.sort(key=lambda x: x.timestamp)
            first_ts = sc_flows[0].timestamp
            last_ts = sc_flows[-1].timestamp

            # Define window boundaries
            w_start = base_timestamp if base_timestamp is not None else first_ts
            window_idx = 0

            # Advance window by step_size
            while w_start <= last_ts:
                w_end = w_start + self.window_size
                # Window condition: [w_start, w_end)
                window_flows = [
                    f for f in sc_flows if w_start <= f.timestamp < w_end
                ]

                if len(window_flows) >= self.min_flows_per_window:
                    features = self._calculate_window_features(window_flows)
                    dominant_label = Counter(f.label for f in window_flows if f.label).most_common(1)
                    label_val = dominant_label[0][0] if dominant_label else "BENIGN"

                    row_dict = {
                        "window_idx": window_idx,
                        "timestamp_start": w_start,
                        "timestamp_end": w_end,
                        "scenario_id": sc_id,
                        "dominant_label": label_val,
                    }
                    row_dict.update(features)
                    all_state_rows.append(row_dict)

                window_idx += 1
                w_start += self.step_size

        df_states = pd.DataFrame(all_state_rows)
        if df_states.empty:
            return pd.DataFrame(columns=["window_idx", "timestamp_start", "timestamp_end", "scenario_id", "dominant_label"] + self.feature_names)

        # Enforce column order
        meta_cols = ["window_idx", "timestamp_start", "timestamp_end", "scenario_id", "dominant_label"]
        cols = meta_cols + self.feature_names
        return df_states[cols].copy()

    def _calculate_window_features(self, flows: List[FlowRecord]) -> Dict[str, float]:
        """Calculates the curated feature vector for a single 30s window."""
        flow_count = len(flows)
        if flow_count == 0:
            return {name: 0.0 for name in self.feature_names}

        total_packets = sum(f.packets for f in flows)
        total_bytes = sum(f.bytes for f in flows)
        pkt_rate = float(total_packets) / self.window_size
        byte_rate = float(total_bytes) / self.window_size

        unique_src_ips = len(set(f.src_ip for f in flows))
        unique_dst_ips = len(set(f.dst_ip for f in flows))
        unique_dst_ports = len(set(f.dst_port for f in flows))

        syn_count = sum(f.syn_flag for f in flows)
        rst_count = sum(f.rst_flag for f in flows)
        fin_count = sum(f.fin_flag for f in flows)

        syn_ratio = float(syn_count) / max(flow_count, 1)
        rst_ratio = float(rst_count) / max(flow_count, 1)

        # Entropies
        dst_ip_counts = list(Counter(f.dst_ip for f in flows).values())
        dst_ip_entropy = shannon_entropy(dst_ip_counts)

        dst_port_counts = list(Counter(f.dst_port for f in flows).values())
        dst_port_entropy = shannon_entropy(dst_port_counts)

        # Failures
        failed_flow_count = sum(1 for f in flows if f.failed)
        failed_flow_ratio = float(failed_flow_count) / max(flow_count, 1)

        # TCP flag pattern diversity
        flag_patterns = Counter(
            (f.syn_flag, f.rst_flag, f.fin_flag, f.ack_flag, f.psh_flag, f.urg_flag)
            for f in flows
        )
        tcp_flag_diversity = shannon_entropy(list(flag_patterns.values()))

        # Targeted service port shares
        port_445_count = sum(1 for f in flows if f.dst_port == 445)
        port_3389_count = sum(1 for f in flows if f.dst_port == 3389)
        port_22_count = sum(1 for f in flows if f.dst_port == 22)
        port_80_443_count = sum(1 for f in flows if f.dst_port in (80, 443, 8080, 8443))

        port_445_share = float(port_445_count) / max(flow_count, 1)
        port_3389_share = float(port_3389_count) / max(flow_count, 1)
        port_22_share = float(port_22_count) / max(flow_count, 1)
        port_80_443_share = float(port_80_443_count) / max(flow_count, 1)

        mean_flow_duration = float(sum(f.duration for f in flows)) / max(flow_count, 1)
        bytes_per_packet = float(total_bytes) / max(total_packets, 1)

        return {
            "flow_count": float(flow_count),
            "total_packets": float(total_packets),
            "total_bytes": float(total_bytes),
            "pkt_rate": pkt_rate,
            "byte_rate": byte_rate,
            "unique_src_ips": float(unique_src_ips),
            "unique_dst_ips": float(unique_dst_ips),
            "unique_dst_ports": float(unique_dst_ports),
            "syn_count": float(syn_count),
            "rst_count": float(rst_count),
            "fin_count": float(fin_count),
            "syn_ratio": syn_ratio,
            "rst_ratio": rst_ratio,
            "dst_ip_entropy": dst_ip_entropy,
            "dst_port_entropy": dst_port_entropy,
            "failed_flow_count": float(failed_flow_count),
            "failed_flow_ratio": failed_flow_ratio,
            "tcp_flag_diversity": tcp_flag_diversity,
            "port_445_share": port_445_share,
            "port_3389_share": port_3389_share,
            "port_22_share": port_22_share,
            "port_80_443_share": port_80_443_share,
            "mean_flow_duration": mean_flow_duration,
            "bytes_per_packet": bytes_per_packet,
        }
