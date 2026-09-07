"""
CyberSentinel AI - Flow CSV Ingestion Engine.

Ingests and normalizes network flow CSV files from sources such as CICIDS2017,
NetFlow/IPFIX logs, or custom synthetic traffic generators.
Handles missing columns, malformed timestamps, infinite values, and duplicates.
"""

from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import logging
import math
import numpy as np
import pandas as pd
from dateutil import parser as date_parser

from network.flow.flow_record import FlowRecord

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when flow dataset violates schema or integrity rules."""
    pass


COLUMN_ALIASES: Dict[str, List[str]] = {
    "timestamp": [
        "timestamp", "time", "start_time", "flow_start", "first_switched",
        "date_time", "datetime", "frame.time_epoch"
    ],
    "src_ip": [
        "src_ip", "source_ip", "source ip", "srcip", "ip.src", "saddr",
        "source_address", "src_addr"
    ],
    "dst_ip": [
        "dst_ip", "destination_ip", "destination ip", "dstip", "ip.dst", "daddr",
        "destination_address", "dst_addr"
    ],
    "src_port": [
        "src_port", "source_port", "source port", "srcport", "tcp.srcport",
        "udp.srcport", "sport", "s_port"
    ],
    "dst_port": [
        "dst_port", "destination_port", "destination port", "dstport", "tcp.dstport",
        "udp.dstport", "dport", "d_port"
    ],
    "protocol": [
        "protocol", "proto", "ip.proto", "ip_proto", "trans_protocol"
    ],
    "packets": [
        "packets", "total_packets", "total fwd packets", "tot_pkts", "pkt_count",
        "total packets", "pkts", "tot_fwd_pkts", "total_fwd_packets"
    ],
    "bytes": [
        "bytes", "total_bytes", "total length of fwd packets", "tot_bytes", "byte_count",
        "total bytes", "tot_fwd_bytes", "total_fwd_bytes", "octets"
    ],
    "duration": [
        "duration", "flow_duration", "flow duration", "dur", "td"
    ],
    "syn_flag": [
        "syn_flag", "syn", "tcp.flags.syn", "syn flag count", "fin_flag_cnt"
    ],
    "rst_flag": [
        "rst_flag", "rst", "tcp.flags.reset", "rst flag count"
    ],
    "fin_flag": [
        "fin_flag", "fin", "tcp.flags.fin", "fin flag count"
    ],
    "ack_flag": [
        "ack_flag", "ack", "tcp.flags.ack", "ack flag count"
    ],
    "psh_flag": [
        "psh_flag", "psh", "tcp.flags.push", "psh flag count"
    ],
    "urg_flag": [
        "urg_flag", "urg", "tcp.flags.urg", "urg flag count"
    ],
    "failed": [
        "failed", "is_failed", "flow_failed", "status_failed"
    ],
    "scenario_id": [
        "scenario_id", "scenario", "trace_id", "session_id", "file_source"
    ],
    "label": [
        "label", "attack", "attack_cat", "class", "tag"
    ]
}


def parse_timestamp_scalar(val: Any) -> float:
    """Safely parses scalar timestamp to Unix epoch float."""
    if pd.isna(val):
        raise ValueError("Timestamp value is NaN or null")
    
    # Check if already a number (epoch float or int)
    if isinstance(val, (int, float, np.integer, np.floating)):
        ts = float(val)
        # Handle millisecond or microsecond epoch timestamps
        if ts > 1e14:  # microseconds/nanoseconds
            return ts / 1e6
        elif ts > 1e11:  # milliseconds
            return ts / 1e3
        return ts
    
    # Try string conversion
    val_str = str(val).strip()
    try:
        ts = float(val_str)
        if ts > 1e14:
            return ts / 1e6
        elif ts > 1e11:
            return ts / 1e3
        return ts
    except ValueError:
        pass
    
    # Attempt standard datetime parsing
    dt = date_parser.parse(val_str)
    return dt.timestamp()


class CSVFlowLoader:
    """
    Robust reader and cleaner for flow-based CSV datasets.
    Maps arbitrary column casing, replaces invalid numerical entries,
    and deduplicates flows.
    """

    def __init__(self, default_scenario: str = "default_scenario") -> None:
        self.default_scenario = default_scenario

    def _resolve_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Resolves raw DataFrame column names to standard schema keys."""
        normalized_cols = {col.strip().lower(): col for col in df.columns}
        resolved: Dict[str, str] = {}

        for canonical_key, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                if alias.lower() in normalized_cols:
                    resolved[canonical_key] = normalized_cols[alias.lower()]
                    break

        # Check required fields
        required_fields = ["timestamp", "src_ip", "dst_ip", "dst_port"]
        missing = [f for f in required_fields if f not in resolved]
        if missing:
            raise DataValidationError(
                f"CSV missing mandatory flow columns: {missing}. Available columns: {list(df.columns)}"
            )

        return resolved

    def load_dataframe(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Loads and cleans raw CSV file into a pandas DataFrame."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Flow CSV file not found: {path}")

        # Check for empty file
        if path.stat().st_size == 0:
            return pd.DataFrame()

        # Read CSV with flexible delimiters
        try:
            df = pd.read_csv(path, skipinitialspace=True, low_memory=False)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except Exception as e:
            raise DataValidationError(f"Failed to parse CSV {path}: {str(e)}") from e

        if df.empty:
            return df

        return self.clean_dataframe(df, scenario_name=path.stem)

    def clean_dataframe(
        self, df: pd.DataFrame, scenario_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Cleans and standardizes raw DataFrame containing flow logs."""
        if df.empty:
            return pd.DataFrame()

        resolved = self._resolve_columns(df)
        cleaned = pd.DataFrame()

        # Extract & parse timestamps
        raw_timestamps = df[resolved["timestamp"]]
        parsed_ts = []
        valid_mask = []
        for idx, val in enumerate(raw_timestamps):
            try:
                parsed_ts.append(parse_timestamp_scalar(val))
                valid_mask.append(True)
            except Exception:
                parsed_ts.append(np.nan)
                valid_mask.append(False)

        cleaned["timestamp"] = parsed_ts
        cleaned = cleaned[valid_mask].copy()
        df_valid = df[valid_mask].copy()

        if cleaned.empty:
            raise DataValidationError("All timestamp values in CSV could not be parsed.")

        def _safe_numeric(series: pd.Series, default_val: float) -> pd.Series:
            s = pd.to_numeric(series, errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan)
            return s.fillna(default_val)

        # Map IP and Port
        cleaned["src_ip"] = df_valid[resolved["src_ip"]].astype(str).str.strip()
        cleaned["dst_ip"] = df_valid[resolved["dst_ip"]].astype(str).str.strip()

        # Src port
        if "src_port" in resolved:
            cleaned["src_port"] = _safe_numeric(df_valid[resolved["src_port"]], 0).astype(int)
        else:
            cleaned["src_port"] = 0

        # Dst port
        cleaned["dst_port"] = _safe_numeric(df_valid[resolved["dst_port"]], 0).astype(int)

        # Ensure valid port boundaries [0, 65535]
        cleaned["src_port"] = cleaned["src_port"].clip(0, 65535)
        cleaned["dst_port"] = cleaned["dst_port"].clip(0, 65535)

        # Protocol
        if "protocol" in resolved:
            cleaned["protocol"] = _safe_numeric(df_valid[resolved["protocol"]], 6).astype(int)
        else:
            cleaned["protocol"] = 6  # Default TCP

        # Packets and Bytes
        if "packets" in resolved:
            cleaned["packets"] = _safe_numeric(df_valid[resolved["packets"]], 1).astype(int)
        else:
            cleaned["packets"] = 1

        if "bytes" in resolved:
            cleaned["bytes"] = _safe_numeric(df_valid[resolved["bytes"]], 64).astype(int)
        else:
            cleaned["bytes"] = cleaned["packets"] * 64

        cleaned["packets"] = cleaned["packets"].clip(lower=1)
        cleaned["bytes"] = cleaned["bytes"].clip(lower=0)

        # Duration
        if "duration" in resolved:
            dur = _safe_numeric(df_valid[resolved["duration"]], 0.0).astype(float)
            # Some datasets (e.g. CICIDS) store duration in microseconds
            if dur.max() > 1e6:
                dur = dur / 1e6
            cleaned["duration"] = dur.clip(lower=0.0)
        else:
            cleaned["duration"] = 0.0

        # Flags
        for flag in ["syn_flag", "rst_flag", "fin_flag", "ack_flag", "psh_flag", "urg_flag"]:
            if flag in resolved:
                val = _safe_numeric(df_valid[resolved[flag]], 0).astype(int)
                cleaned[flag] = (val > 0).astype(int)
            else:
                cleaned[flag] = 0

        # Failed flow status: explicit or inferred from RST or 0 bytes
        if "failed" in resolved:
            cleaned["failed"] = df_valid[resolved["failed"]].astype(bool)
        else:
            cleaned["failed"] = (cleaned["rst_flag"] == 1) | (cleaned["bytes"] == 0)

        # Scenario identifier
        if "scenario_id" in resolved:
            cleaned["scenario_id"] = df_valid[resolved["scenario_id"]].astype(str)
        else:
            cleaned["scenario_id"] = scenario_name if scenario_name else self.default_scenario

        # Label
        if "label" in resolved:
            cleaned["label"] = df_valid[resolved["label"]].astype(str).str.strip()
        else:
            cleaned["label"] = "BENIGN"

        # Handle Infinite and NaN values across all columns
        cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
        cleaned.fillna(0, inplace=True)

        # Deduplicate identical flow records (same 5-tuple + exact timestamp)
        cleaned.drop_duplicates(
            subset=["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol"],
            inplace=True
        )

        # Strictly sort chronologically
        cleaned.sort_values(by="timestamp", ascending=True, inplace=True)
        cleaned.reset_index(drop=True, inplace=True)

        return cleaned

    def load_flows(self, file_path: Union[str, Path]) -> List[FlowRecord]:
        """Loads and converts flow records from CSV file to FlowRecord instances."""
        df = self.load_dataframe(file_path)
        if df.empty:
            return []

        records = []
        for row in df.itertuples(index=False):
            records.append(
                FlowRecord(
                    timestamp=float(row.timestamp),
                    src_ip=str(row.src_ip),
                    dst_ip=str(row.dst_ip),
                    src_port=int(row.src_port),
                    dst_port=int(row.dst_port),
                    protocol=int(row.protocol),
                    packets=int(row.packets),
                    bytes=int(row.bytes),
                    duration=float(row.duration),
                    syn_flag=int(row.syn_flag),
                    rst_flag=int(row.rst_flag),
                    fin_flag=int(row.fin_flag),
                    ack_flag=int(row.ack_flag),
                    psh_flag=int(row.psh_flag),
                    urg_flag=int(row.urg_flag),
                    failed=bool(row.failed),
                    scenario_id=str(row.scenario_id),
                    label=str(row.label),
                )
            )
        return records
