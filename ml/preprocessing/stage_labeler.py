"""
CyberSentinel AI - Derived Attack Stage Labeling Engine.

RESEARCH DISCLAIMER:
Stage labels are derived from scenario timelines and flow characteristics
using a documented mapping policy. They are a research approximation,
NOT authoritative ATT&CK ground truth.

Taxonomy:
- BENIGN (0)
- RECONNAISSANCE (1)
- INITIAL_ACCESS (2)
- EXECUTION (3)
- CREDENTIAL_ACCESS (4)
- DISCOVERY (5)
- LATERAL_MOVEMENT (6)
- COMMAND_AND_CONTROL (7)
- EXFILTRATION (8)
- UNKNOWN (9)
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

STAGE_TAXONOMY = [
    "BENIGN",
    "RECONNAISSANCE",
    "INITIAL_ACCESS",
    "EXECUTION",
    "CREDENTIAL_ACCESS",
    "DISCOVERY",
    "LATERAL_MOVEMENT",
    "COMMAND_AND_CONTROL",
    "EXFILTRATION",
    "UNKNOWN",
]

STAGE_TO_ID: Dict[str, int] = {stage: i for i, stage in enumerate(STAGE_TAXONOMY)}
ID_TO_STAGE: Dict[int, str] = {i: stage for i, stage in enumerate(STAGE_TAXONOMY)}

# Research mapping from common dataset attack tags (e.g. CICIDS2017) to tactical stages
DATASET_LABEL_MAPPING: Dict[str, str] = {
    # Benign
    "benign": "BENIGN",
    "normal": "BENIGN",
    
    # Reconnaissance / Discovery
    "portscan": "RECONNAISSANCE",
    "port scan": "RECONNAISSANCE",
    "nmap": "RECONNAISSANCE",
    "ipsweep": "RECONNAISSANCE",
    "discovery": "DISCOVERY",
    
    # Initial Access / Exploitation
    "dos": "INITIAL_ACCESS",
    "ddos": "INITIAL_ACCESS",
    "heartbleed": "INITIAL_ACCESS",
    "web attack": "INITIAL_ACCESS",
    "web attack - brute force": "CREDENTIAL_ACCESS",
    "web attack - xss": "INITIAL_ACCESS",
    "web attack - sql injection": "INITIAL_ACCESS",
    
    # Credential Access
    "ftp-patator": "CREDENTIAL_ACCESS",
    "ssh-patator": "CREDENTIAL_ACCESS",
    "bruteforce": "CREDENTIAL_ACCESS",
    "brute force": "CREDENTIAL_ACCESS",
    
    # Lateral Movement / Infiltration
    "infiltration": "LATERAL_MOVEMENT",
    "smb_spread": "LATERAL_MOVEMENT",
    "lateral_movement": "LATERAL_MOVEMENT",
    
    # Command & Control
    "bot": "COMMAND_AND_CONTROL",
    "botnet": "COMMAND_AND_CONTROL",
    "c2": "COMMAND_AND_CONTROL",
    "beacon": "COMMAND_AND_CONTROL",
    
    # Exfiltration
    "exfiltration": "EXFILTRATION",
    "data_leak": "EXFILTRATION",
}


class StageLabeler:
    """
    Assigns attack stages and binary indicators to network states S_t
    using scenario metadata, dataset labels, and behavioral heuristics.
    """

    def __init__(self, fallback_to_heuristics: bool = True) -> None:
        self.fallback_to_heuristics = fallback_to_heuristics

    def label_state(self, row: pd.Series) -> Tuple[str, int, int]:
        """
        Derives the attack stage for a given window state row.
        
        Args:
            row: pd.Series representing a state window (includes dominant_label and features).
            
        Returns:
            Tuple[str, int, int]: (stage_name, stage_id, is_attack)
        """
        raw_label = str(row.get("dominant_label", "")).strip().lower()

        # Check explicit label dictionary mapping
        for key, stage in DATASET_LABEL_MAPPING.items():
            if key in raw_label:
                is_attack = 0 if stage == "BENIGN" else 1
                return stage, STAGE_TO_ID[stage], is_attack

        # If unknown or fallback enabled, apply heuristic rules based on network behavioral features
        if self.fallback_to_heuristics:
            return self._apply_behavioral_heuristics(row)

        return "UNKNOWN", STAGE_TO_ID["UNKNOWN"], 0

    def _apply_behavioral_heuristics(self, row: pd.Series) -> Tuple[str, int, int]:
        """
        Heuristic research rules for windows where explicit label is missing.
        """
        # Exfiltration: high byte rate and high bytes per packet with low packet diversity
        byte_rate = float(row.get("byte_rate", 0.0))
        bpp = float(row.get("bytes_per_packet", 0.0))
        if byte_rate > 50000.0 and bpp > 800.0:
            return "EXFILTRATION", STAGE_TO_ID["EXFILTRATION"], 1

        # Lateral Movement: significant internal SMB (port 445) or RDP (port 3389) traffic
        port_445_share = float(row.get("port_445_share", 0.0))
        port_3389_share = float(row.get("port_3389_share", 0.0))
        if port_445_share > 0.30 or port_3389_share > 0.30:
            return "LATERAL_MOVEMENT", STAGE_TO_ID["LATERAL_MOVEMENT"], 1

        # Credential Access: high SSH (port 22) share with high failed flow ratio
        port_22_share = float(row.get("port_22_share", 0.0))
        failed_ratio = float(row.get("failed_flow_ratio", 0.0))
        if port_22_share > 0.25 and failed_ratio > 0.30:
            return "CREDENTIAL_ACCESS", STAGE_TO_ID["CREDENTIAL_ACCESS"], 1

        # Reconnaissance: high unique dst ports, high dst port entropy, high syn ratio
        port_entropy = float(row.get("dst_port_entropy", 0.0))
        syn_ratio = float(row.get("syn_ratio", 0.0))
        unique_ports = float(row.get("unique_dst_ports", 0.0))
        if port_entropy > 2.5 and (syn_ratio > 0.40 or unique_ports > 15):
            return "RECONNAISSANCE", STAGE_TO_ID["RECONNAISSANCE"], 1

        # Default to BENIGN if benign-like activity, else UNKNOWN
        flow_count = float(row.get("flow_count", 0.0))
        if flow_count > 0:
            return "BENIGN", STAGE_TO_ID["BENIGN"], 0

        return "UNKNOWN", STAGE_TO_ID["UNKNOWN"], 0

    def attach_labels_to_dataframe(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches a DataFrame of states S_t with stage_name, stage_id, and is_attack.
        """
        if df_states.empty:
            df_out = df_states.copy()
            df_out["stage_name"] = []
            df_out["stage_id"] = []
            df_out["is_attack"] = []
            return df_out

        stages = []
        stage_ids = []
        attacks = []

        for _, row in df_states.iterrows():
            stage_name, s_id, is_atk = self.label_state(row)
            stages.append(stage_name)
            stage_ids.append(s_id)
            attacks.append(is_atk)

        df_out = df_states.copy()
        df_out["stage_name"] = stages
        df_out["stage_id"] = stage_ids
        df_out["is_attack"] = attacks
        return df_out
