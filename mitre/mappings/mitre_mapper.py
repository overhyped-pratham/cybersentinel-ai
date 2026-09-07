"""
CyberSentinel AI — Deterministic MITRE ATT&CK Mapping Layer (Phase 9).

Maps predicted attack stages to relevant MITRE ATT&CK techniques.

CRITICAL RULES:
  - Every technique ID and name must be verified from MITRE ATT&CK Enterprise v14.
  - No LLM-invented technique IDs.
  - Every entry includes: technique_id, name, tactic, rationale, and mapping_provenance.
  - The mapping is static and inspectable — no dynamic ML-based generation.
  - The mapping layer is SEPARATE from the ML model. The model predicts stages;
    this module deterministically looks up associated techniques.

Source: MITRE ATT&CK Enterprise v14 (https://attack.mitre.org/)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Canonical MITRE Technique Registry
# ---------------------------------------------------------------------------
# Each entry is a dict with:
#   technique_id   : official MITRE ID (e.g. "T1595")
#   name           : official MITRE technique name
#   tactic         : MITRE tactic (e.g. "Reconnaissance")
#   rationale      : why this technique is associated with the predicted stage
#   sub_techniques : list of sub-technique IDs if applicable (may be empty)
#   mapping_provenance : how this mapping was derived (always "MITRE ATT&CK v14 static")

STAGE_TO_MITRE: Dict[str, List[Dict[str, Any]]] = {
    "BENIGN": [],  # No attack techniques associated with benign traffic

    "RECONNAISSANCE": [
        {
            "technique_id": "T1595",
            "name": "Active Scanning",
            "tactic": "Reconnaissance",
            "rationale": (
                "SYN flooding, port entropy spikes, and high unique-destination-port counts "
                "are characteristic of active network scanning (nmap, masscan)."
            ),
            "sub_techniques": ["T1595.001", "T1595.002"],
            "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping",
        },
        {
            "technique_id": "T1046",
            "name": "Network Service Discovery",
            "tactic": "Discovery",
            "rationale": (
                "High unique_dst_ports, elevated syn_ratio, and broad dst_ip_entropy "
                "indicate active service enumeration of discovered hosts."
            ),
            "sub_techniques": [],
            "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping",
        },
        {
            "technique_id": "T1590",
            "name": "Gather Victim Network Information",
            "tactic": "Reconnaissance",
            "rationale": (
                "Network scanning often precedes enumeration of network topology, "
                "live hosts, and exposed services."
            ),
            "sub_techniques": ["T1590.004"],
            "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping",
        },
    ],

    "INITIAL_ACCESS": [
        {
            "technique_id": "T1190",
            "name": "Exploit Public-Facing Application",
            "tactic": "Initial Access",
            "rationale": (
                "High failed_flow_ratio and port_80_443_share indicate exploitation "
                "attempts targeting web-facing services."
            ),
            "sub_techniques": [],
            "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping",
        },
        {
            "technique_id": "T1133",
            "name": "External Remote Services",
            "tactic": "Initial Access",
            "rationale": (
                "port_22_share and port_3389_share spikes indicate exploitation of "
                "SSH/RDP remote service access."
            ),
            "sub_techniques": [],
            "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping",
        },
    ],

    "EXECUTION": [
        {
            "technique_id": "T1059",
            "name": "Command and Scripting Interpreter",
            "tactic": "Execution",
            "rationale": (
                "Increased byte_rate and total_bytes following initial access "
                "indicate payload delivery and remote command execution."
            ),
            "sub_techniques": ["T1059.004"],
            "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping",
        },
    ],

    "CREDENTIAL_ACCESS": [
        {
            "technique_id": "T1110",
            "name": "Brute Force",
            "tactic": "Credential Access",
            "rationale": (
                "Elevated failed_flow_count, failed_flow_ratio, and rst_ratio are "
                "signatures of brute-force authentication attempts (SSH, SMB, RDP)."
            ),
            "sub_techniques": ["T1110.001", "T1110.003"],
            "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping",
        },
        {
            "technique_id": "T1003",
            "name": "OS Credential Dumping",
            "tactic": "Credential Access",
            "rationale": (
                "High port_445_share combined with credential access stage may "
                "indicate SMB-based credential dumping (pass-the-hash patterns)."
            ),
            "sub_techniques": ["T1003.002"],
            "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping",
        },
    ],

    "DISCOVERY": [
        {
            "technique_id": "T1082",
            "name": "System Information Discovery",
            "tactic": "Discovery",
            "rationale": (
                "Internal network scanning with moderate dst_ip_entropy and "
                "unique_dst_ports expansion indicates host enumeration."
            ),
            "sub_techniques": [],
            "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping",
        },
        {
            "technique_id": "T1018",
            "name": "Remote System Discovery",
            "tactic": "Discovery",
            "rationale": (
                "Spikes in unique_dst_ips indicate automated discovery of "
                "adjacent network hosts."
            ),
            "sub_techniques": [],
            "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping",
        },
    ],

    "LATERAL_MOVEMENT": [
        {
            "technique_id": "T1021",
            "name": "Remote Services",
            "tactic": "Lateral Movement",
            "rationale": (
                "Elevated port_445_share (SMB), port_3389_share (RDP), and "
                "port_22_share (SSH) combined with unique_dst_ips growth indicate "
                "lateral movement via remote service abuse."
            ),
            "sub_techniques": ["T1021.001", "T1021.002", "T1021.004"],
            "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping",
        },
        {
            "technique_id": "T1570",
            "name": "Lateral Tool Transfer",
            "tactic": "Lateral Movement",
            "rationale": (
                "Increasing total_bytes and byte_rate during lateral movement "
                "may indicate tool staging across compromised systems."
            ),
            "sub_techniques": [],
            "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping",
        },
    ],

    "COMMAND_AND_CONTROL": [
        {
            "technique_id": "T1071",
            "name": "Application Layer Protocol",
            "tactic": "Command and Control",
            "rationale": (
                "Regular beacon intervals with high port_80_443_share and "
                "long mean_flow_duration indicate C2 over HTTP/S."
            ),
            "sub_techniques": ["T1071.001"],
            "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping",
        },
        {
            "technique_id": "T1573",
            "name": "Encrypted Channel",
            "tactic": "Command and Control",
            "rationale": (
                "Encrypted C2 channels often appear as long-duration, "
                "constant-rate flows with low packet counts."
            ),
            "sub_techniques": [],
            "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping",
        },
    ],

    "EXFILTRATION": [
        {
            "technique_id": "T1048",
            "name": "Exfiltration Over Alternative Protocol",
            "tactic": "Exfiltration",
            "rationale": (
                "Large total_bytes and high byte_rate with unusual destination "
                "ports indicate data exfiltration over non-standard protocols."
            ),
            "sub_techniques": ["T1048.003"],
            "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping",
        },
        {
            "technique_id": "T1041",
            "name": "Exfiltration Over C2 Channel",
            "tactic": "Exfiltration",
            "rationale": (
                "Data may be exfiltrated over established C2 channels "
                "(high bytes_per_packet, sustained byte_rate)."
            ),
            "sub_techniques": [],
            "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping",
        },
    ],

    "UNKNOWN": [
        {
            "technique_id": "UNKNOWN",
            "name": "Unclassified Activity",
            "tactic": "Unknown",
            "rationale": "Stage not classified with sufficient confidence for MITRE mapping.",
            "sub_techniques": [],
            "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping",
        },
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_techniques_for_stage(stage: str) -> List[Dict[str, Any]]:
    """
    Return the list of MITRE ATT&CK techniques associated with a predicted attack stage.

    Args:
        stage: Attack stage string (e.g. "RECONNAISSANCE")

    Returns:
        List of technique dicts. Empty list if stage is BENIGN or unknown.
    """
    return STAGE_TO_MITRE.get(stage, STAGE_TO_MITRE.get("UNKNOWN", []))


def get_primary_technique(stage: str) -> Optional[Dict[str, Any]]:
    """Return the first (highest-priority) technique for a given stage."""
    techniques = get_techniques_for_stage(stage)
    return techniques[0] if techniques else None


def get_mitre_summary(stage: str) -> Dict[str, Any]:
    """
    Return a structured summary of the MITRE mapping for a predicted stage.

    Output:
      {
        "stage": str,
        "technique_count": int,
        "primary_technique_id": str | None,
        "primary_technique_name": str | None,
        "techniques": List[Dict],
      }
    """
    techniques = get_techniques_for_stage(stage)
    primary = techniques[0] if techniques else None
    return {
        "stage": stage,
        "technique_count": len(techniques),
        "primary_technique_id": primary["technique_id"] if primary else None,
        "primary_technique_name": primary["name"] if primary else None,
        "techniques": techniques,
    }


def list_all_stages() -> List[str]:
    """Return all stages in the MITRE mapping."""
    return list(STAGE_TO_MITRE.keys())
