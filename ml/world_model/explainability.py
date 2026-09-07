"""
CyberSentinel AI — Feature Attribution Explainability (Phase 9).

Generates grounded, model-output-derived explanations for "Why does CyberSentinel
believe the attack is progressing?"

Explanations are computed entirely from the predicted next physical state S_hat_{t+1}
vs the current observed state S_t. No LLM-generated conclusions.

Output:
  - Top-K features by absolute change (ranked by |delta|)
  - Stage-specific feature relevance: which features are most indicative per attack stage
  - Human-readable narrative built from structured feature deltas only
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ml.state.state_builder import FEATURE_NAMES

# ---------------------------------------------------------------------------
# Stage-to-feature-relevance map
# Grounded in network security domain knowledge.
# Rationale documented inline.
# ---------------------------------------------------------------------------
STAGE_FEATURE_RELEVANCE: Dict[str, List[str]] = {
    "BENIGN": ["flow_count", "total_packets", "byte_rate", "dst_ip_entropy"],
    "RECONNAISSANCE": [
        "unique_dst_ports",   # scanning many ports
        "pkt_rate",           # rapid low-payload probe bursts
        "syn_count",          # half-open connection probes
        "syn_ratio",          # SYN-heavy traffic (nmap style)
        "dst_port_entropy",   # entropy spikes during wide port scan
        "dst_ip_entropy",     # scanning multiple targets
    ],
    "INITIAL_ACCESS": [
        "failed_flow_count",  # exploit attempts generating failed flows
        "failed_flow_ratio",
        "port_80_443_share",  # HTTP/HTTPS exploitation vectors
        "pkt_rate",
    ],
    "EXECUTION": [
        "byte_rate",          # payload delivery
        "total_bytes",
        "mean_flow_duration",
    ],
    "CREDENTIAL_ACCESS": [
        "failed_flow_count",   # brute-force generates many failed auth flows
        "failed_flow_ratio",
        "rst_count",           # RST-heavy brute-force traffic patterns
        "rst_ratio",
        "port_22_share",       # SSH brute-force
        "syn_ratio",
    ],
    "DISCOVERY": [
        "unique_dst_ips",
        "unique_dst_ports",
        "dst_ip_entropy",
        "dst_port_entropy",
    ],
    "LATERAL_MOVEMENT": [
        "port_445_share",     # SMB lateral movement
        "port_3389_share",    # RDP lateral movement
        "port_22_share",      # SSH lateral movement
        "unique_dst_ips",     # spreading to new hosts
        "tcp_flag_diversity", # different connection types across hosts
    ],
    "COMMAND_AND_CONTROL": [
        "byte_rate",          # C2 beacon regularity
        "mean_flow_duration", # long-lived C2 sessions
        "dst_ip_entropy",     # encrypted C2 tunneling
        "port_80_443_share",  # C2 over HTTP/S
    ],
    "EXFILTRATION": [
        "total_bytes",        # large volume outbound
        "byte_rate",
        "bytes_per_packet",   # large payload packets
        "mean_flow_duration",
    ],
    "UNKNOWN": ["flow_count", "total_packets"],
}


# ---------------------------------------------------------------------------
# Core explainability functions
# ---------------------------------------------------------------------------

def compute_feature_deltas(
    current_state: np.ndarray,
    predicted_state: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute per-feature absolute and relative changes between S_t and S_hat_{t+1}.

    Args:
        current_state:   (input_dim,) — current scaled state vector
        predicted_state: (input_dim,) — predicted next scaled state vector
        feature_names:   list of feature name strings

    Returns:
        List of dicts (one per feature), sorted by abs_change descending:
          {name, current, predicted, abs_change, rel_change_pct}
    """
    names = feature_names or FEATURE_NAMES
    assert len(current_state) == len(names), (
        f"State dim {len(current_state)} != feature count {len(names)}"
    )

    deltas = []
    for i, name in enumerate(names):
        cur = float(current_state[i])
        pred = float(predicted_state[i])
        abs_delta = abs(pred - cur)
        denom = abs(cur) + 1e-8
        rel_pct = (pred - cur) / denom * 100.0
        deltas.append({
            "feature": name,
            "current": round(cur, 4),
            "predicted": round(pred, 4),
            "abs_change": round(abs_delta, 4),
            "rel_change_pct": round(rel_pct, 2),
            "direction": "increase" if pred > cur else "decrease" if pred < cur else "stable",
        })

    # Sort by absolute change, descending
    deltas.sort(key=lambda d: d["abs_change"], reverse=True)
    return deltas


def get_top_features(
    deltas: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Return the top-K features by absolute change magnitude."""
    return deltas[:top_k]


def get_stage_relevant_features(
    deltas: List[Dict[str, Any]],
    predicted_stage: str,
) -> List[Dict[str, Any]]:
    """
    Filter feature deltas to those most relevant to the predicted attack stage.

    Returns only features that appear in STAGE_FEATURE_RELEVANCE for this stage,
    sorted by absolute change.
    """
    relevant_names = set(STAGE_FEATURE_RELEVANCE.get(predicted_stage, []))
    return [d for d in deltas if d["feature"] in relevant_names]


def build_explanation(
    current_state: np.ndarray,
    predicted_state: np.ndarray,
    current_stage: str,
    predicted_stage: str,
    top_k: int = 5,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a complete, grounded explanation for the forecast.

    Output structure:
      {
        "current_stage": str,
        "predicted_stage": str,
        "top_k_changed_features": List[...],   # top-k by abs delta (any stage)
        "stage_relevant_features": List[...],  # features relevant to predicted stage
        "narrative": str,                       # human-readable summary
        "provenance": str,                      # confirms explanation source
      }
    """
    names = feature_names or FEATURE_NAMES
    deltas = compute_feature_deltas(current_state, predicted_state, names)
    top_features = get_top_features(deltas, top_k)
    stage_features = get_stage_relevant_features(deltas, predicted_stage)

    # Build narrative strictly from feature deltas
    lines = []
    if current_stage != predicted_stage:
        lines.append(f"Stage transition predicted: {current_stage} → {predicted_stage}.")
    else:
        lines.append(f"Stage continuation predicted: {predicted_stage}.")

    if stage_features:
        lines.append(f"Key indicators for {predicted_stage}:")
        for f in stage_features[:3]:
            direction = f["direction"]
            lines.append(
                f"  • {f['feature']}: {f['current']:.3f} → {f['predicted']:.3f}"
                f" ({'+' if direction == 'increase' else '-'}{abs(f['rel_change_pct']):.1f}% {direction})"
            )

    if not stage_features:
        lines.append("Top changed features (no stage-specific indicators prominent):")
        for f in top_features[:3]:
            lines.append(
                f"  • {f['feature']}: {f['current']:.3f} → {f['predicted']:.3f}"
                f" ({f['rel_change_pct']:+.1f}%)"
            )

    return {
        "current_stage": current_stage,
        "predicted_stage": predicted_stage,
        "top_k_changed_features": top_features,
        "stage_relevant_features": stage_features,
        "narrative": " ".join(lines),
        "provenance": (
            "Explanation derived from CyberWorldModelV2 predicted physical state "
            "delta S_hat_{t+1} - S_t. No LLM-generated conclusions."
        ),
    }
