"""
CyberSentinel AI — Phase 14 Multi-Host Experiments Runner.

Validates the full dynamic pipeline on 5 multi-host traffic patterns:
  1. normal_background
  2. connection_burst
  3. repeated_attempts
  4. port_diversity
  5. large_data_transfer

Outputs:
  experiments/phase14/multihost_traffic_results.json
"""

import asyncio
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from network.flow.flow_record import FlowRecord
from scripts.multi_host_traffic_generator import generate_pattern_flows, SECONDARY_LAPTOP_IP, CYBERSENTINEL_HOST_IP
from ml.state.state_builder import NetworkStateBuilder, FEATURE_NAMES
from backend.services.model_service import ModelService
from network.telemetry.stream_processor import TelemetryWindowEvent
from backend.services.live_ingest_service import LiveIngestService
from backend.agents.defensive_agent import CyberSentinelDefensiveAgent

_OUTPUT_DIR = _ROOT / "experiments" / "phase14"
_OUTPUT_FILE = _OUTPUT_DIR / "multihost_traffic_results.json"


async def run_pattern_experiment(
    pattern: str,
    model_svc: ModelService,
    agent: CyberSentinelDefensiveAgent,
    num_windows: int = 5,
    window_seconds: float = 30.0,
) -> Dict[str, Any]:
    """Runs a single controlled traffic pattern through the live ingest pipeline."""
    live_svc = LiveIngestService()
    session_id = f"exp_{pattern}_{int(time.time())}"
    q = live_svc.subscribe()

    base_ts = 1000.0
    pattern_events: List[Dict[str, Any]] = []

    for w_idx in range(num_windows):
        w_start = base_ts + (w_idx * window_seconds)
        w_end = w_start + window_seconds

        flows = generate_pattern_flows(
            pattern,
            base_timestamp=w_start,
            duration_seconds=window_seconds,
            source_ip=SECONDARY_LAPTOP_IP,
            target_ip=CYBERSENTINEL_HOST_IP,
            scenario_id=session_id,
        )

        win = TelemetryWindowEvent(
            window_id=f"win_{pattern}_{w_idx}",
            window_start=w_start,
            window_end=w_end,
            flows=flows,
            flow_count=len(flows),
            dropped_malformed=0,
            source_id=f"MultiHostGenerator({SECONDARY_LAPTOP_IP})",
        )

        await live_svc._process_window(win, session_id, model_svc, k_steps=4)

        # Collect event from queue
        try:
            evt = q.get_nowait()
            if evt.get("status") == "FORECAST":
                pattern_events.append(evt)
        except asyncio.QueueEmpty:
            pass

    live_svc.unsubscribe(q)

    if not pattern_events:
        raise RuntimeError(f"No forecast generated for pattern {pattern}")

    last_event = pattern_events[-1]

    # Query defensive agent for an analyst narrative grounded in the forecast event
    agent_resp = agent.answer(
        query="Provide threat summary and priority recommendation for this window.",
        current_forecast=last_event,
        session_id=session_id,
    )

    # Compute raw physical 24-D state for this window
    df_state = NetworkStateBuilder().build_states(flows, scenario_id=session_id, base_timestamp=w_start)
    raw_24d = {name: float(df_state[name].iloc[-1]) for name in FEATURE_NAMES}

    return {
        "pattern": pattern,
        "source_ip": SECONDARY_LAPTOP_IP,
        "target_ip": CYBERSENTINEL_HOST_IP,
        "window_count": num_windows,
        "total_flows_last_window": len(flows),
        "total_packets_last_window": sum(f.packets for f in flows),
        "total_bytes_last_window": sum(f.bytes for f in flows),
        "physical_state_24d": raw_24d,
        "model_forecast": {
            "current_stage": last_event["current_stage"],
            "predicted_next_stage": last_event["predicted_next_stage"],
            "attack_probability": round(last_event["attack_probability"], 4),
            "confidence": round(last_event["confidence"], 4),
            "transition_detected": last_event["transition_detected"],
            "transition_probability": round(last_event.get("transition_probability") or 0.0, 4),
            "risk_score": round(last_event["risk_score"], 2),
            "risk_level": last_event["risk_level"],
            "recommended_priority": last_event["recommended_priority"],
            "rollout_k4_stages": [r["predicted_stage"] for r in last_event.get("rollout_steps", [])],
        },
        "top_feature_deltas": last_event.get("top_features", [])[:4],
        "mitre_techniques": last_event.get("mitre_techniques", [])[:3],
        "agent_response": {
            "answer": agent_resp["answer"],
            "backend": agent_resp.get("llm_backend", "template"),
            "grounded": agent_resp.get("grounded_in_model_output", True),
        },
    }


async def main():
    print("=" * 70)
    print("CYBERSENTINEL AI — PHASE 14 MULTI-HOST DYNAMIC BEHAVIOR EXPERIMENT")
    print("=" * 70)

    model_svc = ModelService.get_instance()
    agent = CyberSentinelDefensiveAgent()
    print(f"[OK] ModelService loaded (temperature={model_svc.temperature:.4f})")

    patterns = [
        "normal_background",
        "connection_burst",
        "repeated_attempts",
        "port_diversity",
        "large_data_transfer",
    ]

    results: Dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_host": SECONDARY_LAPTOP_IP,
            "monitored_host": CYBERSENTINEL_HOST_IP,
            "model_version": "CyberWorldModelV2",
            "device": "cpu",
            "test_environment": "Isolated LAN Multi-Host Emulation",
        },
        "experiments": {},
    }

    print("\nExecuting controlled traffic patterns through the live ML pipeline...\n")

    for pat in patterns:
        print(f"[*] Testing Pattern: {pat.upper()}...")
        exp_res = await run_pattern_experiment(pat, model_svc, agent)
        results["experiments"][pat] = exp_res
        fc = exp_res["model_forecast"]
        print(f"    Current Stage     : {fc['current_stage']}")
        print(f"    Predicted Stage   : {fc['predicted_next_stage']}")
        print(f"    Attack Probability: {fc['attack_probability']:.4f}")
        print(f"    Confidence        : {fc['confidence']:.4f}")
        print(f"    Risk Score        : {fc['risk_score']} ({fc['risk_level']})")
        print(f"    K=4 Rollout       : {' -> '.join(fc['rollout_k4_stages'])}")
        print(f"    Top Feature Delta : {exp_res['top_feature_deltas'][0]['feature']} "
              f"({exp_res['top_feature_deltas'][0]['direction']} {exp_res['top_feature_deltas'][0]['abs_change']:.2f})")
        print()

    # Save to json
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Results successfully written to {_OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
