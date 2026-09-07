"""
CyberSentinel AI — Phase 12 Real Telemetry & Perturbation Sensitivity Experiments.

Executes:
1. Full end-to-end run on genuine CSV telemetry (trace_multistage_01.csv)
   saving complete execution trace to experiments/phase12/real_telemetry_run.json.
2. Controlled telemetry-level perturbation sensitivity test
   saving comparisons to experiments/phase12/telemetry_sensitivity.json.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Workspace path setup
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch

from network.flow.csv_loader import CSVFlowLoader
from network.flow.flow_record import FlowRecord
from ml.state.state_builder import NetworkStateBuilder, FEATURE_NAMES
from ml.preprocessing.scaler import FeatureScaler
from backend.services.model_service import ModelService
from backend.agents.defensive_agent import CyberSentinelDefensiveAgent


def run_real_telemetry_test(output_dir: Path) -> Dict[str, Any]:
    print("\n--- [Step 2] Executing Real Telemetry Test on trace_multistage_01.csv ---")
    trace_path = ROOT_DIR / "datasets" / "sample" / "trace_multistage_01.csv"
    assert trace_path.exists(), f"Trace file missing: {trace_path}"

    loader = CSVFlowLoader()
    flows = loader.load_flows(trace_path)
    print(f"Loaded {len(flows)} raw flow records from {trace_path.name}")

    # Build 30-second states
    builder = NetworkStateBuilder(window_size_seconds=30.0)
    df_states = builder.build_states(flows)
    print(f"Extracted {len(df_states)} 30-second physical network state windows")

    # Load scaler
    scaler_path = ROOT_DIR / "experiments" / "run_20260907_111554" / "scaler.pkl"
    scaler = FeatureScaler.load(scaler_path)
    X_scaled = scaler.transform(df_states)

    # Initialize ModelService and Agent
    model_service = ModelService.get_instance()
    agent = CyberSentinelDefensiveAgent()

    # Step through windows causal buffer
    history_scaled = []
    run_records = []

    # Sequence length requirement is at least 5 windows
    for t_idx in range(len(df_states)):
        history_scaled.append(X_scaled[t_idx].tolist())
        if len(history_scaled) < 5:
            continue

        # Use recent 8 windows (or up to available)
        seq_input = history_scaled[-8:]

        t_start = time.perf_counter()
        fc = model_service.forecast(x_seq=seq_input, k_steps=4)
        inf_time_ms = (time.perf_counter() - t_start) * 1000.0

        # Query agent
        agent_out = agent.answer(
            query="Summarize current threat level and expected stage transition.",
            current_forecast=fc
        )

        record = {
            "window_index": t_idx,
            "window_start_time": float(df_states.index[t_idx]),
            "observed_physical_state": {k: float(df_states[k].iloc[t_idx]) for k in FEATURE_NAMES[:6]},
            "current_stage": fc["current_stage"],
            "predicted_next_stage": fc["predicted_next_stage"],
            "attack_probability": float(fc["attack_probability"]),
            "confidence": float(fc["confidence"]),
            "transition_detected": bool(fc["transition_detected"]),
            "risk_score": float(fc["risk_score"]),
            "risk_level": fc["risk_level"],
            "inference_time_ms": float(inf_time_ms),
            "top_features": fc.get("top_features", [])[:3],
            "mitre_techniques": [t.get("technique_id") for t in fc.get("mitre_techniques", [])[:3]],
            "rollout_k4_stages": [step.get("predicted_next_stage") for step in fc.get("rollout_steps", [])],
            "agent_summary": agent_out["answer"]
        }
        run_records.append(record)

    output_payload = {
        "metadata": {
            "source_trace": trace_path.name,
            "flow_count": len(flows),
            "window_count": len(df_states),
            "evaluated_windows": len(run_records),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_version": "CyberWorldModelV2",
            "device": "cpu"
        },
        "windows": run_records
    }

    out_file = output_dir / "real_telemetry_run.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Saved real telemetry run to {out_file} ({len(run_records)} windows)")
    return output_payload


def run_telemetry_perturbation_test(output_dir: Path) -> Dict[str, Any]:
    print("\n--- [Step 3] Executing Telemetry-Level Perturbation Sensitivity Test ---")
    model_service = ModelService.get_instance()
    builder = NetworkStateBuilder(window_size_seconds=30.0)
    scaler_path = ROOT_DIR / "experiments" / "run_20260907_111554" / "scaler.pkl"
    scaler = FeatureScaler.load(scaler_path)

    # Base telemetry window: 100 normal benign HTTP/DNS flows over 30 seconds
    base_flows: List[FlowRecord] = []
    base_ts = 1700000000.0
    for i in range(100):
        base_flows.append(FlowRecord(
            timestamp=base_ts + (i * 0.25),
            src_ip=f"192.168.1.{10 + (i % 20)}",
            dst_ip="10.0.0.1",
            src_port=40000 + i,
            dst_port=80 if i % 2 == 0 else 443,
            protocol=6,
            packets=10,
            bytes=1500,
            duration=0.2,
            syn_flag=1,
            ack_flag=1,
            failed=False,
            scenario_id="baseline"
        ))

    # Helper to evaluate flow list
    def evaluate_flows(flow_list: List[FlowRecord]) -> Dict[str, Any]:
        df_state = builder.build_states(flow_list, scenario_id="test")
        x_s = scaler.transform(df_state)
        # Use the state vector from the perturbed window
        target_vec = x_s[-1].tolist()
        seq_input = [target_vec] * 8
        fc = model_service.forecast(x_seq=seq_input, k_steps=2)
        return {
            "attack_probability": float(fc["attack_probability"]),
            "confidence": float(fc["confidence"]),
            "predicted_next_stage": fc["predicted_next_stage"],
            "transition_detected": bool(fc["transition_detected"]),
            "risk_score": float(fc["risk_score"]),
            "risk_level": fc["risk_level"],
            "top_features": fc.get("top_features", [])[:3],
            "predicted_physical_state_sample": {
                k: float(fc["predicted_physical_state"][k])
                for k in ["flow_rate", "syn_ratio", "rst_ratio", "byte_rate"]
                if k in fc.get("predicted_physical_state", {})
            }
        }

    # 1. Baseline
    res_baseline = evaluate_flows(base_flows)

    # 2. Perturbation 1: SYN Flood (500 SYN packets with short duration)
    p1_flows = list(base_flows)
    for i in range(500):
        p1_flows.append(FlowRecord(
            timestamp=base_ts + (i * 0.05),
            src_ip=f"192.168.1.{100 + (i % 50)}",
            dst_ip="10.0.0.1",
            src_port=50000 + i,
            dst_port=80,
            protocol=6,
            packets=1,
            bytes=60,
            duration=0.001,
            syn_flag=1,
            ack_flag=0,
            failed=True,
            scenario_id="syn_flood"
        ))
    res_p1 = evaluate_flows(p1_flows)

    # 3. Perturbation 2: Brute Force SSH (200 failed auth flows on port 22 with RST)
    p2_flows = list(base_flows)
    for i in range(200):
        p2_flows.append(FlowRecord(
            timestamp=base_ts + (i * 0.1),
            src_ip="192.168.1.99",
            dst_ip="10.0.0.5",
            src_port=45000 + i,
            dst_port=22,
            protocol=6,
            packets=8,
            bytes=1200,
            duration=0.05,
            syn_flag=1,
            rst_flag=1,
            ack_flag=1,
            failed=True,
            scenario_id="ssh_bruteforce"
        ))
    res_p2 = evaluate_flows(p2_flows)

    # 4. Perturbation 3: Data Exfiltration (Massive byte surge on high port)
    p3_flows = list(base_flows)
    for i in range(20):
        p3_flows.append(FlowRecord(
            timestamp=base_ts + (i * 1.2),
            src_ip="192.168.1.50",
            dst_ip="198.51.100.77",
            src_port=49999,
            dst_port=8080,
            protocol=6,
            packets=15000,
            bytes=20000000,  # 20 MB per flow
            duration=5.0,
            syn_flag=0,
            ack_flag=1,
            failed=False,
            scenario_id="exfiltration"
        ))
    res_p3 = evaluate_flows(p3_flows)

    # 5. Perturbation 4: Port Scan Reconnaissance (150 unique destination ports)
    p4_flows = list(base_flows)
    for i in range(150):
        p4_flows.append(FlowRecord(
            timestamp=base_ts + (i * 0.15),
            src_ip="192.168.1.88",
            dst_ip="10.0.0.10",
            src_port=55000,
            dst_port=1000 + i,
            protocol=6,
            packets=1,
            bytes=44,
            duration=0.002,
            syn_flag=1,
            ack_flag=0,
            failed=True,
            scenario_id="port_scan"
        ))
    res_p4 = evaluate_flows(p4_flows)

    # 6. Perturbation 5: Packet Size Skew / Fragment Anomaly
    p5_flows = list(base_flows)
    for i in range(100):
        p5_flows.append(FlowRecord(
            timestamp=base_ts + (i * 0.25),
            src_ip="192.168.1.77",
            dst_ip="10.0.0.1",
            src_port=42000 + i,
            dst_port=443,
            protocol=6,
            packets=5000,
            bytes=20000,  # Very small packets (4 bytes/packet)
            duration=0.5,
            syn_flag=0,
            ack_flag=1,
            failed=False,
            scenario_id="packet_skew"
        ))
    res_p5 = evaluate_flows(p5_flows)

    perturbation_results = {
        "baseline": {
            "description": "Standard background HTTP/HTTPS traffic (100 flows)",
            "metrics": res_baseline
        },
        "perturbation_1_syn_flood": {
            "description": "Injected 500 SYN packets with zero ACK and high failure rate",
            "metrics": res_p1,
            "delta_risk": res_p1["risk_score"] - res_baseline["risk_score"],
            "delta_attack_prob": res_p1["attack_probability"] - res_baseline["attack_probability"]
        },
        "perturbation_2_failed_ssh_bruteforce": {
            "description": "Injected 200 failed SSH flows on port 22 with RST flags",
            "metrics": res_p2,
            "delta_risk": res_p2["risk_score"] - res_baseline["risk_score"],
            "delta_attack_prob": res_p2["attack_probability"] - res_baseline["attack_probability"]
        },
        "perturbation_3_exfiltration_surge": {
            "description": "Injected 20 massive flows totaling 400 MB to external IP",
            "metrics": res_p3,
            "delta_risk": res_p3["risk_score"] - res_baseline["risk_score"],
            "delta_attack_prob": res_p3["attack_probability"] - res_baseline["attack_probability"]
        },
        "perturbation_4_port_scan_recon": {
            "description": "Injected sweeps against 150 unique destination ports",
            "metrics": res_p4,
            "delta_risk": res_p4["risk_score"] - res_baseline["risk_score"],
            "delta_attack_prob": res_p4["attack_probability"] - res_baseline["attack_probability"]
        },
        "perturbation_5_packet_size_skew": {
            "description": "Injected extreme packet-to-byte ratio skew",
            "metrics": res_p5,
            "delta_risk": res_p5["risk_score"] - res_baseline["risk_score"],
            "delta_attack_prob": res_p5["attack_probability"] - res_baseline["attack_probability"]
        }
    }

    out_file = output_dir / "telemetry_sensitivity.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(perturbation_results, f, indent=2)
    print(f"Saved telemetry sensitivity experiments to {out_file}")
    return perturbation_results


if __name__ == "__main__":
    out_dir = ROOT_DIR / "experiments" / "phase12"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_real_telemetry_test(out_dir)
    run_telemetry_perturbation_test(out_dir)
    print("\nPhase 12 Real Telemetry & Perturbation Tests Complete.")
