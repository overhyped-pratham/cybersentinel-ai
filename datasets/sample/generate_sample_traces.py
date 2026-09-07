"""
CyberSentinel AI - Synthetic Multi-Scenario Trace Generator.

Generates realistic, physically-grounded network flow traces across distinct
attack stages and benign background activity for offline development and testing.
"""

from pathlib import Path
from typing import List, Optional
import random
import pandas as pd
import numpy as np


def generate_scenario_flows(
    scenario_id: str,
    scenario_type: str = "benign",
    base_timestamp: float = 1680000000.0,
    duration_seconds: float = 300.0,
    flow_frequency_hz: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthesizes a realistic sequence of flow events for a given attack or benign scenario.
    
    Args:
        scenario_id: Unique trace string identifier.
        scenario_type: One of 'benign', 'reconnaissance', 'credential_access',
                       'lateral_movement', 'exfiltration', 'multi_stage'.
        base_timestamp: Starting epoch seconds.
        duration_seconds: Total duration of the scenario trace in seconds.
        flow_frequency_hz: Approximate flows generated per second.
        seed: Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Clean flow table.
    """
    rng = np.random.default_rng(seed)
    n_flows = int(duration_seconds * flow_frequency_hz)
    timestamps = np.sort(rng.uniform(base_timestamp, base_timestamp + duration_seconds, size=n_flows))

    records = []

    internal_subnet = "192.168.1."
    external_ips = [f"203.0.113.{i}" for i in range(10, 30)]
    attacker_ip = "198.51.100.55"
    target_server = "192.168.1.50"
    domain_controller = "192.168.1.10"

    for i in range(n_flows):
        ts = float(timestamps[i])
        progress = (ts - base_timestamp) / duration_seconds

        # Determine current dynamic stage for multi_stage scenario
        curr_type = scenario_type
        if scenario_type == "multi_stage":
            if progress < 0.25:
                curr_type = "benign"
            elif progress < 0.50:
                curr_type = "reconnaissance"
            elif progress < 0.75:
                curr_type = "credential_access"
            else:
                curr_type = "lateral_movement"

        # Generate traffic according to scenario type
        if curr_type == "benign":
            src_ip = f"{internal_subnet}{rng.integers(100, 200)}"
            dst_ip = rng.choice(external_ips)
            src_port = int(rng.integers(49152, 65535))
            dst_port = int(rng.choice([80, 443, 8080, 53]))
            protocol = 6 if dst_port != 53 else 17
            packets = int(rng.integers(5, 30))
            bytes_count = packets * int(rng.integers(60, 400))
            duration = float(rng.uniform(0.1, 5.0))
            syn_flag = 1 if protocol == 6 else 0
            rst_flag = 1 if rng.uniform() < 0.02 else 0
            fin_flag = 1 if protocol == 6 and rst_flag == 0 else 0
            ack_flag = 1 if protocol == 6 else 0
            psh_flag = 1 if protocol == 6 and rng.uniform() < 0.3 else 0
            failed = (rst_flag == 1)
            label = "BENIGN"

        elif curr_type == "reconnaissance":
            # Attacker port-scans multiple ports on target
            src_ip = attacker_ip
            dst_ip = target_server
            src_port = int(rng.integers(40000, 60000))
            dst_port = int(rng.integers(20, 10000)) # wide port sweep
            protocol = 6
            packets = int(rng.choice([1, 2]))
            bytes_count = packets * 60
            duration = float(rng.uniform(0.01, 0.2))
            syn_flag = 1
            rst_flag = 1 if rng.uniform() < 0.7 else 0 # closed ports send RST
            fin_flag = 0
            ack_flag = 0
            psh_flag = 0
            failed = (rst_flag == 1)
            label = "PortScan"

        elif curr_type == "credential_access":
            # Brute force attack against SSH or FTP service
            src_ip = attacker_ip
            dst_ip = target_server
            src_port = int(rng.integers(40000, 60000))
            dst_port = int(rng.choice([22, 21]))
            protocol = 6
            packets = int(rng.integers(8, 15))
            bytes_count = packets * int(rng.integers(70, 120))
            duration = float(rng.uniform(0.5, 2.0))
            syn_flag = 1
            # high failure rate due to bad passwords
            rst_flag = 1 if rng.uniform() < 0.85 else 0
            fin_flag = 1 if rst_flag == 0 else 0
            ack_flag = 1
            psh_flag = 1
            failed = (rst_flag == 1)
            label = "SSH-Patator" if dst_port == 22 else "FTP-Patator"

        elif curr_type == "lateral_movement":
            # Compromised target moves laterally to Domain Controller via SMB (445)
            src_ip = target_server
            dst_ip = domain_controller
            src_port = int(rng.integers(49152, 65535))
            dst_port = int(rng.choice([445, 139, 3389]))
            protocol = 6
            packets = int(rng.integers(20, 80))
            bytes_count = packets * int(rng.integers(150, 600))
            duration = float(rng.uniform(1.0, 10.0))
            syn_flag = 1
            rst_flag = 0
            fin_flag = 1
            ack_flag = 1
            psh_flag = 1
            failed = False
            label = "Infiltration"

        elif curr_type == "exfiltration":
            # Large volume outbound data transfer
            src_ip = target_server
            dst_ip = attacker_ip
            src_port = int(rng.integers(49152, 65535))
            dst_port = 443
            protocol = 6
            packets = int(rng.integers(200, 800))
            bytes_count = packets * int(rng.integers(900, 1400)) # heavy payload
            duration = float(rng.uniform(10.0, 28.0))
            syn_flag = 1
            rst_flag = 0
            fin_flag = 1
            ack_flag = 1
            psh_flag = 1
            failed = False
            label = "Exfiltration"

        else:
            raise ValueError(f"Unknown scenario type: {curr_type}")

        records.append({
            "timestamp": ts,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "packets": packets,
            "bytes": bytes_count,
            "duration": duration,
            "syn_flag": syn_flag,
            "rst_flag": rst_flag,
            "fin_flag": fin_flag,
            "ack_flag": ack_flag,
            "psh_flag": psh_flag,
            "urg_flag": 0,
            "failed": failed,
            "scenario_id": scenario_id,
            "label": label,
        })

    return pd.DataFrame(records)


def generate_sample_dataset(output_dir: Path) -> List[Path]:
    """Generates a bundle of CSV scenarios for offline validation and testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    scenarios = [
        ("trace_benign_alpha", "benign", 100),
        ("trace_benign_beta", "benign", 101),
        ("trace_recon_gamma", "reconnaissance", 102),
        ("trace_bruteforce_delta", "credential_access", 103),
        ("trace_lateral_epsilon", "lateral_movement", 104),
        ("trace_exfil_zeta", "exfiltration", 105),
        ("trace_multistage_theta", "multi_stage", 106),
        ("trace_multistage_iota", "multi_stage", 107),
    ]

    for sc_id, sc_type, seed in scenarios:
        df = generate_scenario_flows(
            scenario_id=sc_id,
            scenario_type=sc_type,
            base_timestamp=1700000000.0 + seed * 1000,
            duration_seconds=360.0, # 12 windows of 30 seconds
            flow_frequency_hz=3.0,
            seed=seed,
        )
        csv_path = output_dir / f"{sc_id}.csv"
        df.to_csv(csv_path, index=False)
        generated_files.append(csv_path)

    return generated_files


if __name__ == "__main__":
    out = Path("datasets/sample")
    files = generate_sample_dataset(out)
    print(f"Generated {len(files)} sample trace CSVs in {out.resolve()}")
