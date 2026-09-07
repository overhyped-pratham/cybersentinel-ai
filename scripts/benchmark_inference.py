"""
CyberSentinel AI — Inference Latency & Performance Benchmark (Phase 10).

Measures:
  1. Single forward pass latency (median, p95, p99 across 100 runs)
  2. K=4 step autoregressive rollout latency (median, p95, p99 across 50 runs)
  3. Feature attribution explainability latency
  4. Memory usage (peak RSS in MB)
  5. API round-trip latency (optional, if server is running)

Usage:
  python scripts/benchmark_inference.py
  python scripts/benchmark_inference.py --runs 200 --check-api
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import psutil
import torch

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.services.model_service import ModelService
from ml.world_model.world_model_v2 import INPUT_DIM


def get_memory_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_forward_and_rollout(runs: int = 100) -> Dict[str, any]:
    print("=" * 60)
    print("CYBERSENTINEL AI -- BENCHMARKING INFERENCE PERFORMANCE")
    print("=" * 60)

    # Initialize model service
    mem_before = get_memory_usage_mb()
    t0 = time.perf_counter()
    svc = ModelService.get_instance()
    init_time_s = time.perf_counter() - t0
    mem_after = get_memory_usage_mb()

    if not svc.is_loaded:
        print("ERROR: ModelService could not load or train CyberWorldModelV2.")
        return {"error": "Model not loaded"}

    print(f"[OK] ModelService loaded in {init_time_s:.2f}s | Memory: {mem_after:.1f} MB (Delta: +{mem_after - mem_before:.1f} MB)")

    # Synthetic realistic input (T=8, D=24)
    seq_len = 8
    x_seq = np.random.randn(seq_len, INPUT_DIM).astype(np.float32).tolist()
    mask = [True] * seq_len

    # Warmup
    print("\nWarming up JIT and PyTorch execution paths...")
    for _ in range(5):
        svc.forecast(x_seq=x_seq, mask_list=mask, k_steps=4)

    # Benchmark 1: Single Step Forecast + Full Pipeline (Forward + Explain + MITRE + Risk)
    print(f"\n[1/3] Benchmarking full 1-step forecast pipeline ({runs} runs)...")
    forecast_times: List[float] = []
    for _ in range(runs):
        t_start = time.perf_counter()
        _ = svc.forecast(x_seq=x_seq, mask_list=mask, k_steps=1)
        forecast_times.append((time.perf_counter() - t_start) * 1000.0)  # ms

    # Benchmark 2: K=4 Step Rollout Pipeline (Forward + K=4 Autoregressive + Explain + MITRE + Risk)
    k4_runs = max(20, runs // 2)
    print(f"[2/3] Benchmarking full K=4 rollout forecast pipeline ({k4_runs} runs)...")
    k4_times: List[float] = []
    for _ in range(k4_runs):
        t_start = time.perf_counter()
        _ = svc.forecast(x_seq=x_seq, mask_list=mask, k_steps=4)
        k4_times.append((time.perf_counter() - t_start) * 1000.0)  # ms

    # Benchmark 3: Raw PyTorch Forward Pass Only
    print(f"[3/4] Benchmarking pure PyTorch model forward pass ({runs} runs)...")
    raw_x = torch.randn(1, seq_len, INPUT_DIM).to(svc.trainer.device)
    raw_mask = torch.ones(1, seq_len, dtype=torch.bool).to(svc.trainer.device)
    raw_times: List[float] = []
    svc.trainer.model.eval()
    with torch.no_grad():
        for _ in range(runs):
            t_start = time.perf_counter()
            _ = svc.trainer.model(raw_x, raw_mask)
            raw_times.append((time.perf_counter() - t_start) * 1000.0)

    # Benchmark 4: Complete End-to-End Pipeline (Raw flows -> Builder -> Scaler -> Inference -> Explain -> MITRE -> Risk -> Agent)
    from network.flow.flow_record import FlowRecord
    from ml.state.state_builder import NetworkStateBuilder
    from ml.preprocessing.scaler import FeatureScaler
    from backend.agents.defensive_agent import CyberSentinelDefensiveAgent

    agent = CyberSentinelDefensiveAgent()
    builder = NetworkStateBuilder(window_size_seconds=30.0)
    scaler_path = _ROOT / "experiments" / "run_20260907_120029" / "world_model" / "scaler.pkl"
    scaler = FeatureScaler.load(scaler_path) if scaler_path.exists() else None

    # Sample batch of 50 flows for telemetry ingestion
    sample_flows = [
        FlowRecord(
            timestamp=float(i), src_ip="192.168.1.50", dst_ip="10.0.0.5",
            src_port=50000 + i, dst_port=80, protocol=6,
            packets=10, bytes=1500, duration=0.5,
            syn_flag=1, ack_flag=1, scenario_id="bench"
        ) for i in range(50)
    ]

    e2e_runs = max(15, runs // 3)
    print(f"[4/4] Benchmarking complete end-to-end telemetry pipeline ({e2e_runs} runs)...")
    e2e_times: List[float] = []
    for _ in range(e2e_runs):
        t_start = time.perf_counter()
        # 1. Telemetry -> StateBuilder
        df_states = builder.build_states(sample_flows)
        # 2. Scaler
        X_s = scaler.transform(df_states) if scaler else np.zeros((1, INPUT_DIM), dtype=np.float32)
        # Pad to sequence
        seq_input = [X_s[0].tolist()] * 8
        # 3. Model Inference + Explain + MITRE + Risk
        fc = svc.forecast(x_seq=seq_input, k_steps=2)
        # 4. Agent narrative
        _ = agent.answer(query="What is the current threat status?", current_forecast=fc)
        e2e_times.append((time.perf_counter() - t_start) * 1000.0)

    # Stats calculation
    def calc_stats(times: List[float]) -> Dict[str, float]:
        arr = np.array(times)
        return {
            "median_ms": float(np.median(arr)),
            "mean_ms": float(np.mean(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "throughput_hz": float(1000.0 / np.median(arr)),
        }

    stats_raw = calc_stats(raw_times)
    stats_forecast = calc_stats(forecast_times)
    stats_k4 = calc_stats(k4_times)
    stats_e2e = calc_stats(e2e_times)
    peak_mem = get_memory_usage_mb()

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Device: {svc.trainer.device}")
    print(f"Model Init Time: {init_time_s:.3f} s")
    print(f"Peak Memory RSS: {peak_mem:.1f} MB")
    print("-" * 60)
    print(f"1. Pure Neural Net Forward Pass:")
    print(f"   Median: {stats_raw['median_ms']:.2f} ms | P95: {stats_raw['p95_ms']:.2f} ms | P99: {stats_raw['p99_ms']:.2f} ms")
    print(f"   Throughput: {stats_raw['throughput_hz']:.1f} inferences/sec")
    print("-" * 60)
    print(f"2. Full 1-Step Pipeline (NN + State Predictor + MITRE + Risk + Explain):")
    print(f"   Median: {stats_forecast['median_ms']:.2f} ms | P95: {stats_forecast['p95_ms']:.2f} ms | P99: {stats_forecast['p99_ms']:.2f} ms")
    print(f"   Throughput: {stats_forecast['throughput_hz']:.1f} windows/sec")
    print("-" * 60)
    print(f"3. Full K=4 Autoregressive Rollout Pipeline:")
    print(f"   Median: {stats_k4['median_ms']:.2f} ms | P95: {stats_k4['p95_ms']:.2f} ms | P99: {stats_k4['p99_ms']:.2f} ms")
    print(f"   Throughput: {stats_k4['throughput_hz']:.1f} rollouts/sec")
    print("-" * 60)
    print(f"4. Complete End-to-End Pipeline (Telemetry -> State -> Model -> Risk -> Agent):")
    print(f"   Median: {stats_e2e['median_ms']:.2f} ms | P95: {stats_e2e['p95_ms']:.2f} ms | P99: {stats_e2e['p99_ms']:.2f} ms")
    print(f"   Throughput: {stats_e2e['throughput_hz']:.1f} full cycles/sec")
    print("=" * 60)

    res = {
        "device": str(svc.trainer.device),
        "model_init_time_s": init_time_s,
        "peak_memory_mb": peak_mem,
        "pure_nn_forward": stats_raw,
        "full_forecast_pipeline": stats_forecast,
        "full_k4_rollout_pipeline": stats_k4,
        "complete_end_to_end_pipeline": stats_e2e,
    }
    return res


def check_api_latency() -> None:
    try:
        import requests
        print("\nChecking live API latency on http://localhost:8000/api/v1/health ...")
        t0 = time.perf_counter()
        resp = requests.get("http://localhost:8000/api/v1/health", timeout=3.0)
        dur = (time.perf_counter() - t0) * 1000.0
        if resp.status_code == 200:
            print(f"[OK] API /health roundtrip: {dur:.2f} ms")
        else:
            print(f"[ERR] API returned status {resp.status_code}")
    except Exception as e:
        print(f"Note: API server not running locally on :8000 ({e}). Skipping API test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CyberSentinel AI Inference Benchmark")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--check-api", action="store_true", help="Check live API endpoint latency")
    parser.add_argument("--out", type=Path, default=None, help="Save benchmark JSON output")
    args = parser.parse_args()

    results = benchmark_forward_and_rollout(runs=args.runs)
    if args.check_api:
        check_api_latency()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved benchmark metrics to {args.out}")
