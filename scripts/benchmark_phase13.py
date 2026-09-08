"""
CyberSentinel AI — Phase 13 Performance & Memory Benchmark.

Empirically measures:
  1. Telemetry Ingestion Latency (Flow parsing / ingestion)
  2. 24-D Feature State Extraction Latency (NetworkStateBuilder)
  3. Feature Scaling Latency (FeatureScaler)
  4. CyberWorldModelV2 Inference Latency (Forward + State Predictor)
  5. K=4 Autoregressive Rollout Latency
  6. Complete SOC Forecast Latency (Model + Explainability + MITRE + Risk)
  7. Streaming Dispatch Latency (LiveIngestService window processing)
  8. Long-running Memory Stability (RSS across 1,000 streaming windows)

Usage:
  python scripts/benchmark_phase13.py [--runs 100]
"""

import argparse
import asyncio
import gc
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import psutil

# Ensure repo root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from network.flow.flow_record import FlowRecord
from network.telemetry.sources import ReplaySource
from ml.state.state_builder import NetworkStateBuilder, FEATURE_NAMES
from ml.preprocessing.scaler import FeatureScaler
from backend.services.model_service import ModelService
from network.telemetry.stream_processor import TelemetryWindowEvent
from backend.services.live_ingest_service import LiveIngestService

_CSV_SAMPLE = _ROOT / "datasets" / "sample" / "trace_multistage_01.csv"


def get_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def make_window_flows(n_flows: int = 50, base_ts: float = 0.0) -> List[FlowRecord]:
    flows = []
    for i in range(n_flows):
        flows.append(FlowRecord(
            timestamp=base_ts + (i * 0.5),
            src_ip=f"192.168.1.{10 + (i % 5)}",
            dst_ip=f"10.0.0.{20 + (i % 3)}",
            src_port=1024 + i,
            dst_port=80 if i % 2 == 0 else 443,
            protocol=6,
            packets=10 + (i % 20),
            bytes=500 + (i * 50),
            duration=0.5,
            syn_flag=1 if i % 10 == 0 else 0,
            rst_flag=0,
            fin_flag=0,
            ack_flag=1,
            psh_flag=1 if i % 3 == 0 else 0,
            urg_flag=0,
            failed=False,
            scenario_id="benchmark",
        ))
    return flows


async def run_benchmark(n_runs: int = 100):
    print("=" * 65)
    print("CYBERSENTINEL AI — PHASE 13 PERFORMANCE & MEMORY BENCHMARK")
    print("=" * 65)

    mem_start = get_memory_mb()
    print(f"Initial Memory RSS: {mem_start:.2f} MB")

    # 1. Warm up components
    print("\nWarming up ModelService and StateBuilder...")
    model_svc = ModelService.get_instance()
    state_builder = NetworkStateBuilder()
    warmup_flows = make_window_flows(50, 0.0)
    warmup_df = state_builder.build_states(warmup_flows, scenario_id="warmup", base_timestamp=0.0)
    raw_vec = warmup_df[FEATURE_NAMES].iloc[-1].to_numpy(dtype=np.float32)
    warmup_seq = [raw_vec.tolist() for _ in range(5)]
    _ = model_svc.forecast(warmup_seq, k_steps=4)
    print(f"Warmed up. Model loaded={model_svc.is_loaded}. Memory: {get_memory_mb():.2f} MB")

    # Benchmark 1: Feature Extraction Latency
    print(f"\n[1/7] Benchmarking 24-D State Extraction ({n_runs} runs)...")
    extraction_times = []
    for i in range(n_runs):
        flows = make_window_flows(60, float(i * 30))
        t0 = time.perf_counter()
        df = state_builder.build_states(flows, scenario_id="bench", base_timestamp=float(i * 30))
        _ = df[FEATURE_NAMES].iloc[-1].to_numpy(dtype=np.float32)
        extraction_times.append((time.perf_counter() - t0) * 1000)

    # Benchmark 2: Feature Normalization Latency
    print(f"[2/7] Benchmarking Feature Scaling ({n_runs} runs)...")
    scaler = FeatureScaler(scaler_type="robust").fit(warmup_df)
    scaling_times = []
    sample_df = warmup_df
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = scaler.transform(sample_df)
        scaling_times.append((time.perf_counter() - t0) * 1000)

    # Benchmark 3: Model 1-Step Inference Latency
    print(f"[3/7] Benchmarking Model Forward + Next State ({n_runs} runs)...")
    model_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model_svc.forecast(warmup_seq, k_steps=1)
        model_times.append((time.perf_counter() - t0) * 1000)

    # Benchmark 4: K=4 Rollout Latency
    print(f"[4/7] Benchmarking K=4 Autoregressive Rollout ({n_runs} runs)...")
    rollout_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model_svc.forecast(warmup_seq, k_steps=4)
        rollout_times.append((time.perf_counter() - t0) * 1000)

    # Benchmark 5: Complete Live Pipeline (Window -> Model -> Explain -> Risk -> Event)
    print(f"[5/7] Benchmarking Complete Live Window Pipeline ({n_runs} runs)...")
    live_svc = LiveIngestService()
    # Preload 5 windows into buffer so every step does a full forecast
    for i in range(5):
        w = TelemetryWindowEvent(
            window_id=f"warm_{i}",
            window_start=float(i * 30),
            window_end=float((i + 1) * 30),
            flows=make_window_flows(50, float(i * 30)),
            flow_count=50,
            dropped_malformed=0,
            source_id="bench",
        )
        await live_svc._process_window(w, "bench_session", model_svc, k_steps=4)

    pipeline_times = []
    for i in range(n_runs):
        w = TelemetryWindowEvent(
            window_id=f"win_{i}",
            window_start=float((i + 5) * 30),
            window_end=float((i + 6) * 30),
            flows=make_window_flows(50, float((i + 5) * 30)),
            flow_count=50,
            dropped_malformed=0,
            source_id="bench",
        )
        t0 = time.perf_counter()
        await live_svc._process_window(w, "bench_session", model_svc, k_steps=4)
        pipeline_times.append((time.perf_counter() - t0) * 1000)

    # Benchmark 6: Telemetry Ingestion Latency (Replay CSV reading)
    print(f"[6/7] Benchmarking Replay Telemetry Ingestion...")
    source = ReplaySource(_CSV_SAMPLE, realtime_factor=0.0)
    t0 = time.perf_counter()
    count = 0
    async for _ in source.stream():
        count += 1
    ingest_time_total = (time.perf_counter() - t0) * 1000
    ingest_per_flow_us = (ingest_time_total / count) * 1000 if count else 0

    # Benchmark 7: Continuous Long-Running Memory Stability (1,000 windows)
    print(f"\n[7/7] Benchmarking Long-Running Memory Stability (1,000 streaming windows)...")
    mem_checkpoints = []
    long_live_svc = LiveIngestService()
    mem_checkpoints.append(("Start (w=0)", get_memory_mb()))

    for w_idx in range(1, 1001):
        win = TelemetryWindowEvent(
            window_id=f"long_win_{w_idx}",
            window_start=float(w_idx * 30),
            window_end=float((w_idx + 1) * 30),
            flows=make_window_flows(40, float(w_idx * 30)),
            flow_count=40,
            dropped_malformed=0,
            source_id="long_bench",
        )
        await long_live_svc._process_window(win, "long_session", model_svc, k_steps=2)

        if w_idx in (100, 250, 500, 750, 1000):
            mem = get_memory_mb()
            mem_checkpoints.append((f"Window {w_idx}", mem))
            print(f"  -> Window {w_idx:4d}: RSS = {mem:.2f} MB")

    mem_end = get_memory_mb()
    mem_delta = mem_end - mem_checkpoints[0][1]

    # Print summary statistics
    def stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    s_extract = stats(extraction_times)
    s_scale = stats(scaling_times)
    s_model = stats(model_times)
    s_rollout = stats(rollout_times)
    s_pipe = stats(pipeline_times)

    print("\n" + "=" * 65)
    print("PHASE 13 BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Telemetry Ingestion ({count} flows):")
    print(f"  Total time: {ingest_time_total:.2f} ms | Per flow: {ingest_per_flow_us:.2f} µs | Rate: {count / (ingest_time_total/1000):.1f} flows/sec")
    print("-" * 65)
    print("24-D State Extraction (NetworkStateBuilder):")
    print(f"  Median: {s_extract['median']:.2f} ms | P95: {s_extract['p95']:.2f} ms | P99: {s_extract['p99']:.2f} ms")
    print("-" * 65)
    print("Feature Scaling (FeatureScaler):")
    print(f"  Median: {s_scale['median']:.2f} ms | P95: {s_scale['p95']:.2f} ms | P99: {s_scale['p99']:.2f} ms")
    print("-" * 65)
    print("1-Step Forecast (CyberWorldModelV2 + Explain + Risk + MITRE):")
    print(f"  Median: {s_model['median']:.2f} ms | P95: {s_model['p95']:.2f} ms | P99: {s_model['p99']:.2f} ms")
    print("-" * 65)
    print("K=4 Autoregressive Threat Rollout:")
    print(f"  Median: {s_rollout['median']:.2f} ms | P95: {s_rollout['p95']:.2f} ms | P99: {s_rollout['p99']:.2f} ms")
    print("-" * 65)
    print("Complete End-to-End Live Window Ingestion Pipeline:")
    print(f"  Median: {s_pipe['median']:.2f} ms | P95: {s_pipe['p95']:.2f} ms | P99: {s_pipe['p99']:.2f} ms")
    print(f"  Throughput: {1000.0 / s_pipe['median']:.1f} windows/sec")
    print("-" * 65)
    print("Memory Stability (1,000 Continuous Ingested Windows):")
    for label, mem_val in mem_checkpoints:
        print(f"  {label:20s}: {mem_val:.2f} MB")
    print(f"  Total Memory Delta: {mem_delta:+.2f} MB over 1,000 windows")
    print("=" * 65)

    return {
        "extraction": s_extract,
        "scaling": s_scale,
        "model_1step": s_model,
        "rollout_k4": s_rollout,
        "pipeline": s_pipe,
        "memory_delta_mb": mem_delta,
        "mem_checkpoints": mem_checkpoints,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.runs))
