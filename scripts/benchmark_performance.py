"""
CyberSentinel X — Real-Time Performance & Latency Benchmark.

Measures actual system performance on the current machine:
  1. Feature Scaler Latency
  2. Layer 1: XGBoost Known Attack Classifier Latency
  3. Layer 1: SHAP TreeExplainer Attribution Latency
  4. Layer 2: PyTorch Autoencoder Novelty Detector Latency
  5. Layer 3: Risk Engine Evaluation Latency
  6. Layer 4: CyberWorldModelV2 Forward + K=4 Autoregressive Rollout Latency
  7. Full End-to-End Pipeline Latency per Event
  8. Single-Thread Throughput (Events / sec)
  9. Peak RSS Memory Usage (MB)

Saves results to artifacts/benchmarks/performance_report.json.
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.classifier.known_attack_classifier import KnownAttackClassifier
from ml.defense.risk_engine import ForecastEvent, RiskEngine
from ml.novelty.autoencoder_detector import AutoencoderNoveltyDetector
from ml.pipeline.detection_pipeline import DetectionPipeline
from ml.preprocessing.scaler import FeatureScaler
from ml.world_model.world_model_v2 import CyberWorldModelV2
import torch


def get_memory_mb():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return 0.0


def benchmark_all(num_trials: int = 100):
    print("=" * 65)
    print("      CYBERSENTINEL X — REAL-TIME PERFORMANCE BENCHMARK")
    print("=" * 65)

    pipeline = DetectionPipeline()
    sample_vec = [
        45.0, 2800.0, 3450000.0, 93.3, 115000.0, 1.0, 2.0, 2.0,
        45.0, 0.0, 45.0, 0.016, 0.0, 0.693, 0.693, 0.0, 0.0, 3.0,
        0.0, 0.0, 0.0, 1.0, 28.5, 1232.0,
    ]

    # Warmup
    for _ in range(10):
        pipeline.process_state(sample_vec, is_scaled=False)

    # 1. Feature Scaler
    scaler = pipeline.scaler
    t0 = time.perf_counter()
    import pandas as pd
    for _ in range(num_trials):
        df_tmp = pd.DataFrame([sample_vec], columns=pipeline.feature_names)
        _ = scaler.transform(df_tmp)[0]
    lat_scaler = (time.perf_counter() - t0) * 1000.0 / num_trials

    # Scaled state
    df_tmp = pd.DataFrame([sample_vec], columns=pipeline.feature_names)
    scaled_vec = scaler.transform(df_tmp)[0]

    # 2. Layer 1: XGBoost Classifier (predict_proba)
    clf = pipeline.classifier
    t0 = time.perf_counter()
    for _ in range(num_trials):
        _ = clf.predict_proba(scaled_vec)
    lat_clf = (time.perf_counter() - t0) * 1000.0 / num_trials

    # 3. Layer 1: SHAP Attribution
    t0 = time.perf_counter()
    shap_trials = min(20, num_trials)
    for _ in range(shap_trials):
        _ = clf.explain_instance(scaled_vec, target_class_idx=0, top_k=5)
    lat_shap = (time.perf_counter() - t0) * 1000.0 / shap_trials

    # 4. Layer 2: Autoencoder Novelty Detector
    ae = pipeline.novelty_detector
    t0 = time.perf_counter()
    for _ in range(num_trials):
        _ = ae.detect_single(scaled_vec)
    lat_ae = (time.perf_counter() - t0) * 1000.0 / num_trials

    # 5. Layer 3: Risk Engine
    re_eng = pipeline.risk_engine
    mock_event = ForecastEvent(
        timestamp="2026-09-18T10:00:00Z",
        model_version="2.0",
        horizon_seconds=30,
        current_stage="RECONNAISSANCE",
        current_state=list(scaled_vec),
        predicted_stage="CREDENTIAL_ACCESS",
        predicted_next_state=None,
        attack_probability=0.85,
        stage_probabilities={"RECONNAISSANCE": 0.1, "CREDENTIAL_ACCESS": 0.8},
        confidence=0.85,
        uncertainty_entropy=0.15,
        transition_detected=True,
        top_features=[],
        anomaly_score=0.75,
        is_novel=True,
    )
    t0 = time.perf_counter()
    for _ in range(num_trials):
        _ = re_eng.evaluate(mock_event, horizon_steps=1)
    lat_risk = (time.perf_counter() - t0) * 1000.0 / num_trials

    # 6. Layer 4: CyberWorldModel Forward + Rollout (K=4)
    wm = pipeline.world_model
    x_tensor = torch.from_numpy(np.tile(scaled_vec, (8, 1))).unsqueeze(0).float()
    mask = torch.ones((1, 8), dtype=torch.bool)
    t0 = time.perf_counter()
    wm_trials = min(25, num_trials)
    for _ in range(wm_trials):
        _ = wm.rollout(x_tensor, mask, k_steps=4)
    lat_wm = (time.perf_counter() - t0) * 1000.0 / wm_trials

    # 7. Full E2E Pipeline (Single State Processing)
    t0 = time.perf_counter()
    e2e_trials = min(30, num_trials)
    for _ in range(e2e_trials):
        _ = pipeline.process_state(sample_vec, is_scaled=False, k_steps=4)
    lat_e2e = (time.perf_counter() - t0) * 1000.0 / e2e_trials
    throughput_eps = round(1000.0 / lat_e2e, 1)

    # 8. Fast Path (Layers 1-3 without WorldModel K=4 rollout)
    pipeline_no_wm = DetectionPipeline(world_model=None)
    t0 = time.perf_counter()
    for _ in range(e2e_trials):
        _ = pipeline_no_wm.process_state(sample_vec, is_scaled=False)
    lat_fast_e2e = (time.perf_counter() - t0) * 1000.0 / e2e_trials
    fast_throughput_eps = round(1000.0 / lat_fast_e2e, 1)

    mem_mb = get_memory_mb()

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "torch_device": str(next(wm.parameters()).device) if wm else "cpu",
        "measurements": {
            "feature_scaler_latency_ms": round(lat_scaler, 3),
            "layer1_xgboost_latency_ms": round(lat_clf, 3),
            "layer1_shap_attribution_latency_ms": round(lat_shap, 3),
            "layer2_autoencoder_latency_ms": round(lat_ae, 3),
            "layer3_risk_engine_latency_ms": round(lat_risk, 3),
            "layer4_world_model_k4_rollout_latency_ms": round(lat_wm, 3),
            "full_e2e_pipeline_latency_ms": round(lat_e2e, 3),
            "full_e2e_throughput_events_per_sec": throughput_eps,
            "fast_path_latency_ms": round(lat_fast_e2e, 3),
            "fast_path_throughput_events_per_sec": fast_throughput_eps,
            "process_rss_memory_mb": mem_mb,
        }
    }

    print(f"  Feature Scaler Latency:                {lat_scaler:.3f} ms")
    print(f"  Layer 1 (XGBoost) Latency:             {lat_clf:.3f} ms")
    print(f"  Layer 1 (SHAP TreeExplainer) Latency:  {lat_shap:.3f} ms")
    print(f"  Layer 2 (PyTorch Autoencoder) Latency: {lat_ae:.3f} ms")
    print(f"  Layer 3 (Dynamic Risk Engine) Latency: {lat_risk:.3f} ms")
    print(f"  Layer 4 (WorldModel K=4 Rollout):      {lat_wm:.3f} ms")
    print(f"  Full 4-Layer E2E Pipeline Latency:     {lat_e2e:.3f} ms ({throughput_eps} events/sec)")
    print(f"  Fast-Path Detection Latency:           {lat_fast_e2e:.3f} ms ({fast_throughput_eps} events/sec)")
    print(f"  Process RSS Memory:                    {mem_mb:.1f} MB")
    print("=" * 65)

    out_dir = _ROOT / "artifacts" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "performance_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved performance report to {out_file}")

    return results


if __name__ == "__main__":
    benchmark_all()
