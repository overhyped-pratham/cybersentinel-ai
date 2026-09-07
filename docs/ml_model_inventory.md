# CyberSentinel AI — ML Model Inventory

**Phase 11 Comprehensive Verification**  
**Date:** 2026-09-07  
**Audit Scope:** Repository-wide model artifacts, checkpoints, scalers, and architectures.

---

## 1. Verified Model & Checkpoint Registry

| Model Name | Source Code | Checkpoint Path | Input Shape | Output Shape | Purpose | Checkpoint Size | Verification Status |
|---|---|---|---|---|---|---|---|
| **CyberWorldModelV2** | [`ml/world_model/world_model_v2.py`](file:///d:/uec%20sih/ml/world_model/world_model_v2.py) | `models/world_model_v2.pt` | $(B, T, 24)$ | Stage: $(B, 10)$<br>Attack: $(B,)$<br>Next State: $(B, 24)$ | Primary production world model. Autoregressive physical state transition. | 1.88 MB | **VERIFIED** |
| **WorldModelV1** | [`ml/world_model/cyber_world_model.py`](file:///d:/uec%20sih/ml/world_model/cyber_world_model.py) | `experiments/phase8c_investigation/world_model_v1.pt` | $(B, T, 24)$ | Stage: $(B, 10)$<br>Attack: $(B,)$<br>Next State: $(B, 24)$ | Legacy baseline from Phase 8. Uses latent residual TransitionHead. | 2.15 MB | **VERIFIED** |
| **WorldModelV1 (Phase 8)** | [`ml/world_model/cyber_world_model.py`](file:///d:/uec%20sih/ml/world_model/cyber_world_model.py) | `experiments/run_20260907_120029/world_model/world_model.pt` | $(B, T, 24)$ | Stage: $(B, 10)$<br>Attack: $(B,)$<br>Next State: $(B, 24)$ | Initial Phase 8 training checkpoint. | 2.16 MB | **VERIFIED** |
| **Temporal GRU** | [`ml/temporal/gru_baseline.py`](file:///d:/uec%20sih/ml/temporal/gru_baseline.py) | `experiments/run_20260907_111554/gru_model.pt` | $(B, T, 24)$ | Stage: $(B, 10)$<br>Attack: $(B,)$ | Sequential baseline. Unidirectional causal GRU. | 232 KB | **VERIFIED** |
| **Logistic Regression** | [`ml/baseline/logistic_regression.py`](file:///d:/uec%20sih/ml/baseline/logistic_regression.py) | `experiments/run_20260907_111554/logistic_model.pkl` | $(N, 24)$ | Stage: $(N, 10)$<br>Attack: $(N,)$ | Static point-in-time baseline (SIH mandated). | 3.68 KB | **VERIFIED** |
| **FeatureScaler (Robust)** | [`ml/preprocessing/scaler.py`](file:///d:/uec%20sih/ml/preprocessing/scaler.py) | `experiments/run_20260907_111554/scaler.pkl` | $(N, 24)$ raw | $(N, 24)$ scaled | Feature normalization using median and IQR. | 1.13 KB | **VERIFIED** |
| **FeatureScaler (WM)** | [`ml/preprocessing/scaler.py`](file:///d:/uec%20sih/ml/preprocessing/scaler.py) | `experiments/run_20260907_120029/world_model/scaler.pkl` | $(N, 24)$ raw | $(N, 24)$ scaled | Scaler fitted on Phase 8 training split. | 1.13 KB | **VERIFIED** |
| **Temperature Calibration** | [`ml/calibration/temperature_scaling.py`](file:///d:/uec%20sih/ml/calibration/temperature_scaling.py) | `artifacts/calibration/temperature.json` | Logits $(N, 10)$ | Calibrated Probs $(N, 10)$ | Post-hoc temperature scaling ($T^* = 1.568035$). | 496 B | **VERIFIED** |

---

## 2. Checkpoint Details & Parameter Counts

| Model | Total Parameters | Trainable Parameters | Architecture Summary |
|---|---|---|---|
| **CyberWorldModelV2** | 468,763 | 468,763 | `NetworkStateEncoder` (24->128), `TemporalTransformer` (2 layers, 4 heads, causal mask), `PhysicalNextStatePredictor` (128->128->24), 3x `ClassificationHead` (128->10, 128->1, 128->10). |
| **WorldModelV1** | 534,811 | 534,811 | `NetworkStateEncoder` (24->128), `TemporalTransformer` (2 layers, 4 heads), `TransitionHead` (latent MLP residual 128->128->128), `NextStateHead` (128->24), 2x `ClassificationHead`. |
| **Temporal GRU** | 53,899 | 53,899 | Linear projection (24->64), 2-layer unidirectional `GRU` (hidden_dim=64), `LayerNorm`, 2x classification heads (64->10, 64->1). |
| **Logistic Regression** | ~250 weights | ~250 weights | 3 independent `LogisticRegression(solver='lbfgs')` models for attack detection, current stage, and next stage. |

---

## 3. Training vs Inference Pipeline Verification

```
Raw NetFlow CSVs / Telemetry Streams
         │
         ▼
[NetworkStateBuilder] window_size = 30.0s (Fixed 24 features)
         │
         ▼
[FeatureScaler] RobustScaler (center=median, scale=IQR)
         │
         ▼
[SequenceBuilder] sequence_length = 8, causal padding = True
         │
         ▼
  Tensors: (B, 8, 24) scaled float32
         │
    ┌────┴─────────────────────────────┐
    ▼                                  ▼
[CyberWorldModelV2]            [Baselines (GRU / LR)]
 - Pred current stage: (B, 10)  - Pred next stage: (B, 10)
 - Pred next state S_hat: (B, 24) - Pred attack prob: (B,)
 - Pred next stage: (B, 10)
 - Pred attack prob: (B,)
 - K=4 Autoregressive Rollout
```
