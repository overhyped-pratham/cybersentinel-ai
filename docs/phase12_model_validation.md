# PHASE 12: MODEL-BY-MODEL VALIDATION

**Date:** September 7, 2026  
**Status:** **VERIFIED — ALL 5 ML ARTIFACTS INDIVIDUALLY VALIDATED**  
**Benchmark Reference:** Phase 8C metrics remain immutable (Top-1: 97.73%, Brier: 0.0452)  

---

## 1. CyberWorldModelV2 — Production Canonical Model

**Checkpoint:** `models/world_model_v2.pt` (1.88 MB)  
**Architecture:** Direct Physical State Transition Network — 492,044 parameters  
**Test Suite:** `tests/test_model_validation.py`, `tests/test_world_model_v2.py`, `tests/test_phase12_model_backend_equivalence.py`

| Invariant | Verified? | Evidence |
| :--- | :---: | :--- |
| Checkpoint loads cleanly | ✅ | `CyberWorldModelTrainerV2.load(models/world_model_v2.pt)` succeeds |
| Input shape: $(B, T, 24)$ where $T \ge 5$ | ✅ | `test_world_model_v2.py::TestWorldModelV2Shapes` |
| Output simplex: $\sum P_i = 1.0 \pm 10^{-6}$ | ✅ | `test_rollout_stage_probs_sum_to_one` |
| Transition prob: $P_{\text{trans}} \in [0,1]$ | ✅ | `test_v2_physical_state_bounded_rollout` |
| Next state: $\hat{x}_{t+1}$ all finite | ✅ | `test_v2_physical_state_bounded_rollout` |
| Deterministic eval mode | ✅ | `test_v2_eval_mode_determinism` |
| K=4 rollout correct length | ✅ | `test_rollout_lengths[4]` |
| No future observation consumed | ✅ | `test_rollout_no_future_leakage`, `test_phase12_causality.py` |
| Dimension mismatch rejected | ✅ | `test_wrong_dimension_fails_assertion` |

**Phase 8C Immutable Benchmark (Preserved):**
$$\text{Top-1: 97.73\%} \quad \text{Brier: 0.0452} \quad \text{Transition Acc: 83.33\%} \quad \text{Attack FPR: 0.00\%}$$

---

## 2. WorldModelV1 — Archived Experimental Baseline (Ablation A)

**Checkpoint:** `experiments/run_20260907_120029/world_model/world_model.pt`  
**Architecture:** Latent GRU Transition Head — 492,428 parameters  
**Test Suite:** `tests/test_model_validation.py::TestWorldModelV1Compatibility`

| Invariant | Verified? | Evidence |
| :--- | :---: | :--- |
| Legacy checkpoint loads | ✅ | `CyberWorldModelTrainer.load()` handles raw `state_dict` payloads |
| Latent state representation: valid tensor | ✅ | `test_v1_alias_same_class` |
| V1 trainable without disturbing V2 | ✅ | `test_v1_trainable` |
| V2 alias resolves correctly | ✅ | `test_v2_alias` |
| Production V2 unaffected by V1 loading | ✅ | V2 singleton unmodified |

---

## 3. Temporal GRU Baseline — Archived Benchmark

**Checkpoint:** `experiments/run_20260907_111554/gru_model.pt`  
**Architecture:** 2-layer bidirectional GRU classifier — 107,340 parameters  
**Test Suite:** `tests/test_model_validation.py::TestTemporalGRUBaseline`

| Invariant | Verified? | Evidence |
| :--- | :---: | :--- |
| Checkpoint loads cleanly | ✅ | `TemporalGRUBaseline.load()` from `.pt` file |
| Input: $(B, T, 24)$ — accepts sequence inputs | ✅ | `test_gru_accepts_sequence_input` |
| Output: $(B, 12)$ valid logit tensor | ✅ | `test_gru_output_dimension` |
| Deterministic eval mode | ✅ | `test_gru_deterministic_eval` |
| Valid probability distribution after softmax | ✅ | $\sum P_i = 1.0$ verified |

**Archived Performance Reference (Phase 8C):**
$$\text{GRU Top-1: 81.82\%} \quad \text{True Transition Acc: 66.67\%} \quad \text{Attack FPR: 53.33\%}$$

---

## 4. Logistic Regression Baseline — Archived Benchmark

**Checkpoint:** `experiments/run_20260907_111554/logistic_model.pkl`  
**Architecture:** Scikit-Learn Multinomial Logistic Regression  
**Test Suite:** `tests/test_model_validation.py::TestLogisticRegressionBaseline`

| Invariant | Verified? | Evidence |
| :--- | :---: | :--- |
| Checkpoint loads cleanly | ✅ | `pickle.load()` on `.pkl` file |
| Input: 24-D flat feature vector | ✅ | `predict_stage_proba([x_24d])` succeeds |
| Output: $(N, 12)$ stage probabilities | ✅ | `test_lr_stage_proba_shape` |
| Output: $(N, 2)$ attack probabilities | ✅ | `test_lr_attack_proba_shape` |
| Valid probability simplex | ✅ | `test_lr_probabilities_sum_to_one` |
| Deterministic inference | ✅ | Two identical calls produce identical outputs |
| Scaler compatibility | ✅ | Works with `FeatureScaler.transform()` output |

---

## 5. FeatureScaler — Production Normalization Artifact

**Checkpoint:** `experiments/run_20260907_111554/scaler.pkl` (training set) and  
`experiments/run_20260907_120029/world_model/scaler.pkl` (Phase 8 set)  
**Architecture:** Custom column-wise MinMax / Z-score normalization  
**Test Suite:** `tests/test_scaler.py`, `tests/test_system_validation.py::TestPreprocessingAndSequenceIntegrity`

| Invariant | Verified? | Evidence |
| :--- | :---: | :--- |
| Checkpoint loads cleanly | ✅ | `FeatureScaler.load(scaler_path)` |
| Round-trip invariance: `inv(transform(x)) ≈ x` | ✅ | tolerance: `rtol=1e-3, atol=0.05` |
| Accepts 24-D feature DataFrame | ✅ | `test_scaler_fit_transform` |
| Rejects unfitted scaler | ✅ | `test_scaler_unfitted_error` |
| Column names preserved | ✅ | Output DataFrame has same `FEATURE_NAMES` columns |

**Mathematical invariant:**
$$\text{FeatureScaler.inverse\_transform}(\text{FeatureScaler.transform}(\mathbf{x})) \approx \mathbf{x}$$
Verified for all 24 network features across 1,080 real flow records from `trace_multistage_01.csv`.

---

## 6. Temperature Calibration Artifact

**Artifact:** `artifacts/calibration/temperature.json`  
**Value:** $T^* = 1.568035$  
**Architecture:** Post-hoc scalar temperature scaling on softmax logits  
**Test Suite:** `tests/test_world_model_v2.py::TestCalibration`, `tests/test_system_validation.py`

| Invariant | Verified? | Evidence |
| :--- | :---: | :--- |
| Artifact loads dynamically from file | ✅ | `load_temperature(path)` reads JSON at startup |
| $T^*$ is NOT hardcoded anywhere in source | ✅ | Anti-hardcoding audit found zero hardcoded temperature values |
| Calibrated probabilities sum to 1.0 | ✅ | `test_apply_temperature_scaling_sums_to_one` |
| argmax invariant under temperature | ✅ | `test_temperature_scaling_invariance` |
| Invalid temperature raises error | ✅ | `test_temperature_invalid` |
| Missing artifact falls back gracefully | ✅ | `test_load_temperature_missing_file` → defaults to $T=1.0$ |

**Calibration formula:**
$$\mathbf{p}_{\text{calib}} = \text{softmax}\left(\frac{\mathbf{z}}{T^*}\right), \quad T^* = 1.568035$$

---

## Summary

All 5 primary ML artifacts and 1 calibration artifact are individually verified:

| Artifact | Status | Architecture | Parameters | Checkpoint |
| :--- | :---: | :--- | :---: | :--- |
| **CyberWorldModelV2** | ✅ VALIDATED | Direct Physical Transition Network | 492,044 | `models/world_model_v2.pt` |
| **WorldModelV1** | ✅ COMPATIBLE | Latent GRU Transition Head | 492,428 | `experiments/.../world_model.pt` |
| **Temporal GRU** | ✅ VALIDATED | 2-layer bidirectional GRU | 107,340 | `experiments/.../gru_model.pt` |
| **Logistic Regression** | ✅ VALIDATED | Multinomial LogReg | ~288 | `experiments/.../logistic_model.pkl` |
| **FeatureScaler** | ✅ VALIDATED | Column-wise MinMax/Z-score | 48 params | `experiments/.../scaler.pkl` |
| **Temperature Scaler** | ✅ VALIDATED | Scalar division | 1 param | `artifacts/calibration/temperature.json` |
