# CyberSentinel AI — Cyber World Model V2

## Overview

CyberWorldModelV2 is the production architecture promoted from Phase 8C empirical investigation (Ablation B — Direct Physical State Transition).

**Core architectural innovation:** The transition operator acts in *physical network-state space* (S ∈ ℝ²⁴) rather than latent space. This prevents the identity-collapse failure mode observed in WorldModelV1 and grounds rollout predictions in physically interpretable feature deltas.

---

## Empirical Benchmark (Phase 8C Hard Multi-Stage Holdout)

> [!IMPORTANT]
> These numbers are **immutable** from `experiments/phase8c_investigation/phase8c_summary.json`. They must not be regenerated merely to improve presentation.

| Metric | WorldModelV2 | Temporal GRU | Logistic Reg |
|--------|-------------|-------------|-------------|
| Next-stage Top-1 | **97.73%** | 81.82% | 50.00% |
| Next-stage Top-3 | **100.0%** | 97.73% | 72.73% |
| True transition acc. | **83.33%** (5/6) | 66.67% (4/6) | 0.00% (0/6) |
| Brier (uncalibrated) | **0.0452** | 0.2913 | 0.7206 |
| Attack FPR | **0.00%** | 53.33% | 0.00% |
| K=4 path accuracy | 25.0% | N/A | N/A |

**Test split:** Hard Multi-Stage Holdout — `trace_multistage_03`, `trace_multistage_theta`, `trace_benign_beta`, `trace_recon_gamma` (N=44 sequences, 6 genuine stage transitions at indices [13, 16, 19, 24, 27, 30]).

---

## Architecture

```
S_1, S_2, ..., S_t   (historical scaled network state vectors, ℝ²⁴)
        ↓
NetworkStateEncoder   (MLP: input_dim → hidden_dim)
        ↓
TemporalTransformer   (causal multi-head self-attention, 2 layers)
        ↓
h_t                   (context representation, ℝ¹²⁸)
        ↓                              ↓
PhysicalNextStatePredictor      ClassificationHead
(MLP: hidden → hidden → ℝ²⁴)   → P(current_stage)
        ↓
S_hat_{t+1}           (predicted next state, physical feature space, ℝ²⁴)
        ↓
NetworkStateEncoder   (re-encode predicted state)
        ↓
h_hat_{t+1}           (latent representation of predicted next state)
        ↓                              ↓
ClassificationHead              ClassificationHead
→ P(next_stage)                 → P(attack)
```

### V1 vs V2 Comparison

| Component | WorldModelV1 | WorldModelV2 |
|-----------|-------------|-------------|
| Transition operator | `TransitionHead`: h_t → Δh (latent residual) | `PhysicalNextStatePredictor`: h_t → S_hat_{t+1} |
| Transition space | Latent (hidden_dim=128) | Physical feature space (input_dim=24) |
| State loss target | `‖h_hat - h_true‖²` (latent MSE) | `‖S_hat - S_true‖²` (physical MSE) |
| Physical interpretability | None | Direct feature deltas |
| Transition collapse risk | High (identity-dominant training) | Eliminated |

### Key: Why Physical Space?

The latent MSE transition loss in V1 caused the `TransitionHead` to learn an approximate identity mapping on training data dominated by persistent states (same-stage majority). The network learned to predict "stay in current stage" in latent space, giving correct predictions for persisting stages but completely missing genuine transitions.

Physical MSE forces the model to predict actual feature changes (e.g., `syn_count`, `failed_flow_count`), which are causally related to attack progression and non-trivially different across stage transitions.

---

## Multi-task Loss (V2)

```
L = λ_stage    · CE(ŷ_current, y_current)
  + λ_attack   · BCE(P_attack, y_attack)
  + λ_state    · MSE(S_hat_{t+1}, S_true_{t+1})
  + λ_next_stage · CE(ŷ_next, y_next)
```

**Default weights from Phase 8C:** `λ_stage=1.0, λ_attack=1.0, λ_state=1.0, λ_next_stage=2.0`

> [!NOTE]
> `λ_next_stage=2.0` (double weight) because transition accuracy was the primary weakness of V1 and the core scientific objective of Phase 8C.

The V1 latent transition MSE term `‖h_hat - h_true‖²` is **absent** from V2 loss. This is intentional: it was the direct cause of the identity-collapse failure.

---

## Autoregressive Rollout

K-step rollout consumes **no future observations**:

```
Step 0: x_seq → Encoder → Transformer → h_0
Step 1: h_0 → PhysicalNextStatePredictor → S_hat_1 → Encoder → h_1
Step 2: h_1 → PhysicalNextStatePredictor → S_hat_2 → Encoder → h_2
...
Step K: h_{K-1} → PhysicalNextStatePredictor → S_hat_K → Encoder → h_K
        → P(stage_K), P(attack_K)
```

Physical bound enforcement (`forward_bounded`) is applied at each rollout step:
- Count/rate/duration features: `Softplus(·)` (smooth non-negative)
- Ratio/share features: `Sigmoid(·)` → [0, 1]
- Entropy features: `Softplus(·)`

> [!NOTE]
> Bounds are applied **only during rollout inference**, not during training, to preserve gradient continuity.

### Rollout Safety Monitoring

Every `RolloutResult` includes `safety_flags`:
- `nan_detected`: True if any predicted state contained NaN/Inf
- `collapse_detected`: True if normalized entropy < 0.05 (near-deterministic prediction)
- `feature_dim_valid`: True if predicted state shape matches `input_dim=24`

---

## Calibration

Temperature scaling is applied to next-stage logits:

```
P_calibrated(stage) = softmax(logits / T*)
```

**T* = 1.5680** fitted on validation scenarios (`trace_multistage_01`, `trace_benign_alpha`).

Stored in: `artifacts/calibration/temperature.json`

> [!IMPORTANT]
> Temperature is **loaded from the artifact file** — it must NOT be hard-coded in source code. Load via `CyberWorldModelTrainerV2.apply_calibration_from_artifact()` or `ml.calibration.temperature_scaling.load_temperature()`.

Temperature scaling preserves argmax invariance (argmax before/after calibration identical). Verified in `tests/test_world_model_v2.py::TestCalibration::test_temperature_scaling_invariance`.

---

## Explainability

Feature attributions are computed from the **predicted physical state delta**:

```
Δ_i = |S_hat_{t+1}[i] - S_t[i]|   for feature i ∈ {0..23}
```

Features are ranked by `|Δ_i|` (absolute change in scaled feature space) and filtered by stage-specific relevance lists defined in `ml/world_model/explainability.py::STAGE_FEATURE_RELEVANCE`.

> [!CAUTION]
> Explanations are derived **entirely from model predicted state changes**. No LLM-generated conclusions are permitted. The `provenance` field in every explanation output confirms this.

---

## MITRE ATT&CK Mapping

Deterministic static lookup from `mitre/mappings/mitre_mapper.py`. Each attack stage maps to verified MITRE ATT&CK Enterprise v14 techniques:

| Stage | Primary Technique |
|-------|------------------|
| RECONNAISSANCE | T1595 — Active Scanning |
| INITIAL_ACCESS | T1190 — Exploit Public-Facing Application |
| CREDENTIAL_ACCESS | T1110 — Brute Force |
| LATERAL_MOVEMENT | T1021 — Remote Services |
| COMMAND_AND_CONTROL | T1071 — Application Layer Protocol |
| EXFILTRATION | T1048 — Exfiltration Over Alternative Protocol |

> [!CAUTION]
> Technique IDs must **never** be generated by an LLM. Every technique in the mapper includes `mapping_provenance = "MITRE ATT&CK Enterprise v14 static mapping"`.

---

## Risk Engine

```
risk_score = 100 × (
    0.40 × P(attack)
  + 0.35 × severity(predicted_next_stage)
  + 0.15 × calibrated_confidence
  + 0.10 × urgency(horizon_k)
)
```

**Priority thresholds:** CRITICAL ≥ 75 | HIGH ≥ 50 | MEDIUM ≥ 30 | LOW < 30

**Stage severity scores** (from `configs/default_config.yaml`):

| Stage | Severity |
|-------|---------|
| BENIGN | 0.00 |
| RECONNAISSANCE | 0.25 |
| INITIAL_ACCESS | 0.50 |
| CREDENTIAL_ACCESS | 0.75 |
| LATERAL_MOVEMENT | 0.85 |
| C2 | 0.90 |
| EXFILTRATION | 1.00 |

---

## File Inventory

| File | Purpose |
|------|---------|
| [`ml/world_model/world_model_v2.py`](file:///d:/uec%20sih/ml/world_model/world_model_v2.py) | CyberWorldModelV2 architecture, loss, trainer |
| [`ml/calibration/temperature_scaling.py`](file:///d:/uec%20sih/ml/calibration/temperature_scaling.py) | Temperature loading, ECE, Brier |
| [`ml/world_model/explainability.py`](file:///d:/uec%20sih/ml/world_model/explainability.py) | Physical state delta attribution |
| [`ml/defense/risk_engine.py`](file:///d:/uec%20sih/ml/defense/risk_engine.py) | ForecastEvent, RiskEngine, RiskAssessment |
| [`mitre/mappings/mitre_mapper.py`](file:///d:/uec%20sih/mitre/mappings/mitre_mapper.py) | Deterministic MITRE ATT&CK lookup |
| [`agent/tools/forecasting_tools.py`](file:///d:/uec%20sih/agent/tools/forecasting_tools.py) | Structured agent tool interface |
| [`scripts/replay_scenario.py`](file:///d:/uec%20sih/scripts/replay_scenario.py) | Window-by-window trace replay |
| [`tests/test_world_model_v2.py`](file:///d:/uec%20sih/tests/test_world_model_v2.py) | V2 model tests (29 tests) |
| [`tests/test_defensive_intelligence.py`](file:///d:/uec%20sih/tests/test_defensive_intelligence.py) | Defensive layer tests (47 tests) |
| [`artifacts/calibration/temperature.json`](file:///d:/uec%20sih/artifacts/calibration/temperature.json) | Validated T* = 1.5680 |

**WorldModelV1 is preserved unchanged at:**
- Source: `ml/world_model/cyber_world_model.py` — `CyberWorldModel` class (lines 228–354)
- Checkpoint: `experiments/phase8c_investigation/world_model_v1.pt`
- Alias: `WorldModelV1 = CyberWorldModel` (importable from `ml.world_model.world_model_v2`)

---

## V2 WorldModelV1 Backwards Compatibility

```python
# V1 still importable — UNCHANGED
from ml.world_model.cyber_world_model import CyberWorldModel, CyberWorldModelTrainer

# V1 aliases from V2 module (for convenience)
from ml.world_model.world_model_v2 import WorldModelV1  # == CyberWorldModel

# V2
from ml.world_model.world_model_v2 import CyberWorldModelV2, WorldModelV2
from ml.world_model.world_model_v2 import CyberWorldModelTrainerV2
```

---

## Usage Example

```python
from ml.world_model.world_model_v2 import CyberWorldModelTrainerV2
from ml.defense.risk_engine import ForecastEvent, RiskEngine
from agent.tools.forecasting_tools import CyberSentinelAgentTools

# Load checkpoint
trainer = CyberWorldModelTrainerV2.load("models/world_model_v2.pt")
trainer.apply_calibration_from_artifact("artifacts/calibration/temperature.json")

# Predict next stage (calibrated)
p_next = trainer.predict_next_stage_proba(x_seq, mask)   # (N, 10)

# K=4 autoregressive rollout
rollout = trainer.rollout(x_seq, mask, k_steps=4)
# rollout.predicted_stages[k] : (N,) stage index at step k
# rollout.safety_flags         : {"nan_detected": False, ...}

# Risk scoring
engine = RiskEngine()
assessment = engine.evaluate(event, horizon_steps=1)
print(f"Risk: {assessment.risk_score:.1f} | Priority: {assessment.severity}")

# Agent tools
tools = CyberSentinelAgentTools()
forecast = tools.get_attack_forecast(event)     # JSON-serializable dict
mitre = tools.get_mitre_mapping(event)          # deterministic MITRE lookup
risk = tools.get_risk_assessment(event)         # structured risk output
```
