# PHASE 12: NO-HARDCODING AUDIT

**Date:** September 7, 2026  
**Status:** **CLEAN — ZERO HARDCODED INTELLIGENCE**  
**Methodology:** Static grep scan + manual review of every suspicious line in `backend/`, `ml/`, `agent/`, `mitre/`, `network/`, `frontend/`  
**Pattern Coverage:** 10 anti-hardcoding patterns scanned  

---

## Executive Summary

A comprehensive codebase grep for 10 categories of potential hardcoded intelligence was performed. Every finding was manually reviewed. **Zero instances of hardcoded cybersecurity intelligence were identified.** All flagged patterns were verified as legitimate model architecture definitions, mathematical computation formulas, standard data routing logic, or schema definitions.

---

## Scan Results Table

| Pattern | Occurrences | Category | Verdict |
| :--- | :---: | :--- | :--- |
| `if stage ==` | 1 | Binary label encoding (0 = BENIGN, 1 = attack) | **Legitimate** |
| `if scenario` | 3 | Data routing / null-check / filtering | **Legitimate** |
| `next_stage =` | 6 | Neural layer declaration in model architecture | **Legitimate** |
| `attack_probability = 0` | 0 | — | **CLEAN** |
| `risk_score =` | 3 | Mathematical formula computation | **Legitimate** |
| `prediction = {` | 0 | — | **CLEAN** |
| `fake_` | 0 | — | **CLEAN** |
| `demo_data` | 0 | — | **CLEAN** |
| `= 0.99` | 0 | — | **CLEAN** |
| `static_prob` | 0 | — | **CLEAN** |

---

## Detailed Finding Review

### Finding 1: `if stage == "BENIGN"` — `ml/preprocessing/stage_labeler.py:111`

```python
is_attack = 0 if stage == "BENIGN" else 1
```

**Classification:** **Legitimate binary label encoder.**  
**Reasoning:** This is a preprocessing-time binary encoding of the string stage label into an integer (0 = benign, 1 = attack). It is used only during training data construction, not during inference. The stage string itself is never hardcoded to a prediction — it is derived from the ground-truth CSV label column.

---

### Finding 2: `if scenario` hits — 3 occurrences

1. `backend/services/replay_service.py:129: if scenario_df.empty:` — Null check for empty DataFrame. Legitimate guard.
2. `ml/state/state_builder.py:102: sc_id = scenario_id if scenario_id is not None else f.scenario_id` — Falls back to `FlowRecord.scenario_id` field when no override is provided. Legitimate routing.
3. `network/flow/csv_loader.py:277: cleaned["scenario_id"] = scenario_name if scenario_name else self.default_scenario` — Sets a metadata field for bookkeeping. Not used in inference.

**Classification:** **All legitimate data routing and null-safety guards.**

---

### Finding 3: `next_stage =` hits — 6 occurrences

All 6 hits are neural module declarations or training-time variable names inside `ml/world_model/`:

1. `ml/temporal/gru_baseline.py:64: self.head_next_stage = nn.Sequential(...)` — Linear head for next-stage classification.
2. `ml/world_model/cyber_world_model.py:264: self.head_next_stage = ClassificationHead(...)` — Idem.
3. `ml/world_model/cyber_world_model.py:363: L_next_stage = CrossEntropy(next_stage_logits, y_next_stage)` — Loss computation variable.
4. `ml/world_model/cyber_world_model.py:501: bnext_stage = bnext_stage.to(self.device)` — Tensor device transfer.
5. `ml/world_model/world_model_v2.py:233: self.head_next_stage = ClassificationHead(...)` — Same architecture pattern.
6. `ml/world_model/world_model_v2.py:579: bnext_stage = bnext_stage.to(self.device)` — Same device transfer.

**Classification:** **All legitimate model architecture layer declarations or training-time variables.**

---

### Finding 4: `risk_score =` hits — 3 occurrences

All 3 hits are inside `ml/defense/risk_engine.py`:

1. Line 8: `risk_score = 100 * (...)` — Mathematical formula definition in module docstring/example.
2. Line 215: `risk_score = 100 * (w1 * p_attack + w2 * p_transition * criticality + w3 * delta_norm)` — Core risk computation formula.
3. Line 254: `risk_score = min(100.0, max(0.0, raw_score * 100.0))` — Clamping to [0, 100] range.

**Classification:** **All legitimate mathematical runtime computation.** Every value of `risk_score` is dynamically computed from the model's `attack_probability`, `transition_confidence`, and physical state delta norm, not from a static constant.

---

## Intelligence Provenance Table

| Intelligence Output | Runtime Source | Hardcoded? | Evidence |
| :--- | :--- | :--- | :--- |
| **Current Stage** | `CyberWorldModelV2.logits_current_stage` → argmax → `STAGE_TAXONOMY[idx]` | **NO** | [`model_service.py:262-273`](file:///d:/uec%20sih/backend/services/model_service.py#L262-L273) |
| **Next Stage** | `CyberWorldModelV2.logits_next_stage` → calibrated softmax → argmax | **NO** | [`model_service.py:258-262`](file:///d:/uec%20sih/backend/services/model_service.py#L258-L262) |
| **Attack Probability** | `CyberWorldModelV2.logits_attack_prob` → sigmoid | **NO** | [`model_service.py:260`](file:///d:/uec%20sih/backend/services/model_service.py#L260) |
| **Confidence** | `max(softmax(logits_next_stage / T*))` | **NO** | [`model_service.py:264`](file:///d:/uec%20sih/backend/services/model_service.py#L264) |
| **Risk Score** | `RiskEngine.evaluate(ForecastEvent)` → mathematical formula | **NO** | [`risk_engine.py:215`](file:///d:/uec%20sih/ml/defense/risk_engine.py#L215) |
| **Feature Attribution** | `pred_next_state − current_state` → top-k by `|Δx_i|` | **NO** | [`model_service.py:277-280`](file:///d:/uec%20sih/backend/services/model_service.py#L277-L280) |
| **MITRE Context** | `get_mitre_summary(predicted_stage)` → official ATT&CK v14 taxonomy | **NO (standard ref)** | [`mitre_mapper.py`](file:///d:/uec%20sih/mitre/mappings/mitre_mapper.py) |
| **Agent Narrative** | `_template_answer(route, fc, query)` reads directly from `ForecastEvent` fields | **NO** | [`defensive_agent.py:99-238`](file:///d:/uec%20sih/backend/agents/defensive_agent.py#L99-L238) |
| **Rollout Trajectory** | `CyberWorldModelTrainerV2.rollout()` → autoregressive PyTorch inference | **NO** | [`model_service.py:308`](file:///d:/uec%20sih/backend/services/model_service.py#L308) |
| **Transition Detected** | `predicted_next_stage != current_stage` | **NO** | [`model_service.py:275`](file:///d:/uec%20sih/backend/services/model_service.py#L275) |

---

## Additional Structural Verification

- **Dashboard (`frontend/dashboard.py`):** Static HTML templates contain only format strings `{value}` and empty placeholder cells. No initial demo values, canned tables, or pre-filled risk scores.
- **Replay Engine (`backend/services/replay_service.py`):** Every replay step executes `ModelService.forecast()` on the real buffered window state vectors. Predictions are never pre-cached or replayed from storage.
- **MITRE Mapper (`mitre/mappings/mitre_mapper.py`):** The mapping from stage name → technique catalog is a static reference table encoding the MITRE ATT&CK Enterprise v14 knowledge base. This is a **legitimately static standard reference**, not ML intelligence.

---

## Verdict

**ZERO HARDCODED INTELLIGENCE FOUND.**  
All dynamic cybersecurity predictions (stage, probability, risk, feature, narrative) are produced exclusively by the live ML inference pipeline at runtime.
