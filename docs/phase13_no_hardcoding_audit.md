# PHASE 13: NO-HARDCODING AUDIT & PROVENANCE REPORT

**Date:** September 8, 2026  
**Status:** **CLEAN — ZERO HARDCODED INTELLIGENCE**  
**Audit Scope:** Ingestion adapters, stream processor, live ingest service, WebSocket endpoints, frontend dashboard  

---

## 1. Executive Summary

A repository-wide anti-hardcoding audit was executed across all new and existing components in Phase 13. Ten categories of potential hardcoding were inspected via automated regex search and manual AST inspection. **Zero instances of hardcoded predictions, transition lookup tables, fixed probabilities, or scenario-specific intelligence were found.**

---

## 2. Automated Grep Pattern Results

| Pattern | Occurrences | Classification | Status |
| :--- | :---: | :--- | :---: |
| `if stage ==` | 1 | Binary label encoding in training preprocessing (`ml/preprocessing/stage_labeler.py:111`) | **Legitimate** |
| `if scenario` | 3 | Data routing, null checks, and dataset loading guards | **Legitimate** |
| `next_stage =` | 6 | Neural network layer declarations (`ClassificationHead`) | **Legitimate** |
| `attack_probability = 0` | 0 | Hardcoded probability literal | **CLEAN** |
| `risk_score =` | 3 | Mathematical formula evaluation (`ml/defense/risk_engine.py`) | **Legitimate** |
| `prediction = {` | 0 | Hardcoded prediction dictionary | **CLEAN** |
| `fake_` | 0 | Fake/mock data marker | **CLEAN** |
| `demo_data` | 0 | Demo placeholder marker | **CLEAN** |
| `= 0.99` | 0 | Fixed high probability constant | **CLEAN** |
| `static_prob` | 0 | Static probability variable | **CLEAN** |

---

## 3. Dynamic Intelligence Provenance Table

Every cybersecurity intelligence field displayed to the analyst originates exclusively from real, runtime ML and algorithmic calculation:

| Intelligence Output | Exact Runtime Source | Method / Layer | Hardcoded? | Verification Test |
| :--- | :--- | :--- | :---: | :--- |
| **Current Stage** | Model output: `logits_current_stage` | $\text{argmax}(\mathbf{z}_{\text{cur}})$ mapped to `STAGE_TAXONOMY` | **NO** | `test_live_ingest_accumulates_and_infers` |
| **Predicted Next Stage** | Model output: `logits_next_stage` | $\text{argmax}(\text{softmax}(\mathbf{z}_{\text{next}} / T^*))$ | **NO** | `test_live_ingest_accumulates_and_infers` |
| **Attack Probability** | Model output: `logits_attack_prob` | $\sigma(z_{\text{atk}}) = \frac{1}{1 + e^{-z_{\text{atk}}}}$ | **NO** | `test_live_ingest_accumulates_and_infers` |
| **Forecast Confidence** | Softmax distribution | $\max_i \left(\text{softmax}(\mathbf{z}_{\text{next}} / T^*)_i\right)$ | **NO** | `test_live_ingest_accumulates_and_infers` |
| **Physical State $\hat{S}_{t+1}$** | Predictor: `PhysicalNextStatePredictor` | $\hat{S}_{t+1} = S_t + \Delta S(h_t)$ | **NO** | `test_v2_physical_state_bounded_rollout` |
| **K-Step Simulation** | Rollout: `trainer.rollout(k_steps=4)` | Autoregressive recurrence on predicted $\hat{S}_{t+k}$ | **NO** | `test_rollout_k4_stages` |
| **Feature Attribution** | Physical delta norm | Top-$k$ sorted by $|\hat{S}_{t+1, i} - S_{t, i}|$ | **NO** | `test_explainability_derived_from_physical_deltas` |
| **MITRE Techniques** | ATT&CK v14 Knowledge Base | Static reference lookup keyed on dynamic stage | **NO** (ref catalog) | `test_mitre_mapping_deterministic_and_no_hallucination` |
| **Risk Score** | Mathematical formula | $100 \cdot (w_1 p_{\text{atk}} + w_2 p_{\text{trans}} \cdot c + w_3 \|\Delta S\|)$ | **NO** | `test_risk_engine_mathematical_sensitivity` |
| **Agent Narrative** | Grounded template / Ollama LLM | Synthesized strictly from runtime `ForecastEvent` fields | **NO** | `test_agent_query_current_state` |

---

## 4. Documentation of Legitimate Constants

All constants present in the codebase were audited and categorized as legitimate infrastructure definitions:

1. **`FEATURE_NAMES` (24 items):** Canonical statistical and behavioral column definitions (e.g., `flow_count`, `dst_port_entropy`, `syn_ratio`). Required for tabular schema consistency.
2. **`STAGE_TAXONOMY` (10 items):** The MITRE-aligned stage taxonomy strings (`BENIGN`, `RECONNAISSANCE`, ..., `EXFILTRATION`).
3. **`INPUT_DIM = 24`, `HIDDEN_DIM = 128`:** Architectural dimensions of `CyberWorldModelV2`.
4. **`_CALIBRATION_ARTIFACT` ($T^* = 1.5680$):** Empirically optimized temperature scaling factor loaded dynamically from `artifacts/calibration/temperature.json`.
5. **Phase 8C Benchmark References:** Hardcoded references in documentation and assertion fixtures ensuring non-regression against frozen Phase 8C results.
