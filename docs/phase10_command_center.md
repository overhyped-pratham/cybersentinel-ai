# CyberSentinel AI — SOC Command Center & Defensive Agent (Phase 10)

## Overview

CyberSentinel AI Phase 10 integrates the Phase 8C/9 validated **CyberWorldModelV2** forecasting engine with a production-ready SOC defensive intelligence stack:
- **FastAPI Backend**: 12 structured JSON endpoints exposing inference, rollout, explainability, MITRE mapping, risk scoring, agent Q&A, and replay sessions.
- **Defensive Agent**: Offline-first natural language interface with Ollama integration and deterministic grounded template fallbacks. The agent explains model output; it never invents intelligence.
- **SOC Command Center UI**: Pure HTML5/CSS/vanilla JS dashboard (zero external CDN or npm dependencies) with strict visual segregation between **OBSERVED**, **FORECAST**, and **K=4 SIMULATED** states.

---

## 1. System Architecture

```
                    NETWORK TRACE / LIVE INPUT
                                │
                                ▼
                       ┌──────────────────┐
                       │ Feature Pipeline │ (RobustScaler, 24 features)
                       └────────┬─────────┘
                                ▼
                       ┌──────────────────┐
                       │ World Model V2   │ (Physical State Forward Sim)
                       └────────┬─────────┘
                                │
                    ┌───────────┼────────────┐
                    ▼           ▼            ▼
                 Current     Next Stage   K-Step
                 Attack       Forecast    Simulation
                    │           │            │
                    └───────────┼────────────┘
                                ▼
                      ┌──────────────────┐
                      │ Explainability   │ (Physical state deltas: |S_hat - S_t|)
                      └────────┬─────────┘
                               ▼
                      ┌──────────────────┐
                      │ MITRE ATT&CK     │ (v14 static mapping)
                      └────────┬─────────┘
                               ▼
                      ┌──────────────────┐
                      │ Risk Engine      │ (Deterministic weighted scoring)
                      └────────┬─────────┘
                               ▼
                      ┌──────────────────┐
                      │ CyberSentinel    │ (Ollama + grounded template fallback)
                      │ Defensive Agent  │
                      └────────┬─────────┘
                               ▼
                      SOC COMMAND CENTER (Offline HTML5 Dashboard)
```

---

## 2. API Endpoints

The API runs by default on `http://localhost:8000/api/v1` with interactive OpenAPI docs at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health check (model status, calibration status, dataset status, Ollama status) |
| `GET` | `/model/info` | Architecture specs, hyperparameters, temperature calibration, immutable benchmark metrics |
| `POST` | `/forecast` | Full canonical unified forecast (`CyberSentinelForecast` response) |
| `POST` | `/rollout` | K-step autoregressive physical state simulation with safety monitoring |
| `POST` | `/explain` | Feature delta attribution ranked by magnitude and stage relevance |
| `POST` | `/mitre` | Deterministic MITRE ATT&CK technique mapping for predicted next stage |
| `POST` | `/risk` | Multi-factor risk scoring, severity band, priority level, and transition urgency |
| `POST` | `/agent/query` | Analyst natural-language Q&A grounded in structured tool outputs |
| `GET` | `/replay/scenarios` | List all available multi-stage attack scenarios in the dataset |
| `POST` | `/replay/start` | Initialize an in-memory step-through replay session |
| `POST` | `/replay/step` | Advance replay session by 1 window and run live V2 model inference |
| `GET` | `/replay/status` | Current window progress and elapsed time for active session |

---

## 3. Unified Forecast Response Schema

Every model prediction returns a canonical `CyberSentinelForecast` object. Raw PyTorch tensors are never exposed.

```json
{
  "timestamp": "2026-09-07T08:30:00.000Z",
  "model_version": "CyberWorldModelV2",
  "forecast_horizon": 30,
  "current_stage": "RECONNAISSANCE",
  "attack_probability": 0.9412,
  "predicted_next_stage": "CREDENTIAL_ACCESS",
  "next_stage_probability": 0.8845,
  "confidence": 0.8845,
  "transition_detected": true,
  "transition_confidence": 0.8845,
  "uncertainty_entropy": 0.1241,
  "stage_probabilities": {
    "BENIGN": 0.0012,
    "RECONNAISSANCE": 0.0821,
    "CREDENTIAL_ACCESS": 0.8845,
    "LATERAL_MOVEMENT": 0.0210
  },
  "rollout_steps": [
    {"step": 1, "predicted_stage": "CREDENTIAL_ACCESS", "confidence": 0.8845, "attack_probability": 0.9412},
    {"step": 2, "predicted_stage": "CREDENTIAL_ACCESS", "confidence": 0.8210, "attack_probability": 0.9520},
    {"step": 3, "predicted_stage": "LATERAL_MOVEMENT", "confidence": 0.7410, "attack_probability": 0.9630},
    {"step": 4, "predicted_stage": "LATERAL_MOVEMENT", "confidence": 0.6850, "attack_probability": 0.9710}
  ],
  "top_features": [
    {"feature": "failed_flow_count", "current": 2.15, "predicted": 14.80, "abs_change": 12.65, "rel_change_pct": 588.4, "direction": "increase"}
  ],
  "mitre_techniques": [
    {"technique_id": "T1110", "name": "Brute Force", "tactic": "Credential Access", "mapping_provenance": "MITRE ATT&CK Enterprise v14 static mapping"}
  ],
  "risk_score": 78.5,
  "risk_level": "CRITICAL",
  "safety_flags": {
    "nan_detected": false,
    "collapse_detected": false,
    "feature_dim_valid": true
  },
  "provenance": {
    "prediction": "CyberWorldModelV2",
    "explanation": "physical_state_delta",
    "mitre": "MITRE_ATT&CK_v14_static_mapping",
    "risk": "RiskEngine_deterministic",
    "narrative": "CyberSentinel_defensive_agent"
  }
}
```

---

## 4. Defensive Agent & Offline Operation

### Agent Grounding Contract
1. **The LLM is NOT the detector.** Threat detection and forecasting are computed by `CyberWorldModelV2`.
2. **The LLM NEVER invents:** stages, probabilities, MITRE IDs, or feature shifts.
3. **Tool Execution First:** The agent executes deterministic retrieval (`get_current_state`, `get_attack_forecast`, `get_rollout`, `get_mitre_mapping`, `get_risk_assessment`) before synthesizing answers.
4. **Sanitization Guard:** In Ollama mode, responses are automatically filtered via regex to strip any hallucinated `T####` technique IDs not present in the model's static lookup.

### Offline Fallback Hierarchy
1. **Ollama local runtime**: If Ollama is running on `http://localhost:11434`, the agent prompts a local model (`llama3.2`, `mistral`, `gemma2`) with strict system constraints.
2. **Deterministic template engine**: If Ollama is unavailable, the agent generates structured answers directly from the `ForecastEvent` fields. No external network connectivity is ever required.

---

## 5. Visual Hierarchy: Observed vs. Forecast vs. Simulated

The SOC Command Center enforces visual discipline across three temporal domains:

1. <span style="color:#10b981;font-weight:bold">OBSERVED TELEMETRY (Green)</span>: Actual historical network state and flow measurements up to timestamp $t$.
2. <span style="color:#3b82f6;font-weight:bold">FORECAST (Blue)</span>: Calibrated next-step prediction for window $t+1$ based on historical context $S_{1..t}$.
3. <span style="color:#8b5cf6;font-weight:bold">K=4 SIMULATION (Purple)</span>: Autoregressive forward rollouts $S_{t+2..t+4}$ without future observations. Explicitly badged as **MODEL SIMULATION** to prevent SOC operator confusion with observed events.

---

## 6. Immutable Empirical Benchmark (Phase 8C Reference)

Evaluated on the Hard Multi-Stage Holdout split ($N=44$ test sequences, 6 genuine attack stage transitions):

| Metric | CyberWorldModelV2 | Temporal GRU | Logistic Regression |
|---|---|---|---|
| **Next-stage Top-1** | **97.73%** | 81.82% | 50.00% |
| **Next-stage Top-3** | **100.0%** | 97.73% | 72.73% |
| **True Transition Acc** | **83.33% (5/6)** | 66.67% (4/6) | 0.00% (0/6) |
| **Brier Score (Calibrated)** | **0.0452** | 0.2913 | 0.7206 |
| **Attack FPR** | **0.00%** | 53.33% | 0.00% |
| **K=4 Autoregressive Rollout** | **Supported** | Unsupported | Unsupported |

---

## 7. Security Boundaries

CyberSentinel AI operates strictly as a **defensive decision support system**:
- **Permitted Capabilities**: Passive telemetry ingestion, stage forecasting, risk prioritization, explainability narratives, MITRE mapping, analyst investigation checklists.
- **Prohibited Capabilities**: The defensive agent has no execution tools for offensive actions (no exploit execution, packet injection, port scanning, lateral movement, or unauthorized credential access).

---

## 8. Performance & Latency Targets

- Pure neural network forward pass: **~1.5 - 3.5 ms**
- Full 1-step forecast pipeline (NN + state predictor + explainability + MITRE + risk): **~4.0 - 8.0 ms**
- K=4 step autoregressive rollout: **~10.0 - 18.0 ms**
- Throughput: **>100 windows/sec** on commodity CPU (enabling real-time processing of 30-second windows with minimal system footprint).
