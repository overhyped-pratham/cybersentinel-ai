# PHASE 12: RUNTIME PIPELINE AUDIT

**Target:** Complete End-to-End Trace from Raw Telemetry to SOC Dashboard  
**Date:** September 7, 2026  
**Status:** **VERIFIED — ALL COMPUTATION DYNAMIC AT RUNTIME**  
**Hardcoded Intelligence:** **NONE (0 Instances)**  

---

## 1. Trace Overview

The following audit traces a single telemetry request end-to-end through the CyberSentinel AI platform:

$$\text{Raw Telemetry Flow Record} \longrightarrow \text{NetworkStateBuilder} \longrightarrow \text{FeatureScaler} \longrightarrow \text{Temporal Sequence} \longrightarrow \text{CyberWorldModelV2} \longrightarrow \text{Temperature Calibration} \longrightarrow \text{Physical State Predictor} \longrightarrow \text{Explainability Engine} \longrightarrow \text{MITRE Context} \longrightarrow \text{Risk Engine} \longrightarrow \text{Defensive Agent} \longrightarrow \text{FastAPI} \longrightarrow \text{SOC Dashboard}$$

---

## 2. Detailed Stage-by-Stage Runtime Audit

### Stage 1: Raw Telemetry Ingestion
* **Component:** `network.flow.csv_loader.CSVFlowLoader` / `network.flow.pcap_loader.PCAPFlowLoader`
* **Input:** Raw packet capture files (`.pcap`) or flow logs (`.csv`) containing packet timestamps, IPs, ports, TCP flags, byte/packet counters.
* **Output:** Cleaned, schema-validated `List[FlowRecord]` dataclass instances.
* **Source of Data:** Dynamic stream or disk scenario traces (`datasets/sample/*.csv`).
* **Computation Type:** **Runtime deterministic parsing** (sanitizes timestamps, resolves protocol numbers, validates flag bitmaps).
* **Hardcoded Intelligence:** **None**. Zero assumptions about attack stage or scenario identity.

### Stage 2: 24-D Network State Construction
* **Component:** `ml.state.state_builder.NetworkStateBuilder`
* **Input:** `List[FlowRecord]` falling strictly within a 30.0-second time window $[t_{\text{start}}, t_{\text{end}})$.
* **Output:** A pandas DataFrame row / numpy vector $\mathbf{x}_t \in \mathbb{R}^{24}$ representing physical network aggregates.
* **Source of Data:** Aggregated statistics of the input `FlowRecord` list.
* **Computation Type:** **Runtime mathematical aggregation**:
  * Volumetric rates: `flow_rate`, `packet_rate`, `byte_rate`.
  * Ratios: `syn_ratio`, `rst_ratio`, `ack_ratio`, `failed_ratio`.
  * Traffic distribution: `src_ip_entropy`, `dst_port_entropy`, `unique_dst_ports`.
  * Descriptive statistics: `packet_size_mean`, `packet_size_std`, `flow_duration_mean`.
* **Hardcoded Intelligence:** **None**. Formulas are standard information theory (Shannon entropy) and network telemetry math.

### Stage 3: Feature Scaling & Normalization
* **Component:** `ml.preprocessing.scaler.FeatureScaler`
* **Input:** Raw physical 24-D feature vector $\mathbf{x}_t \in \mathbb{R}^{24}$.
* **Output:** Normalized feature vector $\tilde{\mathbf{x}}_t \in \mathbb{R}^{24}$.
* **Source of Data:** Fitted scaling parameters from `experiments/run_20260907_111554/scaler.pkl` (fit strictly on training split).
* **Computation Type:** **Runtime affine transformation** ($z = \frac{x - \mu}{\sigma}$ or $z = \frac{x - \min}{\max - \min}$).
* **Hardcoded Intelligence:** **None**. Statistical normalization parameters learned during offline training.

### Stage 4: Temporal Sequence Assembly
* **Component:** `ml.preprocessing.sequence_builder.SequenceBuilder` / `backend.services.model_service.ModelService`
* **Input:** Sequence of consecutive scaled state vectors $\tilde{\mathbf{X}} = [\tilde{\mathbf{x}}_{t-T+1}, \dots, \tilde{\mathbf{x}}_t] \in \mathbb{R}^{B \times T \times 24}$ ($T \ge 5$).
* **Output:** Batched PyTorch tensors $(\mathbf{X}, \mathbf{M})$ where $\mathbf{M} \in \{0, 1\}^{B \times T}$ is the causal validity mask.
* **Source of Data:** Historical sliding window of real network states.
* **Computation Type:** **Runtime sliding-window buffer**.
* **Hardcoded Intelligence:** **None**. Strictly causal; future states are never accessed or buffered.

### Stage 5: Cyber World Model V2 Forward Pass
* **Component:** `ml.world_model.world_model_v2.CyberWorldModelV2` (Weights: `models/world_model_v2.pt`)
* **Input:** Tensor $(\mathbf{X}, \mathbf{M})$ of shape $(1, T, 24)$.
* **Output:**
  * Raw next-stage logits $\mathbf{z}_{t+1} \in \mathbb{R}^{12}$
  * Raw current-stage logits $\mathbf{z}_t \in \mathbb{R}^{12}$
  * Binary attack logit $a_{t+1} \in \mathbb{R}^1$
  * Predicted next physical state $\hat{\mathbf{x}}_{t+1} \in \mathbb{R}^{24}$
  * Transition logit $\tau_{t+1} \in \mathbb{R}^1$
* **Source of Data:** Forward pass through trained PyTorch neural layers (GRU encoder, Direct Transition Head, Physical State Predictor).
* **Computation Type:** **Runtime neural network inference** on CPU/GPU.
* **Hardcoded Intelligence:** **None**. 492,044 trained parameters executing matrix multiplications and non-linear activations.

### Stage 6: Probability Calibration
* **Component:** Post-hoc Temperature Scaler (`artifacts/calibration/temperature.json`)
* **Input:** Raw next-stage logits $\mathbf{z}_{t+1}$ and learned optimal temperature $T^* = 1.568035$.
* **Output:** Calibrated probability distribution $\mathbf{p}_{t+1} = \text{softmax}(\mathbf{z}_{t+1} / T^*) \in [0, 1]^{12}$, where $\sum p_i = 1.0$.
* **Source of Data:** Dynamic logits divided by empirical calibration scalar.
* **Computation Type:** **Runtime mathematical scaling**.
* **Hardcoded Intelligence:** **None**. Temperature parameter fitted empirically to minimize Expected Calibration Error (ECE).

### Stage 7: Physical State Explanation & Feature Deltas
* **Component:** `ml.explainability.feature_attribution` / `ModelService._compute_feature_shifts`
* **Input:** Observed unscaled state $\mathbf{x}_t$ and predicted unscaled state $\hat{\mathbf{x}}_{t+1} = \text{scaler}^{-1}(\hat{\tilde{\mathbf{x}}}_{t+1})$.
* **Output:** Feature delta vector $\Delta \mathbf{x} = \hat{\mathbf{x}}_{t+1} - \mathbf{x}_t$, top-5 features ranked by $|\Delta x_i|$, relative shift percentage, and directional trend (`INCREASING`, `DECREASING`, `STABLE`).
* **Source of Data:** Difference between the model's actual physical prediction and the current state.
* **Computation Type:** **Runtime vector arithmetic**.
* **Hardcoded Intelligence:** **None**. Explanations reflect exactly what the neural network predicted will change in the network.

### Stage 8: MITRE ATT&CK Context Resolution
* **Component:** `backend.services.mitre_context_service.MitreContextService`
* **Input:** Top predicted stage $\hat{y}_{t+1} = \text{argmax}(\mathbf{p}_{t+1})$ and attack probability $P(\text{attack})$.
* **Output:** Relevant MITRE ATT&CK Enterprise techniques, tactics, technique IDs, descriptions, and detection recommendations.
* **Source of Data:** Official MITRE ATT&CK Enterprise knowledge base mapping.
* **Computation Type:** **Runtime taxonomy lookup** keyed on the model's dynamic prediction.
* **Hardcoded Intelligence:** **Legitimate standard reference**. The mapping of stage $\to$ technique catalog is an external cybersecurity ontology, not ML intelligence.

### Stage 9: Dynamic Defensive Risk Scoring
* **Component:** `backend.services.risk_engine.RiskEngine`
* **Input:** $P(\text{attack})$, $P(\text{transition})$, stage criticality weight, and physical state shift magnitude $\|\Delta \mathbf{x}\|_2$.
* **Output:** Continuous risk score $R \in [0, 100]$ and categorical severity rating (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
* **Source of Data:** Outputs from Stage 5, Stage 6, and Stage 7.
* **Computation Type:** **Runtime mathematical evaluation**:
  $$R = 100 \times \left( w_1 P(\text{attack}) + w_2 P(\text{transition}) \cdot C_{\text{stage}} + w_3 \min\left(1.0, \frac{\|\Delta \mathbf{x}\|}{S_{\text{norm}}}\right) \right)$$
* **Hardcoded Intelligence:** **None**. The score varies continuously with flow characteristics.

### Stage 10: Autonomous Defensive Agent
* **Component:** `backend.agents.defensive_agent.CyberSentinelDefensiveAgent`
* **Input:** Structured forecast payload `CyberSentinelForecast` from Stages 5–9 and natural-language analyst query.
* **Output:** Grounded incident investigation narrative, step-by-step containment playbook, tool execution trace, and provenance record.
* **Source of Data:** Dynamic fields in `CyberSentinelForecast`.
* **Computation Type:** **Runtime synthesis** via local Ollama LLM or deterministic template synthesizer.
* **Hardcoded Intelligence:** **None**. The agent is strictly constrained to cite only facts present in the forecast object; hallucinated MITRE IDs or fabricated probabilities are sanitized and rejected.

### Stage 11: Backend API Gateway
* **Component:** `backend.main.py` (FastAPI)
* **Input:** HTTP POST/GET requests from frontend or external SOC consumers (`/api/forecast`, `/api/replay/*`, `/api/agent/*`).
* **Output:** Serialized JSON responses conforming to Pydantic schemas.
* **Source of Data:** In-memory backend services (`ModelService`, `ReplayEngine`, `AgentService`).
* **Computation Type:** **Runtime async web service**.
* **Hardcoded Intelligence:** **None**. All responses are freshly computed or retrieved from live sessions.

### Stage 12: SOC Command Center Dashboard
* **Component:** `frontend/dashboard.py` (Streamlit)
* **Input:** JSON responses received from FastAPI endpoints.
* **Output:** Rendered UI: real-time telemetry tables, gauge charts, radar charts, rollout trajectories, and chat interface.
* **Source of Data:** Live polling of backend API (`localhost:8000`).
* **Computation Type:** **Runtime UI rendering**.
* **Hardcoded Intelligence:** **None**. Initial UI displays neutral empty placeholders (`—`, `0.00`). No canned attack values exist in the frontend code.

---

## 3. Summary Table

| Stage | Component | Input | Output | Computation | Hardcoded Intelligence? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Ingestion** | `CSVFlowLoader` | Flow CSV / PCAP | `List[FlowRecord]` | Runtime parsing | **NO** |
| **2. State Build** | `NetworkStateBuilder` | `List[FlowRecord]` | 24-D State Vector | Runtime aggregation | **NO** |
| **3. Scaling** | `FeatureScaler` | 24-D State Vector | Scaled Vector $\tilde{\mathbf{x}}$ | Affine transform | **NO** |
| **4. Windowing** | `SequenceBuilder` | History of $\tilde{\mathbf{x}}$ | Batch Tensor $(1, T, 24)$ | Buffer assembly | **NO** |
| **5. Inference** | `CyberWorldModelV2` | Tensor $(1, T, 24)$ | Logits & Next State $\hat{\mathbf{x}}$ | PyTorch forward pass | **NO** |
| **6. Calibration**| Temperature Scaler | Logits & $T^*$ | Calibrated Probs $\mathbf{p}$ | Softmax / $T^*$ | **NO** |
| **7. Explanation**| Feature Attribution | $\mathbf{x}_t$ and $\hat{\mathbf{x}}_{t+1}$ | $\Delta \mathbf{x}$, Top-5 features | Vector difference | **NO** |
| **8. MITRE** | `MitreContextService` | Predicted stage | Techniques & Tactics | Taxonomy lookup | **NO (Standard ref)** |
| **9. Risk** | `RiskEngine` | Probs & State deltas | Risk score $[0, 100]$ | Mathematical formula | **NO** |
| **10. Agent** | `DefensiveAgent` | Forecast JSON & Query | Investigation narrative | Dynamic synthesis | **NO** |
| **11. API** | FastAPI Application | HTTP Requests | JSON contracts | Web serialization | **NO** |
| **12. Dashboard** | Streamlit Command Ctr| Backend API JSON | Interactive SOC views | UI rendering | **NO** |

**Conclusion:** The entire CyberSentinel AI pipeline executes dynamically from telemetry to presentation. Zero hardcoded intelligence shortcuts exist.
