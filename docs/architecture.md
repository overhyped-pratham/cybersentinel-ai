# CyberSentinel AI — Architecture & System Design

## 1. System Overview

**CyberSentinel AI** is an AI-based network attack forecasting platform for SIH Problem Statement **SIH26153**.
Unlike traditional Intrusion Detection Systems (IDS) that perform static point-in-time classification:

$$\text{traffic} \to \text{classifier} \to \{\text{malicious}, \text{benign}\}$$

CyberSentinel AI formulates proactive network defense through a **Temporal Cyber World Model** that models how network states evolve over discrete temporal intervals:

$$S_t \to P(S_{t+1} \mid S_t)$$

and performs autoregressive multi-step forward simulation:

$$S_t \to S_{t+1} \to S_{t+2} \to \dots \to S_{t+K}$$

This answers four vital questions for SOC analysts:
1. **What is happening now?** (Current window network state $S_t$)
2. **What attack stage is currently occurring?** (Stage classification $\hat{y}_t$)
3. **What is likely to happen next?** (Next-stage forecast $\hat{y}_{t+1}$ and future $K$-step trajectory $\hat{y}_{t+1 \dots t+K}$)
4. **Why does the model believe that?** (Feature attribution, temporal attention, and physical network evidence)

---

## 2. Mathematical Formulation

At discrete time intervals of length $\Delta T = 30\text{ seconds}$, the network state is represented by vector $S_t \in \mathbb{R}^D$ where $D = 24$.
Given historical context of length $H = 8$ (representing 4 minutes of observation):

$$\mathbf{X}_t = [S_{t-H+1}, S_{t-H+2}, \dots, S_t] \in \mathbb{R}^{H \times D}$$

The system optimizes a multi-task objective:
1. **Latent State Transition**: $\hat{h}_{t+1} = f_{\text{transition}}(c_t)$ where $c_t$ is the contextual embedding produced by a Temporal Transformer.
2. **Current Attack Stage**: $\hat{y}_t = \text{Softmax}(W_{\text{stage}} c_t)$
3. **Attack Probability**: $\hat{p}_{\text{attack}, t+1} = \sigma(W_{\text{attack}} c_t)$
4. **Next-Stage Forecast**: $\hat{y}_{t+1} = \text{Softmax}(W_{\text{forecast}} c_t)$

---

## 3. Repository Architecture

```
cybersentinel-ai/
├── backend/
│   ├── api/            # FastAPI route handlers
│   ├── services/       # Inference, replay, and alert services
│   ├── schemas/        # Pydantic v2 validation contracts
│   └── agents/         # AI agent integration
├── ml/
│   ├── preprocessing/  # Scalers, validators, sequence builders, splitters
│   ├── state/          # NetworkStateBuilder & feature definitions
│   ├── baseline/       # Logistic Regression and XGBoost baselines
│   ├── temporal/       # LSTM / GRU sequential models
│   ├── world_model/    # Transformer CyberWorldModel & latent transition head
│   ├── forecasting/    # Autoregressive K-step rollout engine
│   ├── evaluation/     # Lead time, Brier score, calibration metrics
│   └── explainability/ # SHAP and temporal attention attribution
├── network/
│   ├── pcap/           # Pure-Python offline PCAP flow extractor
│   ├── flow/           # FlowRecord definition and CSV parser
│   └── features/       # Protocol and entropy feature functions
├── mitre/
│   ├── data/           # Machine-readable ATT&CK matrix
│   └── mappings/       # Transparent rule-based stage-to-technique mapper
├── agent/
│   ├── tools/          # Read-only SOC query tools
│   ├── prompts/        # Structured reasoning prompts
│   └── fallback/       # Deterministic rule-based response templates
├── dashboard/          # React + Vite + Tailwind UI
├── datasets/
│   ├── sample/         # Synthesized scenario flow CSVs
│   └── processed/      # Cached window states and sequence tensors
├── models/             # Checkpoints for baselines and world models
├── configs/            # Declarative YAML configurations
├── docs/               # Technical specs and research documentation
└── tests/              # Leakage, unit, and integration test suites
```

---

## 4. Five Core Principles

1. **No Fake ML**: No invented confidence, synthetic accuracies, or hardcoded predictions.
2. **ML Before UI**: Validated mathematical pipelines precede frontend views.
3. **Offline First**: Runs completely without internet connectivity or external APIs.
4. **Reproducibility**: Explicit seeds, configs, and deterministic data splits.
5. **No Data Leakage**: Temporal and scenario boundaries are strictly enforced.
