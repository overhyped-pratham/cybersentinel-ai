# CyberSentinel AI — AI-Based Network Attack Forecasting

> **SIH Problem Statement SIH26153**: AI-Based Network Attack Forecasting  
> **Core Innovation**: A Temporal Cyber World Model that learns network state dynamics $S_t \to P(S_{t+1} \mid S_t)$ and executes autoregressive $K$-step forward simulations to forecast an attacker's future campaign trajectory.

---

## 1. What is Novel?

Traditional Intrusion Detection Systems (IDS) evaluate network flows as isolated, point-in-time classification problems:

$$\text{flow} \xrightarrow{\text{classifier}} \{\text{malicious}, \text{benign}\}$$

This approach suffers from high false-alarm rates, zero forward visibility, and no awareness of multi-stage cyber campaign progression.

**CyberSentinel AI** reframes defensive monitoring as **Temporal World Modeling**:
1. Groups network flows into discrete 30-second window state vectors $S_t \in \mathbb{R}^{24}$.
2. Encodes a sequence of historical states $[S_{t-7}, \dots, S_t]$ using a Temporal Transformer into contextual state embeddings $c_t$.
3. Employs a latent state transition head that models network state evolution:
   $$\hat{h}_{t+1} = f_{\text{transition}}(c_t)$$
4. Conducts autoregressive $K$-step forward rollouts:
   $$S_t \to S_{t+1} \to S_{t+2} \to \dots \to S_{t+K}$$
   to predict the adversary's next tactical moves (e.g., Reconnaissance $\to$ Credential Access $\to$ Lateral Movement $\to$ Exfiltration) before they manifest.

---

## 2. Four Operational SOC Questions

CyberSentinel AI provides defensible, mathematically grounded answers to four fundamental questions:
1. **What is happening now?** $\to$ Window network state representation $S_t$.
2. **What attack stage is currently occurring?** $\to$ Stage classification $\hat{y}_t$.
3. **What is likely to happen next?** $\to$ Next-stage forecast $\hat{y}_{t+1}$ and forward trajectory $t+1 \dots t+K$.
4. **Why does the model believe that?** $\to$ Transparent feature attribution, temporal attention, and physical network evidence.

---

## 3. Strict Research Honesty & Taxonomy Separation

The system strictly enforces architectural boundaries:
- **Observed**: Concrete physical metrics extracted directly from packets (e.g. `port_445_share = 0.42`, `byte_rate = 85000`).
- **Derived**: Attack stage label inferred by our documented research labeling policy (`LATERAL_MOVEMENT`).
- **Predicted**: Probabilities produced by the PyTorch Cyber World Model ($\hat{y}_{t+1} = \text{Command \& Control}, p = 0.84$).
- **Mapped**: External MITRE ATT&CK technique IDs associated by the rule-based knowledge engine (`T1021.002 - SMB/Windows Admin Shares`).
- **Generated**: Natural-language text synthesized by the offline AI agent to explain the threat to a SOC analyst.

---

## 4. Repository Structure

```
cybersentinel-ai/
├── backend/            # FastAPI REST services & Pydantic validation schemas
├── ml/
│   ├── preprocessing/  # FeatureScaler, FeatureValidator, SequenceBuilder, ScenarioSplitter
│   ├── state/          # NetworkStateBuilder (24 curated features per 30s window)
│   ├── baseline/       # Logistic Regression & XGBoost baselines
│   ├── temporal/       # LSTM / GRU sequential baseline
│   ├── world_model/    # PyTorch CyberWorldModel with latent transition head
│   ├── forecasting/    # Autoregressive K-step rollout simulation engine
│   ├── evaluation/     # Lead time, Brier score, calibration curves
│   └── explainability/ # SHAP & temporal attention attribution
├── network/
│   ├── flow/           # FlowRecord definition & robust CSVFlowLoader
│   └── pcap/           # Pure-Python offline PCAPFlowLoader (0 external C dependencies)
├── mitre/              # Machine-readable ATT&CK knowledge base and mapper
├── agent/              # Read-only SOC query tools and deterministic fallbacks
├── datasets/           # Multi-scenario flow traces (synthetic and benchmark)
├── configs/            # Declarative pipeline & model configurations
├── docs/               # Architectural, data pipeline, and state specifications
└── tests/              # Leakage, unit, and integration test suites
```

---

## 5. Verification & Testing

Run the automated test suite:
```powershell
pytest -v
```

Run the end-to-end data pipeline validation script:
```powershell
python scripts/validate_pipeline.py
```

---

## 6. Limitations

- **Labeling Policy**: Public datasets (such as CICIDS2017) do not provide standardized MITRE ATT&CK ground truth. Stage labels are derived approximations based on our documented policy.
- **Window Granularity**: High-frequency attacks occurring entirely within $< 1$ second are aggregated into the containing 30-second window.
- **Zero Lookahead Constraint**: The model cannot predict unprecedented external zero-day vectors that do not perturb network traffic patterns.
