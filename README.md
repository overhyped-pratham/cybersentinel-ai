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
├── backend/            # FastAPI REST services, WebSockets, SSE & Pydantic schemas
│   ├── api/            # Route endpoints & WebSocket stream handler (/api/v1/stream/ws)
│   ├── services/       # ModelService, ReplayService, LiveIngestService
│   └── agents/         # CyberSentinelDefensiveAgent (evidence-grounded analyst agent)
├── dashboard/          # Real-time SOC Command Center UI (HTML5 / WebSocket / SVG)
├── network/
│   ├── flow/           # Standardized FlowRecord representation & CSV loader
│   ├── pcap/           # Pure-Python libpcap capture parser
│   └── telemetry/      # TelemetrySource ABC, PCAPSource, NetFlowSource, StreamProcessor
├── ml/
│   ├── world_model/    # CyberWorldModelV2 (Direct Physical Transition Network)
│   ├── state/          # NetworkStateBuilder (24-D physical state vector, 0 leakage)
│   ├── defense/        # Dynamic RiskEngine (multi-factor severity scoring)
│   ├── calibration/    # Temperature scaling calibration artifact (T*=1.5680)
│   └── preprocessing/  # FeatureScaler, SequenceBuilder, StageLabeler
├── mitre/              # MITRE ATT&CK v14 deterministic stage-to-technique mapper
├── datasets/           # Multi-scenario flow traces (multistage attack traces)
├── docs/               # Architecture, audit reports, benchmarks, validation docs
├── scripts/            # Inference benchmarks, training pipelines, start launcher
└── tests/              # Test suite (229/229 passing: causality, live ingest, WS, equivalence, Phase 14)
```

---

## 5. Live Telemetry & Streaming Configuration

CyberSentinel AI supports three ingestion modes via a modular adapter interface (`TelemetrySource`):

1. **PCAP Capture (`PCAPSource`)**: Ingests offline or live-tailed `.pcap` files without third-party C driver dependencies.
2. **NetFlow v5 (`NetFlowSource`)**: Listens on UDP port `9995` for standard NetFlow v5 datagrams with malformed header resilience.
3. **Trace Replay (`ReplaySource`)**: Paced testing playback of recorded CSV/JSON multi-stage network attack traces.

### Starting the Command Center
Launch backend server and SOC UI:
```bash
python scripts/start_server.py
```
- API Docs: `http://localhost:8000/docs`
- SOC Dashboard: `http://localhost:8000/ui/index.html`

### Real-Time Streaming API
- **WebSocket**: Connect to `ws://localhost:8000/api/v1/stream/ws` to receive continuous, JSON-encoded threat forecasts per completed 30s window.
- **Server-Sent Events**: Connect to `http://localhost:8000/api/v1/stream/events` for browser `EventSource` consumption.
- **Session Control**:
  - `POST /api/v1/stream/start`: Start live ingestion session.
  - `POST /api/v1/stream/stop`: Stop active session.
  - `GET /api/v1/stream/status`: Active session statistics and queue depth.
  - `GET /api/v1/stream/health`: System health, mode, scaler/model availability, WebSocket slot usage.

### Multi-Host Traffic Validation (Phase 14)
Transmit controlled traffic patterns from an authorized secondary device:
```powershell
# From secondary laptop (192.168.1.105):
python scripts/multi_host_traffic_generator.py \
    --target-host 192.168.1.100 --target-port 9995 \
    --pattern large_data_transfer --windows 5
```
Available patterns: `normal_background`, `connection_burst`, `repeated_attempts`, `port_diversity`, `large_data_transfer`.

### Two-Machine Live Demonstration (Phase 16)
Execute the continuous 3-phase lifecycle demonstration (Baseline $\to$ High-Activity $\to$ Recovery) over live UDP NetFlow:
```powershell
# From Laptop 2 or demonstration runner:
python scripts/run_phase16_twomachine_demo.py \
    --target-host 192.168.1.100 --target-port 9995 \
    --pattern large_data_transfer \
    --windows-per-phase 6 --window-seconds 10.0 \
    --report docs/phase16_twomachine_demo_report.md
```
See [`docs/phase16_operator_runbook.md`](docs/phase16_operator_runbook.md) for full physical deployment instructions.

---

## 6. Verification & Benchmarking

Run the complete 235-test automated suite:
```powershell
pytest tests/ -v
```

Run the Phase 13 performance and memory stability benchmark:
```powershell
python scripts/benchmark_phase13.py --runs 100
```

**Measured Latency Breakdown (Intel x86_64, PyTorch CPU):**
- 24-D State Extraction: **2.58 ms** median
- Feature Scaling: **0.80 ms** median
- 1-Step Model Forecast + Risk + MITRE: **6.28 ms** median
- K=4 Autoregressive Rollout: **10.25 ms** median
- Complete End-to-End Live Pipeline: **15.14 ms** median (66.0 windows/sec)
- Long-running Memory Stability: +4.53 MB delta over 1,000 continuous windows (~8.3 hours traffic)

---

## 7. Security Boundaries, Hardening & Fail-Closed Behavior

- **Strictly Defensive**: Operates exclusively as a passive observer, forecaster, and explainer. Does not inject packets, perform port scans, or execute offensive actions.
- **Zero Hardcoded Intelligence**: No stage transitions, risk scores, probabilities, or feature importances are hardcoded or scenario-dependent. All intelligence is computed at runtime from observed telemetry.
- **Fail-Closed Design**: If required model checkpoints, scalers, or valid telemetry are unavailable, the system explicitly returns error states (`MODEL_UNAVAILABLE`, `INVALID_TELEMETRY`) and refuses to emit mock or static predictions.
- **Rate Limiting**: Sliding-window limiter (120 requests/minute per client IP), configurable via `CYBERSENTINEL_RATE_LIMIT`.
- **WebSocket Connection Limiter**: Maximum 50 concurrent live streaming subscribers, configurable via `CYBERSENTINEL_MAX_WS_CONNECTIONS`.
- **API Key Authentication**: Optional `X-API-Key` header enforcement via `CYBERSENTINEL_API_KEY` environment variable.
- **Payload Bounding**: Requests bearing `Content-Length > 10 MB` are rejected with HTTP 413.
- **Safe Response Headers**: All HTTP responses include `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 1; mode=block`.
- **Subscriber Backpressure**: Full subscriber queues drop events with warning log rather than blocking the ingestion thread.

---

## 8. Limitations

- **Window Granularity**: High-frequency attacks occurring entirely within < 1 second are aggregated into the containing 30-second window.
- **Zero Lookahead Constraint**: The model cannot predict unprecedented external zero-day vectors that do not perturb network traffic patterns.
- **Replay vs. Live**: CSV replay serves as an evaluation and testing adapter only; production deployments require live NetFlow or PCAP streams.
