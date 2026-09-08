# PHASE 13: LIVE REAL-WORLD TELEMETRY INGESTION

**Date:** September 8, 2026  
**Status:** **OPERATIONAL & VALIDATED**  
**Production Model:** CyberWorldModelV2 (Direct Physical Transition Network, Phase 8C Canonical)  

---

## 1. Overview & Objectives

Phase 13 establishes the production-grade live telemetry ingestion pipeline for CyberSentinel AI, transitioning from offline dataset evaluation and synthetic replay to continuous, real-time network stream ingestion.

```
REAL NETWORK TELEMETRY (PCAP / NetFlow UDP:9995 / Replay Stream)
  ↓
MODULAR TELEMETRY SOURCE ADAPTER (TelemetrySource ABC)
  ↓
ASYNC STREAM PROCESSOR (Causal 30s Windowing, Bounded Deque, Backpressure)
  ↓
24-D PHYSICAL NETWORK STATE BUILDER (Zero Lookahead)
  ↓
FEATURE NORMALIZATION (FeatureScaler Robust/MinMax)
  ↓
CyberWorldModelV2 INFERENCE (Calibrated Softmax T*=1.5680, Physical State Prediction)
  ↓
K-STEP AUTOREGRESSIVE ROLLOUT (Simulation Horizon K=4)
  ↓
PHYSICAL FEATURE EXPLAINABILITY (|S_hat_{t+1} − S_t|)
  ↓
DETERMINISTIC MITRE ATT&CK v14 MAPPING
  ↓
DYNAMIC RISK ENGINE (Multi-factor Risk Score [0, 100])
  ↓
DEFENSIVE AGENT (Evidence-Grounded Recommendations)
  ↓
WEBSOCKET / SSE LIVE STREAM BROADCAST (`/api/v1/stream/ws`)
  ↓
LIVE SOC COMMAND CENTER DASHBOARD
```

---

## 2. Modular Telemetry Adapters (`network/telemetry/sources.py`)

A polymorphic abstraction hierarchy isolates the ML pipeline from packet capture mechanics:

```
                  TelemetrySource (ABC)
                ┌───────────┼───────────┐
                ▼           ▼           ▼
           PCAPSource  NetFlowSource ReplaySource
```

### 2.1 `TelemetrySource` Contract
- `stream() -> AsyncIterator[FlowRecord]`: Emits standardized, immutable flow records.
- `close() -> None`: Gracefully drains and closes socket/file handles.
- `source_id: str`: Observability identifier.

### 2.2 Implemented Adapters
| Adapter | Ingestion Mechanism | Supported Format / Protocol | Fault Handling |
| :--- | :--- | :--- | :--- |
| **`PCAPSource`** | Pure-Python binary libpcap parser (no npcap/C dependencies) | `.pcap` files, offline captures, and live tailing (`tail=True`) | Truncated packets dropped with warning log; skips non-IPv4 frames |
| **`NetFlowSource`** | Async UDP datagram listener | NetFlow v5 (RFC 3954 subset) on UDP port 9995 | Malformed headers and truncated datagrams dropped safely |
| **`ReplaySource`** | File streaming adapter | Multi-stage CSV/JSON scenario traces with rate pacing (`realtime_factor`) | Skips corrupt rows, preserves timestamps, supports scenario overrides |

### 2.3 `FlowRecord` Schema
Every raw datagram or record is mapped to the immutable `FlowRecord` dataclass:
- 5-tuple: `(src_ip, dst_ip, src_port, dst_port, protocol)`
- Volume: `packets`, `bytes`, `duration`
- Flags: `syn_flag`, `rst_flag`, `fin_flag`, `ack_flag`, `psh_flag`, `urg_flag`
- Health: `failed` (boolean indicating abnormal teardown or 0-byte reply)
- Temporal: `timestamp` (Unix epoch UTC seconds)

---

## 3. Real-Time Stream Processor (`network/telemetry/stream_processor.py`)

The `StreamProcessor` continuously consumes flows from any `TelemetrySource` and groups them into discrete temporal windows:

- **Window Sizing:** Configurable window size (default: 30.0s) and stride (default: 30.0s).
- **Strict Temporal Causality:** A flow at timestamp $t$ belongs to window $W_k = [t_k, t_k + \Delta)$ if and only if $t_k \le t < t_k + \Delta$. Future flows are never included.
- **Out-of-Order Handling:** Flows arriving $> 1$ window behind the active window boundary are safely discarded.
- **Bounded Backlog:** Internal buffer capped at `max_flow_backlog=50,000` flows to enforce bounded memory under extreme bursts.
- **Backpressure Protection:** Output queue capped at `max_output_queue=64`. When downstream ML inference cannot keep pace, excess windows are dropped with logged telemetry counters (`total_output_dropped_backpressure`), preventing memory leaks.
- **Malformed Flow Tolerance:** Flows with negative timestamps, NaN/Inf values, invalid port ranges ($>65535$), or missing IPs are dropped before window aggregation.

---

## 4. Live Ingestion & ML Pipeline (`backend/services/live_ingest_service.py`)

For every completed `TelemetryWindowEvent`:

1. **24-D State Construction:** `NetworkStateBuilder` computes statistical and behavioral features from the window's flows.
2. **Sequence Accumulation:** Minimum 5 windows are buffered before generating the first multi-timestep inference sequence for `CyberWorldModelV2`.
3. **ML Inference:** Calls `ModelService.forecast(x_seq, k_steps=4)`.
4. **Enriched Event Composition:** Assembles full telemetry metadata, calibrated probabilities, risk scores, MITRE tactics, top feature deltas, and K=4 rollout predictions.
5. **Real-time Broadcast:** Dispatches the event to all active WebSocket (`/api/v1/stream/ws`) and SSE (`/api/v1/stream/events`) subscribers.

---

## 5. Fail-Closed Principles

If any component in the telemetry-inference chain is degraded or unavailable, the system strictly **fails closed**:

| Fault Condition | System Response | Hardcoded Fallback? |
| :--- | :--- | :--- |
| Model weights uninitialized | Emits `{"status": "MODEL_UNAVAILABLE"}` | **NO** — Never emits fake stages |
| Scaler artifact missing | Emits `{"status": "SCALER_UNAVAILABLE"}` | **NO** — Halts inference |
| Empty / corrupt window | Emits `{"status": "INVALID_TELEMETRY"}` | **NO** — Skips window |
| Calibration artifact missing | Defaults to $T=1.0$ uncalibrated logit scaling | **NO** — Softmax computed live |
| Subscriber queue full | Drops frame for slow client, preserves server health | **NO** — Backpressure enforced |
