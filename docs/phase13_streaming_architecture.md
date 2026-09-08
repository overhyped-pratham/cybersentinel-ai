# PHASE 13: STREAMING ARCHITECTURE & API REFERENCE

**Date:** September 8, 2026  
**Status:** **OPERATIONAL & VALIDATED**  
**Transport Support:** WebSocket (`/api/v1/stream/ws`), Server-Sent Events (`/api/v1/stream/events`), REST Controls  

---

## 1. Architecture Overview

```
                      +-----------------------------+
                      | Live Telemetry Sources      |
                      | (PCAP / NetFlow / Replay)   |
                      +--------------+--------------+
                                     | FlowRecord stream
                                     v
                      +-----------------------------+
                      | StreamProcessor             |
                      | - Window aggregator (30s)   |
                      | - Backpressure protection   |
                      | - Malformed sanitization    |
                      +--------------+--------------+
                                     | TelemetryWindowEvent
                                     v
                      +-----------------------------+
                      | LiveIngestService           |
                      | - Sliding sequence buffer   |
                      | - NetworkStateBuilder       |
                      | - CyberWorldModelV2         |
                      | - RiskEngine + MITRE        |
                      +--------------+--------------+
                                     | Broadcast
                      +--------------+--------------+
                      |                             |
                      v                             v
           +--------------------+         +--------------------+
           | WebSocket Handler  |         | SSE Handler        |
           | /api/v1/stream/ws  |         | /api/v1/stream/events
           +---------+----------+         +---------+----------+
                     |                              |
                     +--------------+---------------+
                                    v
                         SOC Frontend Dashboard
```

---

## 2. Streaming Endpoints

### 2.1 WebSocket Endpoint: `/api/v1/stream/ws`
Bidirectional, low-latency streaming channel.

- **URL:** `ws://<host>:8000/api/v1/stream/ws`
- **Protocol:** JSON messages over WebSocket.

#### Server → Client Messages
- **Connection Greeting:**
  ```json
  {
    "type": "connected",
    "timestamp": "2026-09-08T15:00:00.000Z",
    "message": "CyberSentinel AI live stream active"
  }
  ```
- **Heartbeat (every 15s):**
  ```json
  {
    "type": "heartbeat",
    "timestamp": "2026-09-08T15:00:15.000Z"
  }
  ```
- **Live Forecast Event (`status: "FORECAST"`):**
  ```json
  {
    "status": "FORECAST",
    "event_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "window_id": "c1f7b0f1-4f01-4475-b6d8-9db9e68bc6a4",
    "session_id": "live_session_01",
    "timestamp": "2026-09-08T15:00:30.000Z",
    "window_start": 0.0,
    "window_end": 30.0,
    "flow_count": 62,
    "flows_per_second": 2.07,
    "dropped_malformed": 0,
    "source_id": "PCAPSource(live_capture.pcap)",
    "inference_latency_ms": 15.2,
    "model_version": "CyberWorldModelV2-Phase8C",
    "provenance": "CyberWorldModelV2.forecast()",
    "current_stage": "RECONNAISSANCE",
    "predicted_next_stage": "CREDENTIAL_ACCESS",
    "attack_probability": 0.9842,
    "confidence": 0.8912,
    "transition_detected": true,
    "transition_probability": 0.8912,
    "stage_probabilities": {
      "BENIGN": 0.0012,
      "RECONNAISSANCE": 0.0912,
      "CREDENTIAL_ACCESS": 0.8912,
      "EXFILTRATION": 0.0164
    },
    "risk_score": 78.4,
    "risk_level": "HIGH",
    "recommended_priority": "P1 — High Priority",
    "time_to_transition_hint": "Imminent (1-2 windows)",
    "top_features": [
      {
        "feature": "dst_port_entropy",
        "current": 4.12,
        "predicted": 1.25,
        "abs_change": 2.87,
        "rel_change_pct": -69.66,
        "direction": "decrease"
      }
    ],
    "explanation_narrative": "Attack progression forecast from RECONNAISSANCE to CREDENTIAL_ACCESS...",
    "mitre_techniques": [
      {
        "technique_id": "T1110",
        "name": "Brute Force",
        "tactic": "Credential Access"
      }
    ],
    "rollout_steps": [
      {"step": 1, "predicted_stage": "CREDENTIAL_ACCESS", "confidence": 0.8912},
      {"step": 2, "predicted_stage": "CREDENTIAL_ACCESS", "confidence": 0.8421},
      {"step": 3, "predicted_stage": "LATERAL_MOVEMENT", "confidence": 0.7615},
      {"step": 4, "predicted_stage": "EXFILTRATION", "confidence": 0.6934}
    ],
    "safety_flags": {
      "nan_detected": false,
      "collapse_detected": false,
      "feature_dim_valid": true
    }
  }
  ```

#### Client → Server Control Messages
- `{"type": "ping"}` → responds with `{"type": "pong", "timestamp": ...}`
- `{"type": "unsubscribe"}` → gracefully closes socket

---

### 2.2 Server-Sent Events Endpoint: `/api/v1/stream/events`
Unidirectional SSE stream using the standard browser `EventSource` protocol.

- **URL:** `http://<host>:8000/api/v1/stream/events`
- **Headers:** `Content-Type: text/event-stream`, `Cache-Control: no-cache`
- **Event Types:** `connected`, `buffering`, `forecast`, `heartbeat`, `error`

---

### 2.3 Session Management REST Endpoints

| Method | Path | Description | Request Body | Response |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/stream/start` | Start live telemetry ingestion | `StartSessionRequest` (`source_kind`, `source_path`, `window_seconds`, `k_steps`) | `{"status": "started", "session_id": "..."}` |
| `POST` | `/api/v1/stream/stop` | Stop live session | `StopSessionRequest` (`session_id`) | `{"status": "stopped"}` |
| `GET` | `/api/v1/stream/status` | Active sessions and queue stats | None | Current status and metrics |
| `GET` | `/api/v1/stream/history`| Last N emitted forecast events | Query parameter `limit` | List of recent events |
