# CyberSentinel AI — Phase 14: Security Hardening Architecture

## 1. Security Architecture Overview

Phase 14 delivers production-grade defense-in-depth hardening across all ingestion, API, and WebSocket surfaces. CyberSentinel operates exclusively as a **defensive, read-only threat forecasting system**. It contains no packet injection capabilities, no offensive probe utilities, and adheres strictly to fail-closed design principles.

```
Incoming Request / Packet
          │
          ▼
┌────────────────────────────────────────────────────────┐
│  SecurityMiddleware (Starlette / ASGI Layer)          │
│                                                        │
│  1. Content-Length Check   (Max 10 MB Payload)         │
│  2. API Key Verification   (X-API-Key Header)          │
│  3. Sliding-Window Limiter (120 req/min per Client IP) │
│  4. Safe Response Headers  (nosniff, DENY, XSS block)  │
└────────────────────────┬───────────────────────────────┘
                         │ Passed
                         ▼
┌────────────────────────────────────────────────────────┐
│  WebSocket Connection Guard (Max 50 Concurrent Slots)  │
└────────────────────────┬───────────────────────────────┘
                         │ Slot Acquired
                         ▼
┌────────────────────────────────────────────────────────┐
│  FastAPI Endpoints & Live Telemetry Ingestion Pipeline │
└────────────────────────────────────────────────────────┘
```

---

## 2. Hardening Mechanisms

### 2.1 Sliding-Window Rate Limiter
* **Module**: `backend.middleware.security.SecurityManager`
* **Configuration**: `CYBERSENTINEL_RATE_LIMIT` (Default: `120` requests/minute per client IP).
* **Behavior**:
  * Tracks request timestamps in an in-memory sliding window deque per client IP.
  * Timestamps older than 60 seconds are automatically evicted.
  * When request frequency exceeds the configured limit, the middleware halts processing immediately and returns:
    * HTTP Status: `429 Too Many Requests`
    * Headers: `Retry-After: 60`
    * Body: `{"detail": "Too Many Requests: Rate limit exceeded"}`
  * Client IP tracking is isolated: rate limit exhaustion on IP `A` has zero impact on IP `B`.

### 2.2 WebSocket Connection Limiter
* **Module**: `backend.middleware.security.SecurityManager`
* **Configuration**: `CYBERSENTINEL_MAX_WS_CONNECTIONS` (Default: `50` concurrent connections).
* **Behavior**:
  * Tracks active live telemetry subscribers connected to `/api/v1/stream/ws`.
  * Atomic acquisition via `security_manager.acquire_ws_slot()`.
  * If maximum subscriber capacity is reached, new incoming WebSocket upgrade attempts are rejected with WebSocket close code:
    * Close Code: `1013` (`Try Again Later`)
    * Reason: `Concurrent WebSocket connection limit reached (50 max)`
  * When a client disconnects, `release_ws_slot()` is invoked in a `finally` block to prevent slot leakage.

### 2.3 API Key Authentication
* **Module**: `backend.middleware.security.SecurityManager`
* **Configuration**: `CYBERSENTINEL_API_KEY` (Environment variable).
* **Behavior**:
  * **Open LAN Mode (Default)**: If `CYBERSENTINEL_API_KEY` is unset or empty, the API permits local access without credentials, simplifying authorized lab deployment.
  * **Enforced Mode**: When `CYBERSENTINEL_API_KEY` is configured, all requests to API endpoints (excluding static frontend assets `/ui` and documentation `/docs`) must supply an exact match in the `X-API-Key` request header.
  * Requests with missing or incorrect keys are rejected with:
    * HTTP Status: `401 Unauthorized`
    * Body: `{"detail": "Invalid or missing X-API-Key header"}`

### 2.4 Payload Size Bounding
* **Constraint**: `MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024` (10 Megabytes).
* **Behavior**:
  * Prevents memory-exhaustion Denial-of-Service attacks from oversized JSON or raw telemetry payloads.
  * Any request bearing a `Content-Length` greater than 10 MB is aborted prior to body parsing:
    * HTTP Status: `413 Payload Too Large`
    * Body: `{"detail": "Payload Too Large: Max body size is 10 MB"}`

### 2.5 Safe Security Headers
Every HTTP response emitted by the application carries strict security headers to prevent clickjacking, MIME-type sniffing, and cross-site scripting:
```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
```

### 2.6 Fail-Closed Ingestion & Backpressure
* **Missing Model/Scaler**: If `CyberWorldModelV2` or `FeatureScaler` are missing, the pipeline emits explicit `MODEL_UNAVAILABLE` or `SCALER_UNAVAILABLE` notifications. It **never** returns cached, fabricated, or default predictions.
* **Subscriber Backpressure**: If a downstream WebSocket subscriber queue becomes full, `LiveIngestService` drops the unconsumed event with a warning log rather than blocking the real-time ingestion thread or causing memory growth. The health status reports `queue_backpressure_status: "warning"`.

---

## 3. Operational Observability

System operational health is exposed via `GET /api/v1/stream/health`:

```json
{
  "status": "healthy",
  "mode": "LIVE",
  "telemetry_source": "NetFlowSource(0.0.0.0:9995)",
  "flows_per_second": 42.5,
  "current_window_id": "win_d7a8f1",
  "inference_latency_ms": 3.42,
  "model_version": "CyberWorldModelV2-Phase8C",
  "model_available": true,
  "scaler_available": true,
  "websocket_subscribers": 1,
  "active_ws_connections": 1,
  "max_ws_connections": 50,
  "dropped_malformed_count": 0,
  "queue_backpressure_status": "nominal",
  "last_inference_timestamp": "2026-09-08T16:05:00.123456+00:00",
  "uptime_windows_inferred": 18,
  "timestamp": "2026-09-08T16:05:01.000000+00:00"
}
```

The live SOC Dashboard polls this endpoint every 5 seconds to provide visual health verification.
