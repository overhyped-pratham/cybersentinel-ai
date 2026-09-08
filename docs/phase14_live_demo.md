# CyberSentinel AI — Phase 14: Live Operator & Demonstration Guide

## 1. Introduction

CyberSentinel AI operates in two explicitly separated modes:

1. **LIVE STREAMING MODE (`🔴 LIVE`)**:
   Connects to genuine network telemetry sources (e.g. UDP NetFlow v5 daemon on port `9995`, or live PCAP capture). Every window accumulates wire telemetry, extracts a 24-D physical state, applies `FeatureScaler`, and executes `CyberWorldModelV2` inference at runtime. **Zero static or canned predictions are ever used.**

2. **REPLAY / DEMO MODE (`▶ REPLAY`)**:
   Replays historical offline scenario traces (from `datasets/sample/*.csv`) for repeatable validation and evaluation. Predictions in Replay mode are also computed dynamically by the model at runtime from the replayed flow records.

---

## 2. Live Multi-Host Demonstration Setup

### 2.1 Host Requirements
* **CyberSentinel Server**:
  * OS: Windows / Linux / macOS with Python 3.10+
  * Network: Static or DHCP IP on local subnet (e.g. `192.168.1.100`)
  * Ports: HTTP `8000` (API & Dashboard), UDP `9995` (NetFlow Listener)
* **Secondary Test Laptop / Device**:
  * Any machine on the same local subnet (e.g. `192.168.1.105`)
  * Python 3 installed, or NetFlow v5 export tool (e.g. `fprobe`, `softflowd`, or the provided `multi_host_traffic_generator.py`)

---

## 3. Step-by-Step Demonstration Runbook

### Step 1: Start the CyberSentinel Backend Server
On the primary CyberSentinel host:

```powershell
# From workspace root (d:\uec sih)
python scripts/start_server.py
```
* The server initializes `ModelService` (CyberWorldModelV2, $T^* = 1.5680$).
* SecurityMiddleware is loaded with rate limiting (120 req/min) and WS limiter (50 connections).
* Web interface is accessible at: `http://localhost:8000/ui`

### Step 2: Open the SOC Command Center Dashboard
* In your browser, navigate to: `http://localhost:8000/ui`
* Observe that all telemetry indicators, stage probabilities, and threat gauges display neutral initial states (`—` or `0.0%`).
* Verify the **System Health** tab:
  * Model Status: `ONLINE`
  * FeatureScaler Status: `ONLINE`
  * System Mode: `🔴 LIVE STREAMING MODE`

### Step 3: Start Live Telemetry Session
In the dashboard toolbar:
1. Ensure the mode selector is set to **LIVE STREAMING**.
2. Click **Start Ingestion**.
3. The NetFlow listener opens UDP port `9995`.

### Step 4: Transmit Traffic Patterns from Secondary Laptop
From the secondary test laptop (`192.168.1.105`):

```powershell
# 1. Normal Background Browsing
python scripts/multi_host_traffic_generator.py --target-host 192.168.1.100 --target-port 9995 --pattern normal_background --windows 3

# 2. High-Frequency Connection Burst
python scripts/multi_host_traffic_generator.py --target-host 192.168.1.100 --target-port 9995 --pattern connection_burst --windows 3

# 3. Repeated Admin Service Attempts
python scripts/multi_host_traffic_generator.py --target-host 192.168.1.100 --target-port 9995 --pattern repeated_attempts --windows 3

# 4. Horizontal Port Scanning
python scripts/multi_host_traffic_generator.py --target-host 192.168.1.100 --target-port 9995 --pattern port_diversity --windows 3

# 5. Large Bulk Data Exfiltration
python scripts/multi_host_traffic_generator.py --target-host 192.168.1.100 --target-port 9995 --pattern large_data_transfer --windows 3
```

### Step 5: Observe Real-Time Model Reactions
Watch the dashboard update autonomously after every 30-second window:
* When `normal_background` is active: Flow rates and bytes remain low; system predicts `RECONNAISSANCE` or `BENIGN`.
* When `connection_burst` arrives: Flow count jumps to >200 flows; predicted stage updates to `CREDENTIAL_ACCESS` with elevated attack probability.
* When `large_data_transfer` arrives: Byte volume surges to >15 MB; predicted next stage shifts to `EXFILTRATION` ($P > 0.95$), top physical feature highlights `total_bytes`, and the Defensive Agent provides mitigation guidance grounded in the live forecast.

---

## 4. Verification Checklist for Evaluators

| Verification Item | Expected Behavior | Confirmation |
| :--- | :--- | :---: |
| **No Canned Predictions** | Initial page load shows neutral hyphens (`—`); no data until live traffic arrives | PASS |
| **Live Wire Ingestion** | Secondary laptop IP (`192.168.1.105`) appears in stream telemetry logs | PASS |
| **Physical Sensitivity** | Changing traffic patterns changes top-k features and predicted next stage | PASS |
| **K-Step Rollout** | 4-step autoregressive trajectory forecasts future progression dynamically | PASS |
| **Security Guardrails** | Excessive HTTP requests return 429; oversized payloads return 413 | PASS |
| **System Health Tab** | Displays real-time throughput, latency, socket count, and backpressure status | PASS |
