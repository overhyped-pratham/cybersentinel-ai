# CyberSentinel AI — Mobile Traffic Simulator Guide

## 1. Executive Summary & Objective

The **CyberSentinel Mobile Traffic Simulator** (`dashboard/simulator.html`) is a responsive, touch-friendly Progressive Web App (PWA) designed for interactive cybersecurity demonstrations and evaluations (e.g., hackathon judging, executive briefings, SOC simulations).

It allows an operator to generate controlled, synthetic network telemetry patterns directly from any mobile phone (iOS / Android) or tablet and transmit them over HTTP to the CyberSentinel AI backend running on a host laptop.

> [!IMPORTANT]
> **Defensive Demonstration Only**: The simulator does NOT perform active network port scanning, exploitation, credential stuffing, or packet flooding. All telemetry is generated purely as structured flow records sent directly to the defensive API endpoint (`POST /api/v1/stream/ingest`).

---

## 2. End-to-End Live Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 MOBILE DEVICE (iOS / Android)               │
│  - Operator taps traffic profile (Normal, Recon, Cred, ...) │
│  - Synthesizes flow records with label="UNKNOWN"            │
│  - Submits HTTP POST /api/v1/stream/ingest                 │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP JSON (LAN)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                HOST LAPTOP (CyberSentinel API)              │
│                                                             │
│ 1. Ingestion Layer:                                         │
│    POST /api/v1/stream/ingest → LiveIngestService           │
│                                                             │
│ 2. Feature Pipeline:                                        │
│    NetworkStateBuilder → 24-D State Vector S_t              │
│    Production FeatureScaler (models/scaler.pkl)             │
│                                                             │
│ 3. ML Inference Engine:                                     │
│    CyberWorldModelV2 (Transformer / GRU temporal model)     │
│    - Predicts current attack stage                          │
│    - Forecasts next attack stage                            │
│    - Evaluates attack probability & transition probability  │
│    - K-step rollout simulation horizon                      │
│                                                             │
│ 4. Defensive Coupling:                                      │
│    - Explainability (top physical telemetry deltas)         │
│    - MITRE ATT&CK Mapping (dynamic technique assignment)    │
│    - Mathematical Risk Engine (0–100 calibrated score)      │
│                                                             │
│ 5. Real-Time Distribution:                                  │
│    - Returns HTTP JSON to Mobile Simulator Feedback Card    │
│    - Broadcasts live WebSocket event to SOC Command Center  │
└──────────────────────────────┬──────────────────────────────┘
                               │ WebSocket (/api/v1/stream/ws)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                SOC COMMAND CENTER DASHBOARD                 │
│              (http://localhost:8000/ui/index.html)          │
│  - Live Threat Timeline updates instantly                   │
│  - Attack Lifecycle Stage transitions                       │
│  - Mitre ATT&CK techniques & defensive narratives           │
│  - Risk gauge shifts in real time                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Quick-Start Demonstration Guide

### Step 1: Start CyberSentinel Backend on Laptop
Run the launch script from the project root:
```powershell
python scripts/start_server.py
```
Or start Uvicorn directly:
```powershell
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### Step 2: Open Command Center on Laptop
Open your browser on the laptop:
```
http://localhost:8000/ui/index.html
```
Verify the connection indicator shows `CONNECTED` (green dot).

### Step 3: Connect Mobile Device
1. Connect your smartphone to the same Wi-Fi network as your laptop.
2. Find your laptop's local IPv4 address (e.g. `ipconfig` on Windows → `192.168.1.10`).
3. Open mobile browser (Chrome / Safari) and navigate to:
   ```
   http://192.168.1.10:8000/ui/simulator.html
   ```
4. Tap **Test Ping** — the status pill will turn green (`CONNECTED`).
5. *(Optional)* Tap "Add to Home Screen" to install as a standalone full-screen PWA.

---

## 4. The 6 Synthetic Traffic Profiles

Every profile generates standardized network flows with realistic headers and statistical characteristics:

| Profile | Icon | Telemetry Characteristics | Expected AI Defensive Reaction |
|---|---|---|---|
| **Normal Traffic** | 🌐 | Ports 80, 443, 53; balanced sizes (500–1500 B); SYN/ACK; 0% failed flows | Model maintains `Normal` stage; risk score remains `LOW` (<25) |
| **Connection Burst** | ⚡ | Rapid bursts to web ports; 1–5ms durations; high flow rate; SYN/FIN | Flow volume spikes; model analyzes temporal dynamics |
| **Recon Scanning** | 🔍 | Port spread across 20–1024; high port entropy; small pkts (44B); 40% RST errors | Model detects scanning patterns (`Reconnaissance`); maps MITRE T1046 |
| **Credential Access** | 🔑 | Ports 22 (SSH), 3389 (RDP), 445 (SMB), 1433 (DB); 70% RST/failed flows | Model detects authentication anomalies (`CredentialAccess`); maps MITRE T1110 |
| **Large Transfer** | 📦 | Port 443; 300–1200 packets/flow; full MTU payloads (1420 B); massive byte rate | Byte rate and bytes/packet spike (`Exfiltration` / Data Staging) |
| **Bot Beaconing** | 📡 | Strict 1.0s periodic intervals; constant 128B size; zero variance; persistent | Periodic temporal rhythm indicates C2 beaconing behavior |

---

## 5. Strict Defensive & Anti-Hardcoding Discipline

CyberSentinel AI operates under strict academic and operational integrity rules:

1. **Zero Hardcoded Outcomes**:
   - The simulator frontend JavaScript contains NO hardcoded attack stages, probabilities, or risk scores.
   - All flows generated by the simulator specify:
     ```json
     { "label": "UNKNOWN" }
     ```
2. **Runtime Inference Only**:
   - The host backend converts raw flow records into the standardized 24-D physical network state vector.
   - Vectors are scaled using the production `FeatureScaler` (`models/scaler.pkl`).
   - Predictions are computed on-the-fly by `CyberWorldModelV2` through forward passes and K-step rollouts.
3. **Fail-Closed Safety**:
   - Empty flow batches return HTTP 422 / `INVALID_TELEMETRY`.
   - Out-of-range port numbers or malformed records are dropped.
   - If the model checkpoint or scaler is missing, the backend fails closed without substituting canned outputs.

---

## 6. Live Presentation Script for Judges

Use this script during live demonstrations:

1. **Introduction (30 seconds)**:
   > *"CyberSentinel AI is a proactive temporal threat world model that forecasts multi-step cyber attacks before they reach critical assets. To demonstrate this live, I have our mobile simulator running here on my smartphone, connected over LAN to our AI engine running on the laptop."*

2. **Establish Baseline (30 seconds)**:
   > *"First, I will tap 'Normal Traffic' on my phone and send a batch. Look at the laptop Command Center: notice that the AI correctly identifies legitimate web and DNS traffic, keeping the risk score in the green zone."*

3. **Demonstrate Stage Transition (45 seconds)**:
   > *"Now, I will switch my phone to 'Recon Scanning' and click 'Start Simulation'. Notice the laptop screen: within milliseconds of ingesting the flow telemetry, CyberSentinel's neural network detects the spike in destination port entropy, transitions the threat stage to Reconnaissance, maps MITRE T1046 Network Service Scanning, and raises the risk score."*

4. **Multi-Step Rollout & Explainability (45 seconds)**:
   > *"Notice the K-step rollout horizon in the Command Center: the World Model forecasts the next probable stage (Credential Access or Lateral Movement) with confidence bounds, while the explainability card attributes the forecast to physical telemetry features — failed flow ratios and port diversity."*

5. **Defense Integrity Statement (15 seconds)**:
   > *"Crucially, my phone sent only raw flow data labeled 'UNKNOWN'. There are zero hardcoded rules or canned scripts. The AI model itself performed real-time inference on the scaled 24-D state vector."*
