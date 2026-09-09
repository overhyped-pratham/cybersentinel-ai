# CyberSentinel AI — Phase 16: Two-Machine Live Demonstration Runbook

This runbook guides operators through executing a live demonstration of CyberSentinel AI using **two physical laptops connected to the same isolated local area network (LAN)**.

---

## 1. Network Topology & Pre-Flight Checklist

```
+------------------------------------+         +-------------------------------------+
|       LAPTOP 2: TEST GENERATOR     |         |     LAPTOP 1: CYBERSENTINEL NODE    |
|       (IP: 192.168.1.105)          |         |         (IP: 192.168.1.100)         |
+------------------------------------+         +-------------------------------------+
| - Runs traffic generator script    |         | - Runs FastAPI backend & Uvicorn    |
| - Emits RFC 3954 NetFlow v5 UDP    |         | - NetFlow UDP socket (0.0.0.0:9995) |
| - Controls lifecycle phases        |  UDP    | - 24-D State Builder + FeatureScaler|
| - Label: UNKNOWN (zero hint)       | =======>| - CyberWorldModelV2 Inference Engine|
+------------------------------------+  9995   | - Live HTML5 Dashboard (port 8000)  |
                                               +-------------------------------------+
```

### Pre-Flight Checklist:
1. Connect both laptops to the same Wi-Fi network or Ethernet switch (subnet: `192.168.1.0/24`).
2. On Laptop 1, determine its local IP via PowerShell `ipconfig` (assumed here to be `192.168.1.100`).
3. Ensure firewall rules on Laptop 1 allow inbound traffic:
   - **TCP Port 8000**: HTTP / WebSocket Dashboard
   - **UDP Port 9995**: NetFlow v5 Telemetry Ingestion
   ```powershell
   # On Laptop 1 (Administrator PowerShell):
   netsh advfirewall firewall add rule name="CyberSentinel Web" dir=in action=allow protocol=TCP localport=8000
   netsh advfirewall firewall add rule name="CyberSentinel NetFlow" dir=in action=allow protocol=UDP localport=9995
   ```
4. Verify connectivity from Laptop 2:
   ```powershell
   # On Laptop 2:
   Test-NetConnection -ComputerName 192.168.1.100 -Port 8000
   ```

---

## 2. Laptop 1: Starting the CyberSentinel Command Center

### Step 1: Launch Backend Server
On Laptop 1:
```powershell
cd "d:\uec sih"
python scripts/start_server.py --host 0.0.0.0 --port 8000
```
Verify terminal output confirms:
- `CyberWorldModelV2 loaded (T=1.5680)`
- `Production FeatureScaler loaded from models/scaler.pkl`
- `Uvicorn running on http://0.0.0.0:8000`

### Step 2: Open Command Center Dashboard
In a browser on Laptop 1 (or Laptop 2):
```
http://192.168.1.100:8000/ui/index.html
```

### Step 3: Start Live NetFlow Ingestion Session
You can start the session via the Dashboard UI or with a single curl command:
```powershell
curl -X POST http://127.0.0.1:8000/api/v1/stream/start `
  -H "Content-Type: application/json" `
  -d '{"source_kind":"netflow","host":"0.0.0.0","port":9995,"window_seconds":10.0,"k_steps":4,"mode":"LIVE"}'
```
Notice the Dashboard top banner transitions to:
`MODE: LIVE | SOURCE: NetFlow UDP 0.0.0.0:9995 | STATUS: ACTIVE`

---

## 3. Laptop 2: Transmitting Controlled Telemetry

Laptop 2 executes the 3-phase lifecycle, simulating an adversary or heavy workload amidst normal baseline traffic.

### Phase 1: Baseline Normal Traffic
Transmit 6 windows of normal browsing, DNS queries, and service traffic:
```powershell
python scripts/multi_host_traffic_generator.py `
  --target-host 192.168.1.100 `
  --target-port 9995 `
  --pattern normal_background `
  --windows 6 `
  --interval 10.0
```
**Expected Dashboard Observation:**
- Throughput: ~5–15 KB/s
- Current Stage: `BENIGN`
- Risk Level: `LOW` / `MODERATE`
- MITRE Mapping: `T1595` or none

---

### Phase 2: Controlled High-Activity Traffic
Transmit 6 windows of sustained high-volume data exfiltration:
```powershell
python scripts/multi_host_traffic_generator.py `
  --target-host 192.168.1.100 `
  --target-port 9995 `
  --pattern large_data_transfer `
  --windows 6 `
  --interval 10.0
```
*(Alternative high-activity patterns available: `connection_burst`, `repeated_attempts`, `port_diversity`)*

**Expected Dashboard Observation:**
- Throughput: Spikes to > 500 KB/s
- Current Stage: Shifts dynamically to `EXFILTRATION` / `LATERAL_MOVEMENT`
- Attack Probability: Climbs to `> 99%`
- Risk Score: Elevates to `95 - 100` (`CRITICAL`)
- Primary MITRE Technique: Updates to `T1048` (*Exfiltration Over Alternative Protocol*)
- Feature Attribution: Top increasing feature shows `byte_rate` / `total_bytes`

---

### Phase 3: Recovery to Normal Traffic
Cease the high-activity transfer and resume normal baseline traffic:
```powershell
python scripts/multi_host_traffic_generator.py `
  --target-host 192.168.1.100 `
  --target-port 9995 `
  --pattern normal_background `
  --windows 6 `
  --interval 10.0
```
**Expected Dashboard Observation:**
- Throughput drops back to baseline
- State vector gradually purges high-volume transfer as sliding window advances
- Risk score stabilizes and begins recovery trajectory
- Stage prediction settles back

---

## 4. Automated Two-Machine Verification Script

To automate this entire sequence and generate an audit report with full raw measurements:
```powershell
python scripts/run_phase16_twomachine_demo.py `
  --target-host 192.168.1.100 `
  --target-port 9995 `
  --pattern large_data_transfer `
  --windows-per-phase 6 `
  --window-seconds 10.0 `
  --report docs/phase16_twomachine_demo_report.md
```

Outputs:
1. `docs/phase16_twomachine_demo_report.md` (Markdown validation report)
2. `experiments/phase16/twomachine_live_results.json` (Structured JSON observations)
