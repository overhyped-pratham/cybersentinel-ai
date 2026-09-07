# CyberSentinel AI — Data Pipeline Specification

## 1. Pipeline Overview

The CyberSentinel data ingestion pipeline converts disparate raw packet captures (PCAP) or flow logs (CSV) into clean, non-overlapping 30-second network state vectors $S_t$.

$$\text{PCAP / CSV} \xrightarrow{\text{Extraction}} \text{FlowRecord} \xrightarrow{\text{Cleaning}} \text{Windowing (30s)} \xrightarrow{\text{Feature Engineering}} S_t \xrightarrow{\text{Sequencing}} \mathbf{X}_t$$

---

## 2. Ingestion Stages

### Stage 1: Input Ingestion
- **CSV Formats Supported**:
  - CICIDS2017 flow dumps
  - NetFlow v5/v9 and IPFIX CSV exports
  - Zeek/Bro connection logs (`conn.log`)
  - Synthetic scenario benchmark datasets
- **PCAP Binary Ingestion**:
  - Fully offline, pure-Python binary parser without external OS drivers or libpcap bindings.
  - Reassembles raw Ethernet $\to$ IPv4 $\to$ TCP/UDP packets into 5-tuple flow records using a 15-second flow inactivity timeout.

### Stage 2: Schema Normalization & Cleaning
- Standardizes diverse column naming schemes (`Total Fwd Packets`, `Flow Duration`, `Destination Port`, `Timestamp`, etc.) via a unified alias mapping dictionary.
- **Timestamp Parsing**: Handles Unix epoch numbers (seconds, milliseconds, microseconds) and ISO-8601 string representations.
- **Sanitization**:
  - Replaces all infinite values (`+inf`, `-inf`) and `NaN`s with clean zero/boundary defaults.
  - Replaces out-of-range port numbers ($> 65535$) with safe boundaries.
  - Deduplicates repeated flow logs sharing the exact 5-tuple and microsecond timestamp.
  - Strictly orders all flows chronologically by `timestamp` ascending.

### Stage 3: Temporal Window Aggregation
- Configurable window parameter: `WINDOW_SIZE = 30.0` seconds (default).
- Step parameter: `STEP_SIZE = 30.0` seconds (non-overlapping).
- For scenario trace $k$ starting at $t_0$, the $i$-th window covers the half-open interval:
  $$W_i = [t_0 + i \cdot \Delta T, \; t_0 + (i + 1) \cdot \Delta T)$$
- Flows are partitioned strictly into windows by their flow start timestamp $t_{\text{start}}$:
  $$f \in W_i \iff t_0 + i \Delta T \le f.timestamp < t_0 + (i + 1) \Delta T$$
- Windows with fewer than `min_flows_per_window` (default: 1) are skipped or padded.

---

## 3. Data Leakage Prevention Guarantees

1. **No Temporal Lookahead**: State $S_t$ is computed solely from flows occurring within $[W_t.\text{start}, W_t.\text{end})$. Never uses packet arrivals after $W_t.\text{end}$.
2. **No Cross-Scenario Flow Blending**: Flows belonging to different capture sessions or scenario traces are never aggregated into the same window.
3. **Scaler Statistics Isolation**: Preprocessing normalization transforms (mean, standard deviation, median, IQR) are fit strictly on training scenario splits and frozen before transforming test windows.
