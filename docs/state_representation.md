# CyberSentinel AI — Network State Vector ($S_t$) Specification

Each 30-second temporal window is condensed into a 24-dimensional feature vector $S_t \in \mathbb{R}^{24}$.
Features were deliberately chosen to reflect cyber adversary behavior (reconnaissance scanning, brute force, lateral propagation, data staging) while remaining explainable and mathematically bounded.

---

## Curated Feature Catalog

| # | Feature Name | Datatype | Calculation | Security & Behavioral Rationale | Leakage Safe? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `flow_count` | `float32` | Total flows initiated in window $[t, t+\Delta T)$ | Baseline traffic volume metric. Sudden spikes indicate volumetric attacks or port scans. | Yes (in-window only) |
| 2 | `total_packets` | `float32` | Sum of all packets across active flows in window | Volumetric indicator; distinguishes high-packet scans from quiet beacons. | Yes |
| 3 | `total_bytes` | `float32` | Sum of all bytes transferred in window | Heavy payload transfers indicate staging or exfiltration. | Yes |
| 4 | `pkt_rate` | `float32` | `total_packets / window_size` | Rate-normalized packet intensity per second. | Yes |
| 5 | `byte_rate` | `float32` | `total_bytes / window_size` | Rate-normalized bandwidth consumption per second. | Yes |
| 6 | `unique_src_ips` | `float32` | Count of distinct source IPs | Multi-source spikes indicate distributed attacks or botnets. | Yes |
| 7 | `unique_dst_ips` | `float32` | Count of distinct destination IPs | High count from a single host indicates network mapping or sweeping. | Yes |
| 8 | `unique_dst_ports` | `float32` | Count of distinct destination ports | Key indicator of port scanning and service enumeration. | Yes |
| 9 | `syn_count` | `float32` | Number of flows with SYN flag asserted | TCP connection initiation volume; elevated in SYN floods and stealth scans. | Yes |
| 10 | `rst_count` | `float32` | Number of flows with RST flag asserted | Indicates connection rejection by closed ports or firewall reset actions. | Yes |
| 11 | `fin_count` | `float32` | Number of flows with FIN flag asserted | Graceful connection closures; abnormal ratios indicate evasion scans (FIN scan). | Yes |
| 12 | `syn_ratio` | `float32` | `syn_count / max(flow_count, 1)` | Proportion of flows initiating handshakes. $\approx 1.0$ during scans. | Yes |
| 13 | `rst_ratio` | `float32` | `rst_count / max(flow_count, 1)` | Proportion of aborted flows. High during scanning closed port ranges. | Yes |
| 14 | `dst_ip_entropy` | `float32` | $-\sum p_i \log_2(p_i)$ for destination IP distribution | Measures dispersion of destination hosts. High entropy indicates horizontal scanning. | Yes |
| 15 | `dst_port_entropy` | `float32` | $-\sum p_i \log_2(p_i)$ for destination port distribution | Measures port dispersion. High entropy indicates vertical port scanning. | Yes |
| 16 | `failed_flow_count` | `float32` | Number of flows terminating abnormally | Failed authentication or closed port rejections. | Yes |
| 17 | `failed_flow_ratio` | `float32` | `failed_flow_count / max(flow_count, 1)` | High in brute force attempts (SSH-Patator) and scanning. | Yes |
| 18 | `tcp_flag_diversity`| `float32` | Shannon entropy across 6-tuple TCP flag patterns | Normal traffic clusters on SYN/ACK/FIN. Scanning generates diverse abnormal flag combos. | Yes |
| 19 | `port_445_share` | `float32` | `count(dst_port == 445) / max(flow_count, 1)` | SMB traffic proportion. Spikes strongly correlate with Lateral Movement (EternalBlue, PsExec). | Yes |
| 20 | `port_3389_share` | `float32` | `count(dst_port == 3389) / max(flow_count, 1)`| Remote Desktop Protocol (RDP) traffic. Indicates interactive lateral access or staging. | Yes |
| 21 | `port_22_share` | `float32` | `count(dst_port == 22) / max(flow_count, 1)` | SSH traffic proportion. Spike with high failure indicates Credential Access brute force. | Yes |
| 22 | `port_80_443_share`| `float32` | `count(dst_port in {80,443,8080,8443}) / flow_count` | Web traffic fraction; acts as a baseline anchor for normal business operations. | Yes |
| 23 | `mean_flow_duration`| `float32` | Average flow duration in seconds | Short flows indicate scans; long flows indicate persistent tunnels or large transfers. | Yes |
| 24 | `bytes_per_packet` | `float32` | `total_bytes / max(total_packets, 1)` | Payload density. Small values indicate probes; large values (>800) indicate data exfiltration. | Yes |

---

## Mathematical Guarantees

1. **Numerical Boundedness**:
   - Entropy metrics are lower-bounded by $0.0$.
   - All ratio and share features strictly reside in the interval $[0.0, 1.0]$.
   - Rates and counts are strictly non-negative ($\ge 0.0$).
2. **Deterministic Computation**:
   - Order-independent aggregation ensures that parallel or re-ordered packets within window $[t_0, t_0+\Delta T)$ produce identical feature representations.
