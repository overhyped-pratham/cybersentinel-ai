# CyberSentinel AI — Derived Attack Stage Labeling Policy

> [!IMPORTANT]
> **RESEARCH DISCLAIMER**:
> Stage labels are derived from scenario timelines and flow characteristics using a documented mapping policy. They are a research approximation, **NOT** authoritative ATT&CK ground truth.
> Public network intrusion datasets (such as CICIDS2017, UNSW-NB15, or CSE-CIC-IDS2018) provide attack category tags (e.g., `PortScan`, `FTP-Patator`, `Infiltration`), but **do not** include standardized MITRE ATT&CK tactical sequence ground truth.

---

## 1. Tactical Taxonomy

CyberSentinel AI adopts an eight-stage lifecycle taxonomy rooted in the MITRE Enterprise ATT&CK matrix:

1. **BENIGN**: Normal business operations (HTTP, HTTPS, DNS, standard internal RPC).
2. **RECONNAISSANCE**: Active adversary scanning, network discovery, service probe sweeps.
3. **INITIAL_ACCESS**: Public-facing application exploitation, DoS/DDoS entry vectoring.
4. **EXECUTION**: Malicious payload execution, reverse shells.
5. **CREDENTIAL_ACCESS**: Authentication brute force, password spraying, credential harvesting.
6. **DISCOVERY**: Internal enumeration of active domain controllers, file shares, or subnet boundaries.
7. **LATERAL_MOVEMENT**: Internal pivot from compromised host to peer systems via SMB (445), RDP (3389), or SSH (22).
8. **COMMAND_AND_CONTROL**: Outbound beaconing, interactive command channel maintenance.
9. **EXFILTRATION**: Bulk confidential data transfer, unauthorized outbound staging.
10. **UNKNOWN**: Ambiguous or unclassified window with conflicting or insufficient signatures.

---

## 2. Derivation Policy & Mapping Table

### A. Dataset Tag Mapping

| Dataset Tag / Keyword | Derived Stage | Rationale |
| :--- | :--- | :--- |
| `BENIGN`, `Normal` | **BENIGN** | Normal background network baseline. |
| `PortScan`, `Nmap`, `IPsweep` | **RECONNAISSANCE** | Adversary searching for open network pathways and vulnerable listening services. |
| `DoS`, `DDoS`, `Heartbleed` | **INITIAL_ACCESS** | Network infrastructure stress, perimeter penetration, service disruption. |
| `FTP-Patator`, `SSH-Patator`, `Brute Force` | **CREDENTIAL_ACCESS** | Direct credential cracking against network service daemons. |
| `Infiltration`, `SMB_Spread` | **LATERAL_MOVEMENT** | Propagation through internal corporate subnets toward high-value assets. |
| `Bot`, `Botnet`, `C2`, `Beacon` | **COMMAND_AND_CONTROL** | Automated periodic communication with command nodes. |
| `Exfiltration`, `Data_Leak` | **EXFILTRATION** | Extraction of internal records to external infrastructure. |

### B. Behavioral Heuristic Policy (For Unlabeled or Ambiguous Streams)

When raw labels are missing, the following policy derives stage assignments:

1. **Reconnaissance**:
   $$\text{dst\_port\_entropy} > 2.5 \quad \text{AND} \quad (\text{syn\_ratio} > 0.40 \; \text{OR} \; \text{unique\_dst\_ports} > 15)$$
2. **Credential Access**:
   $$\text{port\_22\_share} > 0.25 \quad \text{AND} \quad \text{failed\_flow\_ratio} > 0.30$$
3. **Lateral Movement**:
   $$\text{port\_445\_share} > 0.30 \quad \text{OR} \quad \text{port\_3389\_share} > 0.30$$
4. **Exfiltration**:
   $$\text{byte\_rate} > 50,000 \text{ bytes/s} \quad \text{AND} \quad \text{bytes\_per\_packet} > 800$$
5. **Benign**:
   $$\text{Default for active flows not satisfying any malicious attack criteria.}$$

---

## 3. Strict Architectural Separation

To prevent deceptive marketing or unscientific claims, the codebase strictly distinguishes:

- **Observed**: Concrete physical metrics measured directly from packets (e.g. `port_445_share = 0.42`, `byte_rate = 85000`).
- **Derived**: Attack stage label inferred by our documented labeling policy (`LATERAL_MOVEMENT`).
- **Predicted**: Probabilities produced by the PyTorch Cyber World Model ($\hat{y}_{t+1} = \text{Command \& Control}, p = 0.84$).
- **Mapped**: External MITRE ATT&CK technique IDs associated by the rule-based knowledge engine (`T1021.002 - SMB/Windows Admin Shares`).
- **Generated**: Natural-language text synthesized by the offline AI agent to explain the threat to a SOC analyst.
