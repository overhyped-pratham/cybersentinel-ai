# CyberSentinel X — Held-Out Attack Family / Novel Behavior Evaluation

## Executive Summary
This experiment demonstrates **unseen attack detection** in accordance with PRD Section 10, 17, and Appendix B.
The Known Attack Classifier was trained **without** exposure to the `EXFILTRATION` attack family.
The system is evaluated on its ability to classify known attacks while correctly isolating held-out behaviors as **Potential Novel Behavior**.

---

## 1. Known Attack Performance (Classifier Layer 1)
- **Evaluated Classes**: `BENIGN, CREDENTIAL_ACCESS, LATERAL_MOVEMENT, RECONNAISSANCE`
- **Accuracy**: `98.85%`
- **Macro-F1**: `0.9872`
- **Precision**: `0.9875`
- **Recall**: `0.9875`
- **False Positive Rate (FPR)**: `0.00%`
- **Inference Latency**: `0.03 ms / sample`

---

## 2. Held-Out / Novel Behavior Evaluation (`EXFILTRATION`)
- **Held-Out Family**: `EXFILTRATION` (36 windows)
- **Anomaly Detection Rate**: `100.00%`
- **AUROC (Benign vs Held-Out)**: `1.0000`
- **Mean Anomaly Score**: `1.0000` / 1.0
- **Mean Calibrated Risk Score**: `77.5` / 100
- **System Threat Classification**: `Potential Novel Behavior` (100.0%)
- **Calibrated Novelty Threshold (P95)**: `0.013547`
- **Pipeline Processing Latency**: `25.60 ms / sample`

---

## 3. Data Leakage & Integrity Controls
1. **Strict Partitioning**: `df_states['stage_name'] == 'EXFILTRATION'` was held out prior to all training splits.
2. **Scaler Integrity**: `FeatureScaler` fit exclusively on `X_train_known`.
3. **Threshold Calibration**: The decision threshold `tau` was empirically derived from unseen benign validation windows (P95).
4. **Human-in-the-Loop Language**: Flagged strictly as `"Potential Novel Behavior"`, not a presumed attack type, enabling safe human triage.
