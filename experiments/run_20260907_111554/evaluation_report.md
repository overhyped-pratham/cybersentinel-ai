# CyberSentinel AI — Baseline Evaluation Report (run_20260907_111554)

## 1. Attack Detection Comparison (Binary)

| Model | Accuracy | Precision (Macro) | Recall (Macro) | Macro F1 | FPR | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Static $S_t$)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| **Temporal GRU ($[S_{t-7}, \dots, S_t]$)** | 0.8182 | 0.8837 | 0.7727 | 0.7872 | 0.4545 | 0.9256 | 0.9450 |

---

## 2. Next-Stage Attack Forecasting Comparison

| Model | Input Formulation | Next-Stage Accuracy (Top-1) | Top-3 Accuracy | Macro F1 | Brier Score (Lower is better) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Current Window $S_t$ | 1.0000 | 1.0000 | 1.0000 | 0.0114 |
| **Temporal GRU Baseline** | Sequence $[S_{t-7}, \dots, S_t]$ | 0.7091 | 0.8909 | 0.7207 | 0.3177 |

---

## 3. Training & Dataset Parameters

- **Train Scenarios**: 22 traces (242 sequences)
- **Validation Scenarios**: 5 traces (55 sequences)
- **Test Scenarios**: 5 traces (55 sequences)
- **Disjoint Split Overlap**: 0.0% (Verified strictly zero scenario leakage)
- **Feature Scaler**: RobustScaler fitted strictly on Train split
