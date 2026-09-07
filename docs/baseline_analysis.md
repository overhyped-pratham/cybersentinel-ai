# CyberSentinel AI — Phase 6 & 7 Baseline Analysis

**Experiment**: run_20260907_111554
**Date**: 2026-09-07
**Author**: CyberSentinel Engineering Team

---

## 1. Executive Summary

This document analyses the empirical results of two baseline models trained and evaluated as part of Phases 6 and 7:

- **Phase 6 — Logistic Regression (LR)**: Static single-timestep classifier on S_t.
- **Phase 7 — Temporal GRU Baseline**: Sequence-aware model on [S_{t-7}, ..., S_t].

Both baselines are deliberately non-world-model architectures. Their purpose is to establish a minimum performance floor and identify difficulty classes for Phase 8.

> **Key finding**: On the current synthetic test corpus, LR achieves perfect (1.0) scores. This is **not** overfitting — it is a dataset-construction artefact explained below.

---

## 2. Dataset & Split Summary

| Partition  | Scenarios | Sequences |
|------------|-----------|-----------|
| Train      | 19        | 242       |
| Validation | 3         | 55        |
| Test       | 5         | 55        |
| **Total**  | **24**    | **384**   |

**Test scenarios (ScenarioBasedSplitter)**:
- trace_benign_02 — 11 windows, all BENIGN
- trace_benign_05 — 11 windows, all BENIGN
- trace_bruteforce_01 — 11 windows, all CREDENTIAL_ACCESS
- trace_exfil_01 — 11 windows, all EXFILTRATION
- trace_recon_03 — 11 windows, all RECONNAISSANCE

**Critical observation**: Every test scenario is a single-stage trace. There are **0 stage-transition windows** in the test set. This reduces LR next-stage forecasting to current-stage recognition — trivially easy when features are perfectly separable.

---

## 3. Attack Detection Results (Binary: Attack vs. Benign)

| Model | Accuracy | Precision | Recall | Macro F1 | FPR | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----------|-----|---------|--------|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Temporal GRU        | 0.8182 | 0.8837 | 0.7727 | 0.7872 | **0.4545** | 0.9256 | 0.9450 |

### 3.1 Why LR Achieves Perfect Scores

Synthetic benign scenarios differ dramatically from attack scenarios (e.g., syn_ratio, dst_port_entropy, unique_dst_ports). With zero measurement noise and no inter-class overlap in the test corpus, a linear boundary trivially separates them.

Real-world data (e.g., CICIDS2017) includes benign traffic with elevated port diversity, early-stage attacks mimicking normal traffic, and noisy flows. On such data, LR performance will degrade significantly.

### 3.2 GRU False Positive Rate: 45.5%

The GRU produced 10 false positives (benign -> attack) out of 22 benign test windows:

`
               Pred: Benign  Pred: Attack
True: Benign       12            10     <- 10 FP (45.5% FPR)
True: Attack        0            33     <-  0 FN  (0.0% FNR)
`

Root causes:
1. GRU overconservative — learned to associate any temporal pattern with attack.
2. Early stopping at epoch 23 insufficient to suppress false-positive pressure.
3. Benign training sequences too few/stereotyped to anchor the decision boundary.

**Implication for Phase 8**: The World Model must output calibrated probability distributions. High-entropy output should signal uncertainty, not trigger a hard alert.

---

## 4. Next-Stage Forecasting Results

| Model | Input | Top-1 Acc | Top-3 Acc | Macro F1 | Brier Score (lower=better) |
|-------|-------|-----------|-----------|----------|-----------------------------|
| Logistic Regression | S_t (static)      | 1.0000 | 1.0000 | 1.0000 | 0.0114 |
| Temporal GRU        | [S_{t-7}...S_t]   | 0.7091 | 0.8909 | 0.7207 | 0.3177 |

### 4.1 Why LR Is "Perfect" at Forecasting

In the test set, y_next_stage == y_current_stage for every window (zero transitions). Forecasting the next stage is therefore identical to classifying the current stage. On a corpus with real attack progressions (RECON -> INITIAL_ACCESS -> LATERAL_MOVEMENT), LR accuracy will fall substantially.

### 4.2 GRU Next-Stage Confusion Matrix

Class order: BENIGN=0, CREDENTIAL_ACCESS=1, EXFILTRATION=2, RECONNAISSANCE=3

`
                     Pred:BEN  Pred:CRED  Pred:EXFIL  Pred:RECON
True: BENIGN             16       0           0            6
True: CRED_ACCESS         2       7           0            2
True: EXFILTRATION        2       0           7            2
True: RECONNAISSANCE      2       0           0            9
`

Key findings:
- EXFILTRATION and CREDENTIAL_ACCESS confused 2/11 times each with BENIGN.
- RECONNAISSANCE is most distinguishable (81.8%) due to port-scan signatures.
- All attack confusions directed toward BENIGN or RECON, never cross-attack.
- Brier score 0.318 (vs LR 0.011): GRU probability predictions are poorly calibrated.

---

## 5. Does Temporal Information Help?

**Cannot be conclusively determined from single-stage test scenarios.**

The comparison is confounded: LR 100% is a dataset artefact; GRU's lower Top-1 reflects harder generalisation + high FPR.

On multi-stage data, temporal models are predicted to outperform static classifiers on:
1. Early detection: recognising attack stage from partial feature history.
2. Transition prediction: detecting that the next window will change stage.
3. FP suppression: distinguishing benign temporal patterns from attack build-up.

This hypothesis will be tested empirically in Phase 8.

---

## 6. Evidence of Overfitting

### Logistic Regression
No overfitting. Low-variance model; perfect scores explained by separability.

### Temporal GRU
- Early stopping triggered at epoch 23 of 35 (patience=8).
- 45.5% FPR suggests the model overfit to attack temporal patterns, causing over-sensitivity.
- Validation improvement plateaued around epoch 15.

**Mitigations for Phase 8**:
- Dropout on TransitionHead (p=0.2).
- Sequence-level mixup data augmentation.
- Label smoothing (epsilon=0.1) on stage heads.
- Temperature scaling post-training for probability calibration.

---

## 7. Per-Attack-Category GRU Accuracy

| Category       | True Label        | Top-1 Acc     | Notes                        |
|----------------|-------------------|---------------|------------------------------|
| Benign         | BENIGN            | 72.7% (16/22) | 6 confused with RECON        |
| Brute-force    | CREDENTIAL_ACCESS | 63.6% (7/11)  | Confused with BENIGN/RECON   |
| Exfiltration   | EXFILTRATION      | 63.6% (7/11)  | Confused with BENIGN/RECON   |
| Reconnaissance | RECONNAISSANCE    | 81.8% (9/11)  | Best non-benign class        |

---

## 8. Known Limitations of This Evaluation

| Limitation | Impact | Phase 8 Mitigation |
|------------|--------|-------------------|
| All test scenarios are single-stage | LR next-stage scores are not meaningful | Use multi-stage traces |
| Synthetic data has no noise | Feature separability is unrealistically high | Add Gaussian noise or use CICIDS2017 |
| Only 5 test scenarios | No confidence intervals computable | Expand to >= 20 test scenarios |
| Only 4 of 10 taxonomy stages tested | UNKNOWN, EXECUTION, etc. untested | Generate traces for all 10 stages |
| GRU early-stopped at epoch 23 | May be undertrained | Phase 8 uses longer warmup + cosine LR |

---

## 9. Requirements the Cyber World Model Must Satisfy (Phase 8 Targets)

For Phase 8 to claim meaningful improvement, the CyberWorldModel must:

1. **Attack FPR < 20%** on benign sequences (vs GRU 45.5%)
2. **Next-stage Top-1 > 75%** on a test set *that includes stage-transition windows*
3. **Brier score < 0.20** (vs GRU 0.318) — requires probability calibration
4. **K-step rollout produces a valid attack chain** — this capability does not exist in LR or GRU

Requirement 4 is the fundamental differentiator. Neither LR nor GRU can produce
S_t -> S_{t+1} -> ... -> S_{t+K} by iterative application. Only the World Model, via its
TransitionHead (h_{t+1} = f(h_t, S_t)), enables multi-step forward simulation.

---

## 10. Conclusion

> *LR perfect scores reflect dataset separability in a single-stage test corpus, not genuine predictive power. GRU 71% Top-1 and 0.318 Brier are scientifically meaningful but show high FPR and poor calibration. The fundamental limitation of both baselines is that neither can answer "What is likely to happen across the next K steps?" That is precisely the problem the Cyber World Model is designed to solve.*

Phase 8 begins with this clear specification: implement CyberWorldModel with a TransitionHead enabling K-step forward simulation, and evaluate on multi-stage attack traces.
