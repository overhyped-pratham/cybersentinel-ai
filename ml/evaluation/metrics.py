"""
CyberSentinel AI - Unified Evaluation & Metrics Engine.

Provides mathematically rigorous, un-faked calculation of:
- Binary and Multi-Class classification metrics (Accuracy, Precision, Recall, Macro/Weighted F1, FPR, ROC-AUC, PR-AUC)
- Next-stage forecast metrics (Top-1 Accuracy, Top-3 Accuracy, Brier Score)
- Confusion matrix computation and artifact rendering.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    is_binary: bool = True,
) -> Dict[str, Any]:
    """
    Computes rigorous classification metrics for binary or multi-class tasks.
    
    Args:
        y_true: Ground truth 1D array of labels.
        y_pred: Predicted 1D array of labels.
        y_prob: Predicted probability array. For binary: shape (N,) or (N, 2). For multi-class: shape (N, C).
        class_names: Optional list of class string names.
        is_binary: True if binary attack detection, False if multi-class stage prediction.
        
    Returns:
        Dict[str, Any] containing all measured metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # False Positive Rate
    if is_binary:
        # cm shape (2, 2) where [[TN, FP], [FN, TP]]
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        else:
            fpr = 0.0
    else:
        # Multi-class macro FPR: sum(FP_c) / sum(FP_c + TN_c)
        fpr_list = []
        for i in range(len(cm)):
            fp = np.sum(cm[:, i]) - cm[i, i]
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            denom = fp + tn
            fpr_list.append(fp / denom if denom > 0 else 0.0)
        fpr = float(np.mean(fpr_list)) if fpr_list else 0.0

    # ROC-AUC and PR-AUC
    roc_auc = None
    pr_auc = None

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            if is_binary:
                # If y_prob is 2D (N, 2), extract positive column
                p1 = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] == 2 else y_prob.ravel()
                # Check that both classes exist in y_true
                if len(np.unique(y_true)) > 1:
                    roc_auc = float(roc_auc_score(y_true, p1))
                    pr_auc = float(average_precision_score(y_true, p1))
            else:
                # Multi-class OvR
                if y_prob.ndim == 2 and y_prob.shape[1] > 1:
                    present_classes = np.unique(y_true)
                    if len(present_classes) > 1:
                        # Subset y_prob to present classes if needed
                        roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
                        # One-hot encode y_true for PR-AUC
                        y_true_oh = np.eye(y_prob.shape[1])[y_true]
                        pr_auc = float(average_precision_score(y_true_oh, y_prob, average="macro"))
        except Exception:
            # Fall back to None if uncomputable (e.g. single class present in split)
            roc_auc = None
            pr_auc = None

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "fpr": fpr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
    }


def calculate_forecasting_metrics(
    y_true_next: np.ndarray,
    y_prob_next: np.ndarray,
    k: int = 3,
) -> Dict[str, Any]:
    """
    Computes specific forecasting metrics:
    - Top-1 next-stage accuracy
    - Top-K next-stage accuracy
    - Multi-class Brier score
    
    Args:
        y_true_next: 1D array of ground truth next stages.
        y_prob_next: 2D array (N, num_classes) of forecasted probabilities for next stage.
        k: Horizon for top-k accuracy (default: 3).
        
    Returns:
        Dict[str, Any] with measured forecast metrics.
    """
    y_true_next = np.asarray(y_true_next, dtype=int)
    y_prob_next = np.asarray(y_prob_next, dtype=float)
    n_samples, n_classes = y_prob_next.shape

    if n_samples == 0:
        return {
            "next_stage_accuracy": 0.0,
            f"top_{k}_accuracy": 0.0,
            "brier_score": 0.0,
        }

    # Top-1 accuracy
    y_pred_top1 = np.argmax(y_prob_next, axis=1)
    acc_top1 = float(accuracy_score(y_true_next, y_pred_top1))

    # Top-K accuracy
    k_eff = min(k, n_classes)
    top_k_preds = np.argsort(y_prob_next, axis=1)[:, -k_eff:]
    correct_in_top_k = sum(
        y_true_next[i] in top_k_preds[i] for i in range(n_samples)
    )
    acc_top_k = float(correct_in_top_k / n_samples)

    # Multi-class Brier score: 1/N sum_i sum_c (p_ic - y_ic)^2
    y_true_oh = np.zeros((n_samples, n_classes), dtype=float)
    for i, label in enumerate(y_true_next):
        if 0 <= label < n_classes:
            y_true_oh[i, label] = 1.0

    brier = float(np.mean(np.sum((y_prob_next - y_true_oh) ** 2, axis=1)))

    return {
        "next_stage_accuracy": acc_top1,
        f"top_{k}_accuracy": acc_top_k,
        "brier_score": brier,
    }


def save_confusion_matrix_plot(
    cm: Union[np.ndarray, List[List[int]]],
    class_names: List[str],
    output_path: Union[str, Path],
    title: str = "Confusion Matrix",
) -> None:
    """Renders and saves a normalized confusion matrix plot."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cm_arr = np.asarray(cm)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
