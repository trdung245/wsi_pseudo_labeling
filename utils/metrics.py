"""
Evaluation metrics for multi-task patch classification.

Functions
---------
compute_classification_metrics:
    Accuracy, per-class precision/recall/F1, macro-F1, AUC-ROC.
compute_regression_metrics:
    MAE, RMSE, Spearman correlation for attention score prediction.
aggregate_metrics:
    Combine classification and regression metrics into a flat dict.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from scipy.stats import spearmanr


def compute_classification_metrics(
    labels: np.ndarray,      # (N,) int
    logits: np.ndarray,      # (N, C) float
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Return accuracy, macro-F1, per-class F1, and macro AUC-ROC."""
    preds = logits.argmax(axis=-1)          # (N,)
    probs = _softmax(logits)                # (N, C)

    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    report = classification_report(
        labels, preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    try:
        auc = roc_auc_score(
            labels, probs,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        auc = float("nan")

    metrics = {
        "accuracy":  acc,
        "macro_f1":  macro_f1,
        "macro_auc": auc,
    }
    # Per-class F1
    for cls_name in (class_names or [str(i) for i in range(logits.shape[1])]):
        if cls_name in report:
            metrics[f"f1_{cls_name}"] = report[cls_name]["f1-score"]

    return metrics


def compute_regression_metrics(
    targets: np.ndarray,   # (N,)
    preds: np.ndarray,     # (N,)
) -> dict[str, float]:
    """Return MAE, RMSE, and Spearman correlation for attention regression."""
    mae  = float(np.abs(targets - preds).mean())
    rmse = float(np.sqrt(((targets - preds) ** 2).mean()))
    rho, _ = spearmanr(targets, preds)
    return {
        "attn_mae":       mae,
        "attn_rmse":      rmse,
        "attn_spearman":  float(rho),
    }


def aggregate_metrics(
    all_labels: list[int],
    all_logits: list[np.ndarray],
    all_attn_pred: list[float],
    all_attn_true: list[float],
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Aggregate raw predictions into a complete metrics dict."""
    labels    = np.array(all_labels)
    logits    = np.array(all_logits)
    attn_pred = np.array(all_attn_pred)
    attn_true = np.array(all_attn_true)

    cls_metrics = compute_classification_metrics(labels, logits, class_names)
    reg_metrics = compute_regression_metrics(attn_true, attn_pred)
    return {**cls_metrics, **reg_metrics}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)
