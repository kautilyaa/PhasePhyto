"""Evaluation metrics for PhasePhyto."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute standard classification metrics.

    Returns dict with accuracy, macro/weighted F1, precision, recall,
    confusion matrix, and full classification report.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        ),
    }


def per_class_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str],
) -> list[dict[str, float | int | str]]:
    """Per-class precision, recall, F1."""
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    return [
        {
            "class": name,
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i, name in enumerate(class_names)
    ]
