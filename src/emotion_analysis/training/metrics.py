"""Multilabel classification metrics.

All metrics computed under a configurable probability threshold (default 0.5).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_multilabel_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    label_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute macro/micro/weighted F1, precision, recall, plus per-label F1.

    `logits` shape: (N, num_labels). Sigmoid applied internally.
    `labels` shape: (N, num_labels), multi-hot {0,1}.
    """
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        hamming_loss,
    )

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    metrics: dict[str, float] = {
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(labels, preds)),
    }

    if label_names is not None:
        per_label_f1 = f1_score(labels, preds, average=None, zero_division=0)
        for name, score in zip(label_names, per_label_f1, strict=True):
            metrics[f"f1_{name}"] = float(score)

    return metrics


def hf_compute_metrics_fn(label_names: list[str], threshold: float = 0.5) -> Any:
    """Build a `compute_metrics` callable for HF Trainer."""

    def _fn(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        return compute_multilabel_metrics(
            logits=np.asarray(logits),
            labels=np.asarray(labels),
            threshold=threshold,
            label_names=label_names,
        )

    return _fn
