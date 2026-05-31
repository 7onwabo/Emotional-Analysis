"""Reporting helpers: per-language breakdown, confusion matrices, error tables.

All functions take hard 0/1 `preds` (not logits) so the transformer and
sklearn-baseline paths share one reporting code path. The caller thresholds
logits into preds before calling here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from emotion_analysis.training.metrics import metrics_from_predictions


def per_language_report(
    preds: np.ndarray,
    labels: np.ndarray,
    languages: list[str],
    label_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Group predictions by language and compute metrics per group + overall.

    Required for the rubric's equity / fairness reflection. Returns a dict
    keyed by language code (plus an `overall` entry).
    """
    preds = np.asarray(preds).astype(int)
    labels = np.asarray(labels).astype(int)
    langs = np.asarray(languages)

    def _zero_support(y: np.ndarray) -> list[str]:
        support = y.sum(axis=0)
        return [label_names[i] for i in range(len(label_names)) if support[i] == 0]

    report: dict[str, dict[str, Any]] = {
        "overall": metrics_from_predictions(preds, labels, label_names=label_names)
    }
    report["overall"]["zero_support_labels"] = _zero_support(labels)
    for lang in sorted(set(languages)):
        mask = langs == lang
        report[lang] = metrics_from_predictions(
            preds[mask], labels[mask], label_names=label_names
        )
        report[lang]["n_examples"] = int(mask.sum())
        report[lang]["zero_support_labels"] = _zero_support(labels[mask])
    return report


def confusion_per_label(
    preds: np.ndarray, labels: np.ndarray, label_names: list[str]
) -> dict[str, dict[str, int]]:
    """Per-label 2x2 confusion counts (tp/fp/fn/tn)."""
    preds = np.asarray(preds).astype(int)
    labels = np.asarray(labels).astype(int)
    out: dict[str, dict[str, int]] = {}
    for i, name in enumerate(label_names):
        p, y = preds[:, i], labels[:, i]
        out[name] = {
            "tp": int(((p == 1) & (y == 1)).sum()),
            "fp": int(((p == 1) & (y == 0)).sum()),
            "fn": int(((p == 0) & (y == 1)).sum()),
            "tn": int(((p == 0) & (y == 0)).sum()),
        }
    return out


def collect_error_examples(
    texts: list[str],
    preds: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    languages: list[str] | None = None,
    n: int = 50,
) -> list[dict[str, Any]]:
    """Return up to `n` misclassified examples, worst-first by Hamming distance.

    Each entry lists the gold vs predicted label names for qualitative analysis.
    """
    preds = np.asarray(preds).astype(int)
    labels = np.asarray(labels).astype(int)
    per_example_err = (preds != labels).sum(axis=1)
    order = np.argsort(-per_example_err)

    out: list[dict[str, Any]] = []
    for idx in order:
        if per_example_err[idx] == 0:
            break
        gold = [label_names[j] for j in range(len(label_names)) if labels[idx, j] == 1]
        pred = [label_names[j] for j in range(len(label_names)) if preds[idx, j] == 1]
        entry: dict[str, Any] = {
            "text": texts[idx],
            "gold": gold,
            "pred": pred,
            "n_wrong": int(per_example_err[idx]),
        }
        if languages is not None:
            entry["language"] = languages[idx]
        out.append(entry)
        if len(out) >= n:
            break
    return out
