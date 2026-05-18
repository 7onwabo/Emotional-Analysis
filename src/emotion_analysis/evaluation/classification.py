"""Reporting helpers: per-language breakdown, confusion matrices, error tables."""

from __future__ import annotations

from typing import Any

import numpy as np


def per_language_report(
    logits: np.ndarray,
    labels: np.ndarray,
    languages: list[str],
    label_names: list[str],
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Split predictions by language and return per-language metric dict.

    Required for the rubric's equity / fairness reflection.
    """
    raise NotImplementedError("TODO: group-by language, call compute_multilabel_metrics each")


def confusion_per_label(
    preds: np.ndarray, labels: np.ndarray, label_names: list[str]
) -> dict[str, np.ndarray]:
    raise NotImplementedError("TODO: per-label 2x2 confusion matrices")


def collect_error_examples(
    texts: list[str],
    preds: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    n: int = 50,
) -> list[dict[str, Any]]:
    """Sample misclassified examples for qualitative error analysis."""
    raise NotImplementedError("TODO: argsort by per-example loss, return top-n")
