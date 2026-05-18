"""sklearn baselines: TF-IDF + (LogReg | LinearSVC), one-vs-rest for multilabel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BaselineConfig:
    vectorizer: str = "tfidf"
    classifier: str = "logistic_regression"
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = 50_000
    C: float = 1.0


def build_baseline(config: BaselineConfig) -> Any:
    """Return a fitted-able sklearn Pipeline. Implementation TBD."""
    raise NotImplementedError("TODO: TfidfVectorizer -> OneVsRest(classifier)")
