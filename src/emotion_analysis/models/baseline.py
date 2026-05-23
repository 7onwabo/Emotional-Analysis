"""sklearn baselines: TF-IDF + (LogReg | LinearSVC), one-vs-rest for multilabel.

`build_baseline` returns an unfitted `sklearn.pipeline.Pipeline`:
    TfidfVectorizer -> OneVsRestClassifier(estimator)

`.predict(texts)` yields a multi-hot matrix in `EMOTION_LABELS` order. LogReg
also exposes `.predict_proba`; LinearSVC does not, so probability-thresholding
is only available for the LogReg variant (the trainer falls back to hard
`.predict` for SVM).
"""

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


def _build_estimator(config: BaselineConfig) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    if config.classifier == "logistic_regression":
        return LogisticRegression(C=config.C, max_iter=1000, class_weight="balanced")
    if config.classifier == "linear_svc":
        return LinearSVC(C=config.C, class_weight="balanced")
    raise ValueError(f"Unknown classifier: {config.classifier}")


def build_baseline(config: BaselineConfig) -> Any:
    """Return an unfitted multilabel sklearn Pipeline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline

    if config.vectorizer != "tfidf":
        raise ValueError(f"Unsupported vectorizer: {config.vectorizer}")

    vectorizer = TfidfVectorizer(
        ngram_range=tuple(config.ngram_range),
        max_features=config.max_features,
        sublinear_tf=True,
        strip_accents=None,  # African-language diacritics are meaningful — keep them
    )
    clf = OneVsRestClassifier(_build_estimator(config))
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])
