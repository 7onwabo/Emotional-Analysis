"""Tests for baselines, metrics, and reporting. sklearn-dependent tests skip if absent."""

from __future__ import annotations

import numpy as np
import pytest

from emotion_analysis import EMOTION_LABELS
from emotion_analysis.evaluation.classification import (
    collect_error_examples,
    confusion_per_label,
    per_language_report,
)

sklearn = pytest.importorskip("sklearn")


def test_metrics_from_predictions_perfect():
    from emotion_analysis.training.metrics import metrics_from_predictions

    # Every label must have at least one positive, else macro-F1 averages in
    # zero-support labels (zero_division=0) and drops below 1.0.
    labels = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
    m = metrics_from_predictions(labels, labels, label_names=EMOTION_LABELS)
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["f1_micro"] == pytest.approx(1.0)
    assert m["hamming_loss"] == pytest.approx(0.0)


def test_build_baseline_fits_and_predicts():
    from emotion_analysis.models.baseline import BaselineConfig, build_baseline

    texts = ["i am angry and disgusted", "what a joyful surprise", "so sad today", "fearful night"]
    labels = np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    )
    pipe = build_baseline(BaselineConfig(max_features=100, ngram_range=(1, 1)))
    pipe.fit(texts, labels)
    preds = np.asarray(pipe.predict(texts))
    assert preds.shape == (4, len(EMOTION_LABELS))
    assert set(np.unique(preds)).issubset({0, 1})


def test_train_baseline_artifacts(tmp_path):
    from emotion_analysis.models.baseline import BaselineConfig, build_baseline
    from emotion_analysis.training.trainer import train_baseline

    texts = ["angry", "joyful", "sad", "fear", "disgust", "surprise"] * 4
    rng = np.random.default_rng(0)
    labels = (rng.random((len(texts), len(EMOTION_LABELS))) > 0.6).astype(int)
    pipe = build_baseline(BaselineConfig(max_features=50, ngram_range=(1, 1)))
    art = train_baseline(
        pipe, texts, labels, texts, labels, output_dir=tmp_path, label_names=EMOTION_LABELS
    )
    assert "f1_macro" in art.metrics
    assert (tmp_path / "pipeline.joblib").exists()


def test_per_language_report_groups():
    preds = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    labels = preds.copy()
    langs = ["afr", "afr", "swa"]
    rep = per_language_report(preds, labels, langs, EMOTION_LABELS)
    assert set(rep) == {"overall", "afr", "swa"}
    assert rep["afr"]["n_examples"] == 2
    assert rep["swa"]["n_examples"] == 1
    # micro-F1 is robust to zero-support labels; perfect preds -> 1.0
    assert rep["overall"]["f1_micro"] == pytest.approx(1.0)
    assert rep["overall"]["hamming_loss"] == pytest.approx(0.0)
    # only anger/disgust/fear fire here -> joy/sadness/surprise are zero-support
    assert set(rep["overall"]["zero_support_labels"]) == {"joy", "sadness", "surprise"}
    # swa slice only has the fear-positive row
    assert set(rep["swa"]["zero_support_labels"]) == {"anger", "disgust", "joy", "sadness", "surprise"}


def test_confusion_and_errors():
    preds = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    labels = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    conf = confusion_per_label(preds, labels, EMOTION_LABELS)
    assert conf["anger"]["tp"] == 1
    assert conf["disgust"]["fn"] == 1
    errs = collect_error_examples(
        ["right", "wrong"], preds, labels, EMOTION_LABELS, languages=["afr", "swa"], n=10
    )
    assert len(errs) == 1
    assert errs[0]["text"] == "wrong"
    assert errs[0]["gold"] == ["disgust"]
    assert errs[0]["language"] == "swa"
