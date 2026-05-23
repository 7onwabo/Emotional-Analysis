"""Smoke tests — verify the package imports and configs parse."""

from __future__ import annotations

from pathlib import Path

import pytest

from emotion_analysis import EMOTION_LABELS, TARGET_LANGUAGES
from emotion_analysis.data.preprocessing import multilabel_to_vector, preprocess_text
from emotion_analysis.training.metrics import compute_multilabel_metrics
from emotion_analysis.utils.config import load_config
from emotion_analysis.utils.seeds import set_seed


def test_label_constants() -> None:
    assert EMOTION_LABELS == ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    assert set(TARGET_LANGUAGES) == {"afr", "swa", "hau", "amh", "tir"}


def test_preprocess_strips_urls_and_mentions() -> None:
    out = preprocess_text("Check https://x.com out @user  ")
    assert "https" not in out
    assert "@user" not in out
    assert "  " not in out


def test_multilabel_to_vector_order() -> None:
    v = multilabel_to_vector(["joy", "anger"], EMOTION_LABELS)
    assert v == [1, 0, 0, 1, 0, 0]


def test_configs_load() -> None:
    cfg = load_config("training", "data", "languages", "models")
    assert cfg.seed == 42
    assert len(cfg.target_languages) == 5
    assert {lang.source for lang in cfg.target_languages} == {"brighter", "ethioemo"}
    assert "afro_xlmr_base" in cfg.models


def test_metrics_shapes() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    logits = rng.standard_normal((20, len(EMOTION_LABELS)))
    labels = (rng.random((20, len(EMOTION_LABELS))) > 0.7).astype(int)
    m = compute_multilabel_metrics(logits, labels, label_names=EMOTION_LABELS)
    assert "f1_macro" in m
    assert all(f"f1_{l}" in m for l in EMOTION_LABELS)


def test_set_seed_runs() -> None:
    set_seed(123)


@pytest.mark.parametrize("name", ["pyproject.toml", "Makefile", "configs/training.yaml"])
def test_repo_files_exist(name: str) -> None:
    assert (Path(__file__).resolve().parents[1] / name).exists()
