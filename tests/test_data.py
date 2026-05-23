"""Tests for the BRIGHTER data layer.

Uses in-memory `datasets.Dataset.from_list` fixtures so they run offline.
"""

from __future__ import annotations

import pytest

from emotion_analysis import EMOTION_LABELS
from emotion_analysis.data.loaders import (
    EmotionExample,
    to_arrays,
    to_emotion_examples,
)
from emotion_analysis.data.preprocessing import preprocess_examples

datasets = pytest.importorskip("datasets")


@pytest.fixture
def afr_like_split():
    # Mirrors the BRIGHTER schema: id, text, 6 emotion columns, plus `emotions`.
    # `surprise` is null for one row to exercise the null-coercion path.
    return datasets.Dataset.from_list(
        [
            {
                "id": "afr_train_000",
                "text": "baie families is in rou gedompel",
                "anger": 0, "disgust": 0, "fear": 0,
                "joy": 0, "sadness": 1, "surprise": None,
                "emotions": ["sadness"],
            },
            {
                "id": "afr_train_001",
                "text": "wonderlike nuus vandag!",
                "anger": 0, "disgust": 0, "fear": 0,
                "joy": 1, "sadness": 0, "surprise": 1,
                "emotions": ["joy", "surprise"],
            },
            {
                "id": "afr_train_002",
                "text": "  ",  # whitespace-only; dropped by preprocess_examples
                "anger": 1, "disgust": 0, "fear": 0,
                "joy": 0, "sadness": 0, "surprise": 0,
                "emotions": ["anger"],
            },
        ]
    )


def test_to_emotion_examples_canonical_order(afr_like_split):
    examples = to_emotion_examples(afr_like_split, language="afr")
    assert len(examples) == 3
    assert all(isinstance(ex, EmotionExample) for ex in examples)
    assert examples[0].labels == [0, 0, 0, 0, 1, 0]  # surprise null -> 0
    assert examples[1].labels == [0, 0, 0, 1, 0, 1]
    assert examples[0].example_id == "afr_train_000"
    assert all(ex.language == "afr" for ex in examples)


def test_to_arrays_shapes(afr_like_split):
    examples = to_emotion_examples(afr_like_split, language="afr")
    texts, labels, langs = to_arrays(examples)
    assert len(texts) == 3
    assert labels.shape == (3, len(EMOTION_LABELS))
    assert labels.dtype.kind in {"i", "u"}
    assert langs == ["afr", "afr", "afr"]


def test_preprocess_drops_short_and_clips(afr_like_split):
    examples = to_emotion_examples(afr_like_split, language="afr")
    cleaned = preprocess_examples(examples, min_chars=3, max_chars=20)
    assert len(cleaned) == 2  # whitespace-only row dropped
    assert all(len(ex.text) <= 20 for ex in cleaned)
    # labels preserved through the clean
    assert cleaned[0].labels == [0, 0, 0, 0, 1, 0]


def test_missing_label_column_treated_as_zero():
    ds = datasets.Dataset.from_list(
        [{"id": "x", "text": "hello", "anger": 1, "joy": 0, "emotions": ["anger"]}]
    )
    examples = to_emotion_examples(ds, language="eng")
    # disgust/fear/sadness/surprise columns absent -> all 0
    assert examples[0].labels == [1, 0, 0, 0, 0, 0]
