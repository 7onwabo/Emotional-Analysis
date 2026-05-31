"""Tests for explainability helpers. LIME-dependent tests skip if `lime` absent."""

from __future__ import annotations

import numpy as np
import pytest

from emotion_analysis import EMOTION_LABELS

pytest.importorskip("lime")

from emotion_analysis.evaluation.explainability import lime_explain  

JOY = EMOTION_LABELS.index("joy")


def _toy_proba_fn(texts):
    """joy probability driven by the token 'good'; other labels flat."""
    out = np.full((len(texts), len(EMOTION_LABELS)), 0.1)
    for i, t in enumerate(texts):
        out[i, JOY] = 0.9 if "good" in t.lower().split() else 0.1
    return out


def test_lime_attributes_the_driving_token():
    attrs = lime_explain(
        _toy_proba_fn,
        "today is a good day",
        label_index=JOY,
        label_names=EMOTION_LABELS,
        num_features=5,
        num_samples=200,
    )
    assert attrs, "expected non-empty attributions"
    assert all(isinstance(tok, str) and isinstance(w, float) for tok, w in attrs)
    top_token = max(attrs, key=lambda kv: abs(kv[1]))[0]
    assert top_token == "good"


def test_baseline_proba_fn_rejects_no_predict_proba():
    from emotion_analysis.evaluation.explainability import make_baseline_proba_fn

    class _NoProba:
        pass

    class _Pipe:
        named_steps = {"clf": _NoProba()}

    with pytest.raises(TypeError):
        make_baseline_proba_fn(_Pipe())
