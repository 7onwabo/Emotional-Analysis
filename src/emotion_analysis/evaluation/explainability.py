"""Model explainability for the error-analysis section of the report.

Two attribution paths, both reducing a multi-label model to a per-label score:

* `lime_explain` — model-agnostic, works for the sklearn baseline (LogReg
  `predict_proba`) and the transformer (sigmoid over logits). Wraps the chosen
  label as a binary problem so `LimeTextExplainer` can attribute tokens.
* `integrated_gradients` — transformer-only token attribution via Captum
  `LayerIntegratedGradients` on the embedding layer.

`shap_explain` is optional (SHAP pulls numba/llvmlite); install with
`pip install -e ".[shap]"`.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from emotion_analysis import EMOTION_LABELS
ProbaFn = Callable[[list[str]], np.ndarray]


def make_baseline_proba_fn(pipeline: Any) -> ProbaFn:
    """Per-label probabilities from a fitted sklearn multilabel pipeline.

    Requires an estimator exposing `predict_proba` (LogReg). LinearSVC does not,
    so use the LogReg baseline for LIME.
    """
    clf = pipeline.named_steps.get("clf") if hasattr(pipeline, "named_steps") else None
    if clf is not None and not hasattr(clf, "predict_proba"):
        raise TypeError(
            "Baseline classifier has no predict_proba (LinearSVC?). "
            "Use the tfidf_logreg model for LIME."
        )

    def _fn(texts: list[str]) -> np.ndarray:
        proba = pipeline.predict_proba(texts)
        return np.asarray(proba)

    return _fn


def make_transformer_proba_fn(
    model: Any, tokenizer: Any, max_length: int = 128, batch_size: int = 16
) -> ProbaFn:
    """Per-label sigmoid probabilities from a transformer classifier."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    def _fn(texts: list[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tokenizer(
                    texts[i : i + batch_size],
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                logits = model(**enc).logits
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.vstack(out)

    return _fn


def lime_explain(
    proba_fn: ProbaFn,
    text: str,
    label_index: int,
    label_names: list[str] | None = None,
    num_features: int = 10,
    num_samples: int = 1000,
) -> list[tuple[str, float]]:
    """LIME token attributions for one label of one text.

    Returns a list of (token, weight) pairs; positive weight pushes toward the
    label. Wraps the multi-label `proba_fn` into the 2-column [P(not), P(label)]
    form LIME expects.
    """
    from lime.lime_text import LimeTextExplainer

    label_names = label_names or list(EMOTION_LABELS)

    def _binary_proba(texts: list[str]) -> np.ndarray:
        p = proba_fn(list(texts))[:, label_index]
        return np.column_stack([1.0 - p, p])

    explainer = LimeTextExplainer(class_names=["not_" + label_names[label_index], label_names[label_index]])
    exp = explainer.explain_instance(
        text,
        _binary_proba,
        num_features=num_features,
        num_samples=num_samples,
        labels=(1,),
    )
    return exp.as_list(label=1)


def integrated_gradients(
    model: Any,
    tokenizer: Any,
    text: str,
    label_index: int,
    max_length: int = 128,
    n_steps: int = 50,
) -> list[tuple[str, float]]:
    """Per-token attribution for one label via Captum LayerIntegratedGradients.

    Attributions are summed over the embedding dimension and L2-normalized.
    Returns (token, score) pairs aligned to the tokenizer's word pieces.
    """
    import torch
    from captum.attr import LayerIntegratedGradients

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    enc = tokenizer(
        text, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    ref_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    baseline_ids = torch.full_like(input_ids, ref_id)
    special_mask = torch.tensor(
        tokenizer.get_special_tokens_mask(input_ids[0].tolist(), already_has_special_tokens=True),
        device=device,
    ).bool()
    baseline_ids[0, special_mask] = input_ids[0, special_mask]

    embeddings = model.get_input_embeddings()

    def _forward(ids: torch.Tensor) -> torch.Tensor:
        logits = model(input_ids=ids, attention_mask=attention_mask).logits
        return logits[:, label_index]

    lig = LayerIntegratedGradients(_forward, embeddings)
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        n_steps=n_steps,
    )
    scores = attributions.sum(dim=-1).squeeze(0)
    norm = torch.norm(scores)
    if norm > 0:
        scores = scores / norm
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    return list(zip(tokens, scores.detach().cpu().tolist(), strict=True))


def shap_explain(proba_fn: ProbaFn, texts: list[str], label_index: int, tokenizer: Any = None) -> Any:
    """SHAP token attributions (optional dependency).

    Returns the SHAP Explanation object for the chosen label.
    """
    try:
        import shap
    except ImportError as e: 
        raise ImportError('SHAP not installed. Run: pip install -e ".[shap]"') from e

    def _scalar_proba(texts_in: list[str]) -> np.ndarray:
        return proba_fn(list(texts_in))[:, label_index]

    masker = shap.maskers.Text(tokenizer) if tokenizer is not None else shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(_scalar_proba, masker)
    return explainer(texts)
