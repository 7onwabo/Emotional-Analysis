"""Model explainability: SHAP, LIME, attention visualization (Captum).

Used for error analysis section of the report.
"""

from __future__ import annotations

from typing import Any


def shap_explain(model: Any, tokenizer: Any, texts: list[str], label_index: int) -> Any:
    """SHAP partition explainer over tokens. Returns SHAP values."""
    raise NotImplementedError("TODO: shap.Explainer(pipeline, masker=shap.maskers.Text(tokenizer))")


def lime_explain(model: Any, tokenizer: Any, text: str, label_index: int, num_features: int = 10) -> Any:
    """LIME explanation for a single text."""
    raise NotImplementedError("TODO: LimeTextExplainer.explain_instance")


def attention_rollout(model: Any, tokenizer: Any, text: str, layer: int = -1) -> Any:
    """Attention rollout via Captum for transformer attribution."""
    raise NotImplementedError("TODO: captum LayerIntegratedGradients on embeddings")
