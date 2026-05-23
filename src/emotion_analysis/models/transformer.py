"""Transformer fine-tuning wrapper around HuggingFace `Trainer`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from emotion_analysis import EMOTION_LABELS


@dataclass
class TransformerConfig:
    hf_id: str
    num_labels: int
    max_length: int = 128
    problem_type: str = "multi_label_classification"
    label_names: list[str] = field(default_factory=lambda: list(EMOTION_LABELS))


def load_model_and_tokenizer(config: TransformerConfig) -> tuple[Any, Any]:
    """Load `AutoModelForSequenceClassification` + `AutoTokenizer`.

    Configures `id2label` / `label2id` and sets `problem_type` so the model
    uses `BCEWithLogitsLoss` for multi-label training. Returns (model, tokenizer).
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    label_names = config.label_names or list(EMOTION_LABELS)
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    tokenizer = AutoTokenizer.from_pretrained(config.hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.hf_id,
        num_labels=config.num_labels,
        problem_type=config.problem_type,
        id2label=id2label,
        label2id=label2id,
    )
    return model, tokenizer


def tokenize_function(examples: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,  # dynamic padding via DataCollatorWithPadding at batch time
    )


def build_hf_dataset(
    texts: list[str],
    labels: Any,
    tokenizer: Any,
    max_length: int,
) -> Any:
    """Build a tokenized `datasets.Dataset` with float multi-hot `labels`.

    Labels must be float for `BCEWithLogitsLoss`. Returns a dataset with
    columns: input_ids, attention_mask, labels.
    """
    import numpy as np
    from datasets import Dataset

    labels_f = np.asarray(labels, dtype="float32")
    ds = Dataset.from_dict({"text": list(texts), "labels": labels_f.tolist()})
    ds = ds.map(
        lambda batch: tokenize_function(batch, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )
    return ds
