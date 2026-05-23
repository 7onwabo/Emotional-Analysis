"""Dataset loaders.

The BRIGHTER loader prefers a local copy under `data/raw/brighter/{lang}/{split}`
written by `scripts/download_data.py`, and falls back to a hub pull. Labels are
normalized to a multi-hot vector in canonical `EMOTION_LABELS` order; columns
that are absent or `null` for a given language are treated as 0 (e.g. afr has
no `surprise` annotations in some splits).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from emotion_analysis import EMOTION_LABELS

BRIGHTER_HF_ID = "brighter-dataset/BRIGHTER-emotion-categories"
DEFAULT_RAW_DIR = Path("data/raw/brighter")


@dataclass
class EmotionExample:
    text: str
    labels: list[int]  # multi-hot over EMOTION_LABELS
    language: str
    example_id: str | None = None


def _coerce_label(value: Any) -> int:
    """null / missing -> 0; otherwise truthy -> 1."""
    if value is None:
        return 0
    try:
        return int(bool(int(value)))
    except (TypeError, ValueError):
        return 0


def _row_to_multihot(row: dict[str, Any], label_order: Iterable[str]) -> list[int]:
    return [_coerce_label(row.get(lbl)) for lbl in label_order]


def load_brighter(
    language: str,
    split: str = "train",
    raw_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    *,
    hub_fallback: bool = True,
) -> Any:
    """Load a BRIGHTER split as a `datasets.Dataset`.

    Prefers local `raw_dir/{language}/{split}` (written by the downloader);
    falls back to `datasets.load_dataset(BRIGHTER_HF_ID, language, split=split)`
    when `hub_fallback=True`.
    """
    from datasets import load_dataset, load_from_disk

    base = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    local = base / language / split
    if local.exists():
        return load_from_disk(str(local))
    if not hub_fallback:
        raise FileNotFoundError(
            f"BRIGHTER split not found at {local}. Run scripts/download_data.py first."
        )
    return load_dataset(BRIGHTER_HF_ID, language, split=split, cache_dir=str(cache_dir) if cache_dir else None)


def to_emotion_examples(
    ds: Any,
    language: str,
    label_order: list[str] | None = None,
    text_column: str = "text",
    id_column: str = "id",
) -> list[EmotionExample]:
    """Materialise a HF Dataset into in-memory `EmotionExample` list."""
    label_order = label_order or EMOTION_LABELS
    examples: list[EmotionExample] = []
    for row in ds:
        examples.append(
            EmotionExample(
                text=row.get(text_column, "") or "",
                labels=_row_to_multihot(row, label_order),
                language=language,
                example_id=row.get(id_column),
            )
        )
    return examples


def to_arrays(
    examples: list[EmotionExample],
) -> tuple[list[str], np.ndarray, list[str]]:
    """Flatten to (texts, label_matrix, languages)."""
    texts = [ex.text for ex in examples]
    labels = np.asarray([ex.labels for ex in examples], dtype=np.int8)
    langs = [ex.language for ex in examples]
    return texts, labels, langs


def load_brighter_examples(
    language: str,
    split: str = "train",
    raw_dir: str | Path | None = None,
    label_order: list[str] | None = None,
    text_column: str = "text",
    id_column: str = "id",
) -> list[EmotionExample]:
    """Convenience: load + materialise in one call."""
    ds = load_brighter(language, split=split, raw_dir=raw_dir)
    return to_emotion_examples(
        ds,
        language=language,
        label_order=label_order,
        text_column=text_column,
        id_column=id_column,
    )


def load_ethioemo(
    language: str,
    split: str = "train",
    cache_dir: str | Path | None = None,
) -> Any:
    """Load EthioEmo split. Schema aligned with BRIGHTER (deferred to phase 2)."""
    raise NotImplementedError("EthioEmo loader is phase 2; BRIGHTER alone covers the target langs.")


def load_auxiliary(name: str, **kwargs: Any) -> Any:
    """Load an auxiliary dataset (AfriSenti, AfriHate). Phase 2."""
    raise NotImplementedError("Auxiliary loaders are phase 2.")
