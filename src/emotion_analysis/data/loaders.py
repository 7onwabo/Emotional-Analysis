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
ETHIOEMO_HF_ID = "Tadesse/EthioEmo"
DEFAULT_RAW_DIR = Path("data/raw/brighter")
DEFAULT_ETHIOEMO_DIR = Path("data/raw/ethioemo")

# Both datasets share the same id/text + 6 int64 emotion-column schema.
SOURCE_REGISTRY: dict[str, tuple[str, Path]] = {
    "brighter": (BRIGHTER_HF_ID, DEFAULT_RAW_DIR),
    "ethioemo": (ETHIOEMO_HF_ID, DEFAULT_ETHIOEMO_DIR),
}


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


def _load_local_or_hub(
    hf_id: str,
    base_dir: Path,
    config: str,
    split: str,
    cache_dir: str | Path | None,
    hub_fallback: bool,
) -> Any:
    """Prefer a local `base_dir/{config}/{split}`; else pull from the hub."""
    from datasets import load_dataset, load_from_disk

    local = base_dir / config / split
    if local.exists():
        return load_from_disk(str(local))
    if not hub_fallback:
        raise FileNotFoundError(
            f"Split not found at {local}. Run scripts/download_data.py first."
        )
    return load_dataset(hf_id, config, split=split, cache_dir=str(cache_dir) if cache_dir else None)


def load_brighter(
    language: str,
    split: str = "train",
    raw_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    *,
    hub_fallback: bool = True,
) -> Any:
    """Load a BRIGHTER split as a `datasets.Dataset` (local-first, hub fallback)."""
    base = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    return _load_local_or_hub(BRIGHTER_HF_ID, base, language, split, cache_dir, hub_fallback)


def load_ethioemo(
    language: str,
    split: str = "train",
    raw_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    *,
    hub_fallback: bool = True,
) -> Any:
    """Load an EthioEmo split (amh/tir/orm/som). Same schema as BRIGHTER."""
    base = Path(raw_dir) if raw_dir is not None else DEFAULT_ETHIOEMO_DIR
    return _load_local_or_hub(ETHIOEMO_HF_ID, base, language, split, cache_dir, hub_fallback)


def load_by_source(
    source: str,
    language: str,
    split: str = "train",
    raw_root: str | Path | None = None,
    *,
    hub_fallback: bool = True,
) -> Any:
    """Dispatch to the right loader by `source` ('brighter' | 'ethioemo')."""
    if source not in SOURCE_REGISTRY:
        raise KeyError(f"Unknown source '{source}'. Known: {list(SOURCE_REGISTRY)}")
    hf_id, default_dir = SOURCE_REGISTRY[source]
    base = (Path(raw_root) / source) if raw_root is not None else default_dir
    return _load_local_or_hub(hf_id, base, language, split, None, hub_fallback)


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


def load_examples_by_source(
    source: str,
    language: str,
    split: str = "train",
    raw_root: str | Path | None = None,
    label_order: list[str] | None = None,
    text_column: str = "text",
    id_column: str = "id",
    *,
    hub_fallback: bool = True,
) -> list[EmotionExample]:
    """Load + materialise a split from any registered source ('brighter'|'ethioemo')."""
    ds = load_by_source(source, language, split=split, raw_root=raw_root, hub_fallback=hub_fallback)
    return to_emotion_examples(
        ds,
        language=language,
        label_order=label_order,
        text_column=text_column,
        id_column=id_column,
    )


AFRISENTI_HF_ID = "shmuhammad/AfriSenti-twitter-sentiment"
AFRIHATE_HF_ID = "shmuhammad/AfriHate"

# Languages in our target set that appear in each auxiliary dataset
AFRISENTI_OVERLAP = ["hau", "amh", "orm"]  # hau=Hausa, amh=Amharic, orm=Oromo
AFRIHATE_OVERLAP = ["hau", "amh"]


@dataclass
class AuxExample:
    text: str
    label: int   # single integer class label
    language: str


_AFRISENTI_BASE = (
    "https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/data/{lang}/{split}.tsv"
)


def load_afrisenti(
    language: str,
    split: str = "train",
    cache_dir: str | Path | None = None,
) -> list[AuxExample]:
    """Load AfriSenti for one language directly from GitHub TSV files.

    Labels: 0=positive, 1=negative, 2=neutral.
    The HF dataset uses a loading script incompatible with datasets>=3.0, so we
    download the raw TSV from the upstream GitHub repository instead.
    """
    import io
    import urllib.request

    hf_split = "dev" if split == "validation" else split
    url = _AFRISENTI_BASE.format(lang=language, split=hf_split)
    label_map: dict[Any, int] = {"positive": 0, "negative": 1, "neutral": 2}

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = resp.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Could not download AfriSenti {language}/{hf_split} from {url}: {e}")

    examples: list[AuxExample] = []
    for line in io.StringIO(content):
        line = line.rstrip("\n")
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        # Format: tweet\tlabel  (no header row in some files; skip if header)
        text_col, label_col = parts[0], parts[-1]
        if label_col in ("label", "labels"):
            continue  # skip header
        label = label_map.get(label_col.strip().lower(), 2)
        if text_col.strip():
            examples.append(AuxExample(text=text_col.strip(), label=label, language=language))
    return examples


def load_afrihate(
    language: str,
    split: str = "train",
    cache_dir: str | Path | None = None,
) -> list[AuxExample]:
    """Load AfriHate for one language. Labels: 0=non-hate, 1=hate."""
    from datasets import load_dataset

    label_map = {"hate": 1, "non-hate": 0, "normal": 0, 0: 0, 1: 1}
    ds = load_dataset(
        AFRIHATE_HF_ID,
        language,
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=True,
    )
    examples: list[AuxExample] = []
    for row in ds:
        text = row.get("tweet") or row.get("text") or ""
        raw_label = row.get("label", 0)
        label = label_map.get(raw_label, 0)
        examples.append(AuxExample(text=text, label=label, language=language))
    return examples


def load_auxiliary(name: str, language: str, split: str = "train", **kwargs: Any) -> list[AuxExample]:
    """Load an auxiliary dataset by name: 'afrisenti' or 'afrihate'."""
    if name == "afrisenti":
        return load_afrisenti(language, split=split)
    if name == "afrihate":
        return load_afrihate(language, split=split)
    raise ValueError(f"Unknown auxiliary dataset '{name}'. Known: afrisenti, afrihate")
