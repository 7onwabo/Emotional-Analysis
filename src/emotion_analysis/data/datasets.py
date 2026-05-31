"""Shared dataset assembly used by the train / evaluate / explain scripts.
Resolves target languages from `configs/languages.yaml` (each carries a
`source`: brighter | ethioemo), loads each language's split via the matching
loader, applies preprocessing, and concatenates into arrays. Centralising this
keeps source-dispatch and preprocessing identical across all entry points.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from emotion_analysis.data.loaders import EmotionExample, load_examples_by_source, to_arrays
from emotion_analysis.data.preprocessing import preprocess_examples


def resolve_languages(cfg: Any, language: str) -> list[str]:
    """Return target language codes: a single code, or all targets for 'all'."""
    targets = [lang.code for lang in cfg.target_languages]
    if language == "all":
        return targets
    if language not in targets:
        raise SystemExit(f"Unknown language '{language}'. Targets: {targets}")
    return [language]


def _source_for(cfg: Any, code: str) -> str:
    for lang in cfg.target_languages:
        if lang.code == code:
            return lang.source
    raise KeyError(f"Language '{code}' not in target_languages.")


def load_examples(cfg: Any, languages: list[str], split: str) -> list[EmotionExample]:
    """Load + preprocess `EmotionExample`s for the given languages and split."""
    pp = cfg.preprocessing
    raw_root = Path(cfg.paths.raw)
    out: list[EmotionExample] = []
    for code in languages:
        source = _source_for(cfg, code)
        examples = load_examples_by_source(source, code, split=split, raw_root=raw_root)
        examples = preprocess_examples(
            examples,
            min_chars=pp.min_chars,
            max_chars=pp.max_chars,
            lowercase=pp.lowercase,
            strip_urls=pp.strip_urls,
            strip_mentions=pp.strip_mentions,
            strip_emoji=pp.strip_emoji,
            normalize_whitespace=pp.normalize_whitespace,
        )
        out.extend(examples)
    return out


def build_split(
    cfg: Any, languages: list[str], split: str
) -> tuple[list[str], np.ndarray, list[str]]:
    """Load + preprocess -> (texts, label_matrix, languages)."""
    return to_arrays(load_examples(cfg, languages, split))
