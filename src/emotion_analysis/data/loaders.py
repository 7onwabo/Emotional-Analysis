"""Dataset loaders.

Scaffold only — implementations to be filled in once HF dataset IDs are
confirmed by the team during the data-exploration phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EmotionExample:
    text: str
    labels: list[int]  # multi-hot over EMOTION_LABELS
    language: str
    example_id: str | None = None


def load_brighter(
    language: str,
    split: str = "train",
    cache_dir: str | Path | None = None,
) -> Any:
    """Load a BRIGHTER language split from the HF hub.

    Returns a `datasets.Dataset`. Caller is expected to map to the
    canonical label order from `EMOTION_LABELS`.
    """
    raise NotImplementedError("TODO: wire to HF datasets.load_dataset once HF id confirmed")


def load_ethioemo(
    language: str,
    split: str = "train",
    cache_dir: str | Path | None = None,
) -> Any:
    """Load EthioEmo split. Schema aligned with BRIGHTER."""
    raise NotImplementedError("TODO: wire EthioEmo loader")


def load_auxiliary(name: str, **kwargs: Any) -> Any:
    """Load an auxiliary dataset (AfriSenti, AfriHate, MasakhaNews)."""
    raise NotImplementedError("TODO: dispatch table for auxiliary datasets")
