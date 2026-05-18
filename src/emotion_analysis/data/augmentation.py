"""Data augmentation strategies for low-resource emotion data.

Planned methods:
    - back-translation (zul -> eng -> zul via NLLB or Google Translate)
    - synonym replacement (limited utility for Bantu morphology)
    - mixup at embedding level (advanced)

Scaffold only.
"""

from __future__ import annotations

from typing import Protocol


class Augmenter(Protocol):
    def augment(self, text: str, language: str) -> list[str]: ...


class BackTranslationAugmenter:
    """Round-trip translation via a pivot language."""

    def __init__(self, pivot: str = "eng", model_id: str | None = None) -> None:
        self.pivot = pivot
        self.model_id = model_id

    def augment(self, text: str, language: str) -> list[str]:
        raise NotImplementedError("TODO: load NLLB or call deep-translator")


class SynonymReplacementAugmenter:
    def __init__(self, p: float = 0.1) -> None:
        self.p = p

    def augment(self, text: str, language: str) -> list[str]:
        raise NotImplementedError("TODO: language-aware synonym swap")
