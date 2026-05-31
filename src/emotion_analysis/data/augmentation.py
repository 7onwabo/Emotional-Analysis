"""Back-translation augmentation for low-resource emotion data.

Uses deep-translator (Google Translate) to round-trip text through English.
Each text is translated source->English->source to produce a paraphrase.
"""

from __future__ import annotations

import time
from typing import Protocol

LANG_TO_GOOGLE: dict[str, str] = {
    "afr": "af",
    "swa": "sw",
    "hau": "ha",
    "amh": "am",
    "tir": "ti",
    "orm": "om",
}


class Augmenter(Protocol):
    def augment(self, text: str, language: str) -> list[str]: ...


class BackTranslationAugmenter:
    """Round-trip translation: source -> English -> source via Google Translate.

    Requires `deep-translator` (included in [augment] extras).
    Rate-limits itself to avoid hitting Google's free-tier limits.
    """

    def __init__(self, delay: float = 0.3) -> None:
        self.delay = delay  

    def augment(self, text: str, language: str) -> list[str]:
        from deep_translator import GoogleTranslator

        src_code = LANG_TO_GOOGLE.get(language)
        if src_code is None:
            return []
        try:
            en_text = GoogleTranslator(source=src_code, target="en").translate(text)
            time.sleep(self.delay)
            back = GoogleTranslator(source="en", target=src_code).translate(en_text)
            time.sleep(self.delay)
            if back and back.strip() != text.strip():
                return [back]
        except Exception:
            pass
        return []


class SynonymReplacementAugmenter:
    def __init__(self, p: float = 0.1) -> None:
        self.p = p

    def augment(self, text: str, language: str) -> list[str]:
        raise NotImplementedError("Synonym replacement not supported for African languages.")
