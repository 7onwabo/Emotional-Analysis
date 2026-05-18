"""Text preprocessing.

Keep minimal — transformers tokenizers handle most normalization.
Only strip social-media noise and enforce length bounds here.
"""

from __future__ import annotations

import re
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")


def preprocess_text(
    text: str,
    *,
    lowercase: bool = False,
    strip_urls: bool = True,
    strip_mentions: bool = True,
    strip_emoji: bool = False,
    normalize_whitespace: bool = True,
) -> str:
    if strip_urls:
        text = URL_RE.sub(" ", text)
    if strip_mentions:
        text = MENTION_RE.sub(" ", text)
    if strip_emoji:
        text = "".join(c for c in text if unicodedata.category(c)[0] != "S")
    if lowercase:
        text = text.lower()
    if normalize_whitespace:
        text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def multilabel_to_vector(labels: list[str], label_order: list[str]) -> list[int]:
    label_set = set(labels)
    return [int(lbl in label_set) for lbl in label_order]
