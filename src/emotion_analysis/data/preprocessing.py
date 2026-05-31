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


def preprocess_examples(
    examples: list,  
    *,
    min_chars: int = 3,
    max_chars: int | None = 512,
    lowercase: bool = False,
    strip_urls: bool = True,
    strip_mentions: bool = True,
    strip_emoji: bool = False,
    normalize_whitespace: bool = True,
) -> list:
    """Apply `preprocess_text` to a list of EmotionExample and drop too-short rows.

    `max_chars` clips overlong text (tokenizer truncation still bounds the
    model-side length; this is a safety net). Returns a new list — input is
    not mutated.
    """
    out = []
    for ex in examples:
        text = preprocess_text(
            ex.text,
            lowercase=lowercase,
            strip_urls=strip_urls,
            strip_mentions=strip_mentions,
            strip_emoji=strip_emoji,
            normalize_whitespace=normalize_whitespace,
        )
        if len(text) < min_chars:
            continue
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars]
        out.append(
            ex.__class__(
                text=text,
                labels=ex.labels,
                language=ex.language,
                example_id=ex.example_id,
            )
        )
    return out
