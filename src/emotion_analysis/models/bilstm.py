"""BiLSTM multilabel classifier with subword (FastText-style) embeddings.

Uses character n-gram hashing so no external vector files are needed — the
embedding table is trained from scratch on the target data, which mirrors
FastText's in-vocab behaviour while keeping setup simple.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class BiLSTMConfig:
    num_labels: int
    vocab_size: int = 200_000
    embed_dim: int = 300
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    max_length: int = 128
    label_names: list[str] = field(default_factory=list)


class BiLSTMClassifier(nn.Module):
    def __init__(self, config: BiLSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            config.embed_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_dim * 2, config.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        offsets: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape

        embedded = self.embedding(input_ids) 
        embedded = self.dropout(embedded)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu() if lengths is not None else torch.full((batch_size,), seq_len),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        fwd = hidden[-2]  
        bwd = hidden[-1]  
        pooled = self.dropout(torch.cat([fwd, bwd], dim=-1))  
        logits = self.classifier(pooled)  

        output: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            output["loss"] = nn.BCEWithLogitsLoss()(logits, labels.float())
        return output


class FastTextTokenizer:
    """Character n-gram tokenizer matching FastText's hashing trick."""

    def __init__(
        self,
        vocab_size: int = 200_000,
        max_length: int = 128,
        ngram_range: tuple[int, int] = (2, 4),
    ) -> None:
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.ngram_range = ngram_range

    def _ngrams(self, word: str) -> list[str]:
        word = f"<{word}>"
        grams: list[str] = [word]
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            grams += [word[i : i + n] for i in range(len(word) - n + 1)]
        return grams

    def _hash(self, s: str) -> int:
        return hash(s) % self.vocab_size

    def encode(self, text: str) -> list[int]:
        tokens: list[int] = []
        for word in text.lower().split()[: self.max_length]:
            tokens.append(self._hash(word))
        return tokens or [0]

    def __call__(
        self, texts: list[str], max_length: int | None = None
    ) -> dict[str, torch.Tensor]:
        ml = max_length or self.max_length
        ids, lengths = [], []
        for text in texts:
            enc = self.encode(text)[:ml]
            lengths.append(len(enc))
            enc += [0] * (ml - len(enc))
            ids.append(enc)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "lengths": torch.tensor(lengths, dtype=torch.long),
        }

    def save_pretrained(self, path: str) -> None:
        import json, pathlib
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        cfg = {"vocab_size": self.vocab_size, "max_length": self.max_length, "ngram_range": list(self.ngram_range)}
        (pathlib.Path(path) / "tokenizer_config.json").write_text(json.dumps(cfg))

    @classmethod
    def from_pretrained(cls, path: str) -> "FastTextTokenizer":
        import json, pathlib
        cfg = json.loads((pathlib.Path(path) / "tokenizer_config.json").read_text())
        return cls(vocab_size=cfg["vocab_size"], max_length=cfg["max_length"], ngram_range=tuple(cfg["ngram_range"]))
