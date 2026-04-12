"""Tiny deterministic vocabulary tokenizer for routing models."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.models.formatting import ASSISTANT_MARKER
from src.models.formatting import SYSTEM_MARKER
from src.models.formatting import USER_MARKER


class VocabTokenizer:
    """Whitespace tokenizer with a small, reproducible vocabulary."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    SPECIAL_TOKENS = (
        PAD_TOKEN,
        UNK_TOKEN,
        SYSTEM_MARKER,
        USER_MARKER,
        ASSISTANT_MARKER,
    )

    def __init__(self, stoi: Mapping[str, int]) -> None:
        self.stoi = dict(stoi)
        self.itos = {index: token for token, index in self.stoi.items()}
        self.pad_id = self.stoi[self.PAD_TOKEN]
        self.unk_id = self.stoi[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1) -> "VocabTokenizer":
        if min_freq <= 0:
            raise ValueError("min_freq must be greater than 0.")

        counter: Counter[str] = Counter()
        first_seen: dict[str, int] = {}
        for text in texts:
            for token in cls._tokenize(text):
                counter[token] += 1
                first_seen.setdefault(token, len(first_seen))

        stoi: dict[str, int] = {}
        for token in cls.SPECIAL_TOKENS:
            if token not in stoi:
                stoi[token] = len(stoi)

        ordered_tokens = sorted(
            (
                token
                for token, count in counter.items()
                if count >= min_freq and token not in stoi
            ),
            key=lambda token: (first_seen[token], token),
        )
        for token in ordered_tokens:
            stoi[token] = len(stoi)
        return cls(stoi)

    @classmethod
    def build_from_texts(cls, texts: Iterable[str], min_freq: int = 1) -> "VocabTokenizer":
        return cls.build(texts, min_freq=min_freq)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(token, self.unk_id) for token in self._tokenize(text)]

    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(self.itos.get(int(index), self.UNK_TOKEN) for index in ids)

    def pad(self, batch: list[list[int]], pad_to: int | None = None) -> list[list[int]]:
        if not batch:
            return []
        max_length = max(len(row) for row in batch)
        target_length = pad_to if pad_to is not None else max_length
        if target_length < max_length:
            raise ValueError(
                f"pad_to={target_length} is smaller than the longest sequence length {max_length}."
            )
        return [row + [self.pad_id] * (target_length - len(row)) for row in batch]

    def save(self, path: str | Path) -> None:
        payload = {"stoi": self.stoi}
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VocabTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "stoi" not in payload:
            raise ValueError("Tokenizer file must contain a mapping with a 'stoi' key.")
        stoi = payload["stoi"]
        if not isinstance(stoi, dict):
            raise ValueError("'stoi' must be a mapping.")
        return cls({str(token): int(index) for token, index in stoi.items()})

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}.")
        return text.split()
