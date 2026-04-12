"""Lightweight text encoders for routing models."""

from __future__ import annotations

import torch
from torch import nn


class LSTMEncoder(nn.Module):
    """Encode padded token sequences with a single-layer LSTM."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        *,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape [batch, seq], got {tuple(tokens.shape)}.")

        lengths = (tokens != self.padding_idx).sum(dim=1).clamp(min=1).to(dtype=torch.long)
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        return hidden[-1]
