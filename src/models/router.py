"""Policy/value network for routing actions."""

from __future__ import annotations

import torch
from torch import nn

from src.models.encoders import LSTMEncoder


class PolicyValueNet(nn.Module):
    """Predict routing logits over three actions and a scalar value."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_actions: int = 3,
        *,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(
            vocab_size,
            embedding_dim,
            hidden_dim,
            padding_idx=padding_idx,
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(tokens)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values
