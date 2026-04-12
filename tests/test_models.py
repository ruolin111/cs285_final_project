"""Tests for the lightweight routing model stack."""

from __future__ import annotations

import torch

from src.models.encoders import LSTMEncoder
from src.models.router import PolicyValueNet


def test_lstm_encoder_returns_batch_embeddings() -> None:
    encoder = LSTMEncoder(vocab_size=32, embedding_dim=16, hidden_dim=24)
    tokens = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]], dtype=torch.long)

    features = encoder(tokens)

    assert features.shape == (2, 24)


def test_policy_value_net_output_shapes() -> None:
    model = PolicyValueNet(vocab_size=32, embedding_dim=16, hidden_dim=24, num_actions=3)
    tokens = torch.randint(0, 32, (2, 8))

    logits, values = model(tokens)

    assert logits.shape == (2, 3)
    assert values.shape == (2,)
