"""Minimal PPO implementation for milestone-scale routing experiments."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from src.datasets.schema import Episode
from src.envs.conversation_routing import ConversationRoutingEnv
from src.models.formatting import ConversationFormatter
from src.models.router import PolicyValueNet
from src.models.tokenizer import VocabTokenizer


@dataclass(slots=True)
class PPOConfig:
    """Hyperparameters for the milestone PPO baseline."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    rollout_episodes: int = 8
    num_updates: int = 20
    update_epochs: int = 4
    minibatch_size: int = 16
    eval_every: int = 1


@dataclass(slots=True)
class RolloutBatch:
    """Flat batch of on-policy transitions."""

    texts: list[str]
    actions: list[int]
    log_probs: list[float]
    rewards: list[float]
    dones: list[bool]
    values: list[float]
    returns: list[float]
    advantages: list[float]
    episode_returns: list[float]
    episode_lengths: list[int]
    final_infos: list[dict[str, Any]]

    @property
    def size(self) -> int:
        return len(self.actions)


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """Compute GAE advantages and returns for a single episode."""

    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("rewards, values, and dones must have the same length.")

    advantages = [0.0] * len(rewards)
    gae = 0.0
    next_value = 0.0
    for index in reversed(range(len(rewards))):
        nonterminal = 1.0 - float(dones[index])
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[index] = gae
        next_value = values[index]

    returns = [advantage + value for advantage, value in zip(advantages, values, strict=True)]
    return advantages, returns


class PPOAgent:
    """Policy/value model plus text preprocessing for routing decisions."""

    def __init__(
        self,
        *,
        formatter: ConversationFormatter,
        tokenizer: VocabTokenizer,
        model: PolicyValueNet,
        device: torch.device,
        config: PPOConfig,
    ) -> None:
        self.formatter = formatter
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    @classmethod
    def build(
        cls,
        *,
        train_episodes: list[Episode],
        formatter: ConversationFormatter,
        embedding_dim: int,
        hidden_dim: int,
        min_freq: int,
        device: torch.device,
        config: PPOConfig,
    ) -> "PPOAgent":
        texts = build_vocab_corpus(train_episodes, formatter)
        tokenizer = VocabTokenizer.build(texts, min_freq=min_freq)
        model = PolicyValueNet(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            padding_idx=tokenizer.pad_id,
        )
        return cls(
            formatter=formatter,
            tokenizer=tokenizer,
            model=model,
            device=device,
            config=config,
        )

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        encoded = [self.tokenizer.encode(text) for text in texts]
        padded = self.tokenizer.pad(encoded)
        return torch.tensor(padded, dtype=torch.long, device=self.device)

    def act(self, observation: dict[str, Any], *, deterministic: bool = False) -> tuple[int, float, float, str]:
        text = self.formatter.format_observation(observation)
        tokens = self.encode_texts([text])
        self.model.eval()
        with torch.no_grad():
            logits, values = self.model(tokens)
            distribution = Categorical(logits=logits)
            action_tensor = (
                torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
            )
            log_prob_tensor = distribution.log_prob(action_tensor)
        return (
            int(action_tensor.item()),
            float(log_prob_tensor.item()),
            float(values.item()),
            text,
        )

    def evaluate_actions(
        self,
        texts: list[str],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.encode_texts(texts)
        logits, values = self.model(tokens)
        distribution = Categorical(logits=logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_probs, entropy, values

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        if batch.size == 0:
            raise ValueError("Cannot update PPO on an empty rollout batch.")

        actions = torch.tensor(batch.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(batch.log_probs, dtype=torch.float32, device=self.device)
        returns = torch.tensor(batch.returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(batch.advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        indices = torch.arange(batch.size, device=self.device)
        clip_fraction_total = 0.0
        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        num_minibatches = 0

        self.model.train()
        for _ in range(self.config.update_epochs):
            permutation = indices[torch.randperm(batch.size, device=self.device)]
            for start in range(0, batch.size, self.config.minibatch_size):
                minibatch_indices = permutation[start : start + self.config.minibatch_size]
                minibatch_actions = actions[minibatch_indices]
                minibatch_old_log_probs = old_log_probs[minibatch_indices]
                minibatch_returns = returns[minibatch_indices]
                minibatch_advantages = advantages[minibatch_indices]
                minibatch_texts = [batch.texts[int(index)] for index in minibatch_indices.cpu().tolist()]

                new_log_probs, entropy, values = self.evaluate_actions(
                    minibatch_texts,
                    minibatch_actions,
                )
                ratios = torch.exp(new_log_probs - minibatch_old_log_probs)
                unclipped = ratios * minibatch_advantages
                clipped = torch.clamp(
                    ratios,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * minibatch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(values, minibatch_returns)
                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_bonus
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                clip_fraction = (
                    (torch.abs(ratios - 1.0) > self.config.clip_ratio)
                    .to(dtype=torch.float32)
                    .mean()
                )
                policy_loss_total += float(policy_loss.item())
                value_loss_total += float(value_loss.item())
                entropy_total += float(entropy_bonus.item())
                clip_fraction_total += float(clip_fraction.item())
                num_minibatches += 1

        return {
            "policy_loss": policy_loss_total / max(1, num_minibatches),
            "value_loss": value_loss_total / max(1, num_minibatches),
            "entropy": entropy_total / max(1, num_minibatches),
            "clip_fraction": clip_fraction_total / max(1, num_minibatches),
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "model_state_dict": self.model.state_dict(),
            "tokenizer": self.tokenizer.stoi,
            "config": asdict(self.config),
        }
        torch.save(payload, path)


def build_vocab_corpus(episodes: list[Episode], formatter: ConversationFormatter) -> list[str]:
    """Build deterministic formatter outputs for vocabulary construction."""

    texts: list[str] = []
    for episode in episodes:
        for turn in episode.turns:
            if turn.role == "user":
                texts.append(formatter.format_prefix(episode, turn.turn_index))
    return texts


def select_rollout_episodes(episodes: list[Episode], num_episodes: int) -> list[Episode]:
    """Choose rollout episodes with shuffled coverage across the dataset."""

    if not episodes:
        raise ValueError("episodes must be non-empty.")
    if num_episodes <= 0:
        raise ValueError("num_episodes must be greater than 0.")

    selected: list[Episode] = []
    while len(selected) < num_episodes:
        shuffled = list(episodes)
        random.shuffle(shuffled)
        selected.extend(shuffled)
    return selected[:num_episodes]


def collect_rollouts(
    agent: PPOAgent,
    episodes: list[Episode],
    *,
    reward_config: Any | None = None,
) -> RolloutBatch:
    """Collect one flat on-policy batch by cycling through stored episodes."""

    rollout_episodes = select_rollout_episodes(episodes, agent.config.rollout_episodes)
    env = ConversationRoutingEnv(rollout_episodes, reward_config=reward_config)
    texts: list[str] = []
    actions: list[int] = []
    log_probs: list[float] = []
    rewards: list[float] = []
    dones: list[bool] = []
    values: list[float] = []
    returns: list[float] = []
    advantages: list[float] = []
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    final_infos: list[dict[str, Any]] = []

    for _ in range(len(rollout_episodes)):
        observation, _ = env.reset()
        done = False
        episode_texts: list[str] = []
        episode_actions: list[int] = []
        episode_log_probs: list[float] = []
        episode_rewards: list[float] = []
        episode_dones: list[bool] = []
        episode_values: list[float] = []
        final_info: dict[str, Any] = {}

        while not done:
            action, log_prob, value, text = agent.act(observation)
            next_observation, reward, done, _, info = env.step(action)

            episode_texts.append(text)
            episode_actions.append(action)
            episode_log_probs.append(log_prob)
            episode_rewards.append(float(reward))
            episode_dones.append(bool(done))
            episode_values.append(float(value))
            final_info = info
            observation = next_observation

        episode_advantages, episode_returns_values = compute_gae(
            episode_rewards,
            episode_values,
            episode_dones,
            gamma=agent.config.gamma,
            gae_lambda=agent.config.gae_lambda,
        )
        texts.extend(episode_texts)
        actions.extend(episode_actions)
        log_probs.extend(episode_log_probs)
        rewards.extend(episode_rewards)
        dones.extend(episode_dones)
        values.extend(episode_values)
        advantages.extend(episode_advantages)
        returns.extend(episode_returns_values)
        episode_returns.append(sum(episode_rewards))
        episode_lengths.append(len(episode_rewards))
        final_infos.append(final_info)

    return RolloutBatch(
        texts=texts,
        actions=actions,
        log_probs=log_probs,
        rewards=rewards,
        dones=dones,
        values=values,
        returns=returns,
        advantages=advantages,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        final_infos=final_infos,
    )


def evaluate_policy(
    agent: PPOAgent,
    episodes: list[Episode],
    *,
    reward_config: Any | None = None,
) -> dict[str, float]:
    """Run greedy episode-level evaluation on a fixed episode list."""

    env = ConversationRoutingEnv(episodes, reward_config=reward_config)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    harmful_total = 0
    harmful_blocked = 0
    harmful_missed = 0
    benign_total = 0
    benign_blocked = 0
    benign_allowed = 0
    intervention_turns: list[int] = []

    for _ in range(len(episodes)):
        observation, _ = env.reset()
        done = False
        total_reward = 0.0
        length = 0
        final_info: dict[str, Any] = {}
        while not done:
            action, _, _, _ = agent.act(observation, deterministic=True)
            observation, reward, done, _, info = env.step(action)
            total_reward += float(reward)
            length += 1
            final_info = info

        episode_returns.append(total_reward)
        episode_lengths.append(length)
        if final_info["harmful"]:
            harmful_total += 1
            if final_info["blocked"]:
                harmful_blocked += 1
            if final_info["final_outcome"] == "harmful_missed":
                harmful_missed += 1
        else:
            benign_total += 1
            if final_info["blocked"]:
                benign_blocked += 1
            if final_info["final_outcome"] == "benign_allowed":
                benign_allowed += 1
        if final_info["block_turn"] is not None:
            intervention_turns.append(int(final_info["block_turn"]))

    return {
        "eval_avg_return": sum(episode_returns) / max(1, len(episode_returns)),
        "eval_avg_episode_length": sum(episode_lengths) / max(1, len(episode_lengths)),
        "attack_success_rate": harmful_missed / max(1, harmful_total),
        "harmful_block_rate": harmful_blocked / max(1, harmful_total),
        "benign_false_positive_rate": benign_blocked / max(1, benign_total),
        "benign_allow_rate": benign_allowed / max(1, benign_total),
        "average_intervention_turn": (
            sum(intervention_turns) / len(intervention_turns) if intervention_turns else -1.0
        ),
    }


def write_metrics_jsonl(path: str | Path, metrics: list[dict[str, Any]]) -> None:
    """Persist one JSON object per update."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(metric, sort_keys=True) for metric in metrics]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
