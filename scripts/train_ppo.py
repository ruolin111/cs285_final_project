#!/usr/bin/env python3
"""Train a milestone-scale PPO routing policy on replayable conversations."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml
import torch

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.algorithms.ppo import PPOAgent
from src.algorithms.ppo import PPOConfig
from src.algorithms.ppo import collect_rollouts
from src.algorithms.ppo import evaluate_policy
from src.algorithms.ppo import write_metrics_jsonl
from src.datasets.io import load_episodes
from src.envs.conversation_routing import RewardConfig
from src.models.formatting import ConversationFormatter
from src.utils.config import load_config
from src.utils.seeding import seed_everything
from src.utils.types import ConfigError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the milestone PPO baseline.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    return parser.parse_args()


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be a mapping, got {type(value).__name__}.")
    return value


def _require_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{context} must be an integer.")
    return value


def _require_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{context} must be a number.")
    return float(value)


def build_run_directory(config: dict[str, Any]) -> Path:
    experiment = _require_mapping(config.get("experiment"), "config.experiment")
    name = experiment.get("name")
    output_dir = experiment.get("output_dir")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("config.experiment.name must be a non-empty string.")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ConfigError("config.experiment.output_dir must be a non-empty string.")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{name}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_formatter(config: dict[str, Any]) -> ConversationFormatter:
    formatter_config = _require_mapping(config.get("formatter", {}), "config.formatter")
    return ConversationFormatter(
        last_k_turns=formatter_config.get("last_k_turns"),
        max_chars=formatter_config.get("max_chars"),
        max_tokens_estimate=formatter_config.get("max_tokens_estimate"),
    )


def build_reward_config(config: dict[str, Any]) -> RewardConfig:
    reward_mapping = _require_mapping(config.get("reward", {}), "config.reward")
    defaults = RewardConfig()
    payload = {
        field_name: _require_number(reward_mapping.get(field_name, getattr(defaults, field_name)), f"config.reward.{field_name}")
        for field_name in RewardConfig.__dataclass_fields__
    }
    return RewardConfig(**payload)


def build_ppo_config(config: dict[str, Any]) -> PPOConfig:
    ppo_mapping = _require_mapping(config.get("ppo"), "config.ppo")
    return PPOConfig(
        learning_rate=_require_number(ppo_mapping.get("learning_rate"), "config.ppo.learning_rate"),
        gamma=_require_number(ppo_mapping.get("gamma"), "config.ppo.gamma"),
        gae_lambda=_require_number(ppo_mapping.get("gae_lambda"), "config.ppo.gae_lambda"),
        clip_ratio=_require_number(ppo_mapping.get("clip_ratio"), "config.ppo.clip_ratio"),
        value_coef=_require_number(ppo_mapping.get("value_coef"), "config.ppo.value_coef"),
        entropy_coef=_require_number(ppo_mapping.get("entropy_coef"), "config.ppo.entropy_coef"),
        max_grad_norm=_require_number(ppo_mapping.get("max_grad_norm"), "config.ppo.max_grad_norm"),
        rollout_episodes=_require_int(ppo_mapping.get("rollout_episodes"), "config.ppo.rollout_episodes"),
        num_updates=_require_int(ppo_mapping.get("num_updates"), "config.ppo.num_updates"),
        update_epochs=_require_int(ppo_mapping.get("update_epochs"), "config.ppo.update_epochs"),
        minibatch_size=_require_int(ppo_mapping.get("minibatch_size"), "config.ppo.minibatch_size"),
        eval_every=_require_int(ppo_mapping.get("eval_every"), "config.ppo.eval_every"),
    )


def plot_training_curves(metrics: list[dict[str, Any]], output_path: Path) -> None:
    updates = [metric["update"] for metric in metrics]
    avg_returns = [metric["train_avg_return"] for metric in metrics]
    attack_success = [metric.get("attack_success_rate", float("nan")) for metric in metrics]
    benign_fp = [metric.get("benign_false_positive_rate", float("nan")) for metric in metrics]

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(updates, avg_returns, marker="o")
    axes[0].set_title("Train Avg Return")
    axes[0].set_xlabel("Update")

    axes[1].plot(updates, attack_success, marker="o")
    axes[1].set_title("Attack Success Rate")
    axes[1].set_xlabel("Update")

    axes[2].plot(updates, benign_fp, marker="o")
    axes[2].set_title("Benign False Positive Rate")
    axes[2].set_xlabel("Update")

    for axis in axes:
        axis.grid(True, alpha=0.3)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def train(config_path: Path) -> Path:
    config = load_config(config_path)
    seed_everything(_require_int(config.get("seed"), "config.seed"))

    device_name = config.get("device", "cpu")
    if not isinstance(device_name, str) or not device_name:
        raise ConfigError("config.device must be a non-empty string.")
    device = torch.device(device_name)

    data_mapping = _require_mapping(config.get("data"), "config.data")
    train_episodes = load_episodes(Path(data_mapping["train_path"]))
    val_episodes = load_episodes(Path(data_mapping.get("val_path", data_mapping["train_path"])))

    formatter = build_formatter(config)
    reward_config = build_reward_config(config)
    ppo_config = build_ppo_config(config)
    model_mapping = _require_mapping(config.get("model"), "config.model")
    embedding_dim = _require_int(model_mapping.get("embedding_dim"), "config.model.embedding_dim")
    hidden_dim = _require_int(model_mapping.get("hidden_dim"), "config.model.hidden_dim")
    min_freq = _require_int(model_mapping.get("min_freq", 1), "config.model.min_freq")

    run_dir = build_run_directory(config)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    agent = PPOAgent.build(
        train_episodes=train_episodes,
        formatter=formatter,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        min_freq=min_freq,
        device=device,
        config=ppo_config,
    )
    agent.tokenizer.save(run_dir / "artifacts" / "tokenizer.json")

    metrics_history: list[dict[str, Any]] = []
    best_attack_success_rate = float("inf")
    best_benign_fp = float("inf")

    for update in range(1, ppo_config.num_updates + 1):
        rollout = collect_rollouts(agent, train_episodes, reward_config=reward_config)
        update_metrics = agent.update(rollout)
        metrics = {
            "update": update,
            "train_avg_return": sum(rollout.episode_returns) / max(1, len(rollout.episode_returns)),
            "train_avg_episode_length": sum(rollout.episode_lengths) / max(1, len(rollout.episode_lengths)),
            **update_metrics,
        }

        if update % ppo_config.eval_every == 0 or update == ppo_config.num_updates:
            eval_metrics = evaluate_policy(agent, val_episodes, reward_config=reward_config)
            metrics.update(eval_metrics)
            is_better = (
                eval_metrics["attack_success_rate"] < best_attack_success_rate
                or (
                    eval_metrics["attack_success_rate"] == best_attack_success_rate
                    and eval_metrics["benign_false_positive_rate"] < best_benign_fp
                )
            )
            if is_better:
                best_attack_success_rate = eval_metrics["attack_success_rate"]
                best_benign_fp = eval_metrics["benign_false_positive_rate"]
                agent.save(run_dir / "checkpoints" / "best.pt")

        metrics_history.append(metrics)

    agent.save(run_dir / "checkpoints" / "last.pt")
    write_metrics_jsonl(run_dir / "metrics.jsonl", metrics_history)
    plot_training_curves(metrics_history, run_dir / "plots" / "training_curves.png")
    (run_dir / "summary.json").write_text(
        json.dumps(metrics_history[-1], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return run_dir


def main() -> int:
    args = parse_args()
    try:
        run_dir = train(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"saved run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
