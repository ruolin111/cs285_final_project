"""Tests for the milestone PPO baseline."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from src.algorithms.ppo import compute_gae
from src.algorithms.ppo import select_rollout_episodes
from src.datasets.io import load_episodes


def test_compute_gae_returns_terminal_episode_values() -> None:
    advantages, returns = compute_gae(
        rewards=[1.0, 2.0],
        values=[0.5, 0.25],
        dones=[False, True],
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert advantages == pytest.approx([2.5, 1.75])
    assert returns == pytest.approx([3.0, 2.0])


def test_select_rollout_episodes_uses_shuffled_coverage() -> None:
    episodes = load_episodes(Path("sample_data/episodes.jsonl"))
    random.seed(1)

    selected = select_rollout_episodes(episodes, 2)

    assert len(selected) == 2
    assert {episode.episode_id for episode in selected} == {
        "harmful-late-001",
        "benign-001",
    }


def test_train_ppo_cli_creates_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.yaml"
    config_path.write_text(
        f"""
experiment:
  name: ppo-smoke
  output_dir: {tmp_path.as_posix()}/outputs
device: cpu
seed: 7
data:
  train_path: sample_data/episodes.jsonl
  val_path: sample_data/episodes.jsonl
  test_path: sample_data/episodes.jsonl
formatter:
  last_k_turns: 4
model:
  embedding_dim: 16
  hidden_dim: 32
  min_freq: 1
reward:
  harmful_block_reward: 1.0
  harmful_success_penalty: -2.0
  harmful_no_success_reward: 0.0
  benign_allow_reward: 0.1
  benign_warn_penalty: -0.05
  benign_block_penalty: -1.0
  harmful_warn_penalty: -0.05
ppo:
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 1.0
  rollout_episodes: 2
  num_updates: 3
  update_epochs: 1
  minibatch_size: 4
  eval_every: 2
""".lstrip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/train_ppo.py", "--config", str(config_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    marker = "saved run_dir="
    assert marker in result.stdout
    run_dir = Path(result.stdout.strip().split(marker, maxsplit=1)[1])

    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "plots" / "training_curves.png").is_file()
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "artifacts" / "tokenizer.json").is_file()

    metrics_lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics_lines) == 3
    last_metrics = json.loads(metrics_lines[-1])
    assert "attack_success_rate" in last_metrics
    assert "benign_false_positive_rate" in last_metrics
    first_metrics = json.loads(metrics_lines[0])
    assert "attack_success_rate" not in first_metrics
