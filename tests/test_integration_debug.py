"""Smoke tests for the bootstrap config loader."""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import logging
import pytest
import torch

from src.utils.config import load_config
from src.utils.config import require_bootstrap_config
from src.utils.logging import configure_logging
from src.utils.types import ConfigError
from src.utils.seeding import seed_everything


def test_base_config_loads() -> None:
    config = load_config(Path("configs/base.yaml"))

    assert isinstance(config, dict)
    assert config["seed"] == 7
    assert config["device"] == "cpu"
    assert config["experiment"]["name"] == "base"
    assert config["experiment"]["output_dir"] == "outputs"
    assert config["data"]["train_path"] == "sample_data/episodes.jsonl"
    assert config["data"]["val_path"] == "sample_data/episodes.jsonl"
    assert config["data"]["test_path"] == "sample_data/episodes.jsonl"


def test_nested_config_loads(tmp_path: Path) -> None:
    config_path = tmp_path / "nested.yaml"
    config_path.write_text(
        """
seed: 13
device: cpu
experiment:
  name: nested-smoke-test
  output_dir: outputs/nested-smoke-test
trainer:
  batch_size: 32
  optimizer:
    name: adamw
    lr: 0.0003
data:
  train_path: data/nested-train.csv
  val_path: data/nested-val.csv
  test_path: data/nested-test.csv
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["seed"] == 13
    assert config["experiment"]["name"] == "nested-smoke-test"
    assert config["trainer"]["optimizer"]["lr"] == 0.0003


def test_missing_required_bootstrap_key_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "missing-seed.yaml"
    config_path.write_text(
        """
device: cpu
experiment:
  name: broken
  output_dir: outputs
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    with pytest.raises(ConfigError, match="seed"):
        require_bootstrap_config(config, context=str(config_path))


def test_train_cli_prints_loaded_experiment() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            "configs/base.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "loaded experiment=base"


def test_module_entrypoint_prints_loaded_experiment() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train",
            "--config",
            "configs/base.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "loaded experiment=base"


def test_train_cli_reports_bad_config_without_traceback(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(
        """
seed: 7
experiment:
  output_dir: outputs
data:
  train_path: sample_data/episodes.jsonl
  val_path: sample_data/episodes.jsonl
  test_path: sample_data/episodes.jsonl
""".lstrip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(bad_config),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_train_cli_reports_directory_path_without_traceback(tmp_path: Path) -> None:
    config_dir = tmp_path / "config-dir"
    config_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(config_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_seed_everything_is_deterministic() -> None:
    seed_everything(123)
    first = random.randint(0, 1_000_000)
    first_tensor = torch.rand(1).item()

    seed_everything(123)
    second = random.randint(0, 1_000_000)
    second_tensor = torch.rand(1).item()

    assert first == second
    assert first_tensor == second_tensor


def test_configure_logging_reconfigures() -> None:
    configure_logging("INFO")
    assert logging.getLogger().level == logging.INFO

    configure_logging("WARNING")
    root = logging.getLogger()

    assert root.level == logging.WARNING


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(ConfigError, match="Unknown logging level"):
        configure_logging("NOT_A_LEVEL")
