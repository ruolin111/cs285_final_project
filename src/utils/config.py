"""Config loading helpers for the bootstrap project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.types import ConfigError


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be a mapping, got {type(value).__name__}.")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return the parsed mapping."""

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read config file {path}: {exc.strerror or exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML in {path}: {exc}") from exc

    return _require_mapping(raw, f"Config file {path}")


def require_bootstrap_config(config: dict[str, Any], context: str = "config") -> tuple[int, str]:
    """Validate the tiny Task 1 CLI contract and return ``(seed, experiment_name)``."""

    config_mapping = _require_mapping(config, context)
    seed = config_mapping.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError(f"{context} key 'seed' must be an integer.")

    experiment = _require_mapping(config_mapping.get("experiment"), f"{context} key 'experiment'")
    name = experiment.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{context} key 'experiment.name' must be a non-empty string.")

    return seed, name.strip()
