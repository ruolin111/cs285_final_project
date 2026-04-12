"""Logging setup helpers for the bootstrap CLI."""

from __future__ import annotations

import logging

from src.utils.types import ConfigError


def _coerce_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    if not isinstance(level, str) or not level.strip():
        raise ConfigError("logging level must be an integer or a non-empty string.")

    normalized = level.strip().upper()
    level_mapping = logging.getLevelNamesMapping()
    if normalized not in level_mapping:
        raise ConfigError(f"Unknown logging level: {level}")
    return level_mapping[normalized]


def configure_logging(level: int | str = logging.INFO) -> None:
    """Install a simple, readable logging configuration."""

    logging.basicConfig(
        level=_coerce_level(level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
