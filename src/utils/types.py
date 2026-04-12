"""Lightweight shared types used by the bootstrap utilities."""

from __future__ import annotations

class ConfigError(ValueError):
    """Raised when a config file is missing required data or has the wrong shape."""
