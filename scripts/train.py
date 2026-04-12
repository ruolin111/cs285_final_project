#!/usr/bin/env python3
"""Tiny CLI entry point for loading config and printing the experiment name."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.config import require_bootstrap_config
from src.utils.types import ConfigError
from src.utils.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CS285 bootstrap entry point.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment config file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        seed, experiment_name = require_bootstrap_config(config, context=str(args.config))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    seed_everything(seed)
    print(f"loaded experiment={experiment_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
