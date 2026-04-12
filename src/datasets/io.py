"""JSONL loading helpers for replayable episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.datasets.schema import DatasetSchemaError
from src.datasets.schema import Episode
from src.datasets.schema import parse_episode


def load_episode_records(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    """Read a JSONL file and return its raw episode records with source line numbers."""

    file_path = Path(path)
    records: list[tuple[int, dict[str, Any]]] = []

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetSchemaError(
                        f"{file_path} line {line_number}: invalid JSON: {exc.msg}."
                    ) from exc

                if not isinstance(record, dict):
                    raise DatasetSchemaError(
                        f"{file_path} line {line_number}: each JSONL row must decode to an object."
                    )
                records.append((line_number, record))
    except FileNotFoundError as exc:
        raise DatasetSchemaError(f"Episode file not found: {file_path}") from exc
    except OSError as exc:
        raise DatasetSchemaError(f"Could not read episode file {file_path}: {exc.strerror or exc}") from exc

    return records


def load_episodes(path: str | Path) -> list[Episode]:
    """Load and validate episodes from a JSONL file."""

    return list(iter_episodes(path))


def iter_episodes(path: str | Path) -> Iterable[Episode]:
    """Iterate over validated episodes from a JSONL file."""

    file_path = Path(path)
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetSchemaError(
                        f"{file_path} line {line_number}: invalid JSON: {exc.msg}."
                    ) from exc

                if not isinstance(record, dict):
                    raise DatasetSchemaError(
                        f"{file_path} line {line_number}: each JSONL row must decode to an object."
                    )
                yield parse_episode(record, f"{file_path} line {line_number}")
    except FileNotFoundError as exc:
        raise DatasetSchemaError(f"Episode file not found: {file_path}") from exc
    except OSError as exc:
        raise DatasetSchemaError(f"Could not read episode file {file_path}: {exc.strerror or exc}") from exc
