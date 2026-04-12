"""Minimal raw-record normalization for canonical replayable episodes."""

from __future__ import annotations

from typing import Any

from src.datasets.schema import DatasetSchemaError
from src.datasets.schema import parse_episode
from src.datasets.schema import REQUIRED_EPISODE_KEYS


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetSchemaError(f"{context} must be a mapping, got {type(value).__name__}.")
    return value


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise DatasetSchemaError(f"{context} must be a string, got {type(value).__name__}.")

    text = value.strip()
    if not text:
        raise DatasetSchemaError(f"{context} must be a non-empty string.")
    return text


def _require_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise DatasetSchemaError(f"{context} must be a boolean, got {type(value).__name__}.")
    return value


def _require_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetSchemaError(f"{context} must be an integer, got {type(value).__name__}.")
    return value


def _normalize_turn(raw_turn: Any, turn_index: int, context: str) -> dict[str, Any]:
    record = _require_mapping(raw_turn, context)
    role = _require_string(record.get("role"), f"{context} key 'role'").lower()
    text_value = record.get("text", record.get("content"))
    text = _require_string(text_value, f"{context} key 'text'")

    normalized_index = record.get("turn_index", turn_index)
    turn_index_value = _require_int(normalized_index, f"{context} key 'turn_index'")

    return {
        "role": role,
        "text": text,
        "turn_index": turn_index_value,
    }


def normalize_episode_record(raw_record: Any, context: str = "episode") -> dict[str, Any]:
    """Normalize a raw episode-like mapping into the canonical schema shape."""

    record = _require_mapping(raw_record, context)

    normalized_turns_source = record.get("turns")
    if not isinstance(normalized_turns_source, list):
        raise DatasetSchemaError(f"{context} key 'turns' must be a list of turn records.")

    normalized = {
        "episode_id": _require_string(record.get("episode_id"), f"{context} key 'episode_id'"),
        "source": _require_string(record.get("source"), f"{context} key 'source'"),
        "split": _require_string(record.get("split"), f"{context} key 'split'"),
        "label": _require_string(record.get("label"), f"{context} key 'label'"),
        "harmful": _require_bool(record.get("harmful"), f"{context} key 'harmful'"),
        "jailbreak_success": _require_bool(
            record.get("jailbreak_success"), f"{context} key 'jailbreak_success'"
        ),
        "success_turn": None
        if record.get("success_turn") is None
        else _require_int(record.get("success_turn"), f"{context} key 'success_turn'"),
        "turns": [
            _normalize_turn(turn, index, f"{context} turn {index}")
            for index, turn in enumerate(normalized_turns_source)
        ],
        "metadata": dict(_require_mapping(record.get("metadata"), f"{context} key 'metadata'")),
    }

    # Keep the output shape exact and predictable, then validate that the
    # normalized record is fully canonical before returning it.
    canonical = {key: normalized[key] for key in REQUIRED_EPISODE_KEYS}
    parse_episode(canonical, context)
    return canonical


def normalize_episode_records(raw_records: list[Any]) -> list[dict[str, Any]]:
    """Normalize a batch of raw episode records."""

    return [normalize_episode_record(record, context=f"episode {index}") for index, record in enumerate(raw_records)]
