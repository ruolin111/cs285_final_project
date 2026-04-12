"""Strict replayable schema for prompt-routing episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class DatasetSchemaError(ValueError):
    """Raised when an episode record does not match the canonical schema."""


ALLOWED_TURN_ROLES = ("user", "assistant", "system")

REQUIRED_EPISODE_KEYS = (
    "episode_id",
    "source",
    "split",
    "label",
    "harmful",
    "jailbreak_success",
    "success_turn",
    "turns",
    "metadata",
)

REQUIRED_TURN_KEYS = ("role", "text", "turn_index")


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


def _require_optional_int(value: Any, context: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, context)


def _require_exact_keys(record: dict[str, Any], required: tuple[str, ...], context: str) -> None:
    missing = [key for key in required if key not in record]
    if missing:
        missing_keys = ", ".join(missing)
        raise DatasetSchemaError(f"{context} is missing required keys: {missing_keys}.")

    extra = sorted(set(record) - set(required))
    if extra:
        extra_keys = ", ".join(extra)
        raise DatasetSchemaError(f"{context} has unexpected keys: {extra_keys}.")


@dataclass(frozen=True, slots=True)
class Turn:
    """A single replayable conversation turn."""

    role: str
    text: str
    turn_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _require_string(self.role, "Turn.role").lower())
        object.__setattr__(self, "text", _require_string(self.text, "Turn.text"))
        object.__setattr__(self, "turn_index", _require_int(self.turn_index, "Turn.turn_index"))

        if self.role not in ALLOWED_TURN_ROLES:
            allowed = ", ".join(ALLOWED_TURN_ROLES)
            raise DatasetSchemaError(f"Turn.role must be one of: {allowed}.")
        if self.turn_index < 0:
            raise DatasetSchemaError("Turn.turn_index must be greater than or equal to 0.")


@dataclass(frozen=True, slots=True)
class Episode:
    """A strict replayable episode with canonical replay metadata.

    `success_turn` is the raw conversation `turn_index` of the first harmful
    assistant response in the stored trajectory. It is never a user decision-step
    index, and it must be null for benign or unsuccessful episodes.
    """

    episode_id: str
    source: str
    split: str
    label: str
    harmful: bool
    jailbreak_success: bool
    success_turn: int | None
    turns: tuple[Turn, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _require_string(self.episode_id, "Episode.episode_id"))
        object.__setattr__(self, "source", _require_string(self.source, "Episode.source"))
        object.__setattr__(self, "split", _require_string(self.split, "Episode.split"))
        object.__setattr__(self, "label", _require_string(self.label, "Episode.label"))
        object.__setattr__(self, "harmful", _require_bool(self.harmful, "Episode.harmful"))
        object.__setattr__(
            self,
            "jailbreak_success",
            _require_bool(self.jailbreak_success, "Episode.jailbreak_success"),
        )
        object.__setattr__(
            self,
            "success_turn",
            _require_optional_int(self.success_turn, "Episode.success_turn"),
        )

        if not isinstance(self.turns, tuple):
            raise DatasetSchemaError(
                f"Episode.turns must be a tuple of Turn objects, got {type(self.turns).__name__}."
            )
        if not self.turns:
            raise DatasetSchemaError("Episode.turns must contain at least one turn.")
        for index, turn in enumerate(self.turns):
            if not isinstance(turn, Turn):
                raise DatasetSchemaError(
                    f"Episode.turns[{index}] must be a Turn, got {type(turn).__name__}."
                )

        metadata = _require_mapping(self.metadata, "Episode.metadata")
        object.__setattr__(self, "metadata", dict(metadata))

        expected_indices = list(range(len(self.turns)))
        actual_indices = [turn.turn_index for turn in self.turns]
        if actual_indices != expected_indices:
            raise DatasetSchemaError(
                "Episode.turns must have contiguous turn_index values starting at 0."
            )

        if self.success_turn is not None and not self.harmful:
            raise DatasetSchemaError("Episode.success_turn must be null when harmful is false.")
        if self.success_turn is not None and not self.jailbreak_success:
            raise DatasetSchemaError(
                "Episode.success_turn must be null when jailbreak_success is false."
            )
        if self.jailbreak_success and self.success_turn is None:
            raise DatasetSchemaError(
                "Episode.success_turn must be non-null when jailbreak_success is true."
            )
        if self.success_turn is not None and not 0 <= self.success_turn < len(self.turns):
            raise DatasetSchemaError("Episode.success_turn must be within the turn range or null.")
        if self.success_turn is not None and self.turns[self.success_turn].role != "assistant":
            raise DatasetSchemaError("Episode.success_turn must point to an assistant turn.")


def parse_turn(data: Any, context: str) -> Turn:
    """Parse and validate a canonical turn record."""

    record = _require_mapping(data, context)
    _require_exact_keys(record, REQUIRED_TURN_KEYS, context)
    return Turn(
        role=record["role"],
        text=record["text"],
        turn_index=record["turn_index"],
    )


def parse_episode(data: Any, context: str) -> Episode:
    """Parse and validate a canonical episode record."""

    record = _require_mapping(data, context)
    _require_exact_keys(record, REQUIRED_EPISODE_KEYS, context)

    turns_value = record["turns"]
    if not isinstance(turns_value, list):
        raise DatasetSchemaError(f"{context} key 'turns' must be a list of turn records.")

    turns = tuple(
        parse_turn(turn_data, f"{context} turn {index}")
        for index, turn_data in enumerate(turns_value)
    )
    metadata = _require_mapping(record["metadata"], f"{context} key 'metadata'")
    return Episode(
        episode_id=record["episode_id"],
        source=record["source"],
        split=record["split"],
        label=record["label"],
        harmful=record["harmful"],
        jailbreak_success=record["jailbreak_success"],
        success_turn=record["success_turn"],
        turns=turns,
        metadata=metadata,
    )
