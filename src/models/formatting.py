"""Deterministic conversation formatting for routing models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from src.datasets.schema import Episode
from src.datasets.schema import Turn


SYSTEM_MARKER = "[SYSTEM]"
USER_MARKER = "[USER]"
ASSISTANT_MARKER = "[ASSISTANT]"
ROUTING_MARKER = "[ROUTING]"

ROLE_MARKERS = {
    "system": SYSTEM_MARKER,
    "user": USER_MARKER,
    "assistant": ASSISTANT_MARKER,
}


@dataclass(slots=True)
class ConversationFormatter:
    """Render a replayable conversation prefix with stable role markers."""

    last_k_turns: int | None = None
    max_chars: int | None = None
    max_tokens_estimate: int | None = None

    def __post_init__(self) -> None:
        active_limits = [
            self.last_k_turns is not None,
            self.max_chars is not None,
            self.max_tokens_estimate is not None,
        ]
        if sum(active_limits) > 1:
            raise ValueError("Use at most one truncation strategy at a time.")
        if self.last_k_turns is not None and self.last_k_turns <= 0:
            raise ValueError("last_k_turns must be greater than 0.")
        if self.max_chars is not None and self.max_chars <= 0:
            raise ValueError("max_chars must be greater than 0.")
        if self.max_tokens_estimate is not None and self.max_tokens_estimate <= 0:
            raise ValueError("max_tokens_estimate must be greater than 0.")

    def format_prefix(
        self,
        episode: Episode,
        decision_turn_index: int,
        *,
        num_warnings: int = 0,
    ) -> str:
        """Format the episode prefix up to and including a decision turn."""

        turns = [turn for turn in episode.turns if turn.turn_index <= decision_turn_index]
        return self._compose_output(
            decision_turn_index=decision_turn_index,
            num_warnings=num_warnings,
            turns=turns,
        )

    def format_observation(self, observation: Mapping[str, Any]) -> str:
        """Format the current environment observation or a compatible mapping."""

        turns = self._extract_turns(observation)
        decision_turn_index = int(
            observation.get("current_turn_index", observation.get("turn_index", 0))
        )
        num_warnings = int(observation.get("num_warnings", 0))
        return self._compose_output(
            decision_turn_index=decision_turn_index,
            num_warnings=num_warnings,
            turns=turns,
        )

    def format_turns(self, turns: Iterable[Turn | Mapping[str, Any]]) -> str:
        """Format an explicit sequence of turns."""

        normalized_turns = [self._coerce_turn(turn) for turn in turns]
        return self._format_turns(normalized_turns)

    def _extract_turns(self, observation: Mapping[str, Any]) -> list[Turn]:
        if "prefix" in observation:
            return [self._coerce_turn(turn) for turn in observation["prefix"]]
        if "turns" in observation:
            return [self._coerce_turn(turn) for turn in observation["turns"]]
        raise ValueError("Observation must contain either 'prefix' or 'turns'.")

    def _coerce_turn(self, turn: Turn | Mapping[str, Any]) -> Turn:
        if isinstance(turn, Turn):
            return turn
        if not isinstance(turn, Mapping):
            raise TypeError(f"Turn must be a Turn or mapping, got {type(turn).__name__}.")
        return Turn(
            role=str(turn["role"]),
            text=str(turn["text"]),
            turn_index=int(turn["turn_index"]),
        )

    def _format_turns(self, turns: list[Turn]) -> str:
        selected_turns = self._apply_truncation(turns)
        pieces = []
        for turn in selected_turns:
            marker = ROLE_MARKERS.get(turn.role)
            if marker is None:
                raise ValueError(f"Unsupported turn role: {turn.role}")
            pieces.append(f"{marker}\n{turn.text}")
        return "\n\n".join(pieces)

    def _compose_output(
        self,
        *,
        decision_turn_index: int,
        num_warnings: int,
        turns: list[Turn],
    ) -> str:
        selected_turns = self._apply_truncation(
            turns,
            decision_turn_index=decision_turn_index,
            num_warnings=num_warnings,
        )
        prefix_text = self._render_without_limits(selected_turns)
        metadata = self._routing_metadata(
            decision_turn_index=decision_turn_index,
            num_warnings=num_warnings,
        )
        if not prefix_text:
            return metadata
        return f"{metadata}\n\n{prefix_text}"

    def _apply_truncation(
        self,
        turns: list[Turn],
        *,
        decision_turn_index: int | None = None,
        num_warnings: int = 0,
    ) -> list[Turn]:
        if not turns:
            return []

        truncated = list(turns)
        if self.last_k_turns is not None:
            truncated = truncated[-self.last_k_turns :]
        if self.max_chars is not None:
            truncated = self._truncate_by_chars(
                truncated,
                decision_turn_index=decision_turn_index,
                num_warnings=num_warnings,
            )
        if self.max_tokens_estimate is not None:
            truncated = self._truncate_by_tokens(
                truncated,
                decision_turn_index=decision_turn_index,
                num_warnings=num_warnings,
            )
        return truncated

    def _truncate_by_chars(
        self,
        turns: list[Turn],
        *,
        decision_turn_index: int | None,
        num_warnings: int,
    ) -> list[Turn]:
        rendered = self._render_with_metadata(
            turns,
            decision_turn_index=decision_turn_index,
            num_warnings=num_warnings,
        )
        if len(rendered) <= self.max_chars:
            return turns
        if (
            len(
                self._render_with_metadata(
                    turns[-1:],
                    decision_turn_index=decision_turn_index,
                    num_warnings=num_warnings,
                )
            )
            > self.max_chars
        ):
            raise ValueError("max_chars is too small to fit even the last turn.")

        kept: list[Turn] = []
        for turn in reversed(turns):
            candidate = [turn, *kept]
            if (
                len(
                    self._render_with_metadata(
                        candidate,
                        decision_turn_index=decision_turn_index,
                        num_warnings=num_warnings,
                    )
                )
                > self.max_chars
            ):
                break
            kept = candidate
        if not kept:
            raise ValueError("max_chars truncation removed every turn.")
        return kept

    def _truncate_by_tokens(
        self,
        turns: list[Turn],
        *,
        decision_turn_index: int | None,
        num_warnings: int,
    ) -> list[Turn]:
        if (
            len(
                self._render_with_metadata(
                    turns,
                    decision_turn_index=decision_turn_index,
                    num_warnings=num_warnings,
                ).split()
            )
            <= self.max_tokens_estimate
        ):
            return turns
        if (
            len(
                self._render_with_metadata(
                    turns[-1:],
                    decision_turn_index=decision_turn_index,
                    num_warnings=num_warnings,
                ).split()
            )
            > self.max_tokens_estimate
        ):
            raise ValueError("max_tokens_estimate is too small to fit even the last turn.")

        kept: list[Turn] = []
        for turn in reversed(turns):
            candidate = [turn, *kept]
            if (
                len(
                    self._render_with_metadata(
                        candidate,
                        decision_turn_index=decision_turn_index,
                        num_warnings=num_warnings,
                    ).split()
                )
                > self.max_tokens_estimate
            ):
                break
            kept = candidate
        if not kept:
            raise ValueError("max_tokens_estimate truncation removed every turn.")
        return kept

    def _render_without_limits(self, turns: list[Turn]) -> str:
        pieces = []
        for turn in turns:
            marker = ROLE_MARKERS.get(turn.role)
            if marker is None:
                raise ValueError(f"Unsupported turn role: {turn.role}")
            pieces.append(f"{marker}\n{turn.text}")
        return "\n\n".join(pieces)

    def _routing_metadata(self, *, decision_turn_index: int, num_warnings: int) -> str:
        return (
            f"{ROUTING_MARKER}\n"
            f"decision_turn_index={decision_turn_index}\n"
            f"num_warnings={num_warnings}"
        )

    def _render_with_metadata(
        self,
        turns: list[Turn],
        *,
        decision_turn_index: int | None,
        num_warnings: int,
    ) -> str:
        if decision_turn_index is None:
            return self._render_without_limits(turns)
        metadata = self._routing_metadata(
            decision_turn_index=decision_turn_index,
            num_warnings=num_warnings,
        )
        prefix_text = self._render_without_limits(turns)
        if not prefix_text:
            return metadata
        return f"{metadata}\n\n{prefix_text}"
