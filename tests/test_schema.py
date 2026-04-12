"""Tests for the canonical replayable episode schema and JSONL loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.datasets.io import load_episode_records
from src.datasets.io import load_episodes
from src.datasets.io import iter_episodes
from src.datasets.preprocess import normalize_episode_record
from src.datasets.schema import DatasetSchemaError
from src.datasets.schema import Episode
from src.datasets.schema import Turn
from src.datasets.schema import parse_episode
from src.datasets.schema import parse_turn


def test_load_episodes_reads_three_canonical_records() -> None:
    episodes = load_episodes(Path("sample_data/episodes.jsonl"))

    assert len(episodes) == 3
    assert [episode.episode_id for episode in episodes] == [
        "harmful-immediate-001",
        "harmful-late-001",
        "benign-001",
    ]
    assert [episode.source for episode in episodes] == ["synthetic", "synthetic", "synthetic"]
    assert [episode.split for episode in episodes] == ["train", "train", "train"]
    assert [episode.label for episode in episodes] == [
        "harmful_immediate",
        "harmful_late",
        "benign",
    ]
    assert [episode.harmful for episode in episodes] == [True, True, False]
    assert [episode.jailbreak_success for episode in episodes] == [True, True, False]
    assert [episode.success_turn for episode in episodes] == [1, 2, None]
    assert episodes[0].turns[episodes[0].success_turn].role == "assistant"
    assert episodes[1].turns[episodes[1].success_turn].role == "assistant"
    assert episodes[2].metadata["scenario"] == "benign"


def test_parse_turn_rejects_unknown_role() -> None:
    with pytest.raises(DatasetSchemaError, match="one of"):
        parse_turn({"role": "critic", "text": "hi", "turn_index": 0}, "turn")


def test_parse_turn_requires_canonical_fields() -> None:
    with pytest.raises(DatasetSchemaError, match="missing required keys"):
        parse_turn({"role": "user", "text": "hi"}, "turn")


def test_parse_episode_rejects_missing_top_level_fields() -> None:
    with pytest.raises(DatasetSchemaError, match="missing required keys"):
        parse_episode(
            {
                "episode_id": "x",
                "source": "synthetic",
                "split": "train",
                "label": "benign",
                "harmful": False,
                "jailbreak_success": False,
                "success_turn": None,
                "turns": [],
            },
            "episode",
        )


def test_parse_episode_rejects_non_boolean_fields() -> None:
    with pytest.raises(DatasetSchemaError, match="must be a boolean"):
        parse_episode(
            {
                "episode_id": "x",
                "source": "synthetic",
                "split": "train",
                "label": "benign",
                "harmful": "false",
                "jailbreak_success": False,
                "success_turn": None,
                "turns": [{"role": "user", "text": "hi", "turn_index": 0}],
                "metadata": {},
            },
            "episode",
        )


def test_parse_episode_rejects_metadata_that_is_not_a_mapping() -> None:
    with pytest.raises(DatasetSchemaError, match="metadata"):
        parse_episode(
            {
                "episode_id": "x",
                "source": "synthetic",
                "split": "train",
                "label": "benign",
                "harmful": False,
                "jailbreak_success": False,
                "success_turn": None,
                "turns": [{"role": "user", "text": "hi", "turn_index": 0}],
                "metadata": [],
            },
            "episode",
        )


def test_parse_episode_rejects_non_contiguous_turn_indices() -> None:
    with pytest.raises(DatasetSchemaError, match="contiguous"):
        parse_episode(
            {
                "episode_id": "x",
                "source": "synthetic",
                "split": "train",
                "label": "benign",
                "harmful": False,
                "jailbreak_success": False,
                "success_turn": None,
                "turns": [
                    {"role": "user", "text": "hi", "turn_index": 0},
                    {"role": "assistant", "text": "ok", "turn_index": 2},
                ],
                "metadata": {},
            },
            "episode",
        )


def test_parse_episode_rejects_success_turn_out_of_range() -> None:
    with pytest.raises(DatasetSchemaError, match="turn range"):
        parse_episode(
            {
                "episode_id": "x",
                "source": "synthetic",
                "split": "train",
                "label": "benign",
                "harmful": True,
                "jailbreak_success": True,
                "success_turn": 3,
                "turns": [{"role": "user", "text": "hi", "turn_index": 0}],
                "metadata": {},
            },
            "episode",
        )


def test_parse_episode_rejects_success_turn_pointing_to_non_assistant_turn() -> None:
    with pytest.raises(DatasetSchemaError, match="assistant turn"):
        parse_episode(
            {
                "episode_id": "x",
                "source": "synthetic",
                "split": "train",
                "label": "benign",
                "harmful": True,
                "jailbreak_success": True,
                "success_turn": 0,
                "turns": [
                    {"role": "user", "text": "hi", "turn_index": 0},
                    {"role": "assistant", "text": "ok", "turn_index": 1},
                ],
                "metadata": {},
            },
            "episode",
        )


@pytest.mark.parametrize(
    ("harmful", "jailbreak_success", "success_turn", "expected_message"),
    [
        (False, True, 0, "harmful is false"),
        (True, False, 0, "jailbreak_success is false"),
        (True, True, None, "non-null when jailbreak_success is true"),
    ],
)
def test_parse_episode_rejects_impossible_success_turn_combinations(
    harmful: bool,
    jailbreak_success: bool,
    success_turn: int | None,
    expected_message: str,
) -> None:
    with pytest.raises(DatasetSchemaError, match=expected_message):
        parse_episode(
            {
                "episode_id": "x",
                "source": "synthetic",
                "split": "train",
                "label": "benign",
                "harmful": harmful,
                "jailbreak_success": jailbreak_success,
                "success_turn": success_turn,
                "turns": [{"role": "user", "text": "hi", "turn_index": 0}],
                "metadata": {},
            },
            "episode",
        )


def test_load_episode_records_reports_line_number_for_bad_jsonl(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text('{"episode_id":"x"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(DatasetSchemaError, match="line 2"):
        load_episode_records(bad_file)


def test_load_episodes_rejects_unexpected_top_level_keys(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad-top-level.jsonl"
    bad_file.write_text(
        (
            '{"episode_id":"x","source":"synthetic","split":"train","label":"benign",'
            '"harmful":false,"jailbreak_success":false,"success_turn":null,'
            '"turns":[{"role":"user","text":"hi","turn_index":0}],"metadata":{},'
            '"extra":"nope"}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetSchemaError, match="unexpected keys"):
        load_episodes(bad_file)


def test_load_episodes_rejects_unexpected_turn_keys(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad-turn-key.jsonl"
    bad_file.write_text(
        (
            '{"episode_id":"x","source":"synthetic","split":"train","label":"benign",'
            '"harmful":false,"jailbreak_success":false,"success_turn":null,'
            '"turns":[{"role":"user","text":"hi","turn_index":0,"extra":"nope"}],'
            '"metadata":{}}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetSchemaError, match="unexpected keys"):
        load_episodes(bad_file)


def test_iter_episodes_streams_and_defers_late_errors(tmp_path: Path) -> None:
    streamed_file = tmp_path / "streamed.jsonl"
    streamed_file.write_text(
        (
            '{"episode_id":"one","source":"synthetic","split":"train","label":"benign",'
            '"harmful":false,"jailbreak_success":false,"success_turn":null,'
            '"turns":[{"role":"user","text":"hi","turn_index":0}],"metadata":{}}\n'
            '{"episode_id":"two","source":"synthetic","split":"train","label":"benign",'
            '"harmful":false,"jailbreak_success":false,"success_turn":null,'
            '"turns":[{"role":"user","text":"hi","turn_index":0,"extra":"nope"}],"metadata":{}}\n'
        ),
        encoding="utf-8",
    )

    episode_iter = iter_episodes(streamed_file)

    first = next(episode_iter)
    assert first.episode_id == "one"

    with pytest.raises(DatasetSchemaError, match="unexpected keys"):
        next(episode_iter)


def test_normalize_episode_record_builds_canonical_shape() -> None:
    normalized = normalize_episode_record(
        {
            "episode_id": "  demo-1  ",
            "source": " synthetic ",
            "split": " train ",
            "label": " benign ",
            "harmful": False,
            "jailbreak_success": False,
            "success_turn": None,
            "turns": [
                {"role": "user", "content": "  hello  "},
                {"role": "assistant", "content": " world "},
            ],
            "metadata": {"scenario": "demo"},
        }
    )

    assert normalized["episode_id"] == "demo-1"
    assert normalized["source"] == "synthetic"
    assert normalized["turns"][0] == {"role": "user", "text": "hello", "turn_index": 0}
    assert normalized["turns"][1] == {"role": "assistant", "text": "world", "turn_index": 1}


def test_normalize_episode_record_rejects_non_canonical_success_turn() -> None:
    with pytest.raises(DatasetSchemaError, match="assistant turn"):
        normalize_episode_record(
            {
                "episode_id": "demo-2",
                "source": "synthetic",
                "split": "train",
                "label": "harmful_bad",
                "harmful": True,
                "jailbreak_success": True,
                "success_turn": 0,
                "turns": [
                    {"role": "user", "content": "bad request"},
                    {"role": "assistant", "content": "unsafe answer"},
                ],
                "metadata": {"scenario": "bad"},
            }
        )


def test_episode_dataclass_rejects_manual_non_canonical_turns() -> None:
    with pytest.raises(DatasetSchemaError, match="contiguous"):
        Episode(
            episode_id="x",
            source="synthetic",
            split="train",
            label="benign",
            harmful=False,
            jailbreak_success=False,
            success_turn=None,
            turns=(
                Turn(role="user", text="hi", turn_index=0),
                Turn(role="assistant", text="ok", turn_index=3),
            ),
            metadata={},
        )
