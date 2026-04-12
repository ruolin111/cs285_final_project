"""Tests for the larger synthetic milestone dataset."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.datasets.io import load_episodes


DATASET_PATH = Path("sample_data/ppo_synthetic_dataset.jsonl")
TRAIN_SPLIT_PATH = Path("sample_data/ppo_synthetic_dataset_train.jsonl")
VAL_SPLIT_PATH = Path("sample_data/ppo_synthetic_dataset_val.jsonl")
TEST_SPLIT_PATH = Path("sample_data/ppo_synthetic_dataset_test.jsonl")


def test_generated_synthetic_dataset_has_expected_size_and_splits() -> None:
    episodes = load_episodes(DATASET_PATH)

    assert len(episodes) == 120

    split_counts = Counter(episode.split for episode in episodes)
    assert split_counts == {
        "train": 84,
        "val": 18,
        "test": 18,
    }


def test_generated_synthetic_dataset_has_expected_label_balance() -> None:
    episodes = load_episodes(DATASET_PATH)

    harmful_success = sum(
        1
        for episode in episodes
        if episode.harmful and episode.jailbreak_success and episode.success_turn is not None
    )
    harmful_no_success = sum(
        1
        for episode in episodes
        if episode.harmful and not episode.jailbreak_success and episode.success_turn is None
    )
    benign = sum(1 for episode in episodes if not episode.harmful)

    assert harmful_success == 45
    assert harmful_no_success == 15
    assert benign == 60


def test_generated_synthetic_dataset_contains_multi_turn_decision_horizons() -> None:
    episodes = load_episodes(DATASET_PATH)

    num_user_turns = [
        sum(1 for turn in episode.turns if turn.role == "user")
        for episode in episodes
    ]
    assert max(num_user_turns) >= 4
    assert sum(1 for count in num_user_turns if count >= 3) >= 60


def test_generated_harmful_success_turns_point_to_assistant_turns() -> None:
    episodes = load_episodes(DATASET_PATH)

    harmful_success_episodes = [
        episode for episode in episodes if episode.harmful and episode.jailbreak_success
    ]

    assert len(harmful_success_episodes) == 45
    assert all(episode.success_turn is not None for episode in harmful_success_episodes)
    assert all(
        episode.turns[episode.success_turn].role == "assistant"
        for episode in harmful_success_episodes
    )


def test_generated_split_files_exist_with_expected_sizes() -> None:
    train_episodes = load_episodes(TRAIN_SPLIT_PATH)
    val_episodes = load_episodes(VAL_SPLIT_PATH)
    test_episodes = load_episodes(TEST_SPLIT_PATH)

    assert len(train_episodes) == 84
    assert len(val_episodes) == 18
    assert len(test_episodes) == 18
    assert {episode.split for episode in train_episodes} == {"train"}
    assert {episode.split for episode in val_episodes} == {"val"}
    assert {episode.split for episode in test_episodes} == {"test"}
