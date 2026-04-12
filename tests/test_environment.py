"""Tests for the replayable conversation routing environment."""

from __future__ import annotations

import pytest

from src.datasets.schema import Episode
from src.datasets.schema import Turn
from src.envs.conversation_routing import ALLOW
from src.envs.conversation_routing import BLOCK
from src.envs.conversation_routing import WARN
from src.envs.conversation_routing import ConversationRoutingEnv


def make_episode(
    *,
    episode_id: str,
    harmful: bool,
    jailbreak_success: bool,
    success_turn: int | None,
    turns: tuple[Turn, ...],
) -> Episode:
    return Episode(
        episode_id=episode_id,
        source="synthetic",
        split="train",
        label="synthetic",
        harmful=harmful,
        jailbreak_success=jailbreak_success,
        success_turn=success_turn,
        turns=turns,
        metadata={"scenario": episode_id},
    )


def test_harmful_episode_block_before_success_is_positive() -> None:
    episode = make_episode(
        episode_id="harmful-block",
        harmful=True,
        jailbreak_success=True,
        success_turn=2,
        turns=(
            Turn(role="system", text="sys", turn_index=0),
            Turn(role="user", text="user", turn_index=1),
            Turn(role="assistant", text="harmful", turn_index=2),
        ),
    )
    env = ConversationRoutingEnv([episode])

    obs, info = env.reset()
    next_obs, reward, done, truncated, step_info = env.step(BLOCK)

    assert obs["current_turn_index"] == 1
    assert done is True
    assert truncated is False
    assert reward > 0.0
    assert next_obs["episode_id"] == episode.episode_id
    assert step_info["blocked"] is True
    assert step_info["block_turn"] == 1
    assert step_info["final_outcome"] == "harmful_blocked"
    assert step_info["success_turn"] == 2
    assert info["episode_id"] == episode.episode_id


def test_harmful_episode_miss_reaches_success_and_penalizes() -> None:
    episode = make_episode(
        episode_id="harmful-miss",
        harmful=True,
        jailbreak_success=True,
        success_turn=2,
        turns=(
            Turn(role="system", text="sys", turn_index=0),
            Turn(role="user", text="user", turn_index=1),
            Turn(role="assistant", text="harmful", turn_index=2),
        ),
    )
    env = ConversationRoutingEnv([episode])

    env.reset()
    next_obs, reward, done, truncated, info = env.step(ALLOW)

    assert done is True
    assert truncated is False
    assert reward < 0.0
    assert reward <= -1.0
    assert info["blocked"] is False
    assert info["jailbreak_success"] is True
    assert info["final_outcome"] == "harmful_success"
    assert next_obs["current_turn_index"] == 2
    assert next_obs["current_turn_role"] == "assistant"
    assert next_obs["turn_text"] == "harmful"


def test_harmful_multi_user_timing_path_warns_then_hits_terminal_success() -> None:
    episode = make_episode(
        episode_id="harmful-multi-user",
        harmful=True,
        jailbreak_success=True,
        success_turn=4,
        turns=(
            Turn(role="system", text="sys", turn_index=0),
            Turn(role="user", text="user-1", turn_index=1),
            Turn(role="assistant", text="assistant-1", turn_index=2),
            Turn(role="user", text="user-2", turn_index=3),
            Turn(role="assistant", text="harmful", turn_index=4),
            Turn(role="user", text="user-3", turn_index=5),
        ),
    )
    env = ConversationRoutingEnv([episode])

    env.reset()
    obs, reward, done, truncated, info = env.step(ALLOW)
    assert done is False
    assert truncated is False
    assert reward == 0.0
    assert obs["current_turn_index"] == 3
    assert obs["current_turn_role"] == "user"

    next_obs, reward, done, truncated, info = env.step(WARN)
    assert done is True
    assert truncated is False
    assert reward < 0.0
    assert info["num_warnings"] == 1
    assert info["final_outcome"] == "harmful_success"
    assert info["blocked"] is False
    assert next_obs["current_turn_index"] == 4
    assert next_obs["current_turn_role"] == "assistant"
    assert next_obs["turn_text"] == "harmful"


def test_benign_episode_block_is_a_false_positive() -> None:
    episode = make_episode(
        episode_id="benign-block",
        harmful=False,
        jailbreak_success=False,
        success_turn=None,
        turns=(
            Turn(role="system", text="sys", turn_index=0),
            Turn(role="user", text="user", turn_index=1),
            Turn(role="assistant", text="ok", turn_index=2),
        ),
    )
    env = ConversationRoutingEnv([episode])

    env.reset()
    _, reward, done, truncated, info = env.step(BLOCK)

    assert done is True
    assert truncated is False
    assert reward < 0.0
    assert info["blocked"] is True
    assert info["final_outcome"] == "benign_blocked"


def test_episode_advances_across_user_turns_until_completion() -> None:
    episode = make_episode(
        episode_id="benign-progress",
        harmful=False,
        jailbreak_success=False,
        success_turn=None,
        turns=(
            Turn(role="system", text="sys", turn_index=0),
            Turn(role="user", text="user-1", turn_index=1),
            Turn(role="assistant", text="assistant-1", turn_index=2),
            Turn(role="user", text="user-2", turn_index=3),
            Turn(role="assistant", text="assistant-2", turn_index=4),
        ),
    )
    env = ConversationRoutingEnv([episode])

    obs, _ = env.reset()
    assert obs["current_turn_index"] == 1

    obs, reward, done, truncated, info = env.step(ALLOW)
    assert done is False
    assert truncated is False
    assert reward == 0.0
    assert obs["current_turn_index"] == 3
    assert info["num_warnings"] == 0

    obs, reward, done, truncated, info = env.step(ALLOW)
    assert done is True
    assert truncated is False
    assert reward >= 0.0
    assert obs["episode_id"] == episode.episode_id
    assert obs["current_turn_index"] == 4
    assert obs["current_turn_role"] == "assistant"
    assert obs["turn_text"] == "assistant-2"
    assert info["final_outcome"] == "benign_allowed"
    assert info["blocked"] is False


def test_environment_rejects_success_turn_before_first_user_turn() -> None:
    episode = make_episode(
        episode_id="invalid-success-turn",
        harmful=True,
        jailbreak_success=True,
        success_turn=0,
        turns=(
            Turn(role="assistant", text="harmful", turn_index=0),
            Turn(role="user", text="user", turn_index=1),
        ),
    )

    with pytest.raises(ValueError, match="before the first user turn"):
        ConversationRoutingEnv([episode])
