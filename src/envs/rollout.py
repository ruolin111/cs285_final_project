"""Small rollout helpers for replayable routing episodes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.envs.conversation_routing import ConversationRoutingEnv


def rollout_episode(
    env: ConversationRoutingEnv,
    policy_fn: Callable[[dict[str, object]], int],
) -> list[dict[str, Any]]:
    """Collect one full trajectory from the environment."""

    observation, _ = env.reset()
    trajectory: list[dict[str, Any]] = []
    done = False

    while not done:
        action = int(policy_fn(observation))
        next_observation, reward, done, truncated, info = env.step(action)
        trajectory.append(
            {
                "obs": observation,
                "action": action,
                "reward": reward,
                "next_obs": next_observation,
                "done": done,
                "truncated": truncated,
                "info": info,
            }
        )
        observation = next_observation

    return trajectory
