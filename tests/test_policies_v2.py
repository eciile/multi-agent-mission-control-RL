"""Unit and integration checks for the V2 adaptive baseline."""

import numpy as np
import pytest

from mamc_rl.env_v2 import DynamicMissionEnv, MOVES
from mamc_rl.policies_v2 import heuristic_action_v2, random_action_v2


def test_actions_are_in_joint_action_space():
    env = DynamicMissionEnv()
    env.reset(seed=7)
    assert env.action_space.contains(heuristic_action_v2(env))
    assert env.action_space.contains(random_action_v2(env))


@pytest.mark.parametrize("seed", [0, 7, 21])
def test_heuristic_avoids_invalid_moves_and_collisions(seed):
    env = DynamicMissionEnv(obstacle_change_prob=1.0)
    env.reset(seed=seed)
    for _ in range(60):
        action = heuristic_action_v2(env)
        before = env.agents.copy()
        pair = [action // 5, action % 5]
        planned = before + MOVES[pair]
        for position in planned:
            assert tuple(position) not in env.obstacles
            assert np.all((position >= 0) & (position < 10))
        assert not np.array_equal(planned[0], planned[1])
        assert not (np.array_equal(planned[0], before[1]) and np.array_equal(planned[1], before[0]))
        _, _, terminated, truncated, info = env.step(action)
        assert info["invalid_actions"] == 0
        assert info["collision"] is False
        if terminated or truncated:
            break


def test_heuristic_is_deterministic_for_identical_seed():
    a, b = DynamicMissionEnv(), DynamicMissionEnv()
    a.reset(seed=55)
    b.reset(seed=55)
    for _ in range(60):
        x = heuristic_action_v2(a)
        y = heuristic_action_v2(b)
        assert x == y
        outcome_a = a.step(x)
        outcome_b = b.step(y)
        np.testing.assert_array_equal(outcome_a[0], outcome_b[0])
        assert outcome_a[1:] == outcome_b[1:]
        if outcome_a[2] or outcome_a[3]:
            break


def test_reacts_to_new_urgent_target():
    env = DynamicMissionEnv(dynamic_obstacles=False, urgent_spawn_step=2)
    env.reset(seed=7)
    # No future target should be available to policy prior to spawn.
    assert not env.active[3]
    for _ in range(2):
        action = heuristic_action_v2(env)
        env.step(action)
    assert env.active[3]
    assert env.action_space.contains(heuristic_action_v2(env))
