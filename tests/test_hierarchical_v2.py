"""Tests for centralized high-level assignment and low-level BFS navigation."""
import numpy as np
import pytest
from mamc_rl.hierarchical_v2 import HierarchicalMissionEnv


def test_action_and_observation_spaces():
    env = HierarchicalMissionEnv()
    obs, _ = env.reset(seed=7)
    assert obs.shape == (125,)
    assert env.action_space.n == 25
    env.close()


def test_reproducibility():
    a = HierarchicalMissionEnv()
    b = HierarchicalMissionEnv()
    x, _ = a.reset(seed=13)
    y, _ = b.reset(seed=13)
    np.testing.assert_array_equal(x, y)
    for cmd in [7, 13, 19, 24, 0]:
        out_a, out_b = a.step(cmd), b.step(cmd)
        np.testing.assert_array_equal(out_a[0], out_b[0])
        assert out_a[1:] == out_b[1:]
        if out_a[2] or out_a[3]:
            break
    a.close()
    b.close()


def test_reaching_goals_avoids_invalid_actions():
    env = HierarchicalMissionEnv()
    env.reset(seed=7)
    for _ in range(60):
        _, _, term, trunc, info = env.step(7) # agent0->target0, agent1->target1
        assert info['invalid_actions'] == 0
        assert info['collision'] is False
        assert 1 <= info['low_level_steps'] <= 3
        if term or trunc:
            break
    env.close()


def test_idle_waits_for_spawn():
    env = HierarchicalMissionEnv(option_horizon=3, dynamic_obstacles=False)
    env.reset(seed=7)
    for _ in range(4):
        _, _, term, trunc, info = env.step(0)
        assert not (term or trunc)
    assert env.base.steps == 12
    assert env.base.active[3]
    assert info['target_spawned']
    env.close()


def test_bad_action_and_horizon():
    with pytest.raises(ValueError):
        HierarchicalMissionEnv(option_horizon=0)
    env = HierarchicalMissionEnv()
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(25)
    env.close()


def test_already_completed_assignment_does_not_end_option_early():
    # Regression: previously completed goals must not trigger an immediate
    # reassignment every primitive step.
    env = HierarchicalMissionEnv(option_horizon=3, dynamic_obstacles=False)
    env.reset(seed=7)
    env.base.completed[0] = True
    # Agent 0 chooses already-completed target 0, Agent 1 stays idle.
    # With no obstacle refresh or target spawn within steps 1..3,
    # the macro-action should run for the full horizon.
    _, _, terminated, truncated, info = env.step(5)
    assert not terminated and not truncated
    assert info["low_level_steps"] == 3
    env.close()
