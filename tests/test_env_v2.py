"""Tests for Version 2; run: python -m pytest tests/test_env_v2.py -q."""
import numpy as np
import pytest
from mamc_rl.env_v2 import DynamicMissionEnv, FIXED_OBSTACLES


def test_initial_state_and_observation():
    env = DynamicMissionEnv()
    obs, _ = env.reset(seed=42)
    assert obs.shape == (125,)
    assert env.observation_space.contains(obs)
    assert env.action_space.n == 25
    assert env.active.tolist() == [True, True, True, False]
    assert env.priorities.tolist() == [1, 2, 3, 0]
    # The inactive target slot must not reveal its future location.
    np.testing.assert_array_equal(obs[104 + 6:104 + 8], [0.0, 0.0])


def test_seed_reproducibility_through_changes():
    a, b = DynamicMissionEnv(obstacle_change_prob=1.0), DynamicMissionEnv(obstacle_change_prob=1.0)
    np.testing.assert_array_equal(a.reset(seed=7)[0], b.reset(seed=7)[0])
    for _ in range(20):
        x = a.step(0)
        y = b.step(0)
        np.testing.assert_array_equal(x[0], y[0])
        assert x[1:] == y[1:]
        if x[2] or x[3]:
            break


def test_urgent_target_appears_without_leaking_future():
    env = DynamicMissionEnv(dynamic_obstacles=False, urgent_spawn_step=3)
    env.reset(seed=11)
    assert not env.active[3]
    for i in range(3):
        obs, _, _, _, info = env.step(0)
        assert info["target_spawned"] == (i == 2)
    assert env.active[3] and env.priorities[3] == 3
    assert env.observation_space.contains(obs)
    assert tuple(env.targets[3]) not in env.obstacles


def test_dynamic_obstacles_do_not_overlap_entities():
    env = DynamicMissionEnv(obstacle_period=1, obstacle_change_prob=1.0)
    env.reset(seed=15)
    for _ in range(20):
        obs, _, term, trunc, _ = env.step(0)
        assert env.observation_space.contains(obs)
        assert all(tuple(p) not in env.obstacles for p in env.agents)
        assert all(tuple(env.targets[i]) not in env.obstacles for i in range(4) if env.active[i])
        assert not np.array_equal(env.agents[0], env.agents[1])
        if term or trunc:
            break


def test_static_configuration_and_time_limit():
    env = DynamicMissionEnv(dynamic_obstacles=False, changing_targets=False, max_steps=1)
    obs, _ = env.reset(seed=19)
    _, _, term, trunc, info = env.step(0)
    assert term or trunc
    assert not info["target_spawned"]
    assert not info["obstacles_changed"]
    with pytest.raises(RuntimeError):
        env.step(0)


def test_all_four_ablation_modes():
    for dyn_obs, dyn_targets in ((False, False), (True, False), (False, True), (True, True)):
        env = DynamicMissionEnv(dynamic_obstacles=dyn_obs, changing_targets=dyn_targets)
        obs, _ = env.reset(seed=3)
        assert env.observation_space.contains(obs)
        for _ in range(18):
            obs, reward, term, trunc, info = env.step(0)
            assert env.observation_space.contains(obs)
            assert isinstance(reward, float)
            assert info["weighted_completed"] <= info["weighted_total"]
            if term or trunc:
                break
