"""Regression tests for deadline-slack feature engineering (V3 unchanged)."""

import numpy as np
import pytest

from mamc_rl.controllers_v3 import HierarchicalDeadlineEnv
from mamc_rl.policies_v2 import _distance_map
from mamc_rl.slack_observation_v3 import SlackObservationV3


def test_observation_and_mask_compatibility():
    env = SlackObservationV3(HierarchicalDeadlineEnv(dynamic_obstacles=False))
    try:
        obs, _ = env.reset(seed=42)
        assert obs.shape == (137,)
        assert env.observation_space.contains(obs)
        assert env.action_masks().shape == (25,)
        assert env.action_masks().sum() == 16
        assert np.all(obs[-8:].reshape(2, 4)[:, 3] == 0)  # no future leak
    finally:
        env.close()


def test_exact_shortest_path_slack_and_agent_order():
    env = SlackObservationV3(HierarchicalDeadlineEnv(dynamic_obstacles=False))
    try:
        obs, _ = env.reset(seed=11)
        base = env.unwrapped.base
        features = obs[-8:].reshape((2, 4))
        for j in range(3):
            dist_map = _distance_map(tuple(base.targets[j]), set(base.obstacles))
            for i in range(2):
                position = tuple(base.agents[i])
                distance = dist_map.get(position)
                expected = (-1.0 if distance is None else
                            np.clip((base.deadlines[j] - base.steps - distance)
                                    / base.max_steps, -1, 1))
                assert features[i, j] == pytest.approx(expected)
    finally:
        env.close()


def test_expired_targets_have_zero_slack():
    env = SlackObservationV3(HierarchicalDeadlineEnv(
        dynamic_obstacles=False, deadlines=(1, 1, 1, 16)
    ))
    try:
        env.reset(seed=12)
        observation, *_ = env.step(0)
        assert np.all(observation[-8:].reshape(2, 4)[:, :3] == 0)
        assert env.action_masks().sum() == 1
    finally:
        env.close()


def test_observations_reproducible_and_urgent_spawn_visible():
    a = SlackObservationV3(HierarchicalDeadlineEnv(dynamic_obstacles=False))
    b = SlackObservationV3(HierarchicalDeadlineEnv(dynamic_obstacles=False))
    try:
        np.testing.assert_array_equal(a.reset(seed=31)[0], b.reset(seed=31)[0])
        assert np.all(a.slack_features().reshape(2, 4)[:, 3] == 0)
        for _ in range(12):
            obs_a, rew_a, term_a, trunc_a, info_a = a.step(0)
            obs_b, rew_b, term_b, trunc_b, info_b = b.step(0)
            np.testing.assert_array_equal(obs_a, obs_b)
            assert (rew_a, term_a, trunc_a) == (rew_b, term_b, trunc_b)
            if info_a['target_spawned']:
                assert a.unwrapped.base.active[3]
                assert np.any(obs_a[-8:].reshape(2, 4)[:, 3] != 0)
                break
        else:
            pytest.fail('urgent objective did not spawn')
    finally:
        a.close()
        b.close()
