"""Regression tests for the V4 training-only reward ablation."""
import numpy as np
import pytest

from mamc_rl.controllers_v4 import HierarchicalResourceEnv
from mamc_rl.success_reward_v4 import SuccessBonusV4


def test_bonus_wrapper_preserves_masks_and_observations():
    with SuccessBonusV4(HierarchicalResourceEnv(dynamic_obstacles=False), bonus=9.) as env:
        obs, _ = env.reset(seed=3)
        assert env.observation_space.contains(obs)
        np.testing.assert_array_equal(env.action_masks(), env.unwrapped.action_masks())
        action = int(np.flatnonzero(env.action_masks())[0])
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert info['success_bonus'] == 0.0  # A single primitive decision cannot finish V4


def test_bonus_only_when_terminal_success():
    seed = 17
    with HierarchicalResourceEnv(dynamic_obstacles=False) as baseline, \
            SuccessBonusV4(HierarchicalResourceEnv(dynamic_obstacles=False), bonus=9.) as shaped:
        obs0, _ = baseline.reset(seed=seed)
        obs1, _ = shaped.reset(seed=seed)
        np.testing.assert_array_equal(obs0, obs1)
        total_added = 0.0
        rng = np.random.default_rng(123)
        for _ in range(60):
            choices = np.flatnonzero(baseline.action_masks())
            action = int(rng.choice(choices))
            assert shaped.action_masks()[action]
            a = baseline.step(action)
            b = shaped.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            assert a[2:4] == b[2:4]
            assert a[4]['success'] == b[4]['success']
            expected = 9.0 if (a[2] or a[3]) and a[4]['success'] else 0.0
            assert b[1] == pytest.approx(a[1] + expected)
            assert b[4]['success_bonus'] == expected
            total_added += expected
            if a[2] or a[3]:
                break
        assert total_added in (0., 9.)


def test_zero_bonus_is_identical_to_base_reward():
    with HierarchicalResourceEnv() as standard, SuccessBonusV4(HierarchicalResourceEnv(), bonus=0.) as shaped:
        standard.reset(seed=42)
        shaped.reset(seed=42)
        for _ in range(20):
            action = int(np.flatnonzero(standard.action_masks())[0])
            a = standard.step(action)
            b = shaped.step(action)
            assert b[1] == a[1]
            if a[2] or a[3]:
                break


def test_negative_bonus_rejected():
    with pytest.raises(ValueError):
        SuccessBonusV4(HierarchicalResourceEnv(), bonus=-1)
