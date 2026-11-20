"""Regression tests for the non-oracle V4 lookahead baseline."""
import numpy as np
import pytest

from mamc_rl.controllers_v4 import HierarchicalResourceEnv
from mamc_rl.lookahead_v4 import lookahead_action, _forecast_copy


def test_action_is_feasible_and_no_duplicate_assignments():
    env = HierarchicalResourceEnv(dynamic_obstacles=False)
    env.reset(seed=4)
    action = lookahead_action(env, depth=2, beam_width=3)
    assert env.action_masks()[action]
    x, y = divmod(action, 5)
    assert x == 0 or x != y


def test_planning_does_not_modify_real_simulation():
    env = HierarchicalResourceEnv()
    before, _ = env.reset(seed=7)
    fuel = env.base.fuel.copy()
    step = env.base.steps
    action = lookahead_action(env, depth=2, beam_width=2)
    assert env.action_masks()[action]
    np.testing.assert_array_equal(before, env.base._observe())
    np.testing.assert_array_equal(fuel, env.base.fuel)
    assert env.base.steps == step
    assert env.base.dynamic_obstacles
    assert env.base.urgent_spawn_step == 12


def test_forecast_hides_future_random_events():
    env = HierarchicalResourceEnv()
    env.reset(seed=23)
    copy = _forecast_copy(env)
    assert copy.base.dynamic_obstacles is False
    assert copy.base.urgent_spawn_step > copy.base.max_steps
    assert env.base.urgent_spawn_step == 12
    # Once spawned, the urgent objective is already known to both controllers.
    env.base.active[3] = True
    subsequent = _forecast_copy(env)
    assert subsequent.base.urgent_spawn_step == 12


def test_reproducible_without_changing_environment():
    a = HierarchicalResourceEnv(dynamic_obstacles=False)
    b = HierarchicalResourceEnv(dynamic_obstacles=False)
    a.reset(seed=8)
    b.reset(seed=8)
    assert lookahead_action(a, depth=1) == lookahead_action(b, depth=1)


def test_bad_parameters():
    env = HierarchicalResourceEnv()
    env.reset(seed=1)
    with pytest.raises(ValueError):
        lookahead_action(env, depth=0)
    with pytest.raises(ValueError):
        lookahead_action(env, beam_width=0)
