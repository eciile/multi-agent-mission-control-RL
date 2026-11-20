"""Regression checks for V4 feasibility and policy-sensitivity audit."""

import numpy as np

from mamc_rl.controllers_v4 import HierarchicalResourceEnv
from audit_v4_feasibility import (
    audit_state, collect_states, feasible_joint_actions,
    transition_sensitivity, visible_feasible_targets,
)


def test_individual_feasibility_respects_capabilities():
    env = HierarchicalResourceEnv(dynamic_obstacles=False)
    env.reset(seed=2000)
    a0, a1 = visible_feasible_targets(env)
    assert 2 not in a0 and 4 not in a0
    assert 1 not in a1
    assert 4 not in a1  # urgent target hidden before spawn
    env.close()


def test_feasible_joint_actions_are_masked_and_include_idle():
    env = HierarchicalResourceEnv(dynamic_obstacles=False)
    env.reset(seed=2000)
    available = feasible_joint_actions(env)
    assert 0 in available
    assert all(env.action_masks()[a] for a in available)
    env.close()


def test_sensitivity_does_not_change_real_environment():
    env = HierarchicalResourceEnv(dynamic_obstacles=False)
    env.reset(seed=2000)
    before = (env.base.steps, env.base.agents.copy(), env.base.fuel.copy(),
              env.base.completed.copy())
    result = transition_sensitivity(env)
    assert result['unique_successors'] >= 1
    assert env.base.steps == before[0]
    np.testing.assert_array_equal(env.base.agents, before[1])
    np.testing.assert_array_equal(env.base.fuel, before[2])
    np.testing.assert_array_equal(env.base.completed, before[3])
    env.close()


def test_audit_counts_match_actual_mask():
    env = HierarchicalResourceEnv(dynamic_obstacles=False)
    env.reset(seed=2000)
    row = audit_state(env, seed=2000, decision=0)
    assert row['legal_joint_actions'] == int(env.action_masks().sum())
    assert 1 <= row['reachable_joint_actions'] <= row['legal_joint_actions']
    assert row['unique_successors'] <= row['legal_joint_actions']
    env.close()


def test_seed_reproducibility_of_audit():
    x = collect_states(seed=2000, n_decisions=2, horizon=3)
    y = collect_states(seed=2000, n_decisions=2, horizon=3)
    assert x == y
