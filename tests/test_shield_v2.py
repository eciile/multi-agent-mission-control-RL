"""Focused tests for the inference-time feasibility shield."""
import numpy as np

from mamc_rl.shield_v2 import valid_joint_mask, select_shielded_action


def test_right_boundary_blocks_right_movement():
    mask = valid_joint_mask(np.array([[0, 9], [9, 1]]), set())
    assert not mask[5 * 4 + 0]
    assert mask[0]


def test_obstacle_blocks_destination():
    mask = valid_joint_mask(np.array([[4, 4], [8, 8]]), {(4, 5)})
    assert not mask[5 * 4 + 0]
    assert mask[5 * 1 + 0]


def test_collision_and_swap_rejected():
    agents = np.array([[4, 4], [4, 5]])
    mask = valid_joint_mask(agents, set())
    assert not mask[5 * 4 + 0]  # Agent 0 moves into staying Agent 1
    assert not mask[5 * 4 + 3]  # Swap places
    assert mask[0]


def test_highest_feasible_probability_selected():
    agents = np.array([[0, 9], [9, 1]])
    mask = valid_joint_mask(agents, set())
    probs = np.zeros(25)
    probs[20] = 0.9  # Invalid: Agent 0 right at boundary
    probs[0] = 0.08
    probs[1] = 0.02
    assert select_shielded_action(probs, mask) == 0


def test_all_masked_fails():
    try:
        select_shielded_action(np.ones(25), np.zeros(25, dtype=bool))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
