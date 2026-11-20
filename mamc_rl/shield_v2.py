"""Inference-time feasibility shield for V2 centralized PPO.

No learning, shortest-path planning or target information is used here.
Only the currently observed agent locations and obstacle map determine feasibility.
The policy's own probabilities rank the feasible joint actions.
"""
from __future__ import annotations

import numpy as np

# Same action encoding as DynamicMissionEnv: 5 * agent0_action + agent1_action.
MOVES = np.asarray([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int64)


def valid_joint_mask(agents: np.ndarray, obstacles: set[tuple[int, int]], size: int = 10) -> np.ndarray:
    """Return a boolean mask of 25 actions feasible under V2 movement rules.

    Both agents move simultaneously. Excludes invalid grid/wall moves,
    shared destinations, and position swaps. Stay/stay remains feasible.
    """
    agents = np.asarray(agents, dtype=np.int64)
    if agents.shape != (2, 2):
        raise ValueError("agents must have shape (2, 2)")
    blocked = set(map(tuple, obstacles))
    mask = np.zeros(25, dtype=bool)
    for joint in range(25):
        pair = [joint // 5, joint % 5]
        dest = agents + MOVES[pair]
        if np.any(dest < 0) or np.any(dest >= size):
            continue
        if any(tuple(p) in blocked for p in dest):
            continue
        if np.array_equal(dest[0], dest[1]):
            continue
        if np.array_equal(dest[0], agents[1]) and np.array_equal(dest[1], agents[0]):
            continue
        mask[joint] = True
    if not mask[0]:
        raise ValueError("stay/stay unexpectedly infeasible; overlapping or blocked agent state")
    return mask


def select_shielded_action(probabilities: np.ndarray, mask: np.ndarray) -> int:
    """Select most probable feasible action, without modifying the policy."""
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if probabilities.shape != (25,) or mask.shape != (25,):
        raise ValueError("probabilities and mask must each have 25 entries")
    if not mask.any() or not np.all(np.isfinite(probabilities)):
        raise ValueError("Invalid action mask or action probabilities")
    return int(np.argmax(np.where(mask, probabilities, -np.inf)))
