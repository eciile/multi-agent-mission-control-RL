"""Non-learning, adaptive controller for DynamicMissionEnv (Version 2).

The controller uses only the CURRENT observable state. At each decision it:
  1. computes obstacle-aware shortest paths via breadth-first search (BFS),
  2. selects distinct active objectives for the two agents using a transparent
     priority/distance tradeoff,
  3. picks the highest-scoring collision-free joint move.

No future target locations or obstacle changes are consulted. This is a
replanning heuristic, NOT a globally optimal planner or an RL algorithm.
"""
from __future__ import annotations

from collections import deque
from itertools import product

import numpy as np

from .env_v2 import MOVES, N_AGENTS, N_TARGETS, SIZE


def random_action_v2(env) -> int:
    """Uniform random joint action (the random baseline)."""
    return int(env.action_space.sample())


def _distance_map(goal: tuple[int, int], blocked: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """BFS distance to goal from every reachable cell, respecting obstacles."""
    if goal in blocked:
        return {}
    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        r, c = queue.popleft()
        for dr, dc in MOVES[1:]:
            nxt = (r + int(dr), c + int(dc))
            if (0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE
                    and nxt not in blocked and nxt not in distances):
                distances[nxt] = distances[(r, c)] + 1
                queue.append(nxt)
    return distances


def _goal_assignment(
    positions: list[tuple[int, int]],
    pending: list[int],
    targets: np.ndarray,
    priorities: np.ndarray,
    maps: dict[int, dict[tuple[int, int], int]],
) -> tuple[int | None, int | None]:
    """Assign distinct goals to agents; permit unassigned idle agents.

    Scores prioritize valuable goals that can be reached soon. The same
    scoring rule applies after urgent target appearances or map changes.
    """
    options: list[int | None] = [None, *pending]
    best_score = -float("inf")
    best_pair: tuple[int | None, int | None] = (None, None)
    for pair in product(options, repeat=N_AGENTS):
        if pair[0] is not None and pair[0] == pair[1]:
            continue
        total = 0.0
        valid = True
        for i, goal_id in enumerate(pair):
            if goal_id is None:
                continue
            d = maps[goal_id].get(positions[i])
            if d is None:
                valid = False
                break
            # Strong task priority signal, moderated by path length.
            total += float(priorities[goal_id]) / (1.0 + 0.25 * d)
        if valid and total > best_score:
            best_score = total
            best_pair = pair
    return best_pair


def _joint_is_legal(
    current: list[tuple[int, int]],
    proposed: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> bool:
    if any(not (0 <= r < SIZE and 0 <= c < SIZE) or (r, c) in blocked for r, c in proposed):
        return False
    if proposed[0] == proposed[1]:
        return False
    if proposed[0] == current[1] and proposed[1] == current[0]:
        return False
    return True


def heuristic_action_v2(env) -> int:
    """Select a collision-free joint action via priority-aware BFS replanning.

    Reads current positions, obstacles, priorities, target locations and target
    activity/completion. It does not inspect the RNG, future target slots, or
    hidden future obstacle layouts. Valid for 2 agents and 5 actions each.
    """
    positions = [tuple(int(v) for v in row) for row in env.agents]
    blocked = set(env.obstacles)
    pending = [i for i in range(N_TARGETS) if env.active[i] and not env.completed[i]]
    if not pending:
        return 0  # Wait for urgent mission to appear or for episode termination.

    maps = {i: _distance_map(tuple(int(v) for v in env.targets[i]), blocked)
            for i in pending}
    goals = _goal_assignment(positions, pending, env.targets, env.priorities, maps)

    best_action = 0
    best_score = -float("inf")
    for first, second in product(range(5), repeat=2):
        actions = (first, second)
        proposed = [
            (positions[i][0] + int(MOVES[a][0]), positions[i][1] + int(MOVES[a][1]))
            for i, a in enumerate(actions)
        ]
        if not _joint_is_legal(positions, proposed, blocked):
            continue

        score = 0.0
        for i, goal_id in enumerate(goals):
            if goal_id is None:
                # Discourage idle agents from wandering without objectives.
                score -= 0.01 if actions[i] else 0.0
                continue
            before = maps[goal_id].get(positions[i])
            after = maps[goal_id].get(proposed[i])
            if before is None or after is None:
                score -= 100.0
                continue
            # Each step along the shortest path is rewarded; target completion
            # breaks ties. Values are for choosing an action, not env rewards.
            score += 2.0 * (before - after)
            if after == 0:
                score += 0.3 * float(env.priorities[goal_id])
            if actions[i] == 0:
                score -= 0.02
        # Stable tie-break: iterate joint actions in ascending ID order.
        if score > best_score:
            best_score = score
            best_action = 5 * first + second
    return int(best_action)
