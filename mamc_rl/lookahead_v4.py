"""Bounded, non-oracle lookahead baseline for V4 mission allocation.

The search expands actual high-level V4 actions in a copied simulator. During
planning only, stochastic obstacle changes and *future* urgent-target spawning
are suppressed: the planner never sees their future realizations. The real
simulation is never modified. This is a short-horizon heuristic search, not an
optimal scheduler or a perfect simulator of future uncertainty.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .controllers_v4 import HierarchicalResourceEnv
from .policies_v2 import _distance_map


@dataclass
class _Node:
    env: HierarchicalResourceEnv
    first_action: int
    score: float
    finished: bool


def _evaluation(env: HierarchicalResourceEnv) -> float:
    """Surrogate state value: realized value, losses, and feasible near-term work."""
    b = env.base
    completed = float(np.sum(b.priorities[b.completed]))
    expired = float(np.sum(b.priorities[b.expired]))
    potential = 0.0
    positions = [tuple(map(int, position)) for position in b.agents]
    blocked = set(b.obstacles)
    for target in range(4):
        if not b.active[target] or b.completed[target] or b.expired[target]:
            continue
        distance_map = _distance_map(tuple(map(int, b.targets[target])), blocked)
        feasible = []
        for agent in range(2):
            if not b.capabilities[agent, target]:
                continue
            distance = distance_map.get(positions[agent])
            if distance is None or distance > b.fuel[agent]:
                continue
            slack = int(b.deadlines[target]) - int(b.steps) - distance
            if slack < 0:
                continue
            feasible.append((distance, slack))
        if feasible:
            distance, slack = min(feasible, key=lambda pair: (pair[0], pair[1]))
            potential += float(b.priorities[target]) * (
                1.0 / (1.0 + 0.2 * distance) + 0.3 / (1.0 + slack)
            )
    used_fuel = float(np.sum(b.fuel_budgets - b.fuel))
    return 12.0 * completed - 12.0 * expired + 4.0 * potential - 0.08 * used_fuel - 0.02 * b.steps


def _candidates(env: HierarchicalResourceEnv) -> list[int]:
    actions = []
    for act in np.flatnonzero(env.action_masks()):
        action = int(act)
        choices = (action // 5, action % 5)
        # Never deliberately allocate two agents to the same target.
        if choices[0] != 0 and choices[0] == choices[1]:
            continue
        actions.append(action)
    return actions


def _forecast_copy(env: HierarchicalResourceEnv) -> HierarchicalResourceEnv:
    forecast = deepcopy(env)
    forecast.base.dynamic_obstacles = False
    # The future target is unknown before spawn. Suppress its appearance only
    # in forecast simulations; real environment behavior is unchanged.
    if not forecast.base.active[3]:
        forecast.base.urgent_spawn_step = 10**9
    return forecast


def lookahead_action(env: HierarchicalResourceEnv, depth: int = 2, beam_width: int = 4) -> int:
    """Return a valid high-level action from depth-limited beam search.

    Evaluates every legal, nonduplicate first assignment; preserves the best
    `beam_width` partial futures; then expands up to `depth` high-level actions.
    The score is a hand-designed heuristic, not optimality certification.
    """
    if depth < 1 or beam_width < 1:
        raise ValueError('depth and beam_width must be positive')
    candidates = _candidates(env)
    if not candidates:
        raise RuntimeError('No feasible high-level actions')
    frontier: list[_Node] = []
    for action in candidates:
        sim = _forecast_copy(env)
        _, _, done, truncated, _ = sim.step(action)
        frontier.append(_Node(sim, action, _evaluation(sim), bool(done or truncated)))
    frontier.sort(key=lambda node: (-node.score, node.first_action))
    frontier = frontier[:beam_width]

    for _ in range(1, depth):
        expanded: list[_Node] = []
        for node in frontier:
            if node.finished:
                expanded.append(node)
                continue
            for action in _candidates(node.env):
                sim = deepcopy(node.env)  # already a non-oracle forecast
                _, _, done, truncated, _ = sim.step(action)
                expanded.append(_Node(sim, node.first_action, _evaluation(sim), bool(done or truncated)))
        if not expanded:
            break
        expanded.sort(key=lambda node: (-node.score, node.first_action))
        frontier = expanded[:beam_width]
    return int(frontier[0].first_action)
