"""Baselines for the joint discrete-action environment."""
from collections import deque
import numpy as np
from .env import MOVES, SIZE, OBSTACLES


def random_action(env):
    return int(env.action_space.sample())


def _shortest_first_move(start, goal, forbidden):
    """BFS shortest-path first step; returns stay if no path is available."""
    start, goal = tuple(start), tuple(goal)
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        point, first = queue.popleft()
        if point == goal:
            return first
        for action, (dr, dc) in enumerate(MOVES[1:], 1):
            q = (point[0] + int(dr), point[1] + int(dc))
            if q not in visited and q not in OBSTACLES and q not in forbidden and 0 <= q[0] < SIZE and 0 <= q[1] < SIZE:
                visited.add(q)
                queue.append((q, action if point == start else first))
    return 0


def heuristic_action(env):
    """Greedy target allocation + obstacle-aware BFS. No learned parameters."""
    pending = [j for j in range(2) if not env.completed[j]]
    if not pending:
        return 0
    # Assign distinct targets when possible; choose globally cheapest assignment.
    if len(pending) == 2:
        cost_01 = np.abs(env.agents[0] - env.targets[pending[0]]).sum() + np.abs(env.agents[1] - env.targets[pending[1]]).sum()
        cost_10 = np.abs(env.agents[0] - env.targets[pending[1]]).sum() + np.abs(env.agents[1] - env.targets[pending[0]]).sum()
        allocation = pending if cost_01 <= cost_10 else pending[::-1]
    else:
        distances = np.abs(env.agents - env.targets[pending[0]]).sum(axis=1)
        best_agent = int(np.argmin(distances))
        allocation = [None, None]
        allocation[best_agent] = pending[0]
    choices = [0, 0]
    for i in range(2):
        if allocation[i] is not None:
            choices[i] = _shortest_first_move(env.agents[i], env.targets[allocation[i]], set())
    proposed = env.agents + MOVES[choices]
    # Conservative collision handling; agent 1 yields.
    if np.array_equal(proposed[0], proposed[1]) or (np.array_equal(proposed[0], env.agents[1]) and np.array_equal(proposed[1], env.agents[0])):
        choices[1] = 0
        proposed = env.agents + MOVES[choices]
        if np.array_equal(proposed[0], proposed[1]):
            choices[0] = 0
    return 5 * choices[0] + choices[1]
