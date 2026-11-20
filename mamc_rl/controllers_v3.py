"""High-level V3 wrappers and non-learning assignment baselines."""
from __future__ import annotations

from itertools import product
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .env_v3 import DeadlineMissionEnv
from .hierarchical_v2 import HierarchicalMissionEnv
from .policies_v2 import _distance_map


class HierarchicalDeadlineEnv(HierarchicalMissionEnv):
    def __init__(self, option_horizon=3, **kwargs):
        # Do not call HierarchicalMissionEnv.__init__: it constructs the V2
        # simulator, which cannot accept V3-only arguments such as deadlines.
        gym.Env.__init__(self)
        if option_horizon < 1:
            raise ValueError("option_horizon must be >= 1")
        self.option_horizon = int(option_horizon)
        self.base = DeadlineMissionEnv(**kwargs)
        self.observation_space = self.base.observation_space
        self.action_space = spaces.Discrete(25)
        self.render_mode = self.base.render_mode

    def _move_for_goals(self, goals):
        # A goal that expires in the middle of a macro-action must not
        # continue attracting an agent toward an impossible objective.
        goals = tuple(None if g is not None and self.base.expired[g] else g
                      for g in goals)
        return super()._move_for_goals(goals)
    def step(self, action):
        # The V2 parent interrupts on target completion/spawn/map changes.
        # For V3, also interrupt immediately when ANY deadline expires.
        # Limit each parent call to one primitive step to observe expiry.
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid high-level action {action}')
        action = int(action)
        goals = tuple(None if c == 0 else c - 1 for c in (action // 5, action % 5))
        started = tuple(g is not None and self.base.active[g]
                        and not self.base.completed[g] and not self.base.expired[g]
                        for g in goals)
        total = 0.0
        for elapsed in range(1, self.option_horizon + 1):
            before_expired = self.base.expired.copy()
            obs, reward, terminated, truncated, info = self.base.step(self._move_for_goals(goals))
            total += float(reward)
            newly_completed = any(started[i] and self.base.completed[g]
                                  for i, g in enumerate(goals) if g is not None)
            newly_expired = bool(np.any(self.base.expired & ~before_expired))
            if (terminated or truncated or info['target_spawned'] or
                    info['obstacles_changed'] or newly_completed or newly_expired):
                break
        info = dict(info, high_level_action=action, low_level_steps=elapsed)
        return obs, total, terminated, truncated, info

    def action_masks(self):
        valid = [0] + [i + 1 for i in range(4)
                       if self.base.active[i] and not self.base.completed[i] and not self.base.expired[i]]
        mask = np.zeros(25, dtype=bool)
        for a, b in product(valid, repeat=2):
            mask[5*a + b] = True
        return mask


def masked_random_action(env, rng):
    return int(rng.choice(np.flatnonzero(env.action_masks())))


def deadline_heuristic_action(env):
    """Choose distinct reachable tasks, scoring reward and deadline slack.

    Uses current observable positions, deadlines, obstacles; does not look ahead
    to future urgent target coordinates or obstacle randomness.
    """
    base = env.base
    positions = [tuple(map(int, row)) for row in base.agents]
    available = [i for i in range(4) if base.active[i] and not base.completed[i] and not base.expired[i]]
    maps = {i: _distance_map(tuple(map(int, base.targets[i])), set(base.obstacles)) for i in available}
    choices = [None] + available
    best_score, best = -float('inf'), (None, None)
    for a, b in product(choices, repeat=2):
        if a is not None and a == b:
            continue
        score = 0.
        for agent, target in enumerate((a, b)):
            if target is None:
                continue
            d = maps[target].get(positions[agent], float('inf'))
            slack = int(base.deadlines[target]) - base.steps - d
            if slack < 0:
                score -= 3.0  # cannot arrive by deadline with current obstacles
            else:
                score += (3.0 * int(base.priorities[target]) / (1 + 0.2*d)
                          + 1.5 / (1.0 + slack))
        if score > best_score:
            best_score, best = score, (a, b)
    a, b = best
    action = 5 * (0 if a is None else a+1) + (0 if b is None else b+1)
    assert env.action_masks()[action]
    return action
