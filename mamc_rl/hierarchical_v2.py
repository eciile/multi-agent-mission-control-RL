"""Hierarchical V2: PPO chooses goals, obstacle-aware BFS chooses movements.

High-level action Discrete(25): 5 * choice_agent0 + choice_agent1.
Each choice: 0 = idle; 1..4 = target index 0..3. Invalid/inactive/
completed target choices become idle until the next PPO decision.
Each high-level action executes up to `option_horizon` low-level V2 steps.
This is centralized hierarchical RL, NOT decentralized MARL.
"""
from __future__ import annotations

from itertools import product
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .env_v2 import DynamicMissionEnv, MOVES
from .policies_v2 import _distance_map, _joint_is_legal


class HierarchicalMissionEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}

    def __init__(self, option_horizon: int = 3, **env_kwargs):
        super().__init__()
        if option_horizon < 1:
            raise ValueError("option_horizon must be >= 1")
        self.option_horizon = int(option_horizon)
        self.base = DynamicMissionEnv(**env_kwargs)
        self.observation_space = self.base.observation_space
        self.action_space = spaces.Discrete(25)
        self.render_mode = self.base.render_mode

    def reset(self, *, seed=None, options=None):
        return self.base.reset(seed=seed, options=options)

    def _move_for_goals(self, goals: tuple[int | None, int | None]) -> int:
        env = self.base
        positions = [tuple(int(x) for x in row) for row in env.agents]
        blocked = set(env.obstacles)
        maps = {}
        active_goals = []
        for choice in goals:
            if choice is None or not env.active[choice] or env.completed[choice]:
                active_goals.append(None)
            else:
                active_goals.append(choice)
                if choice not in maps:
                    maps[choice] = _distance_map(tuple(int(v) for v in env.targets[choice]), blocked)

        best = 0
        best_score = -float("inf")
        for a0, a1 in product(range(5), repeat=2):
            cmds = [a0, a1]
            candidate = [(positions[i][0] + int(MOVES[cmds[i]][0]),
                          positions[i][1] + int(MOVES[cmds[i]][1])) for i in range(2)]
            if not _joint_is_legal(positions, candidate, blocked):
                continue
            score = 0.0
            for i, target in enumerate(active_goals):
                if target is None:
                    score -= 0.01 if cmds[i] else 0.0
                    continue
                old_d = maps[target].get(positions[i])
                new_d = maps[target].get(candidate[i])
                if old_d is None or new_d is None:
                    score -= 100
                else:
                    score += 2.0 * (old_d - new_d)
                    if new_d == 0:
                        score += 0.3 * float(env.priorities[target])
                    if cmds[i] == 0:
                        score -= 0.02
            if score > best_score:
                best_score, best = score, 5 * a0 + a1
        return best

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid high-level action {action}")
        action = int(action)
        choices = (action // 5, action % 5)
        goals = tuple(None if a == 0 else a - 1 for a in choices)
        # Only interrupt the option when an objective that was unfinished at
        # the start of this decision becomes completed during its execution.
        # A previously completed/inactive assignment is treated as idle, not
        # as a fresh completion event.
        initially_unfinished = tuple(
            g is not None and bool(self.base.active[g]) and not bool(self.base.completed[g])
            for g in goals
        )
        reward_sum = 0.0
        elapsed = 0
        terminated = truncated = False
        last_info = {}
        obs = None
        for _ in range(self.option_horizon):
            low_action = self._move_for_goals(goals)
            obs, reward, terminated, truncated, last_info = self.base.step(low_action)
            elapsed += 1
            reward_sum += float(reward)
            # New mission or map changes trigger immediate high-level reconsideration.
            if (terminated or truncated or last_info["target_spawned"] or
                    last_info["obstacles_changed"]):
                break
            # An assigned target has completed: reassign before further movement.
            if any(
                initially_unfinished[i] and bool(self.base.completed[g])
                for i, g in enumerate(goals) if g is not None
            ):
                break
        info = dict(last_info)
        info.update(high_level_action=action, low_level_steps=elapsed)
        return obs, reward_sum, terminated, truncated, info

    def render(self):
        return self.base.render()

    def close(self):
        self.base.close()
