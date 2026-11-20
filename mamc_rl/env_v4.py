"""V4: deadline missions with agent-specific capabilities and movement budgets.

Keeps V3 reward, deadlines, target generation and obstacle-change semantics.
Adds an explicit capability matrix (agent x target) and finite per-agent fuel.
No future urgent target coordinates are exposed. When an agent exhausts fuel,
it may wait but cannot move. A target only counts if a capable agent reaches it.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .env_v2 import MOVES, N_TARGETS, SIZE
from .env_v3 import DeadlineMissionEnv


DEFAULT_CAPABILITIES = ((1, 0, 1, 0), (0, 1, 1, 1))


class ResourceMissionEnv(DeadlineMissionEnv):
    """Two heterogeneous agents, four targets; primitive action Discrete(25).

    Observation: V3 129 features + remaining fuel fraction (2) + visible
    agent-target capability flags (8), total 139. An inactive future target's
    capability flags are hidden until it appears.
    """

    def __init__(self, fuel_budgets=(16, 18), capabilities=DEFAULT_CAPABILITIES, **kwargs):
        super().__init__(**kwargs)
        fuel = np.asarray(fuel_budgets, dtype=np.int64)
        caps = np.asarray(capabilities, dtype=bool)
        if fuel.shape != (2,) or np.any(fuel < 0):
            raise ValueError('fuel_budgets must be two nonnegative integers')
        if caps.shape != (2, N_TARGETS) or not np.all(caps.any(axis=0)):
            raise ValueError('capabilities must be a 2x4 matrix covering each target')
        self.fuel_budgets = fuel.copy()
        self.fuel = fuel.copy()
        self.capabilities = caps.copy()
        self.observation_space = spaces.Box(0., 1., shape=(139,), dtype=np.float32)

    def _observe(self):
        # Parent's observation includes only currently visible target locations
        # and deadlines; the future target capability is also hidden.
        base = super()._observe()
        fuel_frac = self.fuel.astype(np.float32) / np.maximum(1, self.fuel_budgets)
        visible_caps = self.capabilities & self.active[None, :]
        return np.concatenate((base, fuel_frac, visible_caps.astype(np.float32).ravel())).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self.fuel[:] = self.fuel_budgets
        return super().reset(seed=seed, options=options)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f'Action out of bounds: {action}')
        if self._finished:
            raise RuntimeError('Episode finished; call reset()')

        self.steps += 1
        previous = self.agents.copy()
        commands = np.asarray([int(action) // 5, int(action) % 5])
        candidates = previous + MOVES[commands]
        invalid = 0
        for i in range(2):
            r, c = candidates[i]
            if (commands[i] != 0 and self.fuel[i] == 0) or not (0 <= r < SIZE and 0 <= c < SIZE) or (int(r), int(c)) in self.obstacles:
                candidates[i] = previous[i]
                invalid += 1
        collision = bool(np.array_equal(candidates[0], candidates[1]) or
                         (np.array_equal(candidates[0], previous[1]) and np.array_equal(candidates[1], previous[0])))
        if collision:
            candidates = previous
        moved = np.any(candidates != previous, axis=1)
        self.fuel -= moved.astype(np.int64)
        self.agents = candidates

        gained = 0
        for target in range(N_TARGETS):
            if (self.active[target] and not self.completed[target] and not self.expired[target]
                    and self.steps <= self.deadlines[target]
                    and any(self.capabilities[agent, target] and np.array_equal(self.agents[agent], self.targets[target])
                            for agent in range(2))):
                self.completed[target] = True
                gained += int(self.priorities[target])

        target_spawned = False
        if self.changing_targets and self.steps == self.urgent_spawn_step:
            forbidden = self.obstacles | set(map(tuple, self.agents))
            forbidden |= {tuple(self.targets[i]) for i in range(3)}
            choices = [tuple(p) for p in self._free if tuple(p) not in forbidden]
            if not choices:
                raise RuntimeError('No free cell for urgent target')
            self.targets[3] = choices[int(self.np_random.integers(len(choices)))]
            self.priorities[3] = 3
            self.active[3] = True
            target_spawned = True

        newly_expired = self.active & ~self.completed & ~self.expired & (self.steps >= self.deadlines)
        self.expired |= newly_expired

        obstacles_changed = False
        if (self.dynamic_obstacles and self.steps % self.obstacle_period == 0
                and self.np_random.random() < self.obstacle_change_prob):
            obstacles_changed = self._refresh_obstacles()

        all_arrived = not self.changing_targets or self.active[3]
        remaining = self.active & ~self.completed & ~self.expired
        terminated = bool(all_arrived and not remaining.any())
        truncated = bool(self.steps >= self.max_steps and not terminated)
        self._finished = terminated or truncated
        weighted_total = 9 if self.changing_targets else 6
        weighted_completed = int(self.priorities[self.completed].sum())
        reward = 3.0*gained - 0.05 - 0.2*invalid - (0.5 if collision else 0.0)
        info = {
            'success': bool(terminated and weighted_completed == weighted_total),
            'targets_completed': int(self.completed.sum()),
            'weighted_completed': weighted_completed,
            'weighted_total': weighted_total,
            'weighted_fraction': weighted_completed/weighted_total,
            'expired_targets': int(self.expired.sum()),
            'newly_expired': int(newly_expired.sum()),
            'collision': collision,
            'invalid_actions': invalid,
            'obstacles_changed': obstacles_changed,
            'target_spawned': target_spawned,
            'steps': self.steps,
            'fuel_remaining': tuple(map(int, self.fuel)),
            'fuel_spent': int(moved.sum()),
        }
        return self._observe(), float(reward), terminated, truncated, info
