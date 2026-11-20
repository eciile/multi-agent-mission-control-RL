"""V3 deadline-constrained mission benchmark; V2 remains unchanged.

Targets earn priority value only if reached on/before their absolute deadline.
Expired targets remain in the mission accounting but are never valid assignments.
The future urgent target's location/deadline is hidden until spawn.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .env_v2 import DynamicMissionEnv, MOVES, N_TARGETS, SIZE


class DeadlineMissionEnv(DynamicMissionEnv):
    def __init__(self, deadlines=(12, 18, 24, 23), **kwargs):
        kwargs.setdefault('max_steps', 40)
        kwargs.setdefault('urgent_spawn_step', 12)
        super().__init__(**kwargs)
        if len(deadlines) != N_TARGETS:
            raise ValueError('Expected one deadline per target')
        self.deadlines = np.asarray(deadlines, dtype=np.int64)
        if np.any(self.deadlines < 1) or np.any(self.deadlines > self.max_steps):
            raise ValueError('Deadlines must be within the episode horizon')
        if self.changing_targets and self.deadlines[3] <= self.urgent_spawn_step:
            raise ValueError('Urgent target deadline must be after it spawns')
        self.expired = np.zeros(N_TARGETS, dtype=bool)
        self.observation_space = spaces.Box(0., 1., shape=(129,), dtype=np.float32)
        self._finished = False

    def _observe(self):
        base = super()._observe()
        visible = self.active & ~self.completed & ~self.expired
        remaining = np.where(visible, np.maximum(self.deadlines - self.steps, 0), 0)
        return np.concatenate((base, remaining.astype(np.float32) / self.max_steps))

    def reset(self, *, seed=None, options=None):
        # Parent calls self._observe(), so all V3 state must be initialized already.
        self.expired[:] = False
        self._finished = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f'Action out of bounds: {action}')
        if self._finished:
            raise RuntimeError('Episode finished; call reset()')
        self.steps += 1
        previous = self.agents.copy()
        pair = np.asarray([int(action) // 5, int(action) % 5])
        candidates = previous + MOVES[pair]
        invalid = 0
        for i in range(2):
            r, c = candidates[i]
            if not (0 <= r < SIZE and 0 <= c < SIZE) or (int(r), int(c)) in self.obstacles:
                candidates[i] = previous[i]
                invalid += 1
        collision = bool(np.array_equal(candidates[0], candidates[1]) or
                         (np.array_equal(candidates[0], previous[1]) and
                          np.array_equal(candidates[1], previous[0])))
        if collision:
            candidates = previous
        self.agents = candidates

        gained = 0
        for i in range(N_TARGETS):
            if (self.active[i] and not self.completed[i] and not self.expired[i]
                    and self.steps <= self.deadlines[i]
                    and np.any(np.all(self.agents == self.targets[i], axis=1))):
                self.completed[i] = True
                gained += int(self.priorities[i])

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
        # At t=deadline, reaching the target already counted; otherwise it expires now.
        self.expired |= newly_expired

        obstacles_changed = False
        if (self.dynamic_obstacles and self.steps % self.obstacle_period == 0
                and self.np_random.random() < self.obstacle_change_prob):
            obstacles_changed = self._refresh_obstacles()

        all_arrived = not self.changing_targets or self.active[3]
        remaining = self.active & ~self.completed & ~self.expired
        all_resolved = bool(all_arrived and not remaining.any())
        terminated = all_resolved
        truncated = bool(self.steps >= self.max_steps and not terminated)
        self._finished = terminated or truncated
        weighted_total = 9 if self.changing_targets else 6  # 1+2+3 (+3 urgent)
        weighted_completed = int(self.priorities[self.completed].sum())
        success = bool(terminated and weighted_completed == weighted_total)
        reward = 3.0 * gained - 0.05 - 0.2 * invalid - (0.5 if collision else 0.0)
        info = {
            'success': success,
            'targets_completed': int(self.completed.sum()),
            'weighted_completed': weighted_completed,
            'weighted_total': weighted_total,
            'weighted_fraction': weighted_completed / weighted_total,
            'expired_targets': int(self.expired.sum()),
            'newly_expired': int(newly_expired.sum()),
            'collision': collision,
            'invalid_actions': invalid,
            'obstacles_changed': obstacles_changed,
            'target_spawned': target_spawned,
            'steps': self.steps,
        }
        return self._observe(), float(reward), terminated, truncated, info
