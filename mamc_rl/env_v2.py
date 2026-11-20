"""Version 2: dynamic cooperative missions with changing objectives and obstacles.

Independent of the original MissionGridEnv; this preserves baseline comparability.
Centralized action space: Discrete(25), encoding 5*a0 + a1.
"""
from __future__ import annotations

from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

SIZE = 10
N_AGENTS = 2
N_TARGETS = 4  # Three at reset; one may appear later.
FIXED_OBSTACLES = {(2, 2), (2, 3), (2, 4), (6, 4), (6, 5), (6, 6)}
MOVES = np.array([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int64)


class DynamicMissionEnv(gym.Env):
    """Cooperative mission with full observation and dynamic changes.

    Observation, values normalized into [0, 1]:
      100 obstacle flags (fixed + temporary), 4 agent coordinates,
      8 target coordinates, 4 target priorities / 3,
      4 target-active flags, 4 completion flags, 1 elapsed-time fraction.
    Total: 125 features. Inactive target positions are zeros, not future hints.

    Temporary obstacles are updated every `obstacle_period` steps with
    `obstacle_change_prob`. The next map remains traversable for the agents
    and active unfinished targets; unreachable maps are rejected.
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        max_steps: int = 60,
        render_mode: str | None = None,
        dynamic_obstacles: bool = True,
        changing_targets: bool = True,
        obstacle_period: int = 5,
        obstacle_change_prob: float = 0.7,
        urgent_spawn_step: int = 12,
        temporary_obstacle_count: int = 3,
    ):
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Invalid render mode: {render_mode}")
        if max_steps < 1 or obstacle_period < 1 or urgent_spawn_step < 1:
            raise ValueError("Steps and period must be positive")
        if not 0 <= obstacle_change_prob <= 1 or temporary_obstacle_count < 0:
            raise ValueError("Invalid obstacle configuration")
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.dynamic_obstacles = dynamic_obstacles
        self.changing_targets = changing_targets
        self.obstacle_period = obstacle_period
        self.obstacle_change_prob = obstacle_change_prob
        self.urgent_spawn_step = urgent_spawn_step
        self.temporary_obstacle_count = temporary_obstacle_count
        self.action_space = spaces.Discrete(25)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(125,), dtype=np.float32)
        self.agents = np.zeros((2, 2), dtype=np.int64)
        self.targets = np.zeros((4, 2), dtype=np.int64)
        self.priorities = np.zeros(4, dtype=np.int64)
        self.active = np.zeros(4, dtype=bool)
        self.completed = np.zeros(4, dtype=bool)
        self.temporary_obstacles: set[tuple[int, int]] = set()
        self.steps = 0
        self._free = np.array([(r, c) for r in range(SIZE) for c in range(SIZE)
                               if (r, c) not in FIXED_OBSTACLES], dtype=np.int64)

    @property
    def obstacles(self) -> set[tuple[int, int]]:
        return FIXED_OBSTACLES | self.temporary_obstacles

    def _observe(self) -> np.ndarray:
        grid = np.zeros((SIZE, SIZE), dtype=np.float32)
        for r, c in self.obstacles:
            grid[r, c] = 1.0
        # All target slots are fixed-size. Inactive slots never expose future positions.
        target_coords = np.where(self.active[:, None], self.targets, 0)
        return np.concatenate((
            grid.ravel(),
            self.agents.ravel().astype(np.float32) / (SIZE - 1),
            target_coords.ravel().astype(np.float32) / (SIZE - 1),
            self.priorities.astype(np.float32) / 3.0,
            self.active.astype(np.float32),
            self.completed.astype(np.float32),
            np.array([self.steps / self.max_steps], dtype=np.float32),
        )).astype(np.float32)

    def _reachable(self, obstacles: set[tuple[int, int]]) -> bool:
        """Check reachability of each unfinished target from at least one agent."""
        visited = {tuple(self.agents[0]), tuple(self.agents[1])}
        queue = deque(visited)
        while queue:
            r, c = queue.popleft()
            for dr, dc in MOVES[1:]:
                n = (r + int(dr), c + int(dc))
                if 0 <= n[0] < SIZE and 0 <= n[1] < SIZE and n not in obstacles and n not in visited:
                    visited.add(n)
                    queue.append(n)
        return all(tuple(self.targets[i]) in visited for i in range(N_TARGETS)
                   if self.active[i] and not self.completed[i])

    def _refresh_obstacles(self) -> bool:
        forbidden = set(map(tuple, self.agents)) | FIXED_OBSTACLES
        forbidden |= {tuple(self.targets[i]) for i in range(N_TARGETS) if self.active[i]}
        choices = [tuple(p) for p in self._free if tuple(p) not in forbidden]
        count = min(self.temporary_obstacle_count, len(choices))
        for _ in range(25):
            if count == 0:
                new = set()
            else:
                ids = self.np_random.choice(len(choices), size=count, replace=False)
                new = {choices[int(i)] for i in ids}
            if self._reachable(FIXED_OBSTACLES | new):
                changed = new != self.temporary_obstacles
                self.temporary_obstacles = new
                return changed
        # Keep old safe map if generation attempts fail.
        return False

    def _potential(self) -> int:
        return int(sum(int(self.priorities[i]) * min(np.abs(self.agents - self.targets[i]).sum(axis=1))
                       for i in range(N_TARGETS) if self.active[i] and not self.completed[i]))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        ids = self.np_random.choice(len(self._free), size=5, replace=False)
        self.agents = self._free[ids[:2]].copy()
        self.targets[:] = 0
        self.targets[:3] = self._free[ids[2:]].copy()
        self.priorities[:] = [1, 2, 3, 0]
        self.active[:] = [True, True, True, False]
        self.completed[:] = False
        self.temporary_obstacles = set()
        self.steps = 0
        return self._observe(), {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Action out of bounds: {action}")
        if self.steps >= self.max_steps or (self.active.any() and
           self.completed[self.active].all() and (not self.changing_targets or self.active[3])):
            raise RuntimeError("Episode finished; call reset() before step()")
        self.steps += 1
        previous = self.agents.copy()
        old_potential = self._potential()
        pair = np.array([int(action) // 5, int(action) % 5])
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
        progress = old_potential - self._potential()
        gained_value = 0
        for i in range(N_TARGETS):
            if self.active[i] and not self.completed[i] and np.any(np.all(self.agents == self.targets[i], axis=1)):
                self.completed[i] = True
                gained_value += int(self.priorities[i])

        target_spawned = False
        if self.changing_targets and self.steps == self.urgent_spawn_step:
            forbidden = self.obstacles | set(map(tuple, self.agents))
            forbidden |= {tuple(self.targets[i]) for i in range(3)}
            options = [tuple(p) for p in self._free if tuple(p) not in forbidden]
            if options:
                pos = options[int(self.np_random.integers(len(options)))]
                self.targets[3] = pos
                self.priorities[3] = 3
                self.active[3] = True
                target_spawned = True
            else:
                # Extremely unlikely with the default configuration.
                raise RuntimeError("No free cell for urgent target")

        obstacles_changed = False
        if (self.dynamic_obstacles and self.steps % self.obstacle_period == 0
                and self.np_random.random() < self.obstacle_change_prob):
            obstacles_changed = self._refresh_obstacles()

        ready_to_finish = not self.changing_targets or self.active[3]
        success = bool(ready_to_finish and self.completed[self.active].all())
        truncated = bool(self.steps >= self.max_steps and not success)
        reward = (-0.05 - 0.2 * invalid - (0.5 if collision else 0.0)
                  + 0.1 * progress + 3.0 * gained_value + (10.0 if success else 0.0))
        info = {
            "success": success,
            "targets_completed": int(self.completed.sum()),
            "weighted_completed": int(self.priorities[self.completed].sum()),
            "weighted_total": int(self.priorities[self.active].sum()),
            "collision": collision,
            "invalid_actions": invalid,
            "obstacles_changed": obstacles_changed,
            "target_spawned": target_spawned,
            "steps": self.steps,
        }
        return self._observe(), float(reward), success, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            colors = np.full((SIZE, SIZE, 3), 240, dtype=np.uint8)
            for r, c in FIXED_OBSTACLES:
                colors[r, c] = (70, 70, 70)
            for r, c in self.temporary_obstacles:
                colors[r, c] = (180, 80, 80)
            for i in range(N_TARGETS):
                if self.active[i] and not self.completed[i]:
                    r, c = self.targets[i]
                    colors[r, c] = (40, 160, 75)
            for i, (r, c) in enumerate(self.agents):
                colors[r, c] = [(35, 100, 215), (225, 120, 35)][i]
            return np.repeat(np.repeat(colors, 24, axis=0), 24, axis=1)
        board = np.full((SIZE, SIZE), ".", dtype="<U1")
        for r, c in FIXED_OBSTACLES:
            board[r, c] = "#"
        for r, c in self.temporary_obstacles:
            board[r, c] = "X"
        for i in range(N_TARGETS):
            if self.active[i] and not self.completed[i]:
                r, c = self.targets[i]
                board[r, c] = str(self.priorities[i])
        for i, (r, c) in enumerate(self.agents):
            board[r, c] = "AB"[i]
        return "\n".join(" ".join(row) for row in board)
