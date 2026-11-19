"""Small, centralized cooperative multi-agent mission-control Gymnasium environment."""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

SIZE = 8
OBSTACLES = {(2, 2), (2, 3), (2, 4), (5, 3), (5, 4), (5, 5)}
MOVES = np.array([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int64)


class MissionGridEnv(gym.Env):
    """One controller selects a joint action for 2 cooperative agents.

    Action = 5*a0 + a1, where each a_i is stay/up/down/left/right.
    Observation = 64 obstacle flags + 4 agent coordinates + 4 target coordinates
    + 2 completion flags, normalized into [0, 1].
    """
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}

    def __init__(self, max_steps: int = 40, render_mode: str | None = None):
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(25)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(74,), dtype=np.float32)
        self.agents = np.zeros((2, 2), dtype=np.int64)
        self.targets = np.zeros((2, 2), dtype=np.int64)
        self.completed = np.zeros(2, dtype=bool)
        self.steps = 0
        self._obstacle_grid = np.zeros((SIZE, SIZE), dtype=np.float32)
        for r, c in OBSTACLES:
            self._obstacle_grid[r, c] = 1.0
        self._free = np.array([(r, c) for r in range(SIZE) for c in range(SIZE) if (r, c) not in OBSTACLES])

    def _observe(self) -> np.ndarray:
        return np.concatenate([
            self._obstacle_grid.flatten(),
            self.agents.flatten().astype(np.float32) / (SIZE - 1),
            self.targets.flatten().astype(np.float32) / (SIZE - 1),
            self.completed.astype(np.float32),
        ]).astype(np.float32)

    def _potential(self, active: np.ndarray) -> int:
        return sum(min(np.abs(self.agents - t).sum(axis=1)) for t in self.targets[active])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        indices = self.np_random.choice(len(self._free), size=4, replace=False)
        self.agents = self._free[indices[:2]].copy()
        self.targets = self._free[indices[2:]].copy()
        self.completed[:] = False
        self.steps = 0
        return self._observe(), {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Action out of bounds: {action}")
        self.steps += 1
        old_agents = self.agents.copy()
        old_active = ~self.completed
        old_potential = self._potential(old_active)
        candidates = self.agents + MOVES[np.array([action // 5, action % 5])]
        invalid = 0
        for i in range(2):
            r, c = candidates[i]
            if not (0 <= r < SIZE and 0 <= c < SIZE) or (int(r), int(c)) in OBSTACLES:
                candidates[i] = old_agents[i]
                invalid += 1
        collision = bool(np.array_equal(candidates[0], candidates[1]) or
                         (np.array_equal(candidates[0], old_agents[1]) and
                          np.array_equal(candidates[1], old_agents[0])))
        if collision:
            candidates = old_agents
        self.agents = candidates
        new_potential = self._potential(old_active)
        newly_completed = 0
        for j in range(2):
            if not self.completed[j] and np.any(np.all(self.agents == self.targets[j], axis=1)):
                self.completed[j] = True
                newly_completed += 1
        success = bool(self.completed.all())
        truncated = self.steps >= self.max_steps and not success
        reward = (-0.05 - 0.2 * invalid - (0.5 if collision else 0.0)
                  + 0.15 * (old_potential - new_potential) + 5.0 * newly_completed
                  + (10.0 if success else 0.0))
        info = {"success": success, "targets_completed": int(self.completed.sum()),
                "collision": collision, "invalid_actions": invalid, "steps": self.steps}
        return self._observe(), float(reward), success, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            colors = np.zeros((SIZE, SIZE, 3), dtype=np.uint8) + 240
            for r, c in OBSTACLES:
                colors[r, c] = (80, 80, 80)
            for j, (r, c) in enumerate(self.targets):
                if not self.completed[j]:
                    colors[r, c] = (30, 170, 80)
            for i, (r, c) in enumerate(self.agents):
                colors[r, c] = [(35, 100, 215), (225, 120, 35)][i]
            return np.repeat(np.repeat(colors, 30, axis=0), 30, axis=1)
        board = np.full((SIZE, SIZE), ".", dtype="<U1")
        for r, c in OBSTACLES:
            board[r, c] = "#"
        for j, (r, c) in enumerate(self.targets):
            if not self.completed[j]:
                board[r, c] = "T"
        for i, (r, c) in enumerate(self.agents):
            board[r, c] = str(i)
        return "\n".join(" ".join(row) for row in board)
