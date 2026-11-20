"""Deadline-slack observation ablation for V3 (no changes to rewards/actions).

Appends eight features ordered by agent (agent 0, then agent 1) and target
(A, B, C, urgent D). For a visible, pending and reachable target, feature is
(deadline - primitive_step - obstacle-aware BFS distance) / max_steps,
clipped to [-1, 1]. An unreachable pending target is encoded as -1.
Inactive/completed/expired targets are encoded as 0 to avoid exposing the
unspawned urgent target. Target activity is already available in base obs.

The distances use current obstacles only, not future obstacle layouts.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .controllers_v3 import HierarchicalDeadlineEnv
from .policies_v2 import _distance_map


class SlackObservationV3(gym.ObservationWrapper):
    """Append agent-specific deadline slack to a hierarchical V3 observation."""

    def __init__(self, env: HierarchicalDeadlineEnv):
        super().__init__(env)
        if not isinstance(env, HierarchicalDeadlineEnv):
            raise TypeError('SlackObservationV3 requires HierarchicalDeadlineEnv')
        base_space = env.observation_space
        if not isinstance(base_space, spaces.Box) or len(base_space.shape) != 1:
            raise ValueError('Expected a flat Box observation space')
        self.observation_space = spaces.Box(
            low=np.concatenate((np.asarray(base_space.low, dtype=np.float32),
                                np.full(8, -1.0, dtype=np.float32))),
            high=np.concatenate((np.asarray(base_space.high, dtype=np.float32),
                                 np.full(8, 1.0, dtype=np.float32))),
            dtype=np.float32,
        )

    def slack_features(self) -> np.ndarray:
        base = self.env.unwrapped.base
        features = np.zeros((2, 4), dtype=np.float32)
        blocked = set(base.obstacles)
        for target in range(4):
            if not base.active[target] or base.completed[target] or base.expired[target]:
                continue
            goal = tuple(int(v) for v in base.targets[target])
            distance_map = _distance_map(goal, blocked)
            for agent in range(2):
                pos = tuple(int(v) for v in base.agents[agent])
                distance = distance_map.get(pos)
                if distance is None:
                    features[agent, target] = -1.0
                else:
                    slack = int(base.deadlines[target]) - int(base.steps) - distance
                    features[agent, target] = np.clip(
                        slack / float(base.max_steps), -1.0, 1.0
                    )
        return features.reshape(8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        augmented = np.concatenate((np.asarray(observation, dtype=np.float32),
                                    self.slack_features())).astype(np.float32)
        return augmented

    def action_masks(self) -> np.ndarray:
        """Forward the dynamic mask for MaskablePPO/SubprocVecEnv compatibility."""
        return self.env.action_masks()
