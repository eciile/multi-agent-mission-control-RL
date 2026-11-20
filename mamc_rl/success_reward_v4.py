"""Training-only terminal success bonus for the V4 reward-objective ablation.

The bonus is paid once, only when all objectives are completed. Evaluation must
use the original HierarchicalResourceEnv, not this reward wrapper.
"""
from __future__ import annotations

import gymnasium as gym

from .controllers_v4 import HierarchicalResourceEnv


class SuccessBonusV4(gym.Wrapper):
    def __init__(self, env: HierarchicalResourceEnv, bonus: float = 9.0):
        if bonus < 0:
            raise ValueError('bonus must be nonnegative')
        super().__init__(env)
        self.bonus = float(bonus)

    def action_masks(self):
        """Expose the same mask that MaskablePPO uses on standard V4."""
        return self.env.action_masks()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        addition = self.bonus if (terminated or truncated) and info['success'] else 0.0
        info['success_bonus'] = addition
        return obs, float(reward) + addition, terminated, truncated, info
