"""Hierarchical mission control with state-dependent valid assignment masks.

Idle is always valid, as are active unfinished target assignments.
Duplicate target assignments remain valid intentionally.
"""
from __future__ import annotations

import numpy as np

from .hierarchical_v2 import HierarchicalMissionEnv


class MaskedHierarchicalMissionEnv(HierarchicalMissionEnv):
    def action_masks(self) -> np.ndarray:
        valid_choices = [0] + [
            i + 1 for i in range(4)
            if bool(self.base.active[i]) and not bool(self.base.completed[i])
        ]
        mask = np.zeros(self.action_space.n, dtype=bool)
        for a0 in valid_choices:
            for a1 in valid_choices:
                mask[5 * a0 + a1] = True
        return mask
