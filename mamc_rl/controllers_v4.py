"""V4 high-level controller and non-learning assignment baselines."""
from __future__ import annotations

from itertools import product
import numpy as np

from .controllers_v3 import HierarchicalDeadlineEnv
from .env_v4 import ResourceMissionEnv
from .env_v2 import MOVES
from .policies_v2 import _distance_map, _joint_is_legal


class HierarchicalResourceEnv(HierarchicalDeadlineEnv):
    """Same 25 joint target-assignment actions, using V4 base simulator."""

    def __init__(self, option_horizon=3, **kwargs):
        # V3 wrapper is initialized with a temporary V3 base; replace with V4.
        # This retains the V3 macro-action/interrupt semantics unchanged.
        super().__init__(option_horizon=option_horizon, **{k:v for k,v in kwargs.items()
                                                           if k not in ('fuel_budgets','capabilities')})
        self.base.close()
        self.base = ResourceMissionEnv(**kwargs)
        self.observation_space = self.base.observation_space

    def action_masks(self):
        base = self.base
        choices = []
        for agent in range(2):
            options = [0]
            for target in range(4):
                if not (base.active[target] and not base.completed[target] and not base.expired[target]
                        and base.capabilities[agent, target]):
                    continue
                if base.fuel[agent] > 0 or np.array_equal(base.agents[agent], base.targets[target]):
                    options.append(target + 1)
            choices.append(options)
        mask = np.zeros(25, dtype=bool)
        for a, b in product(*choices):
            mask[5*a+b] = True
        return mask

    def _move_for_goals(self, goals):
        base = self.base
        blocked = set(base.obstacles)
        positions = [tuple(map(int, row)) for row in base.agents]
        maps = {}
        valid = []
        for agent, goal in enumerate(goals):
            if (goal is None or not base.active[goal] or base.completed[goal]
                    or base.expired[goal] or not base.capabilities[agent, goal]):
                valid.append(None)
            else:
                valid.append(goal)
                if goal not in maps:
                    maps[goal] = _distance_map(tuple(map(int, base.targets[goal])), blocked)
        best, best_score = 0, -float('inf')
        for a, b in product(range(5), repeat=2):
            commands = (a, b)
            if any(commands[i] != 0 and base.fuel[i] == 0 for i in range(2)):
                continue
            candidate = [(positions[i][0]+int(MOVES[commands[i]][0]),
                          positions[i][1]+int(MOVES[commands[i]][1])) for i in range(2)]
            if not _joint_is_legal(positions, candidate, blocked):
                continue
            score = 0.
            for agent, goal in enumerate(valid):
                if goal is None:
                    score -= 0.01 if commands[agent] else 0.0
                    continue
                old = maps[goal].get(positions[agent]); new = maps[goal].get(candidate[agent])
                if old is None or new is None:
                    score -= 100.
                else:
                    score += 2*(old-new)
                    if new == 0: score += 0.3*int(base.priorities[goal])
                    if commands[agent] == 0: score -= 0.02
            if score > best_score:
                best, best_score = 5*a+b, score
        return best


def masked_random_action(env, rng):
    return int(rng.choice(np.flatnonzero(env.action_masks())))


def resource_heuristic_action(env):
    """Greedy, deadline- and resource-aware baseline; not an optimal solver."""
    base = env.base
    positions = [tuple(map(int, p)) for p in base.agents]
    valid_targets = [t for t in range(4) if base.active[t] and not base.completed[t] and not base.expired[t]]
    maps = {t:_distance_map(tuple(map(int,base.targets[t])),set(base.obstacles)) for t in valid_targets}
    candidate_choices = []
    for agent in range(2):
        candidate_choices.append([None]+[t for t in valid_targets if base.capabilities[agent,t]])
    best_score, best_action = -float('inf'), 0
    for a,b in product(*candidate_choices):
        if a is not None and a == b: continue
        action = 5*(0 if a is None else a+1)+(0 if b is None else b+1)
        if not env.action_masks()[action]: continue
        score = 0.
        for agent, target in enumerate((a,b)):
            if target is None: continue
            d = maps[target].get(positions[agent], float('inf'))
            slack = float(base.deadlines[target] - base.steps - d)
            fuel_slack = float(base.fuel[agent] - d)
            if slack < 0 or fuel_slack < 0:
                score -= 3.
            else:
                score += (3*int(base.priorities[target])/(1+0.2*d)
                          +1.5/(1+slack) + 0.2/(1+fuel_slack))
        if score > best_score:
            best_score,best_action = score,action
    return int(best_action)
