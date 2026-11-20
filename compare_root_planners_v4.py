"""Compare global-beam and root-preserving V4 planners on DEVELOPMENT missions.

Use seed range outside the held-out final test set (2000..2199).
Both planners use depth 4, beam width 4, option horizon 3, and identical
initial mission seeds. Root-preserving scores are heuristic, not true Q-values.

Examples:
  python compare_root_planners_v4.py --episodes 3 --seed-start 16000
  python compare_root_planners_v4.py --episodes 30 --seed-start 16000
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from audit_root_scores_v4 import root_preserving_scores
from mamc_rl.controllers_v4 import HierarchicalResourceEnv, resource_heuristic_action
from mamc_rl.lookahead_v4 import lookahead_action


def pick_root_action(env: HierarchicalResourceEnv, depth: int, beam_width: int) -> int:
    scores = root_preserving_scores(env, depth=depth, beam_width=beam_width)
    admissible = np.flatnonzero(np.isfinite(scores))
    if len(admissible) == 0:
        raise RuntimeError('No planner-admissible root actions')
    return int(admissible[np.argmax(scores[admissible])])


def run_episode(policy: str, seed: int, depth: int, beam_width: int) -> dict:
    env = HierarchicalResourceEnv(option_horizon=3)
    env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    decisions = 0
    planning_seconds = 0.0
    spawned = False
    try:
        while True:
            start = perf_counter()
            if policy == 'resource_heuristic':
                action = resource_heuristic_action(env)
            elif policy == 'global_beam':
                action = lookahead_action(env, depth=depth, beam_width=beam_width)
            elif policy == 'root_preserving':
                action = pick_root_action(env, depth=depth, beam_width=beam_width)
            else:
                raise ValueError(f'Unknown policy: {policy}')
            planning_seconds += perf_counter() - start
            if not env.action_masks()[action]:
                raise AssertionError(f'Illegal action {action} on seed {seed}')
            _, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += int(info['low_level_steps'])
            decisions += 1
            spawned |= bool(info['target_spawned'])
            if done or truncated:
                return {
                    'seed': seed, 'policy': policy,
                    'success': int(info['success']),
                    'weighted_fraction': float(info['weighted_fraction']),
                    'expired_targets': int(info['expired_targets']),
                    'return_value': total_reward,
                    'primitive_steps': steps,
                    'high_level_decisions': decisions,
                    'planning_seconds': planning_seconds,
                    'planning_ms_per_decision': 1000.0 * planning_seconds / decisions,
                    'urgent_spawned': int(spawned),
                }
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--seed-start', type=int, default=16000)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--beam-width', type=int, default=4)
    p.add_argument('--output', default='results/v4_root_planner_dev.csv')
    args = p.parse_args()
    if min(args.episodes, args.depth, args.beam_width) < 1:
        p.error('Episodes, depth, and beam width must be positive')
    test_seeds = set(range(2000, 2200))
    if test_seeds.intersection(range(args.seed_start, args.seed_start + args.episodes)):
        p.error('Development comparison must not use final test seeds 2000-2199')
    policies = ('resource_heuristic', 'global_beam', 'root_preserving')
    rows = [run_episode(policy, seed, args.depth, args.beam_width)
            for seed in range(args.seed_start, args.seed_start + args.episodes)
            for policy in policies]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {path} ({len(rows)} episode records)')
    print(f'Development seeds {args.seed_start}..{args.seed_start + args.episodes - 1}; '
          f'depth={args.depth}, beam_width={args.beam_width}')
    print('Policy                 Success  Weighted  Expired  Steps  Planning ms/decision')
    for policy in policies:
        group = [row for row in rows if row['policy'] == policy]
        mean = lambda key: float(np.mean([r[key] for r in group]))
        decision_count = sum(r['high_level_decisions'] for r in group)
        latency = 1000.0 * sum(r['planning_seconds'] for r in group) / decision_count
        print(f'{policy:<23}{mean("success"):>6.1%}    {mean("weighted_fraction"):>6.1%}'
              f'      {mean("expired_targets"):>4.2f}  {mean("primitive_steps"):>5.1f}'
              f'            {latency:>8.2f}')
    print('Times are measured on this hardware and exclude environment execution.')
    print('Both searches use approximate non-oracle forecasts; scores are not expected returns.')
    print('These DEVELOPMENT missions must not be used as a final unbiased test.')


if __name__ == '__main__':
    main()
