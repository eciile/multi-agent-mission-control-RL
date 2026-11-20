"""Calibrate V3 before RL. No learned policy is used.

Run: python calibrate_v3.py --episodes 200 --seed-start 2000
"""
import argparse
import csv
from pathlib import Path
import numpy as np

from mamc_rl.controllers_v3 import HierarchicalDeadlineEnv, masked_random_action, deadline_heuristic_action


def episode(policy, seed, deadlines):
    env = HierarchicalDeadlineEnv(deadlines=deadlines)
    rng = np.random.default_rng(seed + 1_000_000)
    _, _ = env.reset(seed=seed)
    ret, steps, decisions, spawn = 0., 0, 0, False
    try:
        while True:
            if policy == 'masked_random':
                action = masked_random_action(env, rng)
            else:
                action = deadline_heuristic_action(env)
            _, rew, terminated, truncated, info = env.step(action)
            ret += rew
            steps += info['low_level_steps']
            decisions += 1
            spawn |= info['target_spawned']
            if terminated or truncated:
                return dict(policy=policy, seed=seed, success=int(info['success']),
                            weighted_fraction=info['weighted_fraction'],
                            weighted_completed=info['weighted_completed'],
                            expired_targets=info['expired_targets'],
                            primitive_steps=steps, decisions=decisions,
                            return_value=ret, urgent_spawned=int(spawn))
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--seed-start', type=int, default=2000)
    p.add_argument('--deadlines', type=int, nargs=4, default=[12, 18, 24, 23])
    p.add_argument('--output', default='results/v3_calibration.csv')
    args = p.parse_args()
    if args.episodes < 1: p.error('episodes must be >= 1')
    rows = [episode(name, seed, tuple(args.deadlines))
            for seed in range(args.seed_start, args.seed_start + args.episodes)
            for name in ('masked_random', 'deadline_heuristic')]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f'Saved {path} ({len(rows)} episode records)')
    print('Policy                Success   Weighted   Expired   Steps   Return   Spawned')
    for policy in ('masked_random', 'deadline_heuristic'):
        subset = [r for r in rows if r['policy'] == policy]
        avg = lambda key: np.mean([r[key] for r in subset])
        print(f'{policy:<21} {avg("success"):>6.1%}   {avg("weighted_fraction"):>7.1%}'
              f'   {avg("expired_targets"):>7.2f}  {avg("primitive_steps"):>6.1f}'
              f'   {avg("return_value"):>6.2f}    {avg("urgent_spawned"):>6.1%}')

if __name__ == '__main__':
    main()
