"""Evaluate a trained V4 MaskablePPO on held-out resource-constrained missions.

Example:
  python evaluate_masked_v4.py --model models/v4_masked_seed42.zip --episodes 200 --seed-start 2000

Uses deterministic masked actions and the same default deadlines and fuel as calibrate_v4.py.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from mamc_rl.controllers_v4 import HierarchicalResourceEnv


def evaluate(model, episodes, seed_start, option_horizon, deadlines, fuel):
    env = HierarchicalResourceEnv(
        option_horizon=option_horizon, deadlines=tuple(deadlines),
        fuel_budgets=tuple(fuel),
    )
    rows = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            observation, _ = env.reset(seed=seed)
            total_reward = 0.0
            primitive_steps = 0
            decisions = 0
            invalid_assignments = 0
            urgent_spawned = False
            while True:
                mask = env.action_masks()
                action, _ = model.predict(
                    observation, deterministic=True, action_masks=mask,
                )
                action = int(action)
                if not mask[action]:
                    invalid_assignments += 1
                observation, reward, terminated, truncated, info = env.step(action)
                decisions += 1
                primitive_steps += int(info['low_level_steps'])
                total_reward += float(reward)
                urgent_spawned |= bool(info['target_spawned'])
                if terminated or truncated:
                    break
            rows.append({
                'fuel_agent0_remaining': info['fuel_remaining'][0],
                'fuel_agent1_remaining': info['fuel_remaining'][1],
                'policy': 'masked_ppo',
                'seed': seed,
                'success': int(info['success']),
                'weighted_fraction': float(info['weighted_fraction']),
                'weighted_completed': int(info['weighted_completed']),
                'expired_targets': int(info['expired_targets']),
                'primitive_steps': primitive_steps,
                'decisions': decisions,
                'return_value': total_reward,
                'urgent_spawned': int(urgent_spawned),
                'invalid_assignments': invalid_assignments,
            })
    finally:
        env.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed-start', type=int, default=2000)
    parser.add_argument('--option-horizon', type=int, default=3)
    parser.add_argument('--deadlines', type=int, nargs=4, default=(12, 18, 24, 23))
    parser.add_argument('--fuel', type=int, nargs=2, default=(16, 18))
    parser.add_argument('--output', default='results/v4_masked_ppo.csv')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    if args.option_horizon < 1:
        parser.error('--option-horizon must be positive')

    model = MaskablePPO.load(args.model, device='cpu')
    rows = evaluate(model, args.episodes, args.seed_start, args.option_horizon, args.deadlines, args.fuel)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    avg = lambda key: float(np.mean([row[key] for row in rows]))
    print(f'Saved {path} ({len(rows)} episode records)')
    print(f'Success: {avg("success"):.1%}; weighted completion: {avg("weighted_fraction"):.1%}; '
          f'expired objectives: {avg("expired_targets"):.2f}; mean return: {avg("return_value"):.2f}; '
          f'primitive steps: {avg("primitive_steps"):.1f}; '
          f'high-level decisions: {avg("decisions"):.1f}; '
          f'urgent target spawned: {avg("urgent_spawned"):.1%}; '
          f'invalid assignments: {avg("invalid_assignments"):.2f}; '
          f'fuel remaining: ({avg("fuel_agent0_remaining"):.1f}, {avg("fuel_agent1_remaining"):.1f})')


if __name__ == '__main__':
    main()
