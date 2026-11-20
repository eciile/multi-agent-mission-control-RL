"""Evaluate hierarchical PPO on the 100 held-out V2 combined-scenario seeds.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from mamc_rl.hierarchical_v2 import HierarchicalMissionEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/hier_v2.zip')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed-start', type=int, default=1000)
    parser.add_argument('--option-horizon', type=int, default=3)
    parser.add_argument('--output', default='results/hier_v2.csv')
    parser.add_argument('--stochastic', action='store_true')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')

    model = PPO.load(args.model, device='cpu')
    records = []
    env = HierarchicalMissionEnv(option_horizon=args.option_horizon)
    try:
        for seed in range(args.seed_start, args.seed_start + args.episodes):
            obs, _ = env.reset(seed=seed)
            total_reward = 0.0
            decisions = 0
            low_steps = 0
            collisions = 0
            invalid = 0
            while True:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(int(action))
                decisions += 1
                low_steps += info['low_level_steps']
                total_reward += reward
                # Shielded BFS produces zero invalids/collisions under V2 rules.
                collisions += int(info['collision'])
                invalid += int(info['invalid_actions'])
                if terminated or truncated:
                    break
            records.append({
                'seed': seed, 'policy': 'hierarchical_ppo',
                'success': int(info['success']), 'steps': low_steps,
                'high_level_decisions': decisions, 'return': total_reward,
                'targets_completed': info['targets_completed'],
                'weighted_completed': info['weighted_completed'],
                'weighted_total': info['weighted_total'],
                'weighted_fraction': info['weighted_completed'] / max(1, info['weighted_total']),
                'collisions': collisions, 'invalid_actions': invalid,
            })
    finally:
        env.close()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    avg = lambda col: float(np.mean([r[col] for r in records]))
    print(f'Saved {path} ({len(records)} episode records)')
    print(f"Success: {avg('success'):.1%}; mean primitive steps: {avg('steps'):.1f}; "
          f"mean return: {avg('return'):.2f}; weighted completion: {avg('weighted_fraction'):.1%}")
    print(f"Mean high-level decisions: {avg('high_level_decisions'):.1f}; "
          f"collisions: {avg('collisions'):.2f}; invalid moves: {avg('invalid_actions'):.2f}")


if __name__ == '__main__':
    main()
