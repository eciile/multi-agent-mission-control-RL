"""Evaluate a saved centralized PPO policy on MAMC-RL V2.

Examples:
    python evaluate_ppo_v2.py --model models/ppo_v2_combined.zip --episodes 100
    python evaluate_ppo_v2.py --model models/ppo_v2_combined.zip --episodes 100 --all-scenarios

Uses the same scenario definitions and metric schema as evaluate_v2.py.
Deterministic policy inference is the default; pass --stochastic to sample actions.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from mamc_rl.env_v2 import DynamicMissionEnv
from evaluate_v2 import SCENARIOS, FIELDS


def run_episode(model: PPO, scenario: str, seed: int, deterministic: bool = True) -> dict:
    env = DynamicMissionEnv(**SCENARIOS[scenario])
    try:
        obs, _ = env.reset(seed=seed)
        # Explicitly seed the action space for consistency if stochastic sampling is used.
        env.action_space.seed(seed)
        total_return = 0.0
        collisions = invalid_actions = obstacle_changes = target_spawns = 0
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
            total_return += float(reward)
            collisions += int(info['collision'])
            invalid_actions += int(info['invalid_actions'])
            obstacle_changes += int(info['obstacles_changed'])
            target_spawns += int(info['target_spawned'])
            if terminated or truncated:
                weighted_total = int(info['weighted_total'])
                return {
                    'scenario': scenario,
                    'policy': 'ppo',
                    'seed': seed,
                    'success': int(info['success']),
                    'steps': int(info['steps']),
                    'return': round(total_return, 4),
                    'targets_completed': int(info['targets_completed']),
                    'targets_total': int(env.active.sum()),
                    'weighted_completed': int(info['weighted_completed']),
                    'weighted_total': weighted_total,
                    'weighted_fraction': round(int(info['weighted_completed']) / weighted_total, 6)
                        if weighted_total else 0.0,
                    'collisions': collisions,
                    'invalid_actions': invalid_actions,
                    'obstacle_changes': obstacle_changes,
                    'target_spawn_events': target_spawns,
                }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/ppo_v2_combined.zip')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed-start', type=int, default=1000)
    parser.add_argument('--all-scenarios', action='store_true')
    parser.add_argument('--stochastic', action='store_true', help='Sample from policy instead of taking the most likely action')
    parser.add_argument('--output', default='results/ppo_v2.csv')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    if not Path(args.model).is_file():
        parser.error(f'Model file not found: {args.model}')

    model = PPO.load(args.model, device='cpu')
    scenarios = tuple(SCENARIOS) if args.all_scenarios else ('combined',)
    rows = [run_episode(model, scenario, seed, deterministic=not args.stochastic)
            for scenario in scenarios
            for seed in range(args.seed_start, args.seed_start + args.episodes)]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Saved {out} ({len(rows)} episode records)')
    print(f'Model: {args.model} | Inference: {"stochastic" if args.stochastic else "deterministic"}')
    print(f'{"Scenario":16s} {"Policy":10s} {"Success":>9s} {"Steps":>8s} {"Return":>9s} {"Weighted":>9s} {"Coll.":>7s} {"Invalid":>8s}')
    print('-' * 85)
    for scenario in scenarios:
        subset = [r for r in rows if r['scenario'] == scenario]
        mean = lambda metric: float(np.mean([r[metric] for r in subset]))
        print(f'{scenario:16s} {"ppo":10s} {mean("success"):8.1%} '
              f'{mean("steps"):8.1f} {mean("return"):9.2f} '
              f'{mean("weighted_fraction"):8.1%} {mean("collisions"):7.2f} '
              f'{mean("invalid_actions"):8.2f}')


if __name__ == '__main__':
    main()
