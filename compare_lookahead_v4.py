"""Compare V4 random, greedy heuristic, and short-horizon lookahead on paired seeds."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from mamc_rl.controllers_v4 import (
    HierarchicalResourceEnv, masked_random_action, resource_heuristic_action,
)
from mamc_rl.lookahead_v4 import lookahead_action

def run_episode(policy: str, seed: int, fuel: tuple[int, int], depth: int, width: int) -> dict:
    env = HierarchicalResourceEnv(fuel_budgets=fuel)
    rng = np.random.default_rng(seed + 1_000_000)
    env.reset(seed=seed)
    total_reward = 0.0
    total_primitive = 0
    decisions = 0
    spawned = False
    planning_s = 0.0
    try:
        while True:
            start = time.perf_counter()
            if policy == 'masked_random':
                action = masked_random_action(env, rng)
            elif policy == 'resource_heuristic':
                action = resource_heuristic_action(env)
            else:
                action = lookahead_action(env, depth=depth, beam_width=width)
            planning_s += time.perf_counter() - start
            _, reward, done, truncated, info = env.step(action)
            total_reward += reward
            total_primitive += int(info['low_level_steps'])
            decisions += 1
            spawned |= bool(info['target_spawned'])
            if done or truncated:
                return {
                    'policy': policy, 'seed': seed,
                    'success': int(info['success']),
                    'weighted_fraction': float(info['weighted_fraction']),
                    'expired_targets': int(info['expired_targets']),
                    'primitive_steps': total_primitive,
                    'high_level_decisions': decisions,
                    'return_value': total_reward,
                    'urgent_spawned': int(spawned),
                    'planning_seconds': planning_s,
                    'planning_ms_per_decision': 1000.0 * planning_s / decisions,
                }
    finally:
        env.close()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--seed-start', type=int, default=2000)
    parser.add_argument('--fuel', type=int, nargs=2, default=(16, 18))
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--beam-width', type=int, default=4)
    parser.add_argument('--output', default='results/v4_lookahead_comparison.csv')
    args = parser.parse_args()
    if args.episodes < 1 or args.depth < 1 or args.beam_width < 1:
        parser.error('episodes, depth and beam-width must be positive')

    policies = ('masked_random', 'resource_heuristic', 'lookahead')
    rows = [run_episode(policy, seed, tuple(args.fuel), args.depth, args.beam_width)
            for seed in range(args.seed_start, args.seed_start + args.episodes)
            for policy in policies]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {output} ({len(rows)} episode records)')
    print('Policy                Success  Weighted  Expired   Steps   Planning ms/decision   Spawned')
    for policy in policies:
        group = [row for row in rows if row['policy'] == policy]
        avg = lambda k: float(np.mean([row[k] for row in group]))
        print(f'{policy:<22}{avg("success"):>6.1%}    {avg("weighted_fraction"):>6.1%}'
              f'     {avg("expired_targets"):>4.2f}    {avg("primitive_steps"):>4.1f}'
              f'     {avg("planning_ms_per_decision"):>9.2f}             {avg("urgent_spawned"):>5.1%}')
    print('Timing measures controller computation only on this machine; not a hardware-independent benchmark.')
    print('Planner forecasts suppress future stochastic obstacle changes and hidden target spawning.')


if __name__ == '__main__':
    main()
