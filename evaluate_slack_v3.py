"""Evaluate a slack-observation V3 MaskablePPO on fixed initial seeds.

Example:
  python evaluate_slack_v3.py --model models/v3_slack_seed42_50k.zip --episodes 200 --seed-start 2000
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from mamc_rl.controllers_v3 import HierarchicalDeadlineEnv
from mamc_rl.slack_observation_v3 import SlackObservationV3


def evaluate(model, episodes, seed_start, option_horizon, deadlines):
    env = SlackObservationV3(HierarchicalDeadlineEnv(
        option_horizon=option_horizon, deadlines=tuple(deadlines)
    ))
    rows = []
    try:
        if tuple(model.observation_space.shape) != tuple(env.observation_space.shape):
            raise ValueError(f'Model expects {model.observation_space.shape}; '
                             f'slack environment provides {env.observation_space.shape}')
        for seed in range(seed_start, seed_start + episodes):
            obs, _ = env.reset(seed=seed)
            rewards = 0.0
            steps = 0
            decisions = 0
            invalid = 0
            urgent_spawned = False
            while True:
                masks = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=masks)
                action = int(action)
                invalid += int(not masks[action])
                obs, reward, terminated, truncated, info = env.step(action)
                rewards += float(reward)
                steps += int(info['low_level_steps'])
                decisions += 1
                urgent_spawned |= bool(info['target_spawned'])
                if terminated or truncated:
                    break
            rows.append({
                'policy': 'slack_masked_ppo', 'seed': seed,
                'success': int(info['success']),
                'weighted_fraction': float(info['weighted_fraction']),
                'weighted_completed': int(info['weighted_completed']),
                'expired_targets': int(info['expired_targets']),
                'target_A_expired': int(env.unwrapped.base.expired[0]),
                'primitive_steps': steps, 'decisions': decisions,
                'return_value': rewards, 'urgent_spawned': int(urgent_spawned),
                'invalid_assignments': invalid,
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
    parser.add_argument('--output', default='results/v3_slack_ppo.csv')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    if args.option_horizon < 1:
        parser.error('--option-horizon must be positive')
    model = MaskablePPO.load(args.model, device='cpu')
    rows = evaluate(model, args.episodes, args.seed_start,
                    args.option_horizon, args.deadlines)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    avg = lambda name: float(np.mean([r[name] for r in rows]))
    print(f'Saved {path} ({len(rows)} episode records)')
    print(f'Success: {avg("success"):.1%}; weighted completion: {avg("weighted_fraction"):.1%}; '
          f'expired objectives: {avg("expired_targets"):.2f}; '
          f'target A expired: {avg("target_A_expired"):.1%}; '
          f'mean return: {avg("return_value"):.2f}; '
          f'primitive steps: {avg("primitive_steps"):.1f}; '
          f'high-level decisions: {avg("decisions"):.1f}; '
          f'urgent target spawned: {avg("urgent_spawned"):.1%}; '
          f'invalid assignments: {avg("invalid_assignments"):.2f}')


if __name__ == '__main__':
    main()
