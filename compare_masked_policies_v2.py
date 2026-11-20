"""Compare trained MaskablePPO against uniformly random VALID assignments.

Both use the identical MaskedHierarchicalMissionEnv and BFS option executor.
Examples:
  python compare_masked_policies_v2.py --model models/hier_masked_10k.zip
  python compare_masked_policies_v2.py --model models/hier_masked_10k.zip --episodes 100 --seed-start 1000
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from mamc_rl.masked_hierarchical_v2 import MaskedHierarchicalMissionEnv


def rollout(policy_name, model, seed, option_horizon):
    env = MaskedHierarchicalMissionEnv(option_horizon=option_horizon)
    rng = np.random.default_rng(seed + 1_000_000)
    obs, _ = env.reset(seed=seed)
    total_return = 0.0
    decisions = 0
    primitive_steps = 0
    spawn_seen = False
    try:
        while True:
            mask = env.action_masks()
            valid = np.flatnonzero(mask)
            if not len(valid):
                raise RuntimeError('No valid joint assignment; idle should always be available')
            if policy_name == 'masked_random':
                action = int(rng.choice(valid))
            else:
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                action = int(action)
            if not mask[action]:
                raise AssertionError('Invalid high-level action selected')
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            decisions += 1
            primitive_steps += int(info['low_level_steps'])
            spawn_seen |= bool(info['target_spawned'])
            if terminated or truncated:
                return {
                    'seed': seed,
                    'policy': policy_name,
                    'success': int(info['success']),
                    'completed_targets': int(info['targets_completed']),
                    'weighted_completed': int(info['weighted_completed']),
                    'weighted_total': int(info['weighted_total']),
                    'weighted_fraction': float(info['weighted_completed']) / max(1, int(info['weighted_total'])),
                    'primitive_steps': primitive_steps,
                    'high_level_decisions': decisions,
                    'episode_return': total_return,
                    'urgent_target_spawned': int(spawn_seen),
                }
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed-start', type=int, default=1000)
    p.add_argument('--option-horizon', type=int, default=3)
    p.add_argument('--output', default='results/masked_allocation_comparison_v2.csv')
    args = p.parse_args()
    if args.episodes < 1:
        p.error('--episodes must be >= 1')
    model = MaskablePPO.load(args.model, device='cpu')
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        for name in ('masked_random', 'masked_ppo'):
            rows.append(rollout(name, model, seed, args.option_horizon))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {path} ({len(rows)} episode records)')
    print(f'{"Policy":<17} {"Success":>8} {"Weighted":>10} {"Return":>9} {"Steps":>8} {"Decisions":>10} {"Spawned":>9}')
    for name in ('masked_random', 'masked_ppo'):
        subset = [r for r in rows if r['policy'] == name]
        avg = lambda key: float(np.mean([r[key] for r in subset]))
        print(f'{name:<17} {avg("success"):>7.1%} {avg("weighted_fraction"):>9.1%} '
              f'{avg("episode_return"):>9.2f} {avg("primitive_steps"):>8.1f} '
              f'{avg("high_level_decisions"):>10.1f} {avg("urgent_target_spawned"):>8.1%}')


if __name__ == '__main__':
    main()
