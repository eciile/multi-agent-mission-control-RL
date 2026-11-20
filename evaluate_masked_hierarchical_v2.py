"""Evaluate a MaskablePPO checkpoint on 100 held-out V2 missions.

Usage:
  python evaluate_masked_hierarchical_v2.py --model models/hier_masked_smoke.zip
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from mamc_rl.masked_hierarchical_v2 import MaskedHierarchicalMissionEnv


def evaluate(model, episodes, seed_start, option_horizon):
    env = MaskedHierarchicalMissionEnv(option_horizon=option_horizon)
    rows = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            obs, _ = env.reset(seed=seed)
            total_reward = 0.0
            primitive_steps = 0
            decisions = 0
            invalid_choices = 0
            while True:
                mask = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                action = int(action)
                choices = (action // 5, action % 5)
                invalid_choices += sum(
                    c > 0 and (not env.base.active[c-1] or env.base.completed[c-1])
                    for c in choices
                )
                obs, reward, terminated, truncated, info = env.step(action)
                decisions += 1
                primitive_steps += int(info['low_level_steps'])
                total_reward += float(reward)
                if terminated or truncated:
                    break
            rows.append(dict(seed=seed, success=int(info['success']),
                             weighted_fraction=float(info['weighted_completed']) / max(1, int(info['weighted_total'])),
                             episode_return=total_reward, primitive_steps=primitive_steps,
                             high_level_decisions=decisions, invalid_assignments=int(invalid_choices)))
    finally:
        env.close()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed-start', type=int, default=1000)
    p.add_argument('--option-horizon', type=int, default=3)
    p.add_argument('--output', default='results/masked_hierarchical_v2.csv')
    args = p.parse_args()
    if args.episodes < 1:
        p.error('--episodes must be >= 1')
    model = MaskablePPO.load(args.model, device='cpu')
    rows = evaluate(model, args.episodes, args.seed_start, args.option_horizon)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    avg = lambda k: float(np.mean([r[k] for r in rows]))
    print(f'Saved {out} ({len(rows)} episode records)')
    print(f'Success: {avg("success"):.1%}; weighted completion: {avg("weighted_fraction"):.1%}; '
          f'mean return: {avg("episode_return"):.2f}; primitive steps: {avg("primitive_steps"):.1f}; '
          f'high-level decisions: {avg("high_level_decisions"):.1f}; '
          f'invalid assignments: {avg("invalid_assignments"):.2f}')


if __name__ == '__main__':
    main()
