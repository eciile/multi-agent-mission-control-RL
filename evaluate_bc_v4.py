"""Closed-loop V4 evaluation of a masked behavioral-cloning policy.

Usage:
  python evaluate_bc_v4.py --model models/v4_bc_planner.pt --episodes 200 --seed-start 2000
Reports controller forward-pass time separately from environment execution.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time
import numpy as np
import torch

from mamc_rl.controllers_v4 import HierarchicalResourceEnv
from train_bc_v4 import BCPolicy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='models/v4_bc_planner.pt')
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--seed-start', type=int, default=2000)
    p.add_argument('--option-horizon', type=int, default=3)
    p.add_argument('--output', default='results/v4_bc_eval.csv')
    args = p.parse_args()
    if args.episodes < 1 or args.option_horizon < 1:
        p.error('episodes and option horizon must be positive')
    # This is a locally generated checkpoint from our own supervised script.
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
    policy = BCPolicy(int(checkpoint['obs_dim']), int(checkpoint['n_actions']))
    policy.load_state_dict(checkpoint['state_dict'])
    policy.eval()
    torch.set_num_threads(1)
    env = HierarchicalResourceEnv(option_horizon=args.option_horizon)
    if env.observation_space.shape != (int(checkpoint['obs_dim']),):
        raise ValueError('Model/environment observation mismatch')
    rows = []
    try:
        for seed in range(args.seed_start, args.seed_start + args.episodes):
            obs, _ = env.reset(seed=seed)
            total_return = 0.0
            primitive_steps = 0
            decisions = 0
            planning_seconds = 0.0
            urgent_spawned = False
            while True:
                mask = env.action_masks()
                with torch.inference_mode():
                    start = time.perf_counter()
                    logits = policy(torch.from_numpy(np.asarray(obs, dtype=np.float32))[None],
                                    torch.from_numpy(np.asarray(mask, dtype=np.bool_))[None])
                    action = int(logits.argmax(dim=-1).item())
                    planning_seconds += time.perf_counter() - start
                if not mask[action]:
                    raise AssertionError('BC selected illegal action')
                obs, reward, done, truncated, info = env.step(action)
                total_return += float(reward)
                primitive_steps += int(info['low_level_steps'])
                decisions += 1
                urgent_spawned |= bool(info['target_spawned'])
                if done or truncated:
                    rows.append({'seed': seed, 'policy':'masked_bc_depth4', 'success':int(info['success']),
                                 'weighted_fraction':float(info['weighted_fraction']),
                                 'expired_targets':int(info['expired_targets']),
                                 'return_value':total_return, 'primitive_steps':primitive_steps,
                                 'high_level_decisions':decisions, 'urgent_spawned':int(urgent_spawned),
                                 'planning_seconds':planning_seconds,
                                 'planning_ms_per_decision':1000 * planning_seconds / decisions})
                    break
    finally:
        env.close()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mean = lambda key: float(np.mean([row[key] for row in rows]))
    total_dec = sum(row['high_level_decisions'] for row in rows)
    total_s = sum(row['planning_seconds'] for row in rows)
    print(f'Saved {out} ({len(rows)} rows)')
    print(f'BC: success={mean("success"):.1%}; weighted={mean("weighted_fraction"):.1%}; '
          f'expired={mean("expired_targets"):.2f}; steps={mean("primitive_steps"):.1f}; '
          f'planning ms/decision={1000 * total_s / total_dec:.3f}; spawned={mean("urgent_spawned"):.1%}')
    print('Planner timing excludes simulator execution; compare on the same hardware.')


if __name__ == '__main__':
    main()
