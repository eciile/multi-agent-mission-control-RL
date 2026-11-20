"""Collect non-oracle depth-4 V4 planner demonstrations on disjoint mission seeds.

Usage:
  python collect_expert_v4.py --train-episodes 20 --val-episodes 10
  python collect_expert_v4.py --train-episodes 500 --val-episodes 100

No final-test seeds (2000-2199) are used by default. The data is gathered from
expert rollouts, not from PPO states; closed-loop BC evaluation is essential.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np

from mamc_rl.controllers_v4 import HierarchicalResourceEnv
from mamc_rl.lookahead_v4 import lookahead_action


def collect(episodes: int, start_seed: int, depth: int, beam_width: int):
    observations, masks, actions, seeds = [], [], [], []
    returns, successes, weighted, latencies = [], [], [], []
    env = HierarchicalResourceEnv(option_horizon=3)
    try:
        for seed in range(start_seed, start_seed + episodes):
            obs, _ = env.reset(seed=seed)
            total = 0.0
            while True:
                mask = np.asarray(env.action_masks(), dtype=np.bool_)
                start = time.perf_counter()
                action = int(lookahead_action(env, depth=depth, beam_width=beam_width))
                latencies.append(time.perf_counter() - start)
                if not mask[action]:
                    raise AssertionError(f'Expert chose invalid action {action} on seed {seed}')
                observations.append(np.asarray(obs, dtype=np.float32).copy())
                masks.append(mask.copy())
                actions.append(action)
                seeds.append(seed)
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    returns.append(total)
                    successes.append(int(info['success']))
                    weighted.append(float(info['weighted_fraction']))
                    break
    finally:
        env.close()
    return {
        'observations': np.asarray(observations, dtype=np.float32),
        'masks': np.asarray(masks, dtype=np.bool_),
        'actions': np.asarray(actions, dtype=np.int64),
        'seeds': np.asarray(seeds, dtype=np.int64),
        'episode_seeds': np.arange(start_seed, start_seed + episodes, dtype=np.int64),
        'episode_returns': np.asarray(returns, dtype=np.float32),
        'episode_success': np.asarray(successes, dtype=np.int64),
        'episode_weighted': np.asarray(weighted, dtype=np.float32),
        'planner_depth': np.asarray(depth, dtype=np.int64),
        'beam_width': np.asarray(beam_width, dtype=np.int64),
        'mean_planning_ms': np.asarray(1000 * np.mean(latencies), dtype=np.float64),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-episodes', type=int, default=500)
    p.add_argument('--val-episodes', type=int, default=100)
    p.add_argument('--train-seed-start', type=int, default=10000)
    p.add_argument('--val-seed-start', type=int, default=15000)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--beam-width', type=int, default=4)
    p.add_argument('--output-dir', default='results/expert_v4')
    args = p.parse_args()
    if min(args.train_episodes, args.val_episodes, args.depth, args.beam_width) < 1:
        p.error('Episode counts, depth and beam width must be positive')
    train_set = set(range(args.train_seed_start, args.train_seed_start + args.train_episodes))
    val_set = set(range(args.val_seed_start, args.val_seed_start + args.val_episodes))
    final_test = set(range(2000, 2200))
    if train_set & val_set or (train_set | val_set) & final_test:
        p.error('Train/validation seeds must be disjoint and exclude final test seeds 2000-2199')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, episodes, start_seed in (('train', args.train_episodes, args.train_seed_start),
                                       ('val', args.val_episodes, args.val_seed_start)):
        dataset = collect(episodes, start_seed, args.depth, args.beam_width)
        dest = out / f'{name}.npz'
        np.savez_compressed(dest, **dataset)
        print(f'{name}: {episodes} missions, {len(dataset["actions"])} expert decisions; '
              f'success {np.mean(dataset["episode_success"]):.1%}, '
              f'weighted {np.mean(dataset["episode_weighted"]):.1%}, '
              f'planner {float(dataset["mean_planning_ms"]):.2f} ms/decision; saved {dest}')
    print('Final evaluation seeds 2000-2199 were excluded from training and validation.')


if __name__ == '__main__':
    main()
