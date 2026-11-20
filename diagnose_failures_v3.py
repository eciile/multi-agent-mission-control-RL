"""Compare V3 masked PPO with the deadline heuristic, recording primitive-step events.

Run:
 python diagnose_failures_v3.py --model models/v3_masked_seed42_50k.zip --episodes 200 --seed-start 2000

Creates per-episode and per-target CSV files in results/. No simulator edits.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from mamc_rl.controllers_v3 import HierarchicalDeadlineEnv, deadline_heuristic_action

TARGET_NAMES = ('A', 'B', 'C', 'D_urgent')


def run_episode(policy, model, seed, option_horizon, deadlines):
    env = HierarchicalDeadlineEnv(option_horizon=option_horizon, deadlines=deadlines)
    obs, _ = env.reset(seed=seed)
    base = env.base
    completed_at = [None] * 4
    expired_at = [None] * 4
    obstacle_changes = 0
    spawn_at = None
    # Hook the underlying simulator's real primitive step to timestamp events
    # even when a high-level action spans multiple primitive transitions.
    original_step = base.step

    def observed_step(action):
        nonlocal obstacle_changes, spawn_at
        was_completed = base.completed.copy()
        was_expired = base.expired.copy()
        outcome = original_step(action)
        _, _, _, _, info = outcome
        for i in range(4):
            if not was_completed[i] and base.completed[i]:
                completed_at[i] = int(base.steps)
            if not was_expired[i] and base.expired[i]:
                expired_at[i] = int(base.steps)
        if info['obstacles_changed']:
            obstacle_changes += 1
        if info['target_spawned']:
            spawn_at = int(base.steps)
        return outcome

    base.step = observed_step
    total_return, decisions, steps = 0.0, 0, 0
    try:
        while True:
            if policy == 'ppo':
                mask = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                action = int(action)
                if not mask[action]:
                    raise AssertionError('PPO selected a masked-out action')
            else:
                action = deadline_heuristic_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            decisions += 1
            steps += int(info['low_level_steps'])
            if terminated or truncated:
                break
        if steps != base.steps:
            raise AssertionError(f'Primitive-step accounting mismatch: {steps} != {base.steps}')
        episode = dict(
            policy=policy, seed=seed, success=int(info['success']),
            weighted_fraction=float(info['weighted_fraction']),
            weighted_completed=int(info['weighted_completed']),
            expired_targets=int(info['expired_targets']),
            primitive_steps=steps, decisions=decisions,
            return_value=total_return,
            obstacle_changes=obstacle_changes,
            urgent_spawned=int(spawn_at is not None),
        )
        details = []
        for i, name in enumerate(TARGET_NAMES):
            details.append(dict(
                policy=policy, seed=seed, target=name,
                priority=int(base.priorities[i]), deadline=int(base.deadlines[i]),
                completed=int(base.completed[i]), expired=int(base.expired[i]),
                completion_step=completed_at[i] if completed_at[i] is not None else '',
                expiry_step=expired_at[i] if expired_at[i] is not None else '',
                spawned_step=spawn_at if i == 3 and spawn_at is not None else '',
            ))
        return episode, details
    finally:
        env.close()


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {path} ({len(rows)} rows)')


def summarize(episodes, targets):
    print('\nPolicy               Success  Weighted   Expired  Steps  Obstacle changes')
    for policy in ('ppo', 'heuristic'):
        rows = [r for r in episodes if r['policy'] == policy]
        avg = lambda key: float(np.mean([r[key] for r in rows]))
        print(f'{policy:<20} {avg("success"):>6.1%}   {avg("weighted_fraction"):>7.1%}'
              f'   {avg("expired_targets"):>6.2f}   {avg("primitive_steps"):>5.1f}'
              f'      {avg("obstacle_changes"):>5.2f}')

    print('\nTarget expiration rate (percentage of episodes)')
    print('Target       Deadline  Priority       PPO   Heuristic')
    for name in TARGET_NAMES:
        p = [r for r in targets if r['policy'] == 'ppo' and r['target'] == name]
        h = [r for r in targets if r['policy'] == 'heuristic' and r['target'] == name]
        print(f'{name:<14} {p[0]["deadline"]:>3}       {p[0]["priority"]:>3}'
              f'       {np.mean([r["expired"] for r in p]):>6.1%}'
              f'    {np.mean([r["expired"] for r in h]):>6.1%}')

    print('\nCompletion time among episodes completing that target (conditional means)')
    print('Target               PPO   Heuristic  (primitive step)')
    for name in TARGET_NAMES:
        values = []
        for policy in ('ppo', 'heuristic'):
            done = [int(r['completion_step']) for r in targets
                    if r['policy'] == policy and r['target'] == name and r['completed']]
            values.append(f'{np.mean(done):.1f}' if done else 'n/a')
        print(f'{name:<16} {values[0]:>6}    {values[1]:>6}')

    by_seed = {(r['seed'], r['policy']): r for r in episodes}
    worse = [s for s in sorted({r['seed'] for r in episodes})
             if by_seed[(s, 'ppo')]['weighted_fraction'] < by_seed[(s, 'heuristic')]['weighted_fraction']]
    better = [s for s in sorted({r['seed'] for r in episodes})
              if by_seed[(s, 'ppo')]['weighted_fraction'] > by_seed[(s, 'heuristic')]['weighted_fraction']]
    equal = len(episodes) // 2 - len(worse) - len(better)
    print(f'\nSeed-paired weighted outcome: PPO worse on {len(worse)}, '
          f'better on {len(better)}, tied on {equal} seeds.')
    print('First 15 PPO-underperforming seeds:', ', '.join(map(str, worse[:15])) or 'none')
    print('Note: initial seeds are paired; obstacle histories can differ across policies.')

    for policy in ('ppo', 'heuristic'):
        rows = [r for r in episodes if r['policy'] == policy]
        successes = [r for r in rows if r['success']]
        failures = [r for r in rows if not r['success']]
        if successes:
            print(f'{policy} successful episodes: mean steps {np.mean([r["primitive_steps"] for r in successes]):.1f}')
        if failures:
            print(f'{policy} failed episodes: mean weighted completion '
                  f'{np.mean([r["weighted_fraction"] for r in failures]):.1%}; '
                  f'mean steps {np.mean([r["primitive_steps"] for r in failures]):.1f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed-start', type=int, default=2000)
    parser.add_argument('--option-horizon', type=int, default=3)
    parser.add_argument('--deadlines', nargs=4, type=int, default=(12, 18, 24, 23))
    parser.add_argument('--output-prefix', default='results/v3_failure_diagnostic')
    args = parser.parse_args()
    if args.episodes < 1 or args.option_horizon < 1:
        parser.error('episodes and option-horizon must be positive')
    model = MaskablePPO.load(args.model, device='cpu')
    episodes, targets = [], []
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        for policy in ('ppo', 'heuristic'):
            episode, detail = run_episode(policy, model, seed, args.option_horizon,
                                          tuple(args.deadlines))
            episodes.append(episode)
            targets.extend(detail)
    save_csv(args.output_prefix + '_episodes.csv', episodes)
    save_csv(args.output_prefix + '_targets.csv', targets)
    summarize(episodes, targets)


if __name__ == '__main__':
    main()
