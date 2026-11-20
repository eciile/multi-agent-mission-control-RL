"""Single-decision V4 intervention: route Agent 0 to target A when feasible.

Compare a frozen MaskablePPO policy with a run where ONLY its first high-level
assignment is replaced by (Agent 0 -> A, Agent 1 -> original choice). After
that, the policy acts normally. Both runs start from the same seed.

This is an exploratory intervention, NOT a causal estimate: altered actions may
change subsequent stochastic obstacle histories. No model or simulator edits.

Example:
  python intervene_early_A_v4.py --model models/v4_masked_seed123_50k.zip \
      --episodes 200 --seed-start 2000
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from mamc_rl.controllers_v4 import HierarchicalResourceEnv
from mamc_rl.policies_v2 import _distance_map


def early_a_feasible(env) -> tuple[bool, float]:
    """Agent 0 can reach active target A now, ignoring future changes/collisions."""
    b = env.base
    t = 0
    if not (b.active[t] and not b.completed[t] and not b.expired[t]
            and b.capabilities[0, t]):
        return False, float('inf')
    dmap = _distance_map(tuple(map(int, b.targets[t])), set(b.obstacles))
    distance = dmap.get(tuple(map(int, b.agents[0])), float('inf'))
    feasible = distance <= int(b.fuel[0]) and distance <= int(b.deadlines[t]) - int(b.steps)
    return bool(feasible), float(distance)


def action_name(action: int) -> str:
    labels = ('idle', 'A', 'B', 'C', 'D')
    return f'{labels[action // 5]}/{labels[action % 5]}'


def first_action_intervention(env, original_action: int) -> tuple[int, bool, bool, float]:
    """Keep agent 1's choice; change agent 0 to target A only when legal/feasible.

    Returns (chosen_action, eligible, changed, distance_to_A).
    """
    mask = env.action_masks()
    if not mask[original_action]:
        raise AssertionError('PPO produced a masked-out original action')
    eligible, distance = early_a_feasible(env)
    proposed = 5 * 1 + (original_action % 5)
    if not eligible or not mask[proposed]:
        return original_action, False, False, distance
    return int(proposed), True, int(proposed) != original_action, distance


def run_episode(model, seed: int, intervention: bool, option_horizon: int,
                deadlines: tuple[int, ...], fuel: tuple[int, ...]):
    env = HierarchicalResourceEnv(
        option_horizon=option_horizon,
        deadlines=deadlines,
        fuel_budgets=fuel,
    )
    try:
        obs, _ = env.reset(seed=seed)
        b = env.base
        first_original = first_selected = None
        first_eligible = first_changed = False
        initial_a_distance = float('inf')
        total_reward = 0.0
        steps = decisions = 0
        spawned = False
        while True:
            mask = env.action_masks()
            proposed, _ = model.predict(obs, deterministic=True, action_masks=mask)
            proposed = int(proposed)
            if not bool(mask[proposed]):
                raise AssertionError('Model predicted an invalid high-level action')
            action = proposed
            if decisions == 0:
                first_original = proposed
                new_action, first_eligible, would_change, initial_a_distance = first_action_intervention(env, proposed)
                first_selected = new_action if intervention else proposed
                first_changed = bool(intervention and would_change)
                if intervention:
                    action = new_action
            obs, reward, terminated, truncated, info = env.step(action)
            steps += int(info['low_level_steps'])
            decisions += 1
            total_reward += float(reward)
            spawned |= bool(info['target_spawned'])
            if terminated or truncated:
                break
        return dict(
            seed=int(seed),
            mode='intervention' if intervention else 'original',
            eligible=int(first_eligible),
            changed=int(first_changed),
            original_first_action=int(first_original),
            selected_first_action=int(first_selected),
            original_first_labels=action_name(first_original),
            selected_first_labels=action_name(first_selected),
            initial_A_distance=initial_a_distance if np.isfinite(initial_a_distance) else '',
            success=int(info['success']),
            weighted_fraction=float(info['weighted_fraction']),
            weighted_completed=int(info['weighted_completed']),
            A_completed=int(b.completed[0]),
            A_expired=int(b.expired[0]),
            D_completed=int(b.completed[3]),
            D_expired=int(b.expired[3]),
            expired_targets=int(info['expired_targets']),
            primitive_steps=int(steps),
            high_level_decisions=int(decisions),
            total_return=float(total_reward),
            urgent_spawned=int(spawned),
            fuel0_remaining=int(b.fuel[0]),
            fuel1_remaining=int(b.fuel[1]),
        )
    finally:
        env.close()


def save_csv(path, rows):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {out} ({len(rows)} rows)')


def summarize(rows, label):
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    print(f'{label:<15} {mean("success"):>7.1%} {mean("weighted_fraction"):>9.1%}'
          f' {mean("A_completed"):>8.1%} {mean("D_completed"):>8.1%}'
          f' {mean("expired_targets"):>8.2f} {mean("primitive_steps"):>7.1f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed-start', type=int, default=2000)
    parser.add_argument('--option-horizon', type=int, default=3)
    parser.add_argument('--deadlines', type=int, nargs=4, default=(12, 18, 24, 23))
    parser.add_argument('--fuel', type=int, nargs=2, default=(16, 18))
    parser.add_argument('--output', default='results/v4_early_A_intervention.csv')
    args = parser.parse_args()
    if args.episodes < 1 or args.option_horizon < 1:
        parser.error('--episodes and --option-horizon must be positive')

    model = MaskablePPO.load(args.model, device='cpu')
    pairs = []
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        original = run_episode(model, seed, False, args.option_horizon,
                               tuple(args.deadlines), tuple(args.fuel))
        test = run_episode(model, seed, True, args.option_horizon,
                           tuple(args.deadlines), tuple(args.fuel))
        if original['original_first_action'] != test['original_first_action']:
            raise AssertionError('First PPO decisions did not reproduce between seeded runs')
        pairs.append((original, test))

    rows = [row for pair in pairs for row in pair]
    save_csv(args.output, rows)
    eligible = [p for p in pairs if p[1]['eligible']]
    changed = [p for p in pairs if p[1]['changed']]
    print(f'Initially feasible A assignments: {len(eligible)}/{len(pairs)}; '
          f'first decisions actually changed: {len(changed)}/{len(pairs)}')
    print('Group           Success  Weighted  A done   D done  Expired    Steps')
    summarize([p[0] for p in pairs], 'all / original')
    summarize([p[1] for p in pairs], 'all / test')
    if changed:
        print('\nOnly seeds where the first assignment actually changed:')
        summarize([p[0] for p in changed], 'changed / orig')
        summarize([p[1] for p in changed], 'changed / test')
        improved = sum(p[1]['weighted_fraction'] > p[0]['weighted_fraction'] + 1e-10 for p in changed)
        worsened = sum(p[1]['weighted_fraction'] < p[0]['weighted_fraction'] - 1e-10 for p in changed)
        a_recovered = sum(p[0]['A_completed'] == 0 and p[1]['A_completed'] == 1 for p in changed)
        d_lost = sum(p[0]['D_completed'] == 1 and p[1]['D_completed'] == 0 for p in changed)
        print(f'Weighted better/worse/tied: {improved}/{worsened}/{len(changed)-improved-worsened}; '
              f'A recovered: {a_recovered}; D lost: {d_lost}')
    print('Note: same initial seeds, but intervention may change future obstacle randomness.')
    print('A-feasibility is a current-state BFS/fuel/deadline lower bound, not a joint-mission guarantee.')


if __name__ == '__main__':
    main()
