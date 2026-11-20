"""Trace PPO and deadline-heuristic decisions on selected V3 mission seeds.

Run:
 python trace_decisions_v3.py --model models/v3_masked_seed42_50k.zip \
   --seeds 2001 2007 2010 2013 2026

Writes one decision-level CSV and one primitive-step CSV. All BFS distances
use currently visible targets and currently present obstacles, never future
urgent-target positions or future obstacle configurations.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sb3_contrib import MaskablePPO

from mamc_rl.controllers_v3 import HierarchicalDeadlineEnv, deadline_heuristic_action
from mamc_rl.policies_v2 import _distance_map

TARGET_NAMES = ('A', 'B', 'C', 'D_urgent')


def _positions(base):
    return [tuple(map(int, p)) for p in base.agents]


def _a_distances(base):
    """Current shortest-path distances, treating other agent as in BFS planner."""
    a_position = tuple(map(int, base.targets[0]))
    dmap = _distance_map(a_position, set(base.obstacles))
    return [dmap.get(p) for p in _positions(base)]


def _choices(action):
    return (int(action) // 5, int(action) % 5)


def _label(choice):
    return 'idle' if choice == 0 else TARGET_NAMES[choice - 1]


def _as_text(value):
    return '' if value is None else value


def _write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {path} ({len(rows)} rows)')


def trace_one(policy, model, seed, option_horizon, deadlines):
    env = HierarchicalDeadlineEnv(option_horizon=option_horizon, deadlines=deadlines)
    obs, _ = env.reset(seed=seed)
    base = env.base
    decisions = []
    primitive = []
    first_a_assignment = None
    first_a_feasible_assignment = None
    completed_a_step = None
    expired_a_step = None
    active_decision = None
    original_step = base.step

    def tracked_primitive_step(joint_action):
        nonlocal completed_a_step, expired_a_step
        t_before = int(base.steps)
        a_was_completed = bool(base.completed[0])
        a_was_expired = bool(base.expired[0])
        before_positions = _positions(base)
        before_d = _a_distances(base)
        before_slack = int(base.deadlines[0] - t_before)
        result = original_step(joint_action)
        _, reward, _, _, info = result
        if bool(base.completed[0]) and not a_was_completed:
            completed_a_step = int(base.steps)
        if bool(base.expired[0]) and not a_was_expired:
            expired_a_step = int(base.steps)
        if t_before < int(base.deadlines[0]):
            primitive.append(dict(
                seed=seed, policy=policy, primitive_step=int(base.steps),
                decision_index=active_decision['decision_index'],
                assigned_agent0=active_decision['assigned_agent0'],
                assigned_agent1=active_decision['assigned_agent1'],
                agent0_before=str(before_positions[0]),
                agent1_before=str(before_positions[1]),
                agent0_after=str(_positions(base)[0]),
                agent1_after=str(_positions(base)[1]),
                a_distance_agent0_before=_as_text(before_d[0]),
                a_distance_agent1_before=_as_text(before_d[1]),
                a_slack_before=before_slack,
                a_reachable_before=int(any(d is not None and d <= before_slack for d in before_d)),
                a_completed_after=int(base.completed[0]),
                a_expired_after=int(base.expired[0]),
                obstacles_changed=int(info['obstacles_changed']),
                urgent_spawned=int(info['target_spawned']),
                low_level_action=int(joint_action), reward=float(reward),
            ))
        return result

    # Instrument the existing simulator without editing its implementation.
    base.step = tracked_primitive_step
    try:
        while True:
            mask = env.action_masks()
            if policy == 'ppo':
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                action = int(action)
            else:
                action = int(deadline_heuristic_action(env))
            if not mask[action]:
                raise AssertionError(f'{policy} chose masked-out action {action}')
            t_before = int(base.steps)
            before_positions = _positions(base)
            before_distances = _a_distances(base)
            before_slack = int(base.deadlines[0] - t_before)
            choices = _choices(action)
            assigned = tuple(_label(c) for c in choices)
            a_is_available = bool(base.active[0] and not base.completed[0] and not base.expired[0])
            a_assignment = a_is_available and 1 in choices
            if a_assignment and first_a_assignment is None:
                first_a_assignment = t_before
            if (a_assignment and first_a_feasible_assignment is None and
                    any(choices[i] == 1 and before_distances[i] is not None and
                        before_distances[i] <= before_slack for i in range(2))):
                first_a_feasible_assignment = t_before
            active_decision = dict(decision_index=len(decisions) + 1,
                                   assigned_agent0=assigned[0], assigned_agent1=assigned[1])
            obs, reward, terminated, truncated, info = env.step(action)
            decisions.append(dict(
                seed=seed, policy=policy, decision_index=active_decision['decision_index'],
                start_step=t_before, end_step=int(base.steps),
                assigned_agent0=assigned[0], assigned_agent1=assigned[1],
                agent0_start=str(before_positions[0]), agent1_start=str(before_positions[1]),
                a_position=str(tuple(map(int, base.targets[0]))),
                a_deadline=int(base.deadlines[0]), a_slack_start=before_slack,
                a_distance_agent0_start=_as_text(before_distances[0]),
                a_distance_agent1_start=_as_text(before_distances[1]),
                a_reachable_from_either=int(a_is_available and any(
                    d is not None and d <= before_slack for d in before_distances)),
                assigned_a=int(a_assignment),
                assigned_a_feasibly=int(a_assignment and any(
                    choices[i] == 1 and before_distances[i] is not None and
                    before_distances[i] <= before_slack for i in range(2))),
                a_completed_end=int(base.completed[0]),
                a_expired_end=int(base.expired[0]),
                duration=int(info['low_level_steps']),
                reward=float(reward),
            ))
            if terminated or truncated:
                summary = dict(seed=seed, policy=policy,
                               success=int(info['success']),
                               weighted_completion=float(info['weighted_fraction']),
                               a_completed=int(base.completed[0]),
                               a_expired=int(base.expired[0]),
                               first_a_assignment=_as_text(first_a_assignment),
                               first_feasible_a_assignment=_as_text(first_a_feasible_assignment),
                               a_completed_step=_as_text(completed_a_step),
                               a_expired_step=_as_text(expired_a_step),
                               final_step=int(base.steps))
                return decisions, primitive, summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[2001, 2007, 2010, 2013, 2026])
    parser.add_argument('--option-horizon', type=int, default=3)
    parser.add_argument('--deadlines', type=int, nargs=4, default=[12, 18, 24, 23])
    parser.add_argument('--output-prefix', default='results/v3_decision_trace')
    args = parser.parse_args()
    if args.option_horizon < 1:
        parser.error('option-horizon must be >= 1')
    model = MaskablePPO.load(args.model, device='cpu')
    decisions, primitive, summaries = [], [], []
    for seed in args.seeds:
        for policy in ('ppo', 'heuristic'):
            decision_rows, primitive_rows, summary = trace_one(
                policy, model, seed, args.option_horizon, tuple(args.deadlines))
            decisions.extend(decision_rows)
            primitive.extend(primitive_rows)
            summaries.append(summary)
    _write_csv(args.output_prefix + '_decisions.csv', decisions, list(decisions[0]))
    _write_csv(args.output_prefix + '_primitive.csv', primitive, list(primitive[0]))
    print('\nPer-seed outcome and target-A assignment timing')
    print('Seed   Policy       Success  Weighted   A state   First assigned  Feasible first  Completed/expired at')
    for r in summaries:
        state = 'done' if r['a_completed'] else ('expired' if r['a_expired'] else 'unresolved')
        t_event = r['a_completed_step'] if r['a_completed'] else r['a_expired_step']
        print(f"{r['seed']:<6} {r['policy']:<12} {r['success']:>3}     "
              f"{r['weighted_completion']:>6.1%}   {state:<10} "
              f"{str(r['first_a_assignment']):>6}          "
              f"{str(r['first_feasible_a_assignment']):>6}         {t_event}")
    print('\nFirst differing high-level decision on each seed (matched decision index, not necessarily same time):')
    for seed in args.seeds:
        p = [r for r in decisions if r['seed'] == seed and r['policy'] == 'ppo']
        h = [r for r in decisions if r['seed'] == seed and r['policy'] == 'heuristic']
        diff = next(((a, b) for a, b in zip(p, h)
                     if (a['assigned_agent0'], a['assigned_agent1']) !=
                        (b['assigned_agent0'], b['assigned_agent1'])), None)
        if diff:
            a, b = diff
            print(f"  {seed}: decision #{a['decision_index']} | PPO t={a['start_step']} "
                  f"({a['assigned_agent0']}, {a['assigned_agent1']}) | "
                  f"heuristic t={b['start_step']} ({b['assigned_agent0']}, {b['assigned_agent1']})")
        else:
            print(f'  {seed}: no differing decision in shared decision indices')
    print('Note: differing assignments alone do not prove causality; policies can enter different states.')


if __name__ == '__main__':
    main()
