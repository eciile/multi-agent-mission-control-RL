"""One-decision interventions in frozen V3 MaskablePPO trajectories.

The baseline is deterministic masked PPO. In the intervention rollout, the
first selected PPO decision is replaced with the deadline-aware heuristic's
recommendation *from the PPO rollout's own current state*. After the one
replaced macro-action, PPO resumes for the remainder of the episode.

Examples:
 python counterfactual_assignments_v3.py --model models/v3_masked_seed42_50k.zip
 python counterfactual_assignments_v3.py --model models/v3_masked_seed42_50k.zip --mode first-a-diff

A changed action can alter the timing/sequence of later random draws. These
are seeded intervention comparisons, not controlled identical stochastic
obstacle trajectories or estimates of an average treatment effect.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sb3_contrib import MaskablePPO
from mamc_rl.controllers_v3 import HierarchicalDeadlineEnv, deadline_heuristic_action
from mamc_rl.policies_v2 import _distance_map

NAMES = ('idle', 'A', 'B', 'C', 'D_urgent')


def labels(action):
    return (NAMES[int(action) // 5], NAMES[int(action) % 5])


def a_assignment(action):
    a, b = int(action) // 5, int(action) % 5
    return (a == 1, b == 1)


def a_distance_and_slack(env):
    base = env.base
    distance_map = _distance_map(tuple(int(v) for v in base.targets[0]), set(base.obstacles))
    distances = [distance_map.get(tuple(int(v) for v in pos)) for pos in base.agents]
    slack = [None if d is None else int(base.deadlines[0] - base.steps - d)
             for d in distances]
    return distances, slack


def policy_action(model, observation, env):
    mask = env.action_masks()
    action, _ = model.predict(observation, deterministic=True, action_masks=mask)
    action = int(action)
    if not bool(mask[action]):
        raise AssertionError(f'Model returned infeasible action {action}')
    return action


def candidate_matches(mode, ppo_action, heuristic_action):
    if ppo_action == heuristic_action:
        return False
    if mode == 'first-diff':
        return True
    # A more focused intervention: does the heuristic choose a different
    # agent-to-A assignment, including assignment vs no assignment?
    return a_assignment(ppo_action) != a_assignment(heuristic_action)


def simulate(model, seed, option_horizon, deadlines, mode, intervene_at=None):
    env = HierarchicalDeadlineEnv(option_horizon=option_horizon, deadlines=tuple(deadlines))
    rows = []
    chosen_intervention = None
    try:
        observation, _ = env.reset(seed=seed)
        total_reward = 0.0
        spawned = False
        while True:
            t = int(env.base.steps)
            index = len(rows) + 1
            proposed = policy_action(model, observation, env)
            heuristic = int(deadline_heuristic_action(env))
            if not bool(env.action_masks()[heuristic]):
                raise AssertionError(f'Heuristic returned infeasible action {heuristic}')
            matched = candidate_matches(mode, proposed, heuristic)
            # A baseline rollout picks the first eligible decision index.
            if intervene_at is None and chosen_intervention is None and matched:
                chosen_intervention = index
            replaced = bool(intervene_at is not None and index == intervene_at)
            executed = heuristic if replaced else proposed
            distances, slack = a_distance_and_slack(env)
            a_before = bool(env.base.completed[0])
            expired_before = bool(env.base.expired[0])
            observation, reward, terminated, truncated, info = env.step(executed)
            spawned |= bool(info['target_spawned'])
            total_reward += float(reward)
            rows.append({
                'seed': seed, 'variant': 'intervention' if intervene_at is not None else 'baseline',
                'decision': index, 'step_start': t, 'step_end': int(env.base.steps),
                'ppo_recommendation': '/'.join(labels(proposed)),
                'heuristic_recommendation': '/'.join(labels(heuristic)),
                'action_executed': '/'.join(labels(executed)),
                'intervened': int(replaced),
                'eligible_difference': int(matched),
                'a_distance_agent0': '' if distances[0] is None else distances[0],
                'a_distance_agent1': '' if distances[1] is None else distances[1],
                'a_slack_agent0': '' if slack[0] is None else slack[0],
                'a_slack_agent1': '' if slack[1] is None else slack[1],
                'a_completed_before': int(a_before),
                'a_expired_before': int(expired_before),
                'a_completed_after': int(env.base.completed[0]),
                'a_expired_after': int(env.base.expired[0]),
                'primitive_duration': int(info['low_level_steps']),
                'reward': float(reward),
            })
            if terminated or truncated:
                return {
                    'seed': seed,
                    'success': int(info['success']),
                    'weighted_fraction': float(info['weighted_fraction']),
                    'weighted_completed': int(info['weighted_completed']),
                    'a_completed': int(env.base.completed[0]),
                    'a_expired': int(env.base.expired[0]),
                    'expired_targets': int(info['expired_targets']),
                    'primitive_steps': int(env.base.steps),
                    'high_level_decisions': len(rows),
                    'return_value': float(total_reward),
                    'urgent_spawned': int(spawned),
                    'first_eligible_decision': chosen_intervention,
                    'intervened_decision': intervene_at,
                    'intervention_applied': int(any(x['intervened'] for x in rows)),
                }, rows
    finally:
        env.close()


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {path} ({len(rows)} rows)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2001, 2007, 2010, 2013, 2026])
    parser.add_argument('--mode', choices=['first-diff', 'first-a-diff'], default='first-a-diff',
                        help='Intervene at first divergent agent-to-A assignment, or any first action difference')
    parser.add_argument('--option-horizon', type=int, default=3)
    parser.add_argument('--deadlines', type=int, nargs=4, default=[12, 18, 24, 23])
    parser.add_argument('--output-prefix', default='results/v3_counterfactual')
    args = parser.parse_args()
    if args.option_horizon < 1:
        parser.error('option-horizon must be at least 1')
    model = MaskablePPO.load(args.model, device='cpu')
    summaries, all_decisions = [], []
    print('Seed  Decision  Intervention                    A base->test  Weighted base->test  Success base->test')
    for seed in args.seeds:
        baseline, baseline_rows = simulate(model, seed, args.option_horizon, args.deadlines, args.mode)
        at = baseline['first_eligible_decision']
        if at is not None:
            intervention, intervention_rows = simulate(
                model, seed, args.option_horizon, args.deadlines, args.mode, intervene_at=at)
            if not intervention['intervention_applied']:
                raise AssertionError('Selected decision index was not replayed')
            # Strict prefix consistency checks catch accidental changes to
            # the pre-intervention environment state or stochastic sequence.
            for before, after in zip(baseline_rows[:at-1], intervention_rows[:at-1]):
                for key in ('step_start', 'step_end', 'ppo_recommendation',
                            'action_executed', 'a_completed_after', 'a_expired_after'):
                    if before[key] != after[key]:
                        raise AssertionError(f'Pre-intervention mismatch seed={seed}, {key}')
            ppo_choice = baseline_rows[at-1]['ppo_recommendation']
            heuristic_choice = baseline_rows[at-1]['heuristic_recommendation']
            label = f'{ppo_choice} => {heuristic_choice}'
        else:
            # No eligible difference: outcome unchanged, no intervention.
            intervention = dict(baseline, intervention_applied=0)
            intervention_rows = []
            label = 'none available'
        summary = {
            'seed': seed, 'mode': args.mode, 'intervention_decision': '' if at is None else at,
            'intervention_assignment': label,
            'intervention_applied': int(intervention['intervention_applied']),
        }
        for prefix, record in (('baseline', baseline), ('intervention', intervention)):
            for field in ('success', 'weighted_fraction', 'weighted_completed',
                          'a_completed', 'a_expired', 'expired_targets',
                          'primitive_steps', 'high_level_decisions', 'return_value', 'urgent_spawned'):
                summary[f'{prefix}_{field}'] = record[field]
        summary['delta_weighted'] = intervention['weighted_fraction'] - baseline['weighted_fraction']
        summary['delta_return'] = intervention['return_value'] - baseline['return_value']
        summaries.append(summary)
        all_decisions.extend(baseline_rows)
        all_decisions.extend(intervention_rows)
        print(f'{seed:<6} {str(at):<9} {label:<31} '
              f"{baseline['a_completed']}->{intervention['a_completed']}        "
              f"{baseline['weighted_fraction']:.1%}->{intervention['weighted_fraction']:.1%}      "
              f"{baseline['success']}->{intervention['success']}")
    write_rows(args.output_prefix + '_summary.csv', summaries)
    write_rows(args.output_prefix + '_decisions.csv', all_decisions)
    applicable = [r for r in summaries if r['intervention_applied']]
    improved = sum(r['delta_weighted'] > 1e-9 for r in applicable)
    worse = sum(r['delta_weighted'] < -1e-9 for r in applicable)
    tied = len(applicable) - improved - worse
    recovered_a = sum(r['baseline_a_expired'] and r['intervention_a_completed'] for r in applicable)
    print(f'\nInterventions applied: {len(applicable)}/{len(summaries)}; '
          f'weighted improvement/worse/tie: {improved}/{worse}/{tied}; '
          f'target A recovered: {recovered_a}')
    print('Caution: action changes may alter later stochastic obstacle updates. '
          'These are single-seed intervention traces, not a randomized causal estimate.')


if __name__ == '__main__':
    main()
