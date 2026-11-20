"""Frozen V4 controller comparison with paired bootstrap uncertainty intervals.

Use a previously trained, preselected MaskablePPO checkpoint. This script NEVER
trains or selects models. All methods see the same initial mission seeds; their
subsequent stochastic histories need not coincide because policies differ.

Smoke: python final_evaluation_v4.py --ppo-model models/v4_horizon3_matched_seed42.zip --episodes 5 --seed-start 29000 --out-dir results/final_smoke
Final: python final_evaluation_v4.py --ppo-model models/v4_horizon3_matched_seed42.zip --episodes 400 --seed-start 30000 --out-dir results/final_v4
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from mamc_rl.controllers_v4 import HierarchicalResourceEnv, masked_random_action, resource_heuristic_action
from mamc_rl.lookahead_v4 import lookahead_action

POLICIES = ('masked_random', 'resource_heuristic', 'masked_ppo', 'lookahead_d4')
METRICS = ('success', 'weighted_fraction', 'expired_targets', 'primitive_steps')
# Data ranges already used in model development or earlier comparisons.
DEVELOPMENT_RANGES = ((2000, 2199), (10000, 10499), (11000, 11299),
                      (15000, 15099), (16000, 16029))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_mean(x: np.ndarray, resamples: int, rng: np.random.Generator):
    mean = float(x.mean())
    # Process in batches to avoid large memory use at higher episode counts.
    samples = []
    left = resamples
    while left:
        k = min(left, 512)
        idx = rng.integers(0, len(x), size=(k, len(x)))
        samples.append(x[idx].mean(axis=1))
        left -= k
    boot = np.concatenate(samples)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi)


def run_episode(policy, seed, model, args):
    env = HierarchicalResourceEnv(option_horizon=args.option_horizon,
                                  fuel_budgets=tuple(args.fuel), deadlines=tuple(args.deadlines))
    rng = np.random.default_rng(seed + 1_000_000)
    obs, _ = env.reset(seed=seed)
    reward_sum = 0.0
    primitives = 0
    decision_count = 0
    planning_seconds = 0.0
    invalid = 0
    spawned = False
    try:
        while True:
            mask = np.asarray(env.action_masks(), dtype=bool)
            start = time.perf_counter()
            if policy == 'masked_random':
                action = masked_random_action(env, rng)
            elif policy == 'resource_heuristic':
                action = resource_heuristic_action(env)
            elif policy == 'masked_ppo':
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                action = int(action)
            elif policy == 'lookahead_d4':
                action = lookahead_action(env, depth=4, beam_width=args.beam_width)
            else:
                raise ValueError(f'Unknown controller: {policy}')
            planning_seconds += time.perf_counter() - start
            action = int(action)
            if not mask[action]:
                invalid += 1
                raise AssertionError(f'Illegal action {action} by {policy} on seed {seed}')
            obs, reward, done, truncated, info = env.step(action)
            reward_sum += float(reward)
            primitives += int(info['low_level_steps'])
            decision_count += 1
            spawned |= bool(info['target_spawned'])
            if done or truncated:
                return {
                    'policy': policy, 'seed': seed,
                    'success': int(info['success']),
                    'weighted_fraction': float(info['weighted_fraction']),
                    'expired_targets': int(info['expired_targets']),
                    'primitive_steps': primitives,
                    'high_level_decisions': decision_count,
                    'return_value': reward_sum,
                    'urgent_spawned': int(spawned),
                    'invalid_assignments': invalid,
                    'planning_seconds': planning_seconds,
                    'fuel_agent0_remaining': int(info['fuel_remaining'][0]),
                    'fuel_agent1_remaining': int(info['fuel_remaining'][1]),
                }
    finally:
        env.close()


def summarize(rows, n_boot, bootstrap_seed):
    groups = {p: {int(r['seed']): r for r in rows if r['policy'] == p} for p in POLICIES}
    seeds = sorted(groups[POLICIES[0]])
    if any(sorted(groups[p]) != seeds for p in POLICIES):
        raise ValueError('Unpaired seed set')
    rng = np.random.default_rng(bootstrap_seed)
    summary = {'n_missions': len(seeds), 'bootstrap_resamples': n_boot,
               'bootstrap_seed': bootstrap_seed, 'controllers': {}, 'paired_differences_vs_heuristic': {}}
    for policy in POLICIES:
        group = [groups[policy][s] for s in seeds]
        entry = {}
        for metric in METRICS:
            values = np.array([float(r[metric]) for r in group], dtype=float)
            mean, lower, upper = bootstrap_mean(values, n_boot, rng)
            entry[metric] = {'mean': mean, 'ci95': [lower, upper]}
        total_time = sum(r['planning_seconds'] for r in group)
        total_decisions = sum(r['high_level_decisions'] for r in group)
        entry['mean_ms_per_decision'] = 1000 * total_time / total_decisions
        entry['total_planning_seconds'] = total_time
        entry['mean_decisions_per_episode'] = total_decisions / len(group)
        entry['invalid_assignments'] = sum(r['invalid_assignments'] for r in group)
        summary['controllers'][policy] = entry
        if policy != 'resource_heuristic':
            deltas = {}
            for metric in ('success', 'weighted_fraction'):
                diff = np.array([float(groups[policy][s][metric]) -
                                 float(groups['resource_heuristic'][s][metric]) for s in seeds])
                mean, lower, upper = bootstrap_mean(diff, n_boot, rng)
                deltas[metric] = {'mean': mean, 'ci95': [lower, upper]}
            summary['paired_differences_vs_heuristic'][policy] = deltas
    return summary


def render_summary(summary, protocol):
    lines = ['# V4 frozen final evaluation', '',
             f"Missions: {summary['n_missions']} (seeds {protocol['seed_start']}–{protocol['seed_end']}); ",
             f"paired bootstrap: {summary['bootstrap_resamples']} resamples, 95% percentile CI.", '',
             '| Controller | Success (95% CI) | Weighted (95% CI) | Expired | ms/decision |',
             '|---|---:|---:|---:|---:|']
    for p in POLICIES:
        d = summary['controllers'][p]
        suc = d['success']; w = d['weighted_fraction']
        lines.append(f"| {p} | {suc['mean']:.1%} [{suc['ci95'][0]:.1%}, {suc['ci95'][1]:.1%}] | "
                     f"{w['mean']:.1%} [{w['ci95'][0]:.1%}, {w['ci95'][1]:.1%}] | "
                     f"{d['expired_targets']['mean']:.2f} | {d['mean_ms_per_decision']:.3f} |")
    lines += ['', '## Paired differences relative to resource heuristic', '',
              '| Controller | Success difference (95% CI) | Weighted difference (95% CI) |',
              '|---|---:|---:|']
    for p, diffs in summary['paired_differences_vs_heuristic'].items():
        s = diffs['success']; w = diffs['weighted_fraction']
        fmt = lambda v: f"{100*v['mean']:+.2f} pp [{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}]"
        lines.append(f'| {p} | {fmt(s)} | {fmt(w)} |')
    lines += ['', '## Interpretation guardrails', '',
              '- Bootstrap intervals reflect variability across mission seeds for the specified simulator configuration; they are not proof of real-world generalization.',
              '- Runs share initial mission seeds, not necessarily identical full stochastic trajectories.',
              '- Timing covers controller computation only on this machine, excludes simulation execution, and is not a hardware-independent benchmark.',
              '- PPO is deterministic at inference and runs on CPU; the model checkpoint was selected before this evaluation.',
              '- Lookahead is an approximate, non-oracle forecast (future obstacle changes and hidden target spawning suppressed).',
              '- This seed range must not be reused for model or hyperparameter selection after inspecting results.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ppo-model', required=True)
    p.add_argument('--episodes', type=int, default=400)
    p.add_argument('--seed-start', type=int, default=30000)
    p.add_argument('--out-dir', default='results/final_v4')
    p.add_argument('--option-horizon', type=int, default=3)
    p.add_argument('--deadlines', nargs=4, type=int, default=(12, 18, 24, 23))
    p.add_argument('--fuel', nargs=2, type=int, default=(16, 18))
    p.add_argument('--beam-width', type=int, default=4)
    p.add_argument('--bootstrap-resamples', type=int, default=10000)
    p.add_argument('--bootstrap-seed', type=int, default=20261120)
    p.add_argument('--overwrite', action='store_true', help='Explicitly replace earlier output files')
    args = p.parse_args()
    if args.episodes < 1 or args.seed_start < 0 or args.option_horizon < 1 or args.beam_width < 1 or args.bootstrap_resamples < 100:
        p.error('Invalid counts: episodes/horizon/beam width must be positive; bootstrap resamples >= 100')
    if any(x < 0 for x in args.fuel):
        p.error('Fuel budgets must be nonnegative')
    end = args.seed_start + args.episodes - 1
    for lo, hi in DEVELOPMENT_RANGES:
        if args.seed_start <= hi and end >= lo:
            p.error(f'Proposed evaluation range overlaps known development seeds {lo}–{hi}')
    model_path = Path(args.ppo_model)
    if not model_path.is_file():
        p.error(f'Model not found: {model_path}')
    out = Path(args.out_dir)
    paths = [out / name for name in ('episodes.csv', 'summary.json', 'summary.md', 'protocol.json')]
    if not args.overwrite and any(path.exists() for path in paths):
        p.error(f'Existing output in {out}; choose a new --out-dir or explicitly pass --overwrite')
    from sb3_contrib import MaskablePPO
    import torch
    model = MaskablePPO.load(str(model_path), device='cpu')
    model.policy.eval()
    torch.set_num_threads(1)
    protocol = {
        'seed_start': args.seed_start, 'seed_end': end, 'episodes': args.episodes,
        'controllers': list(POLICIES), 'option_horizon': args.option_horizon,
        'deadlines': list(args.deadlines), 'fuel_budgets': list(args.fuel),
        'lookahead_depth': 4, 'beam_width': args.beam_width,
        'ppo_model_path': str(model_path), 'ppo_model_sha256': file_sha256(model_path),
        'python': sys.version, 'platform': platform.platform(),
        'numpy': np.__version__, 'torch': torch.__version__,
        'bootstrap_seed': args.bootstrap_seed,
        'bootstrap_resamples': args.bootstrap_resamples,
    }
    for filename in ('final_evaluation_v4.py', 'mamc_rl/controllers_v4.py', 'mamc_rl/env_v4.py', 'mamc_rl/lookahead_v4.py'):
        path = Path(__file__).resolve().parent / filename
        if path.is_file():
            protocol.setdefault('code_sha256', {})[filename] = file_sha256(path)
    # Ensure model is compatible before embarking on lengthy search evaluations.
    dummy = HierarchicalResourceEnv(option_horizon=args.option_horizon, fuel_budgets=tuple(args.fuel), deadlines=tuple(args.deadlines))
    try:
        obs, _ = dummy.reset(seed=args.seed_start)
        prediction, _ = model.predict(obs, deterministic=True, action_masks=dummy.action_masks())
        if not dummy.action_masks()[int(prediction)]:
            raise ValueError('The PPO checkpoint selected an illegal action on the compatibility check')
    finally:
        dummy.close()
    out.mkdir(parents=True, exist_ok=True)
    with paths[3].open('w', encoding='utf-8') as f:
        json.dump(protocol, f, indent=2)
    rows = []
    # Stream episode rows immediately to preserve partial work on a long run.
    with paths[0].open('w', newline='', encoding='utf-8') as f:
        writer = None
        for i, seed in enumerate(range(args.seed_start, end + 1), 1):
            for policy in POLICIES:
                rec = run_episode(policy, seed, model, args)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(rec))
                    writer.writeheader()
                writer.writerow(rec)
                rows.append(rec)
            if i % 25 == 0 or i == args.episodes:
                f.flush()
                print(f'Completed {i}/{args.episodes} paired missions', flush=True)
    summary = summarize(rows, args.bootstrap_resamples, args.bootstrap_seed)
    with paths[1].open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    markdown = render_summary(summary, protocol)
    paths[2].write_text(markdown, encoding='utf-8')
    print('\n' + markdown)
    print(f'Outputs: {out}')


if __name__ == '__main__':
    main()
