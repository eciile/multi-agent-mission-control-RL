"""V4 lookahead depth ablation on paired seeds, with a fixed beam width.

Uses the existing non-oracle lookahead_v4 planner and unchanged V4 simulator.
Search depth changes; evaluation scoring, beam width, and missions do not.

Examples:
    python compare_lookahead_depths_v4.py --episodes 5 --seed-start 2000
    python compare_lookahead_depths_v4.py --episodes 200 --seed-start 2000 --beam-width 4
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from compare_lookahead_v4 import run_episode


def summarize(rows: list[dict], policies: list[str]) -> None:
    print('Policy           Success  Weighted  Expired  Steps  Mean ms/decision  Total planning s  Spawned')
    for policy in policies:
        group = [r for r in rows if r['policy'] == policy]
        if not group:
            continue
        mean = lambda col: float(np.mean([float(r[col]) for r in group]))
        total_seconds = sum(float(r['planning_seconds']) for r in group)
        total_decisions = sum(int(r['high_level_decisions']) for r in group)
        # Weighted by decisions, not episodes; each decision contributes equally.
        ms_per_decision = 1000.0 * total_seconds / total_decisions
        print(f'{policy:<17}{mean("success"):>6.1%}   {mean("weighted_fraction"):>6.1%}'
              f'     {mean("expired_targets"):>5.2f}  {mean("primitive_steps"):>5.1f}'
              f'         {ms_per_decision:>9.2f}         {total_seconds:>9.2f}'
              f'     {mean("urgent_spawned"):>5.1%}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed-start', type=int, default=2000)
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 4, 6])
    parser.add_argument('--beam-width', type=int, default=4)
    parser.add_argument('--fuel', type=int, nargs=2, default=(16, 18))
    parser.add_argument('--output', default='results/v4_lookahead_depths.csv')
    args = parser.parse_args()

    if args.episodes < 1 or args.beam_width < 1:
        parser.error('--episodes and --beam-width must be positive')
    if not args.depths or any(d < 1 for d in args.depths) or len(set(args.depths)) != len(args.depths):
        parser.error('--depths must contain unique positive integers')
    if any(f < 0 for f in args.fuel):
        parser.error('fuel budgets must be nonnegative')

    depths = args.depths
    names = ['resource_heuristic'] + [f'lookahead_d{depth}' for depth in depths]
    rows: list[dict] = []
    fuel = tuple(args.fuel)
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        # Same initial seed and physical environment for all controllers.
        baseline = run_episode('resource_heuristic', seed, fuel, depths[0], args.beam_width)
        baseline['depth'] = 0
        baseline['beam_width'] = 0
        rows.append(baseline)
        for depth in depths:
            record = run_episode('lookahead', seed, fuel, depth, args.beam_width)
            record['policy'] = f'lookahead_d{depth}'
            record['depth'] = depth
            record['beam_width'] = args.beam_width
            rows.append(record)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Saved {output} ({len(rows)} episode records)')
    print(f'Paired seeds: {args.seed_start}..{args.seed_start + args.episodes - 1}; '
          f'beam width: {args.beam_width}; depths: {depths}')
    summarize(rows, names)
    print('Controller computation times are hardware-dependent and include Python simulation copies.')
    print('Lookahead suppresses unknown future obstacles/urgent spawning in forecasts; it is not optimal.')
    print('Search depth is in high-level decisions, not primitive movements.')


if __name__ == '__main__':
    main()
