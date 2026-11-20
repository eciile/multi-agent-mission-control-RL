"""Audit V4 global-beam labels versus per-root lookahead values.

Each permitted first assignment receives its own depth-limited beam search,
so roots are not eliminated by competition with other roots. Scores are
surrogate heuristic scores, NOT calibrated returns or optimal Q-values.
Forecasts suppress unknown future urgent-target spawning and obstacle changes.

Usage:
    python audit_root_scores_v4.py --episodes 5 --max-decisions 3
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np

from mamc_rl.controllers_v4 import HierarchicalResourceEnv
from mamc_rl.lookahead_v4 import (
    _Node, _candidates, _evaluation, _forecast_copy, lookahead_action,
)


def root_preserving_scores(env: HierarchicalResourceEnv, depth: int = 4,
                           beam_width: int = 4) -> np.ndarray:
    """Return 25 scores; NaN means a root is excluded by planner candidates.

    Within each root, keep at most beam_width forecast nodes per depth.  This
    makes every admissible root receive a depth-d search budget.  The score is
    the best surrogate terminal/leaf score found for that root, not a Q-value.
    """
    if depth < 1 or beam_width < 1:
        raise ValueError('depth and beam_width must be positive')
    scores = np.full(25, np.nan, dtype=np.float64)
    candidates = _candidates(env)
    if not candidates:
        raise RuntimeError('No planner-admissible high-level actions')

    for first in candidates:
        sim = _forecast_copy(env)
        _, _, done, truncated, _ = sim.step(first)
        frontier = [_Node(sim, first, _evaluation(sim), bool(done or truncated))]
        for _ in range(1, depth):
            expanded: list[_Node] = []
            for node in frontier:
                if node.finished:
                    expanded.append(node)
                    continue
                for action in _candidates(node.env):
                    child = deepcopy(node.env)
                    _, _, done, truncated, _ = child.step(action)
                    expanded.append(_Node(child, first, _evaluation(child),
                                          bool(done or truncated)))
            if not expanded:
                break
            expanded.sort(key=lambda node: -node.score)
            frontier = expanded[:beam_width]
        scores[first] = max(node.score for node in frontier)
    return scores


def audit(episodes: int, seed_start: int, max_decisions: int, depth: int,
          beam_width: int, output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    times: list[float] = []
    env = HierarchicalResourceEnv(option_horizon=3)
    try:
        for seed in range(seed_start, seed_start + episodes):
            env.reset(seed=seed)
            for decision in range(max_decisions):
                mask = np.asarray(env.action_masks(), dtype=np.bool_)
                old_action = int(lookahead_action(env, depth=depth, beam_width=beam_width))
                start = perf_counter()
                scores = root_preserving_scores(env, depth=depth, beam_width=beam_width)
                times.append((perf_counter() - start) * 1000.0)
                choices = np.flatnonzero(np.isfinite(scores))
                # np.argmax uses lowest action id to break an exact score tie.
                root_action = int(choices[np.argmax(scores[choices])])
                sorted_scores = np.sort(scores[choices])[::-1]
                margin = (float(sorted_scores[0] - sorted_scores[1])
                          if len(sorted_scores) > 1 else float('nan'))
                if not mask[old_action] or not mask[root_action]:
                    raise AssertionError('Planner selected an illegal action')
                record = {
                    'seed': seed, 'decision': decision,
                    'legal_count': int(mask.sum()),
                    'admissible_count': len(choices),
                    'global_beam_action': old_action,
                    'root_best_action': root_action,
                    'agreement': int(old_action == root_action),
                    'global_action_root_score': float(scores[old_action]),
                    'root_best_score': float(scores[root_action]),
                    'root_score_gap': float(scores[root_action] - scores[old_action]),
                    'best_second_margin': margin,
                }
                record.update({f'root_score_{i}': float(scores[i]) for i in range(25)})
                rows.append(record)
                _, _, terminated, truncated, _ = env.step(old_action)
                if terminated or truncated:
                    break
    finally:
        env.close()

    if not rows:
        raise RuntimeError('No decision states audited')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    agreement = np.mean([row['agreement'] for row in rows])
    gaps = np.array([row['root_score_gap'] for row in rows])
    margins = np.array([row['best_second_margin'] for row in rows])
    print(f'Saved {path} ({len(rows)} decision states)')
    print(f'Global-beam versus per-root action agreement: {agreement:.1%}')
    print(f'Mean roots scored/state: {np.mean([r["admissible_count"] for r in rows]):.2f}')
    print(f'Mean best-versus-global score gap: {np.mean(gaps):.3f}')
    print(f'Mean top-two score margin: {np.nanmean(margins):.3f}'
          if np.isfinite(margins).any() else 'Top-two margin: unavailable')
    print(f'Mean per-root labeling time: {np.mean(times):.1f} ms/state')
    print('WARNING: root scores are heuristic surrogate scores from approximate forecasts.')
    print('Per-root selection is a DIFFERENT planner than the existing global beam.')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--seed-start', type=int, default=16000)
    p.add_argument('--max-decisions', type=int, default=3)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--beam-width', type=int, default=4)
    p.add_argument('--output', default='results/v4_root_score_audit.csv')
    args = p.parse_args()
    if min(args.episodes, args.max_decisions, args.depth, args.beam_width) < 1:
        p.error('Episodes, max decisions, depth, and width must all be positive')
    if set(range(args.seed_start, args.seed_start + args.episodes)) & set(range(2000, 2200)):
        p.error('Do not audit final evaluation seeds 2000-2199 while designing supervision')
    audit(args.episodes, args.seed_start, args.max_decisions,
          args.depth, args.beam_width, args.output)


if __name__ == '__main__':
    main()
