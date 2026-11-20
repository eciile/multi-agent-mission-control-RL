"""V4 diagnostic: legal versus reachable assignments, state sensitivity, and optional search.

Does not change the environment or train policies. Only currently visible targets are
considered when checking individual reachability. A feasible *pair* of assignments
is NOT a certificate that both can be completed: movement conflicts and future
obstacles remain unmodeled in the lower bound.

Optional offline beam search intentionally simulates future events on deep copies.
It can establish a *witness* for an initial seed (a successfully simulated plan),
but it is not a deployable non-oracle controller and failure proves nothing.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np

from mamc_rl.controllers_v4 import HierarchicalResourceEnv, resource_heuristic_action
from mamc_rl.policies_v2 import _distance_map


def visible_feasible_targets(env: HierarchicalResourceEnv) -> tuple[set[int], set[int]]:
    """Return individual feasible target *choices* (1-4) for agents 0 and 1."""
    b = env.base
    distances = {}
    out = [set(), set()]
    for t in range(4):
        if not b.active[t] or b.completed[t] or b.expired[t]:
            continue
        distances[t] = _distance_map(tuple(map(int, b.targets[t])), set(b.obstacles))
        for a in range(2):
            if not b.capabilities[a, t]:
                continue
            d = distances[t].get(tuple(map(int, b.agents[a])))
            if d is not None and d <= int(b.fuel[a]) and d <= int(b.deadlines[t] - b.steps):
                out[a].add(t + 1)
    return out[0], out[1]


def feasible_joint_actions(env: HierarchicalResourceEnv) -> list[int]:
    """Legal actions whose non-idle assignments are individually reachable now."""
    f0, f1 = visible_feasible_targets(env)
    result = []
    for action in np.flatnonzero(env.action_masks()):
        a, b = divmod(int(action), 5)
        if (a == 0 or a in f0) and (b == 0 or b in f1):
            result.append(int(action))
    return result


def state_signature(env: HierarchicalResourceEnv) -> tuple:
    """Observable realized state after an action, excluding the selected action itself."""
    b = env.base
    return (int(b.steps), tuple(map(tuple, b.agents)), tuple(map(int, b.fuel)),
            tuple(map(bool, b.completed)), tuple(map(bool, b.expired)),
            tuple(map(bool, b.active)))


def transition_sensitivity(env: HierarchicalResourceEnv) -> dict:
    """Exact one-macro-action successors across all currently legal assignments.

    Deep copies start with identical RNG state; future draws can nonetheless
    diverge when a different action changes the simulator's call sequence.
    """
    mask = env.action_masks()
    legal = [int(a) for a in np.flatnonzero(mask)]
    states = {}
    gains = {}
    for action in legal:
        clone = deepcopy(env)
        before = int(clone.base.priorities[clone.base.completed].sum())
        _, _, _, _, info = clone.step(action)
        states[action] = state_signature(clone)
        gains[action] = int(clone.base.priorities[clone.base.completed].sum()) - before
        clone.close()
    unique = len(set(states.values()))
    return dict(unique_successors=unique, sensitive=int(unique > 1),
                distinct_immediate_gains=len(set(gains.values())),
                noop_action_changes=int(states.get(0) != states.get(legal[-1])) if legal else 0)


def audit_state(env: HierarchicalResourceEnv, seed: int, decision: int) -> dict:
    mask = env.action_masks()
    legal = [int(a) for a in np.flatnonzero(mask)]
    feasible = feasible_joint_actions(env)
    f0, f1 = visible_feasible_targets(env)
    meaningful = [a for a in feasible if a != 0]
    duplicated = [a for a in legal if a // 5 == a % 5 and a % 5 != 0]
    sensitivity = transition_sensitivity(env)
    b = env.base
    return dict(seed=seed, decision=decision, primitive_step=int(b.steps),
                legal_joint_actions=len(legal), reachable_joint_actions=len(feasible),
                actionable_reachable=len(meaningful), duplicated_legal=len(duplicated),
                choices_agent0=len(f0), choices_agent1=len(f1),
                active_objectives=int((b.active & ~b.completed & ~b.expired).sum()),
                fuel0=int(b.fuel[0]), fuel1=int(b.fuel[1]), **sensitivity)


def collect_states(seed: int, n_decisions: int, horizon: int) -> list[dict]:
    env = HierarchicalResourceEnv(option_horizon=horizon)
    rows = []
    try:
        env.reset(seed=seed)
        for decision in range(n_decisions):
            rows.append(audit_state(env, seed, decision))
            action = resource_heuristic_action(env)
            _, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        return rows
    finally:
        env.close()


def search_score(env: HierarchicalResourceEnv) -> float:
    """Search ranking heuristic; not an optimality bound."""
    b = env.base
    gained = float(b.priorities[b.completed].sum())
    missed = float(b.priorities[b.expired].sum())
    f0, f1 = visible_feasible_targets(env)
    possible = set(f0) | set(f1)
    potential = sum(float(b.priorities[t - 1]) for t in possible)
    remaining_fuel = float(b.fuel.sum())
    return 100.0 * gained - 120.0 * missed + 10.0 * potential + 0.1 * remaining_fuel - 0.01 * b.steps


def offline_witness_search(seed: int, horizon: int = 3, width: int = 12,
                           depth: int = 14) -> dict:
    """Offline, simulator-privileged bounded beam search for a success witness.

    Searches realized futures with cloned RNG, including future spawned targets.
    Not a valid policy baseline. If no solution is found, no impossibility claim.
    """
    if width < 1 or depth < 1:
        raise ValueError('width and depth must be positive')
    env = HierarchicalResourceEnv(option_horizon=horizon)
    env.reset(seed=seed)
    frontier = [(env, ())]
    expansions = 0
    best_weighted = 0
    try:
        for layer in range(depth):
            candidates = []
            for node, path in frontier:
                legal = feasible_joint_actions(node)
                # Include all actually legal assignments: a locally unreachable
                # goal might still lead to beneficial movement. Preserve idle.
                legal = sorted(set(legal) | {int(a) for a in np.flatnonzero(node.action_masks())})
                for action in legal:
                    next_env = deepcopy(node)
                    _, _, term, trunc, info = next_env.step(action)
                    expansions += 1
                    best_weighted = max(best_weighted, int(info['weighted_completed']))
                    new_path = path + (action,)
                    if info['success']:
                        next_env.close()
                        return dict(seed=seed, witness_found=1, depth_used=layer + 1,
                                    expansions=expansions, best_weighted=best_weighted,
                                    action_sequence=' '.join(map(str, new_path)))
                    if not (term or trunc):
                        candidates.append((next_env, new_path))
                    else:
                        next_env.close()
            if not candidates:
                break
            candidates.sort(key=lambda pair: search_score(pair[0]), reverse=True)
            kept = candidates[:width]
            for discarded, _ in candidates[width:]:
                discarded.close()
            # Discard references to prior frontier (no more exploration there).
            for old, _ in frontier:
                old.close()
            frontier = kept
        return dict(seed=seed, witness_found=0, depth_used=min(depth, layer + 1),
                    expansions=expansions, best_weighted=best_weighted,
                    action_sequence='')
    finally:
        for node, _ in frontier:
            node.close()


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {path} ({len(rows)} rows)')


def print_summary(rows):
    def avg(key): return float(np.mean([r[key] for r in rows]))
    initial = [r for r in rows if r['decision'] == 0]
    print(f'Audited {len(rows)} high-level states across {len(initial)} initial seeds.')
    print(f'Initial legal joint actions: {avg_key(initial, "legal_joint_actions"):.2f}; '
          f'individually reachable joint actions: {avg_key(initial, "reachable_joint_actions"):.2f}')
    print(f'All visited states: legal={avg("legal_joint_actions"):.2f}, '
          f'individually reachable={avg("reachable_joint_actions"):.2f}, '
          f'unique next states={avg("unique_successors"):.2f}')
    print(f'States with >1 distinct immediate successor: {100*avg("sensitive"):.1f}%')
    print(f'States with >1 possible immediate priority gain: '
          f'{100*np.mean([r["distinct_immediate_gains"]>1 for r in rows]):.1f}%')
    print(f'Initial states with >1 feasible non-idle joint assignment: '
          f'{100*np.mean([r["actionable_reachable"]>1 for r in initial]):.1f}%')
    print('NOTE: Individual reachability ignores movement conflicts and future obstacles.')


def avg_key(rows, key):
    return float(np.mean([r[key] for r in rows]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--seed-start', type=int, default=2000)
    p.add_argument('--decisions-per-episode', type=int, default=4)
    p.add_argument('--option-horizon', type=int, default=3)
    p.add_argument('--output', default='results/v4_feasibility_audit.csv')
    p.add_argument('--search-seeds', type=int, nargs='*', default=[])
    p.add_argument('--search-width', type=int, default=12)
    p.add_argument('--search-depth', type=int, default=14)
    p.add_argument('--search-output', default='results/v4_offline_search_witnesses.csv')
    args = p.parse_args()
    if args.episodes < 1 or args.decisions_per_episode < 1 or args.option_horizon < 1:
        p.error('episodes, decisions-per-episode and option-horizon must be positive')
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        rows.extend(collect_states(seed, args.decisions_per_episode, args.option_horizon))
    write_csv(args.output, rows)
    print_summary(rows)
    if args.search_seeds:
        results = []
        for seed in args.search_seeds:
            result = offline_witness_search(seed, args.option_horizon,
                                            args.search_width, args.search_depth)
            results.append(result)
            status = 'WITNESS FOUND' if result['witness_found'] else 'NO WITNESS FOUND'
            print(f'Seed {seed}: {status}; expansions={result["expansions"]}; '
                  f'best weighted={result["best_weighted"]}/9')
        write_csv(args.search_output, results)
        print('WARNING: Search simulates hidden future events on copies; it is an OFFLINE')
        print('oracle-assisted feasibility probe, NOT a deployable planning baseline.')
        print('A failed search does NOT prove a mission is impossible.')


if __name__ == '__main__':
    main()
