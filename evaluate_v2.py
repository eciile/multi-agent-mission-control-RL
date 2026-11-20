"""Evaluate V2 non-learning baselines across four reproducible mission scenarios.

Run from repository root:
    python evaluate_v2.py --episodes 100 --seed-start 1000

Each controller is evaluated on the same environment seeds in each scenario.
The random controller uses a separate, seeded generator so its action choices
never consume the environment RNG used for obstacle and target generation.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mamc_rl.env_v2 import DynamicMissionEnv
from mamc_rl.policies_v2 import heuristic_action_v2


SCENARIOS = {
    "static": dict(dynamic_obstacles=False, changing_targets=False),
    "obstacles_only": dict(dynamic_obstacles=True, changing_targets=False),
    "targets_only": dict(dynamic_obstacles=False, changing_targets=True),
    "combined": dict(dynamic_obstacles=True, changing_targets=True),
}
POLICIES = ("random", "heuristic")
FIELDS = (
    "scenario", "policy", "seed", "success", "steps", "return",
    "targets_completed", "targets_total", "weighted_completed", "weighted_total",
    "weighted_fraction", "collisions", "invalid_actions",
    "obstacle_changes", "target_spawn_events",
)


def run_episode(scenario: str, policy: str, seed: int) -> dict:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")

    env = DynamicMissionEnv(**SCENARIOS[scenario])
    # Keep random actions independent of the environment's random events.
    policy_rng = np.random.default_rng(seed + 12345)
    try:
        env.reset(seed=seed)
        total_return = 0.0
        collisions = invalid_actions = obstacle_changes = target_spawns = 0
        while True:
            if policy == "random":
                action = int(policy_rng.integers(env.action_space.n))
            else:
                action = int(heuristic_action_v2(env))
            _, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            collisions += int(info["collision"])
            invalid_actions += int(info["invalid_actions"])
            obstacle_changes += int(info["obstacles_changed"])
            target_spawns += int(info["target_spawned"])
            if terminated or truncated:
                weighted_total = int(info["weighted_total"])
                return {
                    "scenario": scenario,
                    "policy": policy,
                    "seed": seed,
                    "success": int(info["success"]),
                    "steps": int(info["steps"]),
                    "return": round(total_return, 4),
                    "targets_completed": int(info["targets_completed"]),
                    "targets_total": int(env.active.sum()),
                    "weighted_completed": int(info["weighted_completed"]),
                    "weighted_total": weighted_total,
                    "weighted_fraction": round(int(info["weighted_completed"]) / weighted_total, 6)
                    if weighted_total else 0.0,
                    "collisions": collisions,
                    "invalid_actions": invalid_actions,
                    "obstacle_changes": obstacle_changes,
                    "target_spawn_events": target_spawns,
                }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100, help="Number of seeds per policy/scenario")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--output", default="results/baselines_v2.csv")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    rows = [run_episode(scenario, policy, seed)
            for scenario in SCENARIOS for policy in POLICIES
            for seed in range(args.seed_start, args.seed_start + args.episodes)]

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {path} ({len(rows)} episode records)")
    print(f"{'Scenario':16s} {'Policy':10s} {'Success':>9s} {'Steps':>8s} {'Return':>9s} {'Weighted':>9s} {'Coll.':>7s} {'Invalid':>8s}")
    print("-" * 85)
    for scenario in SCENARIOS:
        for policy in POLICIES:
            subset = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            mean = lambda field: float(np.mean([r[field] for r in subset]))
            print(f"{scenario:16s} {policy:10s} {mean('success'):8.1%} "
                  f"{mean('steps'):8.1f} {mean('return'):9.2f} "
                  f"{mean('weighted_fraction'):8.1%} {mean('collisions'):7.2f} "
                  f"{mean('invalid_actions'):8.2f}")
    print("\nNote: 'static' and 'obstacles_only' have 3 targets (priority total 6); ")
    print("      'targets_only' and 'combined' have 4 targets (priority total 9).")
    print("      Compare matched scenario settings and use weighted fraction across scenarios;")
    print("      episode returns and step counts are not directly comparable across target counts.")


if __name__ == "__main__":
    main()
