"""Paired comparison: original deterministic PPO vs inference-only feasibility shield.

Run from repository root:
  python evaluate_shield_v2.py --model models/ppo_v2_combined.zip --episodes 100

No retraining. The policy parameters are identical in both conditions.
Metrics are reported on matched initial seeds; subsequent dynamic obstacle
placements may diverge because agent trajectories affect eligible cells.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from mamc_rl.env_v2 import DynamicMissionEnv
from mamc_rl.shield_v2 import select_shielded_action, valid_joint_mask

FIELDS = ["mode", "seed", "success", "steps", "return", "weighted_fraction",
          "targets_completed", "collisions", "invalid_actions", "shield_overrides"]


def policy_probabilities(model: PPO, obs: np.ndarray) -> np.ndarray:
    """Get 25 discrete-action probabilities through the SB3 policy API."""
    import torch

    with torch.no_grad():
        tensor_obs, _ = model.policy.obs_to_tensor(obs)
        distribution = model.policy.get_distribution(tensor_obs)
        probabilities = distribution.distribution.probs.detach().cpu().numpy()
    if probabilities.shape != (1, 25):
        raise ValueError(f"Expected 25 categorical probabilities, got {probabilities.shape}")
    return probabilities[0]


def run_episode(model: PPO, mode: str, seed: int) -> dict:
    env = DynamicMissionEnv(dynamic_obstacles=True, changing_targets=True)
    try:
        obs, _ = env.reset(seed=seed)
        total_return = 0.0
        invalid = collisions = overrides = 0
        while True:
            probs = policy_probabilities(model, obs)
            original_action = int(np.argmax(probs))
            action = original_action
            if mode == "shielded":
                mask = valid_joint_mask(env.agents, env.obstacles)
                action = select_shielded_action(probs, mask)
                overrides += int(action != original_action)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            invalid += int(info["invalid_actions"])
            collisions += int(info["collision"])
            if terminated or truncated:
                total = int(info["weighted_total"])
                return {
                    "mode": mode,
                    "seed": seed,
                    "success": int(info["success"]),
                    "steps": int(info["steps"]),
                    "return": round(total_return, 4),
                    "weighted_fraction": round(int(info["weighted_completed"]) / total, 6) if total else 0.0,
                    "targets_completed": int(info["targets_completed"]),
                    "collisions": collisions,
                    "invalid_actions": invalid,
                    "shield_overrides": overrides,
                }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/ppo_v2_combined.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--output", default="results/ppo_shield_comparison_v2.csv")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not Path(args.model).is_file():
        parser.error(f"Model file not found: {args.model}")

    model = PPO.load(args.model, device="cpu")
    rows = [run_episode(model, mode, seed)
            for mode in ("original", "shielded")
            for seed in range(args.seed_start, args.seed_start + args.episodes)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {output} ({len(rows)} episode records)")
    print("All runs use the same PPO checkpoint and deterministic action ranking.")
    print(f"{'Mode':12s} {'Success':>8s} {'Steps':>8s} {'Return':>9s} {'Weighted':>10s} {'Coll.':>8s} {'Invalid':>8s} {'Overrides':>10s}")
    print("-" * 85)
    for mode in ("original", "shielded"):
        subset = [row for row in rows if row["mode"] == mode]
        mean = lambda key: float(np.mean([row[key] for row in subset]))
        print(f"{mode:12s} {mean('success'):7.1%} {mean('steps'):8.1f} "
              f"{mean('return'):9.2f} {mean('weighted_fraction'):9.1%} "
              f"{mean('collisions'):8.2f} {mean('invalid_actions'):8.2f} "
              f"{mean('shield_overrides'):10.2f}")


if __name__ == "__main__":
    main()
