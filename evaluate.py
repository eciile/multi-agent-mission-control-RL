"""Evaluate all policies on identical seeded scenarios and export raw CSV."""
import argparse
import csv
from pathlib import Path
import numpy as np
from mamc_rl import MissionGridEnv
from mamc_rl.policies import heuristic_action


def run_episode(kind, seed, model=None):
    env = MissionGridEnv()
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 12345)
    total, collisions, invalid = 0.0, 0, 0
    while True:
        if kind == "random":
            action = int(rng.integers(25))
        elif kind == "heuristic":
            action = int(heuristic_action(env))
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        collisions += int(info["collision"])
        invalid += info["invalid_actions"]
        if term or trunc:
            return {"policy": kind, "seed": seed, "success": int(info["success"]),
                    "steps": info["steps"], "reward": round(total, 4),
                    "targets_completed": info["targets_completed"],
                    "collisions": collisions, "invalid_actions": invalid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--model", type=str, default="models/ppo_mamc.zip")
    parser.add_argument("--output", type=str, default="results/evaluation.csv")
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()
    model = None
    policies = ["random", "heuristic"]
    if not args.baselines_only:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
        policies.append("ppo")
    rows = [run_episode(kind, seed, model) for kind in policies
            for seed in range(args.seed_start, args.seed_start + args.episodes)]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output}")
    for kind in policies:
        block = [r for r in rows if r["policy"] == kind]
        print(f"{kind:10s}: success={np.mean([r['success'] for r in block]):.1%} "
              f"mean_steps={np.mean([r['steps'] for r in block]):.1f} "
              f"mean_reward={np.mean([r['reward'] for r in block]):.2f}")


if __name__ == "__main__":
    main()
