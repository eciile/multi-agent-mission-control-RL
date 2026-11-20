"""Train centralized PPO on V2 dynamic cooperative missions.

Examples (run from the repository root):
    python train_v2.py --timesteps 10000 --output models/ppo_v2_smoke
    python train_v2.py --timesteps 100000 --output models/ppo_v2_combined

The trained policy selects one of 25 joint actions for two agents.
This is centralized RL, not decentralized multi-agent RL.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from mamc_rl.env_v2 import DynamicMissionEnv

SCENARIOS = {
    "static": dict(dynamic_obstacles=False, changing_targets=False),
    "obstacles_only": dict(dynamic_obstacles=True, changing_targets=False),
    "targets_only": dict(dynamic_obstacles=False, changing_targets=True),
    "combined": dict(dynamic_obstacles=True, changing_targets=True),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=SCENARIOS, default="combined")
    parser.add_argument("--output", default="models/ppo_v2_combined")
    parser.add_argument("--check-env", action="store_true", help="Validate Gymnasium API before training")
    args = parser.parse_args()
    if args.timesteps < 1:
        parser.error("--timesteps must be positive")

    config = SCENARIOS[args.scenario]
    if args.check_env:
        test_env = DynamicMissionEnv(**config)
        try:
            check_env(test_env, warn=True)
        finally:
            test_env.close()

    def factory():
        return Monitor(DynamicMissionEnv(**config))

    env = make_vec_env(factory, n_envs=4, seed=args.seed)
    try:
        model = PPO(
            "MlpPolicy", env, verbose=1, seed=args.seed,
            n_steps=256, batch_size=256, learning_rate=3e-4,
            gamma=0.99, policy_kwargs={"net_arch": [128, 128]},
        )
        model.learn(total_timesteps=args.timesteps)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output))
        print(f"Saved PPO model: {output}.zip")
        print(f"Training scenario: {args.scenario}; requested timesteps: {args.timesteps}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
