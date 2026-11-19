"""Train PPO on centralized joint actions. Example: python train.py --timesteps 50000"""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from mamc_rl import MissionGridEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="models/ppo_mamc")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(lambda: Monitor(MissionGridEnv()), n_envs=4, seed=args.seed)
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, n_steps=256,
                batch_size=256, learning_rate=3e-4, gamma=0.99,
                policy_kwargs={"net_arch": [128, 128]})
    model.learn(total_timesteps=args.timesteps)
    model.save(args.output)
    env.close()
    print(f"Saved to {args.output}.zip")


if __name__ == "__main__":
    main()
