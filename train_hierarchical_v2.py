"""Train high-level PPO (mission assignment) with BFS navigation.

Example:
  python train_hierarchical_v2.py --timesteps 10000 --output models/hier_smoke
  python train_hierarchical_v2.py --timesteps 100000 --output models/hier_v2
"""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from mamc_rl.hierarchical_v2 import HierarchicalMissionEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--option-horizon', type=int, default=3)
    parser.add_argument('--output', default='models/hier_v2')
    parser.add_argument('--check-env', action='store_true')
    args = parser.parse_args()
    if args.timesteps < 1:
        parser.error('--timesteps must be positive')
    if args.check_env:
        env = HierarchicalMissionEnv(option_horizon=args.option_horizon)
        try:
            check_env(env, warn=True)
        finally:
            env.close()

    def factory():
        return Monitor(HierarchicalMissionEnv(option_horizon=args.option_horizon))

    vec = make_vec_env(factory, n_envs=4, seed=args.seed)
    try:
        model = PPO('MlpPolicy', vec, verbose=1, seed=args.seed,
                    n_steps=256, batch_size=256, learning_rate=3e-4,
                    gamma=0.99, policy_kwargs={'net_arch': [128, 128]})
        model.learn(total_timesteps=args.timesteps)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
        print(f'Saved hierarchical PPO model: {out}.zip')
        print('NOTE: timesteps count high-level decisions, not primitive movements.')
    finally:
        vec.close()


if __name__ == '__main__':
    main()
