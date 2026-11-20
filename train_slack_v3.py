"""Train V3 MaskablePPO with eight agent-target deadline-slack features.

Example:
  python train_slack_v3.py --timesteps 50000 --seed 42 --output models/v3_slack_seed42_50k
  python train_slack_v3.py --primitive-budget 112000 --seed 42 --output models/v3_slack_112k

This mirrors train_masked_v3.py; PPO timesteps are high-level decisions and
primitive interactions are counted across all workers. A primitive budget may
overshoot by one collection batch. Preserve the baseline checkpoints.
"""
import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from mamc_rl.controllers_v3 import HierarchicalDeadlineEnv
from mamc_rl.slack_observation_v3 import SlackObservationV3
from train_masked_v3 import PrimitiveStepCounter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument('--timesteps', type=int, default=None)
    limits.add_argument('--primitive-budget', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--option-horizon', type=int, default=3)
    parser.add_argument('--deadlines', type=int, nargs=4, default=(12, 18, 24, 23))
    parser.add_argument('--output', default='models/v3_slack_seed42_50k')
    args = parser.parse_args()
    if args.timesteps is not None and args.timesteps < 1:
        parser.error('--timesteps must be positive')
    if args.primitive_budget is not None and args.primitive_budget < 1:
        parser.error('--primitive-budget must be positive')
    if args.option_horizon < 1:
        parser.error('--option-horizon must be positive')

    def factory():
        env = HierarchicalDeadlineEnv(
            option_horizon=args.option_horizon, deadlines=tuple(args.deadlines)
        )
        return Monitor(SlackObservationV3(env))

    env = make_vec_env(factory, n_envs=4, seed=args.seed)
    counter = PrimitiveStepCounter(args.primitive_budget)
    try:
        model = MaskablePPO(
            'MlpPolicy', env, verbose=1, seed=args.seed,
            n_steps=256, batch_size=256, learning_rate=3e-4,
            gamma=0.99, policy_kwargs={'net_arch': [128, 128]},
        )
        cap = (args.timesteps if args.timesteps is not None else
               args.primitive_budget * 2 if args.primitive_budget is not None else 10000)
        model.learn(total_timesteps=cap, callback=counter)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output))
        print(f'Saved slack-enhanced V3 model: {output}.zip')
        print(f'High-level training decisions: {model.num_timesteps}')
        print(f'Primitive environment interactions (all workers): {counter.primitive_steps}')
        print(f'Deadlines: {tuple(args.deadlines)}; option horizon: {args.option_horizon}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
