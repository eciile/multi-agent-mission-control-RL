"""Train MaskablePPO for deadline-constrained V3 mission assignment.

Examples:
  python train_masked_v3.py --timesteps 10000 --seed 42 --output models/v3_masked_seed42
  python train_masked_v3.py --primitive-budget 100000 --seed 42 --output models/v3_masked_100k

A PPO timestep is one high-level decision. Primitive interactions are counted
across all vectorized workers; the callback stops at a rollout boundary and
may exceed a requested primitive budget slightly.
"""
import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from mamc_rl.controllers_v3 import HierarchicalDeadlineEnv


class PrimitiveStepCounter(BaseCallback):
    def __init__(self, budget=None):
        super().__init__()
        self.budget = budget
        self.primitive_steps = 0

    def _on_step(self) -> bool:
        self.primitive_steps += sum(
            int(info.get('low_level_steps', 0)) for info in self.locals.get('infos', [])
        )
        return self.budget is None or self.primitive_steps < self.budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument('--timesteps', type=int, default=None,
                        help='High-level PPO decisions across all workers (default: 10000)')
    limits.add_argument('--primitive-budget', type=int, default=None,
                        help='Approximate primitive interaction budget across workers')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--option-horizon', type=int, default=3)
    parser.add_argument('--deadlines', type=int, nargs=4, default=(12, 18, 24, 23))
    parser.add_argument('--output', default='models/v3_masked_seed42')
    args = parser.parse_args()

    if args.timesteps is not None and args.timesteps < 1:
        parser.error('--timesteps must be positive')
    if args.primitive_budget is not None and args.primitive_budget < 1:
        parser.error('--primitive-budget must be positive')
    if args.option_horizon < 1:
        parser.error('--option-horizon must be positive')

    def factory():
        return Monitor(HierarchicalDeadlineEnv(
            option_horizon=args.option_horizon,
            deadlines=tuple(args.deadlines),
        ))

    env = make_vec_env(factory, n_envs=4, seed=args.seed)
    counter = PrimitiveStepCounter(args.primitive_budget)
    try:
        model = MaskablePPO(
            'MlpPolicy', env, verbose=1, seed=args.seed,
            n_steps=256, batch_size=256, learning_rate=3e-4,
            gamma=0.99, policy_kwargs={'net_arch': [128, 128]},
        )
        high_level_cap = (
            args.timesteps if args.timesteps is not None else
            args.primitive_budget * 2 if args.primitive_budget is not None else 10000
        )
        model.learn(total_timesteps=high_level_cap, callback=counter)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output))
        print(f'Saved V3 masked PPO model: {output}.zip')
        print(f'High-level training decisions: {model.num_timesteps}')
        print(f'Primitive environment interactions (all workers): {counter.primitive_steps}')
        print(f'Deadlines: {tuple(args.deadlines)}; option horizon: {args.option_horizon}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
