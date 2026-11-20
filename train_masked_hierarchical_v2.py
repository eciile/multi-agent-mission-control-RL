"""Train MaskablePPO on hierarchical mission assignments.

Examples:
  python train_masked_hierarchical_v2.py --timesteps 10000 --output models/hier_masked_smoke
  python train_masked_hierarchical_v2.py --primitive-budget 100000 --output models/hier_masked_100k_primitive

The primitive budget is counted across all parallel environments and checked
at callback boundaries, so it may be exceeded slightly.
"""
import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from mamc_rl.masked_hierarchical_v2 import MaskedHierarchicalMissionEnv


class PrimitiveCounter(BaseCallback):
    def __init__(self, budget=None):
        super().__init__()
        self.budget = budget
        self.primitive_steps = 0

    def _on_step(self) -> bool:
        self.primitive_steps += sum(int(info.get('low_level_steps', 0))
                                    for info in self.locals.get('infos', []))
        return self.budget is None or self.primitive_steps < self.budget


def main():
    p = argparse.ArgumentParser(description=__doc__)
    budget = p.add_mutually_exclusive_group()
    budget.add_argument('--timesteps', type=int, default=None,
                        help='Requested high-level decisions, across all environments')
    budget.add_argument('--primitive-budget', type=int, default=None,
                        help='Stop after approximately this many primitive environment steps')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--option-horizon', type=int, default=3)
    p.add_argument('--output', default='models/hier_masked_v2')
    args = p.parse_args()
    if args.timesteps is not None and args.timesteps < 1:
        p.error('--timesteps must be positive')
    if args.primitive_budget is not None and args.primitive_budget < 1:
        p.error('--primitive-budget must be positive')

    def factory():
        return Monitor(MaskedHierarchicalMissionEnv(option_horizon=args.option_horizon))

    env = make_vec_env(factory, n_envs=4, seed=args.seed)
    callback = PrimitiveCounter(args.primitive_budget)
    try:
        model = MaskablePPO('MlpPolicy', env, verbose=1, seed=args.seed,
                            n_steps=256, batch_size=256, learning_rate=3e-4,
                            gamma=0.99, policy_kwargs={'net_arch': [128, 128]})
        # Use a generous high-level cap when the primitive budget is the stopping rule.
        high_level_cap = args.timesteps or (args.primitive_budget * 2 if args.primitive_budget else 10000)
        model.learn(total_timesteps=high_level_cap, callback=callback)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
        print(f'Saved masked PPO model: {out}.zip')
        print(f'High-level training decisions: {model.num_timesteps}')
        print(f'Primitive environment interactions (all workers): {callback.primitive_steps}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
