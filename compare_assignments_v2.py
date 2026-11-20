"""Diagnostic: random high-level assignments vs corrected hierarchical PPO.

Same BFS navigation, scenario seeds, and option horizon for both policies.
The random policy is not an optimized task allocator. Results are diagnostic.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from mamc_rl.hierarchical_v2 import HierarchicalMissionEnv


def evaluate(kind, model, seed_start, episodes, horizon):
    env = HierarchicalMissionEnv(option_horizon=horizon)
    rows = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            obs, _ = env.reset(seed=seed)
            rng = np.random.default_rng(seed + 100_000)
            total_reward = 0.0
            decisions = 0
            primitive_steps = 0
            invalid_assignment_choices = 0
            duplicate_assignments = 0
            while True:
                if kind == 'ppo':
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)
                else:
                    action = int(rng.integers(env.action_space.n))
                choices = [action // 5, action % 5]
                for choice in choices:
                    if choice:
                        idx = choice - 1
                        if not env.base.active[idx] or env.base.completed[idx]:
                            invalid_assignment_choices += 1
                if choices[0] == choices[1] and choices[0] != 0:
                    duplicate_assignments += 1
                obs, reward, terminated, truncated, info = env.step(action)
                decisions += 1
                primitive_steps += int(info['low_level_steps'])
                total_reward += float(reward)
                if terminated or truncated:
                    break
            rows.append(dict(policy=kind, seed=seed, success=int(info['success']),
                             weighted_fraction=info['weighted_completed']/max(info['weighted_total'],1),
                             episode_return=total_reward, primitive_steps=primitive_steps,
                             high_level_decisions=decisions,
                             invalid_assignment_choices=invalid_assignment_choices,
                             duplicate_assignments=duplicate_assignments))
    finally:
        env.close()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='models/hier_corrected_10k.zip')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed-start', type=int, default=1000)
    p.add_argument('--option-horizon', type=int, default=3)
    p.add_argument('--output', default='results/assignment_diagnostic_v2.csv')
    args = p.parse_args()
    if args.episodes < 1:
        p.error('--episodes must be >= 1')
    model = PPO.load(args.model, device='cpu')
    rows = []
    for kind in ('random_assignment', 'ppo'):
        rows.extend(evaluate(kind, model, args.seed_start, args.episodes, args.option_horizon))
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {dest} ({len(rows)} episode records)')
    print('Policy                 Success  Weighted  Return   Steps  Decisions  Invalid assignments / ep  Duplicate assignments / ep')
    for kind in ('random_assignment', 'ppo'):
        group = [r for r in rows if r['policy'] == kind]
        avg = lambda k: float(np.mean([r[k] for r in group]))
        print(f'{kind:<22} {avg("success"):>6.1%}  {avg("weighted_fraction"):>7.1%}  {avg("episode_return"):>6.2f}  {avg("primitive_steps"):>6.1f}  {avg("high_level_decisions"):>9.1f}  {avg("invalid_assignment_choices"):>24.2f}  {avg("duplicate_assignments"):>26.2f}')


if __name__ == '__main__':
    main()
