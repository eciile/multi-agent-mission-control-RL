"""Controlled inference-time feasibility mask for hierarchical PPO.

Compares the same frozen policy with/without a mask that removes assignments
pointing to inactive or already completed targets. No retraining, no reward changes.

Usage:
  python evaluate_assignment_mask_v2.py --model models/hier_corrected_10k.zip
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from mamc_rl.hierarchical_v2 import HierarchicalMissionEnv


def valid_joint_actions(active, completed):
    """Idle (0) always valid; targets 1..4 valid iff active and unfinished.

    Duplicate target assignments intentionally remain legal: this experiment
    isolates *only* inactive/completed target choices.
    """
    valid = [0] + [i + 1 for i in range(4) if active[i] and not completed[i]]
    return np.array([5 * a0 + a1 for a0 in valid for a1 in valid], dtype=np.int64)


def choose_masked(model, observation, env):
    valid = valid_joint_actions(env.base.active, env.base.completed)
    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(observation)
        dist = model.policy.get_distribution(obs_tensor)
        logits = dist.distribution.logits.detach().cpu().numpy()[0]
    return int(valid[np.argmax(logits[valid])])


def evaluate(model, mode, episodes, seed_start, option_horizon):
    env = HierarchicalMissionEnv(option_horizon=option_horizon)
    rows = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            obs, _ = env.reset(seed=seed)
            total_reward = 0.0
            decisions = 0
            primitive_steps = 0
            invalid_choices = 0
            altered = 0
            while True:
                raw_action, _ = model.predict(obs, deterministic=True)
                raw_action = int(raw_action)
                if mode == 'masked':
                    action = choose_masked(model, obs, env)
                    altered += int(action != raw_action)
                else:
                    action = raw_action
                choices = (action // 5, action % 5)
                invalid_choices += sum(
                    int(c > 0 and (not env.base.active[c - 1] or env.base.completed[c - 1]))
                    for c in choices
                )
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                decisions += 1
                primitive_steps += int(info['low_level_steps'])
                if terminated or truncated:
                    break
            rows.append({
                'mode': mode, 'seed': seed,
                'success': int(info['success']),
                'weighted_fraction': info['weighted_completed'] / max(1, info['weighted_total']),
                'return': total_reward,
                'primitive_steps': primitive_steps,
                'decisions': decisions,
                'invalid_choices': invalid_choices,
                'overrides': altered,
            })
    finally:
        env.close()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='models/hier_corrected_10k.zip')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--seed-start', type=int, default=1000)
    ap.add_argument('--option-horizon', type=int, default=3)
    ap.add_argument('--output', default='results/assignment_mask_v2.csv')
    args = ap.parse_args()
    if args.episodes < 1:
        ap.error('--episodes must be >= 1')
    model = PPO.load(args.model, device='cpu')
    rows = evaluate(model, 'original', args.episodes, args.seed_start, args.option_horizon)
    rows += evaluate(model, 'masked', args.episodes, args.seed_start, args.option_horizon)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {path} ({len(rows)} episode records)')
    print('Mode        Success  Weighted   Return  Steps  Decisions  Invalid choices  Overrides')
    for mode in ('original', 'masked'):
        group = [r for r in rows if r['mode'] == mode]
        mean = lambda field: float(np.mean([r[field] for r in group]))
        print(f'{mode:<11} {mean("success"):>6.1%}  {mean("weighted_fraction"):>7.1%}  '
              f'{mean("return"):>7.2f}  {mean("primitive_steps"):>5.1f}  '
              f'{mean("decisions"):>9.1f}  {mean("invalid_choices"):>15.2f}  '
              f'{mean("overrides"):>9.2f}')
    print('Mask changes only assignment feasibility. Duplicate assignments remain allowed.')


if __name__ == '__main__':
    main()
