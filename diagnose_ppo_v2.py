"""Diagnose V2 PPO navigation and coordination failures.

Run:
    python diagnose_ppo_v2.py --model models/ppo_v2_combined.zip --episodes 100
    python diagnose_ppo_v2.py --model models/ppo_v2_combined.zip --episodes 100 --stochastic

Uses the V2 combined mission environment, with held-out episode seeds.
Diagnostics separate attempts to move into obstacles from agent-to-agent collisions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from mamc_rl.env_v2 import DynamicMissionEnv, MOVES, SIZE


def run(model: PPO, episodes: int, seed_start: int, deterministic: bool):
    action_counts = Counter()
    agent_actions = [Counter(), Counter()]
    total_steps = 0
    total_invalid = 0
    total_collision_events = 0
    stationary_commands = [0, 0]
    blocked_commands = [0, 0]
    targets_counts = Counter()
    success_count = 0
    weighted_fractions = []
    total_returns = []
    first_example = None

    for seed in range(seed_start, seed_start + episodes):
        env = DynamicMissionEnv(dynamic_obstacles=True, changing_targets=True)
        try:
            obs, _ = env.reset(seed=seed)
            env.action_space.seed(seed)
            episode_return = 0.0
            while True:
                before = env.agents.copy()
                blocked_map = env.obstacles.copy()
                action, _ = model.predict(obs, deterministic=deterministic)
                joint = int(np.asarray(action).item())
                pair = (joint // 5, joint % 5)
                action_counts[joint] += 1
                total_steps += 1
                for i in range(2):
                    a = pair[i]
                    agent_actions[i][a] += 1
                    if a == 0:
                        stationary_commands[i] += 1
                    else:
                        r, c = before[i] + MOVES[a]
                        if not (0 <= r < SIZE and 0 <= c < SIZE) or (int(r), int(c)) in blocked_map:
                            blocked_commands[i] += 1
                obs, reward, terminated, truncated, info = env.step(joint)
                episode_return += float(reward)
                total_invalid += int(info['invalid_actions'])
                total_collision_events += int(info['collision'])
                if terminated or truncated:
                    success_count += int(info['success'])
                    targets_counts[int(info['targets_completed'])] += 1
                    weighted_fractions.append(info['weighted_completed'] / info['weighted_total'])
                    total_returns.append(episode_return)
                    if first_example is None:
                        first_example = (seed, int(info['steps']), int(info['targets_completed']), bool(info['success']))
                    break
        finally:
            env.close()

    print(f'Evaluation mode: {"deterministic" if deterministic else "stochastic"}')
    print(f'Episodes: {episodes} | seeds: {seed_start}..{seed_start + episodes - 1}')
    print(f'Success: {success_count}/{episodes} ({success_count / episodes:.1%})')
    print(f'Mean return: {np.mean(total_returns):.2f}')
    print(f'Mean weighted completion: {np.mean(weighted_fractions):.1%}')
    print(f'Mean episode length: {total_steps / episodes:.2f}')
    print(f'Invalid agent movements per episode: {total_invalid / episodes:.2f}')
    print(f'Collision events per episode: {total_collision_events / episodes:.2f}')
    print(f'Invalid per agent command: {total_invalid / (2 * total_steps):.1%}')
    print('Completed-target distribution:', dict(sorted(targets_counts.items())))
    for i in range(2):
        counts = agent_actions[i]
        print(f'Agent {i} action counts (stay/up/down/left/right): {[counts.get(j,0) for j in range(5)]}')
        print(f'Agent {i}: stay commands {stationary_commands[i] / total_steps:.1%}, '
              f'blocked movement attempts {blocked_commands[i] / total_steps:.1%}')
    print('Top joint actions (code, count, share):')
    for code, count in action_counts.most_common(8):
        print(f'  {code:2d} ({code//5},{code%5}): {count:5d} ({count / total_steps:.1%})')
    print('Example episode (seed, steps, completed targets, success):', first_example)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/ppo_v2_combined.zip')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed-start', type=int, default=1000)
    parser.add_argument('--stochastic', action='store_true')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    if not Path(args.model).is_file():
        parser.error(f'Model not found: {args.model}')
    model = PPO.load(args.model, device='cpu')
    run(model, args.episodes, args.seed_start, not args.stochastic)


if __name__ == '__main__':
    main()
