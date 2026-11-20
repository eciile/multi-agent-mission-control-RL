"""Run offline DAgger rounds for the V4 masked behavioral-cloning controller.

The learner drives each rollout. A depth-limited, non-oracle planner labels each
learner-visited state but NEVER changes the executed action. After each round,
the same 128x128 BC model is retrained from scratch on accumulated labels.

Example:
  python run_dagger_v4.py --rounds 3 --episodes-per-round 100 --device auto

The final test seeds (2000-2199) are never used for collection or model
selection. Evaluate each saved round checkpoint separately with evaluate_bc_v4.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from mamc_rl.controllers_v4 import HierarchicalResourceEnv
from mamc_rl.lookahead_v4 import lookahead_action
from train_bc_v4 import BCPolicy, read_dataset

TEST_SEEDS = set(range(2000, 2200))


def require_disjoint(*groups: set[int]) -> None:
    for i, group in enumerate(groups):
        if group & TEST_SEEDS:
            raise ValueError('Final test seeds 2000-2199 must not appear in DAgger data')
        for other in groups[i + 1:]:
            if group & other:
                raise ValueError('Train, validation and DAgger mission seeds must be disjoint')


def load_learner(path: str, expected_obs_dim: int) -> BCPolicy:
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if int(checkpoint['obs_dim']) != expected_obs_dim or int(checkpoint['n_actions']) != 25:
        raise ValueError(f'Checkpoint {path} does not match V4 observation/action dimensions')
    model = BCPolicy(expected_obs_dim, 25)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def learner_action(model: BCPolicy, obs: np.ndarray, mask: np.ndarray) -> int:
    with torch.inference_mode():
        scores = model(torch.from_numpy(np.asarray(obs, dtype=np.float32))[None],
                       torch.from_numpy(np.asarray(mask, dtype=np.bool_))[None])
        action = int(scores.argmax(dim=-1).item())
    if not bool(mask[action]):
        raise AssertionError('Learner selected an illegal action')
    return action


def collect_learner_states(model: BCPolicy, episode_seeds: range, *,
                           depth: int = 4, beam_width: int = 4,
                           option_horizon: int = 3) -> dict[str, np.ndarray]:
    """Return oracle-labeled observations visited by the learner itself."""
    observations, masks, labels, sample_seeds = [], [], [], []
    successes, weighted, expert_agreement = [], [], []
    oracle_seconds = 0.0
    env = HierarchicalResourceEnv(option_horizon=option_horizon)
    try:
        for seed in episode_seeds:
            obs, _ = env.reset(seed=int(seed))
            matches = 0
            steps = 0
            while True:
                mask = np.asarray(env.action_masks(), dtype=np.bool_)
                if not mask.any():
                    raise AssertionError('No legal actions in reachable state')
                start = time.perf_counter()
                oracle_action = int(lookahead_action(env, depth=depth, beam_width=beam_width))
                oracle_seconds += time.perf_counter() - start
                action = learner_action(model, obs, mask)
                if not bool(mask[oracle_action]):
                    raise AssertionError('Planner returned illegal expert label')
                observations.append(np.asarray(obs, dtype=np.float32).copy())
                masks.append(mask.copy())
                labels.append(oracle_action)
                sample_seeds.append(int(seed))
                matches += int(action == oracle_action)
                steps += 1
                # DAgger: execute learner action, NOT the oracle action.
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    successes.append(int(info['success']))
                    weighted.append(float(info['weighted_fraction']))
                    expert_agreement.append(matches / steps)
                    break
    finally:
        env.close()
    return {
        'observations': np.asarray(observations, dtype=np.float32),
        'masks': np.asarray(masks, dtype=np.bool_),
        'actions': np.asarray(labels, dtype=np.int64),
        'seeds': np.asarray(sample_seeds, dtype=np.int64),
        'episode_seeds': np.asarray(list(episode_seeds), dtype=np.int64),
        'episode_success': np.asarray(successes, dtype=np.int64),
        'episode_weighted': np.asarray(weighted, dtype=np.float32),
        'mean_learner_expert_agreement': np.asarray(np.mean(expert_agreement)),
        'oracle_seconds': np.asarray(oracle_seconds, dtype=np.float64),
    }


def merge_labels(base: dict[str, np.ndarray], collected: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    result = {}
    for key in ('observations', 'masks', 'actions', 'seeds', 'episode_seeds'):
        result[key] = np.concatenate([base[key]] + [round_data[key] for round_data in collected], axis=0)
    if not np.all(result['masks'][np.arange(len(result['actions'])), result['actions']]):
        raise ValueError('Aggregated data includes illegal expert labels')
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial-model', default='models/v4_bc_planner.pt')
    p.add_argument('--train', default='results/expert_v4/train.npz')
    p.add_argument('--val', default='results/expert_v4/val.npz')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--episodes-per-round', type=int, default=100)
    p.add_argument('--seed-start', type=int, default=11000)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--beam-width', type=int, default=4)
    p.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='auto')
    p.add_argument('--output-dir', default='results/dagger_v4')
    p.add_argument('--model-dir', default='models')
    args = p.parse_args()
    if min(args.rounds, args.episodes_per_round, args.epochs, args.depth, args.beam_width) < 1:
        p.error('rounds, episodes, epochs, depth and beam width must be positive')
    torch.set_num_threads(1)
    x_train, masks_train, y_train, train_seeds = read_dataset(args.train)
    _, _, _, val_seeds = read_dataset(args.val)
    start = args.seed_start
    dagger_seeds = set(range(start, start + args.rounds * args.episodes_per_round))
    require_disjoint(set(map(int, train_seeds)), set(map(int, val_seeds)), dagger_seeds)
    train_base = {
        'observations': x_train, 'masks': masks_train, 'actions': y_train,
        'seeds': np.load(args.train, allow_pickle=False)['seeds'].copy(),
        'episode_seeds': train_seeds,
    }
    data_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.initial_model
    aggregated = []
    total_queries, total_oracle_seconds = 0, 0.0
    for round_i in range(1, args.rounds + 1):
        episode_start = start + (round_i - 1) * args.episodes_per_round
        episode_range = range(episode_start, episode_start + args.episodes_per_round)
        policy = load_learner(model_path, x_train.shape[1])
        new_data = collect_learner_states(policy, episode_range,
                                          depth=args.depth, beam_width=args.beam_width)
        aggregated.append(new_data)
        total_queries += len(new_data['actions'])
        total_oracle_seconds += float(new_data['oracle_seconds'])
        round_data_file = data_dir / f'round{round_i}_learner_states.npz'
        np.savez_compressed(round_data_file, **new_data)
        merged = merge_labels(train_base, aggregated)
        merged_file = data_dir / f'round{round_i}_train.npz'
        np.savez_compressed(merged_file, **merged)
        model_path = str(model_dir / f'v4_dagger_round{round_i}.pt')
        print(f'Round {round_i}: {args.episodes_per_round} learner missions; '
              f'{len(new_data["actions"])} expert queries; '
              f'learner success {np.mean(new_data["episode_success"]):.1%}; '
              f'learner weighted {np.mean(new_data["episode_weighted"]):.1%}; '
              f'on-policy expert agreement {float(new_data["mean_learner_expert_agreement"]):.1%}; '
              f'accumulated labels {len(merged["actions"])}', flush=True)
        # Reuse the exact existing BC training recipe and expert validation set.
        # No final evaluation seeds enter model selection.
        command = [sys.executable, 'train_bc_v4.py', '--train', str(merged_file),
                   '--val', args.val, '--epochs', str(args.epochs),
                   '--seed', str(args.seed), '--device', args.device,
                   '--output', model_path]
        subprocess.run(command, check=True)
        print(f'Round {round_i} checkpoint: {model_path}', flush=True)
    print(f'DAgger complete: additional expert queries={total_queries}; '
          f'oracle computation={total_oracle_seconds:.1f}s; last model={model_path}')
    print('Evaluate checkpoints on final seeds 2000-2199 using evaluate_bc_v4.py; '
          'the test set has not been used to select or train checkpoints.')


if __name__ == '__main__':
    main()
