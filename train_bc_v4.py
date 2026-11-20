"""Train a small masked behavioral-cloning policy from V4 planner demonstrations.

Usage: python train_bc_v4.py --train results/expert_v4/train.npz --val results/expert_v4/val.npz
The network has the same 128x128 hidden-layer sizes as the PPO MLP, but this
is supervised learning, not PPO fine-tuning. Selects best epoch on validation
expert-action accuracy; final evaluation seeds remain untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 25):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, observations: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        logits = self.net(observations)
        return logits.masked_fill(~masks.bool(), -1e9)


def read_dataset(path: str):
    with np.load(path, allow_pickle=False) as data:
        x = data['observations'].copy()
        m = data['masks'].copy()
        y = data['actions'].copy()
        seeds = data['episode_seeds'].copy()
    if x.ndim != 2 or x.shape[0] == 0 or m.shape != (len(x), 25) or y.shape != (len(x),):
        raise ValueError(f'Invalid demonstration shapes in {path}')
    if x.shape[1] != 139:
        raise ValueError(f'Expected V4 observation dimension 139, got {x.shape[1]}')
    if np.any(y < 0) or np.any(y >= 25) or not np.all(m[np.arange(len(y)), y]):
        raise ValueError(f'Dataset contains illegal expert actions: {path}')
    return x, m, y, seeds


def accuracy(net, x, m, y, device):
    net.eval()
    with torch.inference_mode():
        hits, total = 0, 0
        for start in range(0, len(y), 4096):
            obs = torch.as_tensor(x[start:start+4096], device=device)
            mask = torch.as_tensor(m[start:start+4096], device=device)
            target = torch.as_tensor(y[start:start+4096], device=device)
            pred = net(obs, mask).argmax(dim=1)
            hits += int((pred == target).sum().item())
            total += len(target)
    return hits / max(total, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', default='results/expert_v4/train.npz')
    p.add_argument('--val', default='results/expert_v4/val.npz')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    p.add_argument('--output', default='models/v4_bc_planner.pt')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        p.error('epochs/batch-size must be positive, lr must be > 0')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable to PyTorch')
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    xtr, mtr, ytr, train_seeds = read_dataset(args.train)
    xval, mval, yval, val_seeds = read_dataset(args.val)
    if set(train_seeds) & set(val_seeds):
        raise ValueError('Overlapping training and validation mission seeds')
    if set(train_seeds) & set(range(2000, 2200)) or set(val_seeds) & set(range(2000, 2200)):
        raise ValueError('Final evaluation seeds leaked into demonstration data')
    net = BCPolicy(xtr.shape[1]).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(mtr), torch.from_numpy(ytr)),
                        batch_size=args.batch_size, shuffle=True)
    best_acc = -1.0
    best_epoch = 0
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'Training: {len(ytr)} expert decisions; validation: {len(yval)}; device: {device}')
    for epoch in range(1, args.epochs + 1):
        net.train()
        sum_loss = 0.0
        for obs, masks, target in loader:
            obs, masks, target = obs.to(device), masks.to(device), target.to(device)
            loss = nn.functional.cross_entropy(net(obs, masks), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += float(loss.item()) * len(target)
        val_acc = accuracy(net, xval, mval, yval, device)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({'state_dict': net.state_dict(), 'obs_dim': xtr.shape[1],
                        'n_actions': 25, 'training_seed': args.seed,
                        'best_epoch': epoch, 'validation_accuracy': best_acc,
                        'train_seeds': [int(train_seeds.min()), int(train_seeds.max())],
                        'val_seeds': [int(val_seeds.min()), int(val_seeds.max())]}, dest)
        if epoch in (1, args.epochs) or epoch % 10 == 0:
            print(f'Epoch {epoch:>3}: train CE={sum_loss/len(ytr):.4f}; '
                  f'validation expert agreement={val_acc:.1%}')
    print(f'Saved {dest}; best epoch={best_epoch}; validation agreement={best_acc:.1%}')
    print('Expert-action agreement is not closed-loop mission performance. Evaluate separately.')


if __name__ == '__main__':
    main()
