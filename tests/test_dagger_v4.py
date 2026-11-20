"""Small regression tests for DAgger data integrity and legal action selection."""

import numpy as np
import pytest
import torch

from run_dagger_v4 import require_disjoint, merge_labels, learner_action
from train_bc_v4 import BCPolicy


def _dataset(seeds, actions):
    actions = np.asarray(actions, dtype=np.int64)
    masks = np.zeros((len(actions), 25), dtype=np.bool_)
    masks[:, 0] = True
    masks[np.arange(len(actions)), actions] = True
    return {
        'observations': np.zeros((len(actions), 139), dtype=np.float32),
        'masks': masks,
        'actions': actions,
        'seeds': np.asarray(seeds, dtype=np.int64),
        'episode_seeds': np.unique(seeds).astype(np.int64),
    }


def test_seed_split_rejects_overlap():
    require_disjoint({10000, 10001}, {15000}, {11000, 11001})
    with pytest.raises(ValueError, match='disjoint'):
        require_disjoint({10000}, {10000}, {11000})


def test_final_test_seeds_never_allowed_in_collection():
    with pytest.raises(ValueError, match='2000-2199'):
        require_disjoint({10000}, {15000}, {2000})


def test_aggregate_keeps_original_and_new_labels():
    base = _dataset([10000, 10000], [1, 2])
    round1 = _dataset([11000], [3])
    round2 = _dataset([11100, 11100], [4, 5])
    merged = merge_labels(base, [round1, round2])
    assert merged['observations'].shape == (5, 139)
    assert merged['actions'].tolist() == [1, 2, 3, 4, 5]
    assert set(merged['episode_seeds'].tolist()) == {10000, 11000, 11100}


def test_aggregation_rejects_illegal_expert_action():
    base = _dataset([10000], [1])
    new = _dataset([11000], [3])
    new['masks'][0, 3] = False
    with pytest.raises(ValueError, match='illegal'):
        merge_labels(base, [new])


def test_learner_respects_mask():
    net = BCPolicy(139, 25)
    for p in net.parameters():
        torch.nn.init.zeros_(p)
    mask = np.zeros(25, dtype=np.bool_)
    mask[7] = True
    action = learner_action(net, np.zeros(139, dtype=np.float32), mask)
    assert action == 7
