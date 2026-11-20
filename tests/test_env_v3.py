import numpy as np
import pytest

from mamc_rl.env_v3 import DeadlineMissionEnv
from mamc_rl.controllers_v3 import HierarchicalDeadlineEnv, deadline_heuristic_action


def test_reset_and_no_future_deadline_leak():
    e = DeadlineMissionEnv(dynamic_obstacles=False)
    obs, _ = e.reset(seed=42)
    assert obs.shape == (129,) and e.observation_space.contains(obs)
    assert obs[-1] == 0.0
    assert not e.active[3]
    np.testing.assert_allclose(obs[-4:-1], np.array([12, 18, 24])/40, rtol=1e-6)


def test_deadline_expiry_and_stable_denominator():
    e = DeadlineMissionEnv(deadlines=(1, 1, 1, 16), dynamic_obstacles=False)
    e.reset(seed=3)
    # Even if an agent happens to collect a target at step 1, every remaining
    # initial target expires. The total achievable mission value stays 9.
    _, _, terminated, _, info = e.step(0)
    assert info['weighted_total'] == 9
    assert sum(e.completed[:3]) + sum(e.expired[:3]) == 3
    assert not terminated
    assert np.all(~(e.completed & e.expired))


def test_arrival_exactly_on_deadline_counts():
    e = DeadlineMissionEnv(deadlines=(1, 2, 3, 16), dynamic_obstacles=False)
    e.reset(seed=11)
    e.targets[0] = e.agents[0].copy()
    _, _, _, _, info = e.step(0)
    assert e.completed[0] and not e.expired[0]
    assert info['weighted_completed'] >= 1


def test_spawn_and_expiry_mask():
    e = HierarchicalDeadlineEnv(dynamic_obstacles=False, deadlines=(1, 1, 1, 16))
    e.reset(seed=12)
    assert e.action_masks().sum() == 16  # idle + three visible objectives per agent
    e.step(0)
    assert e.action_masks().sum() == 1  # initial three expired, urgent not spawned
    for _ in range(5):
        _, _, terminated, truncated, info = e.step(0)
        if info['target_spawned']:
            break
        assert not (terminated or truncated)
    assert info['target_spawned']
    assert e.action_masks().sum() == 4  # idle + urgent target per agent


def test_heuristic_actions_legal_and_reproducible():
    a = HierarchicalDeadlineEnv()
    b = HierarchicalDeadlineEnv()
    np.testing.assert_array_equal(a.reset(seed=10)[0], b.reset(seed=10)[0])
    for _ in range(30):
        x = deadline_heuristic_action(a)
        y = deadline_heuristic_action(b)
        assert x == y and a.action_masks()[x]
        out_a, out_b = a.step(x), b.step(y)
        np.testing.assert_array_equal(out_a[0], out_b[0])
        assert out_a[1:] == out_b[1:]
        if out_a[2] or out_a[3]:
            break


def test_invalid_deadline_config():
    with pytest.raises(ValueError): DeadlineMissionEnv(deadlines=(1, 2, 3))
    with pytest.raises(ValueError): DeadlineMissionEnv(deadlines=(1, 2, 3, 12))
