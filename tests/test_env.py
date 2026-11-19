import numpy as np
from mamc_rl import MissionGridEnv


def test_seed_reproducible():
    a, b = MissionGridEnv(), MissionGridEnv()
    np.testing.assert_array_equal(a.reset(seed=12)[0], b.reset(seed=12)[0])


def test_observation_and_action_spaces():
    env = MissionGridEnv()
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    for _ in range(10):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        if term or trunc:
            break


def test_horizon():
    env = MissionGridEnv(max_steps=1)
    env.reset(seed=5)
    _, _, term, trunc, _ = env.step(0)
    assert term or trunc


def test_no_obstacle_occupancy():
    from mamc_rl.env import OBSTACLES
    env = MissionGridEnv()
    env.reset(seed=31)
    for _ in range(30):
        env.step(env.action_space.sample())
        assert all(tuple(p) not in OBSTACLES for p in env.agents)


def test_no_occupied_same_cell():
    env = MissionGridEnv()
    env.reset(seed=6)
    for _ in range(20):
        env.step(env.action_space.sample())
        assert not np.array_equal(env.agents[0], env.agents[1])
