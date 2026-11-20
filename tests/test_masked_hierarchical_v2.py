import numpy as np

from mamc_rl.masked_hierarchical_v2 import MaskedHierarchicalMissionEnv


def test_mask_at_reset():
    env = MaskedHierarchicalMissionEnv()
    try:
        env.reset(seed=7)
        mask = env.action_masks()
        assert mask.shape == (25,)
        assert mask.dtype == np.bool_
        assert mask[0] and mask[5 * 1 + 2] and mask[5 * 2 + 2]
        assert not mask[4] and not mask[20] and not mask[24]
        assert mask.sum() == 16  # idle + three active targets, for each agent
    finally:
        env.close()


def test_completion_and_spawn_update_masks():
    env = MaskedHierarchicalMissionEnv()
    try:
        env.reset(seed=7)
        env.base.completed[1] = True
        mask = env.action_masks()
        assert not mask[2]  # idle + completed target 1
        assert not mask[10]  # completed target 1 + idle
        assert mask.sum() == 9  # idle and two remaining active targets
        env.base.active[3] = True
        env.base.priorities[3] = 3
        mask = env.action_masks()
        assert mask[4] and mask[20]
        assert mask.sum() == 16
    finally:
        env.close()


def test_masked_actions_can_execute():
    env = MaskedHierarchicalMissionEnv()
    try:
        obs, _ = env.reset(seed=7)
        for _ in range(5):
            legal = np.flatnonzero(env.action_masks())
            assert len(legal) >= 1
            obs, reward, terminated, truncated, info = env.step(int(legal[0]))
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break
    finally:
        env.close()
