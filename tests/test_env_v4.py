import numpy as np
import pytest

from mamc_rl.env_v4 import ResourceMissionEnv
from mamc_rl.controllers_v4 import HierarchicalResourceEnv, resource_heuristic_action


def test_shape_and_future_information_hidden():
    e=ResourceMissionEnv(dynamic_obstacles=False)
    obs,_=e.reset(seed=42)
    assert obs.shape==(139,) and e.observation_space.contains(obs)
    assert not e.active[3]
    assert obs[-1]==0 and obs[-5]==0  # future target's two capability flags
    np.testing.assert_array_equal(e.fuel,e.fuel_budgets)


def test_capability_gates_collection():
    e=ResourceMissionEnv(dynamic_obstacles=False)
    e.reset(seed=5)
    # A requires agent 0. Agent 1 visiting it must not complete the mission.
    e.targets[0]=e.agents[1].copy()
    e.step(0)
    assert not e.completed[0]
    # Completing on a deadline is allowed for a capable agent.
    e.targets[0]=e.agents[0].copy()
    e.step(0)
    assert e.completed[0]


def test_fuel_limits_movement():
    e=ResourceMissionEnv(fuel_budgets=(0,0),dynamic_obstacles=False)
    e.reset(seed=7)
    previous=e.agents.copy()
    _,_,_,_,info=e.step(5*1+1)
    np.testing.assert_array_equal(e.agents,previous)
    assert info['fuel_remaining']==(0,0)


def test_masks_respect_capabilities_and_fuel():
    e=HierarchicalResourceEnv(dynamic_obstacles=False)
    e.reset(seed=2)
    assert e.action_masks()[5*1+0]  # agent 0 -> A
    assert not e.action_masks()[5*0+1]  # agent 1 cannot do A
    assert e.action_masks()[5*0+2]  # agent 1 -> B
    assert not e.action_masks()[5*2+0]  # agent 0 cannot do B
    e.base.fuel[:]=0
    # With no fuel, only idle or a target already under the agent is feasible.
    for a in np.flatnonzero(e.action_masks()):
        c=(int(a)//5,int(a)%5)
        for agent,choice in enumerate(c):
            if choice:
                assert np.array_equal(e.base.agents[agent],e.base.targets[choice-1])


def test_heuristic_legal_and_reproducible():
    a=HierarchicalResourceEnv(); b=HierarchicalResourceEnv()
    np.testing.assert_array_equal(a.reset(seed=12)[0],b.reset(seed=12)[0])
    for _ in range(20):
        x=resource_heuristic_action(a); y=resource_heuristic_action(b)
        assert x==y and a.action_masks()[x]
        outa=a.step(x); outb=b.step(y)
        np.testing.assert_array_equal(outa[0],outb[0])
        assert outa[1:]==outb[1:]
        if outa[2] or outa[3]: break


def test_bad_config():
    with pytest.raises(ValueError): ResourceMissionEnv(fuel_budgets=(1,))
    with pytest.raises(ValueError): ResourceMissionEnv(capabilities=((0,0,0,0),(0,1,1,1)))
