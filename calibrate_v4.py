"""Calibrate V4 without RL: capability-constrained random vs greedy scheduling."""
import argparse
import csv
from pathlib import Path
import numpy as np

from mamc_rl.controllers_v4 import HierarchicalResourceEnv, masked_random_action, resource_heuristic_action


def episode(name, seed, fuel):
    env = HierarchicalResourceEnv(fuel_budgets=tuple(fuel))
    rng = np.random.default_rng(seed + 1_000_000)
    env.reset(seed=seed)
    ret, steps, decisions, spawned = 0.,0,0,False
    try:
        while True:
            action = masked_random_action(env,rng) if name == 'masked_random' else resource_heuristic_action(env)
            _, reward, terminated, truncated, info = env.step(action)
            ret += reward; steps += info['low_level_steps']; decisions += 1
            spawned |= info['target_spawned']
            if terminated or truncated:
                return dict(policy=name,seed=seed,success=int(info['success']),
                            weighted_fraction=info['weighted_fraction'],expired_targets=info['expired_targets'],
                            primitive_steps=steps,high_level_decisions=decisions,return_value=ret,
                            urgent_spawned=int(spawned),fuel_agent0=info['fuel_remaining'][0],
                            fuel_agent1=info['fuel_remaining'][1])
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=200)
    p.add_argument('--seed-start',type=int,default=2000)
    p.add_argument('--fuel',type=int,nargs=2,default=(16,18))
    p.add_argument('--output',default='results/v4_calibration.csv')
    args=p.parse_args()
    if args.episodes < 1: p.error('episodes must be >= 1')
    rows=[episode(name,s,args.fuel) for s in range(args.seed_start,args.seed_start+args.episodes)
          for name in ('masked_random','resource_heuristic')]
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f'Saved {path} ({len(rows)} episode records)')
    print('Policy                Success  Weighted  Expired  Steps  Return  Spawned')
    for name in ('masked_random','resource_heuristic'):
        sub=[r for r in rows if r['policy']==name]
        avg=lambda key:np.mean([r[key] for r in sub])
        print(f'{name:<22}{avg("success"):>6.1%}   {avg("weighted_fraction"):>7.1%}'
              f'   {avg("expired_targets"):>5.2f}  {avg("primitive_steps"):>5.1f}'
              f'  {avg("return_value"):>6.2f}   {avg("urgent_spawned"):>6.1%}')

if __name__=='__main__': main()
