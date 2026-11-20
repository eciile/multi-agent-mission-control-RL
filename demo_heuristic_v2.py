"""Inspect the V2 adaptive baseline: python demo_heuristic_v2.py --seed 7"""
import argparse
from mamc_rl.env_v2 import DynamicMissionEnv
from mamc_rl.policies_v2 import heuristic_action_v2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    env = DynamicMissionEnv(render_mode="ansi")
    env.reset(seed=args.seed)
    print("Initial mission:")
    print(env.render())
    total_reward = 0.0
    for _ in range(args.steps):
        action = heuristic_action_v2(env)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info["target_spawned"] or info["obstacles_changed"] or terminated or truncated:
            print(f"\nStep {info['steps']}, joint action {action}, reward {reward:.2f}")
            print(f"Event flags: obstacle_change={info['obstacles_changed']}, "
                  f"new_target={info['target_spawned']}, success={info['success']}")
            print(env.render())
        if terminated or truncated:
            break
    print(f"\nFinal: steps={env.steps}, targets={int(env.completed.sum())}, "
          f"weighted_completed={int(env.priorities[env.completed].sum())}, "
          f"total_reward={total_reward:.2f}, success={bool(info['success'])}")
    env.close()

if __name__ == "__main__":
    main()