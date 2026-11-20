"""Inspect Version 2 changes with a reproducible sequence of joint stay actions."""
import argparse
from mamc_rl.env_v2 import DynamicMissionEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    env = DynamicMissionEnv(render_mode="ansi", obstacle_change_prob=1.0)
    observation, _ = env.reset(seed=args.seed)
    print("Observation shape:", observation.shape)
    print("Action space:", env.action_space)
    print("Legend: A/B = agents; 1/2/3 = target priority; # = wall; X = temporary obstacle")
    print("\nInitial grid:\n" + env.render())
    for _ in range(args.steps):
        _, reward, terminated, truncated, info = env.step(0)
        if info["target_spawned"] or info["obstacles_changed"]:
            print(f"\nStep {env.steps}: reward={reward:.2f}, events={info}")
            print(env.render())
        if terminated or truncated:
            break
    env.close()

if __name__ == "__main__":
    main()
