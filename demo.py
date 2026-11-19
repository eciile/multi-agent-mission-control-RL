"""Terminal demonstration: python demo.py --policy heuristic --seed 7"""
import argparse
from mamc_rl import MissionGridEnv
from mamc_rl.policies import heuristic_action


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--policy", choices=["random", "heuristic", "ppo"], default="heuristic")
    p.add_argument("--model", default="models/ppo_mamc.zip")
    a = p.parse_args()
    env = MissionGridEnv(render_mode="ansi")
    obs, _ = env.reset(seed=a.seed)
    model = None
    if a.policy == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(a.model)
    print("Initial world:\n" + env.render())
    for step in range(env.max_steps):
        if a.policy == "random":
            action = env.action_space.sample()
        elif a.policy == "heuristic":
            action = heuristic_action(env)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        print(f"\nStep {step+1} | reward={reward:.2f}\n{env.render()}")
        if terminated or truncated:
            print("Result:", info)
            break


if __name__ == "__main__":
    main()
