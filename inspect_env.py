
from mamc_rl import MissionGridEnv

env = MissionGridEnv(render_mode="ansi")
obs, info = env.reset(seed=7)

print("Observation shape:", obs.shape)
print("Action space:", env.action_space)
print("Agent positions:", env.agents)
print("Target positions:", env.targets)
print("\nInitial grid:")
print(env.render())

# Agent 0 moves right (4); Agent 1 moves up (1).
joint_action = 5 * 4 + 1

obs, reward, terminated, truncated, info = env.step(joint_action)

print("\nAfter one joint action:")
print(env.render())
print("New agent positions:", env.agents)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)
print("Info:", info)

env.close()
