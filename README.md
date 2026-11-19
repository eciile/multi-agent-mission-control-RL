# MAMC-RL — Multi-Agent Mission Control with Reinforcement Learning

A compact **research-oriented** prototype for cooperative mission control in a simplified 2D grid. Two virtual agents must visit two targets while avoiding obstacles and each other. The learned policy uses **centralized PPO with a joint action space**, compared with a random controller and a shortest-path-inspired heuristic.

> **Scope:** This is *centralized multi-entity control*, **not** a decentralized multi-agent RL algorithm or a flight-dynamics simulator. It is intended to study mission modeling, sequential decision-making, experimental evaluation, and scalability questions.

## Research question

Can a centralized PPO controller learn to complete cooperative multi-entity missions reliably, and how does it compare to random and heuristic controllers on unseen initial configurations?

## Environment and MDP

| Component | Definition |
|---|---|
| Grid | 8×8 with 6 fixed blocked cells |
| Agents / targets | 2 movable agents and 2 targets; random distinct initial positions |
| Observation | 74 normalized features: 64 obstacle indicators, 4 agent coordinates, 4 target coordinates, 2 completion flags |
| Joint action | 25 discrete possibilities = 5 actions per agent (stay, up, down, left, right) |
| Transition | Both agents move simultaneously. Invalid moves stay; collisions/swaps cancel both moves |
| Reward | −0.05 each step, −0.2 per invalid action, −0.5 collision, +0.15 potential-based progress proxy, +5 per target, +10 mission success |
| Termination | Both targets visited; time-limit truncation at 40 steps |
| Policy | PPO (`MlpPolicy`, 2×128 network) |
| Baselines | Uniform random; greedy target assignment + obstacle-aware BFS |

**Reward nuance:** The progress term is a heuristic shaping signal based on Manhattan distances to targets active before the step; it is not guaranteed to be policy-invariant potential-based shaping. Report this limitation and, if time permits, test an ablation without shaping.

**Simplification:** All entities have perfect shared access to the grid state. This version does not model partial observability, aircraft kinematics, inter-agent communication, or moving hazards.

## Quickstart

Use **Python 3.11**. No paid software or API keys required.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
python demo.py --policy heuristic --seed 7
python evaluate.py --baselines-only --episodes 100
python train.py --timesteps 50000 --seed 42
python evaluate.py --episodes 100 --seed-start 1000
python demo.py --policy ppo --seed 7
```

**Training time varies by hardware.** If slow, run `--timesteps 10000` first as a smoke test, then increase to 50,000 or 100,000 only if time allows. Learning and superior performance are **not guaranteed**.

## Experimental protocol

1. Check environment consistency with `pytest`.
2. Evaluate random and heuristic controllers on the same 100 deterministic episode seeds (1000–1099).
3. Train PPO on procedurally sampled initial positions, seed 42.
4. Evaluate PPO on those 100 held-out seeds. Use the same environment and horizon for all policies.
5. Report mean success rate, mean episode length, mean cumulative reward, collision count and invalid action count.
6. Repeat training with at least 3 training seeds **if there is time**. A single seed is exploratory, not robust evidence.

**Fairness note:** `steps` over all episodes includes failed runs that hit the 40-step horizon. When interpreting efficiency, additionally compute steps conditional on success. Heuristic BFS planning has privileged full-map knowledge, just like the centralized RL observation.

## Suggested repo structure

```text
mamc_rl/env.py       # Gymnasium environment
mamc_rl/policies.py  # random / BFS heuristic baselines
train.py             # PPO training
 evaluate.py         # seeded benchmark + CSV export
 demo.py             # ASCII rollout
 tests/test_env.py   # basic environment integrity tests
 results/            # local metrics (CSV ignored by Git)
 models/             # local trained policies (ignored by Git)
```

## Results

**Not measured yet.** Run the commands above and replace this section with an evidence-based table. Never claim that PPO beats the heuristic before verifying it.

| Controller | Success rate | Mean episode steps | Mean reward |
|---|---:|---:|---:|
| Random | TODO | TODO | TODO |
| Greedy + BFS | TODO | TODO | TODO |
| PPO | TODO | TODO | TODO |

Add a short section discussing failure cases, reward sensitivity and scalability: the joint-action count increases exponentially as `5^N`, so centralized joint-action PPO in this exact form will not scale comfortably to many agents.

## Reproducibility and ethics

Open-source Python stack only, no external datasets or cloud accounts. MIT-licensed code. No weapons modeling or real operational flight-control claims.
