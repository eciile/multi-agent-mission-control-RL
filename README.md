# MAMC-RL Multi-Agent Mission Control

A small, reproducible study of **cooperative mission control** in a 2D simulation. Two agents navigate obstacles and complete prioritized objectives, including a new urgent target during an episode.

The project compares **random control**, **adaptive BFS-based planning**, **centralized PPO**, and **PPO with an inference-time feasibility shield**. It studies which decisions should be learned and which can be handled by classical planning.

## Environment

- **V1:** 8×8 static grid; two agents and two targets.
- **V2:** 10×10 grid; three initial targets, optional urgent target at step 12, changing obstacles, 60-step horizon.
- **Control:** 25 joint actions (five moves per agent); fully observable state.
- **Tools:** Python, Gymnasium, Stable-Baselines3, PyTorch, pytest. All open source.

## Results

V2 **combined** scenario, 100 episodes per controller (seeds 1000–1099):

| Controller | Success | Weighted completion | Mean return | Invalid moves / episode |
|---|---:|---:|---:|---:|
| Random | 2% | 36.2% | 4.79 | 15.97 |
| Adaptive heuristic | 100% | 100% | 40.62 | 0 |
| PPO (100k training steps) | 5% | 43.1% | 3.63 | 32.76 |
| Shielded PPO (same model) | 6% | 49.8% | 13.45 | 0 |

**Finding:** The feasibility shield eliminated invalid movements and collisions in these tests, but full mission success barely improved. This motivates investigating hierarchical control: learned mission assignment with classical navigation.

These are exploratory single-checkpoint results, not evidence that RL outperforms planning. The environment is simplified and does not model realistic aircraft dynamics.

## Run locally

Python 3.11 recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python evaluate_v2.py --episodes 100 --seed-start 1000
python train_v2.py --timesteps 100000 --seed 42 --output models/ppo_v2_combined
python evaluate_shield_v2.py --model models/ppo_v2_combined.zip --episodes 100 --seed-start 1000
```

Results CSVs are in [`results/`](results/); V1 scripts and benchmarks are retained for comparison. PPO checkpoints are generated locally and are not included in Git.

## Next step

Evaluate **hierarchical PPO for mission assignment + BFS for movement** against the adaptive heuristic on dynamic objectives and obstacles.

MIT licensed; see [LICENSE](LICENSE).
