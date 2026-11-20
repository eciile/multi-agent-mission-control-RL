# MAMC-RL Multi-Agent Mission Control

A research prototype exploring **hierarchical reinforcement learning for cooperative mission control** in a simplified 2D simulator. Two agents coordinate to complete prioritized objectives while obstacles and mission requirements change.

## Approach

- **V2:** Dynamic targets and obstacles; compare flat PPO, an adaptive heuristic, and hierarchical control.
- **Hierarchical controller:** MaskablePPO selects target assignments, state-dependent masks exclude unavailable objectives, and BFS handles navigation.
- **V3:** Adds deadlines and target expiry to study time-critical allocation decisions. V2 is preserved for comparison.

## Results

| Benchmark | Controller | Result |
|---|---|---|
| V2, 500 evaluation scenarios | Masked hierarchical PPO | **100% success** across each of 3 training seeds; ~21.2–21.6 primitive steps/episode |
| V2, same 500 scenarios | Masked random assignment + BFS | 91.8% success; 34.3 primitive steps/episode |
| V3, 200 evaluation scenarios | Deadline-aware heuristic + BFS | **98.9%** priority-weighted completion; 93.5% success |
| V3, same 200 scenarios | Masked random assignment + BFS | 63.6% priority-weighted completion; 15.5% success |

**Interpretation:** Feasibility masking and classical navigation make hierarchical learning effective on V2. V3 introduces meaningful deadline tradeoffs; **V3 PPO has not yet been trained or evaluated**. These are results from a toy simulator.

## Run locally

Python 3.11 recommended. Install `requirements.txt` and `sb3-contrib`:

```powershell
python -m pip install -r requirements.txt sb3-contrib
python -m pytest -q
python calibrate_v3.py --episodes 200 --seed-start 2000
python train_masked_hierarchical_v2.py --timesteps 10000 --seed 42 --output models/hier_masked_v2
python evaluate_masked_hierarchical_v2.py --model models/hier_masked_v2.zip --episodes 100 --seed-start 1000
```

Results are saved under `results/`; trained model checkpoints are generated locally.

**Next:** Train and evaluate deadline-aware masked PPO on V3 against the calibrated classical baselines.

MIT licensed — see [LICENSE](LICENSE).
