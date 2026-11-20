# V4 frozen final evaluation

Missions: 400 (seeds 30000–30399); 
paired bootstrap: 10000 resamples, 95% percentile CI.

| Controller | Success (95% CI) | Weighted (95% CI) | Expired | ms/decision |
|---|---:|---:|---:|---:|
| masked_random | 6.8% [4.5%, 9.2%] | 55.4% [53.0%, 57.7%] | 1.92 | 0.018 |
| resource_heuristic | 44.0% [39.2%, 48.8%] | 86.1% [84.6%, 87.6%] | 0.64 | 0.413 |
| masked_ppo | 33.0% [28.2%, 37.5%] | 85.1% [83.4%, 86.8%] | 0.85 | 0.450 |
| lookahead_d4 | 51.0% [46.2%, 56.0%] | 86.7% [85.1%, 88.2%] | 0.56 | 35.968 |

## Paired differences relative to resource heuristic

| Controller | Success difference (95% CI) | Weighted difference (95% CI) |
|---|---:|---:|
| masked_random | -37.25 pp [-42.25, -32.25] | -30.72 pp [-33.11, -28.39] |
| masked_ppo | -11.00 pp [-14.75, -7.25] | -0.94 pp [-2.11, +0.19] |
| lookahead_d4 | +7.00 pp [+3.00, +11.00] | +0.58 pp [-0.61, +1.81] |

## Interpretation guardrails

- Bootstrap intervals reflect variability across mission seeds for the specified simulator configuration; they are not proof of real-world generalization.
- Runs share initial mission seeds, not necessarily identical full stochastic trajectories.
- Timing covers controller computation only on this machine, excludes simulation execution, and is not a hardware-independent benchmark.
- PPO is deterministic at inference and runs on CPU; the model checkpoint was selected before this evaluation.
- Lookahead is an approximate, non-oracle forecast (future obstacle changes and hidden target spawning suppressed).
- This seed range must not be reused for model or hyperparameter selection after inspecting results.
