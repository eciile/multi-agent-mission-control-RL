"""V4 failure analysis: capability, fuel, deadlines, and action agreement.

Example:
 python diagnose_failures_v4.py --models models/v4_masked_seed42_50k.zip models/v4_masked_seed123_50k.zip models/v4_masked_seed456_50k.zip --episodes 200 --seed-start 2000

Creates three CSVs: *_episodes.csv, *_targets.csv, *_decisions.csv.
No V4 environment or model changes. Paths use current working directory.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from mamc_rl.controllers_v4 import HierarchicalResourceEnv, resource_heuristic_action
from mamc_rl.policies_v2 import _distance_map

TARGETS = ("A", "B", "C", "D_urgent")


def feasible_agents(base, target):
    """Feasible *now* by a shortest path, fuel and deadline; ignores other-agent conflicts.

    Only call for an active, unfinished, unexpired target. No future information.
    """
    if not (base.active[target] and not base.completed[target] and not base.expired[target]):
        return []
    dmap = _distance_map(tuple(map(int, base.targets[target])), set(base.obstacles))
    deadline_left = int(base.deadlines[target]) - int(base.steps)
    result = []
    for i in range(2):
        dist = dmap.get(tuple(map(int, base.agents[i])))
        if (base.capabilities[i, target] and dist is not None
                and dist <= int(base.fuel[i]) and dist <= deadline_left):
            result.append(i)
    return result


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path} ({len(rows)} rows)")


def label_action(action):
    parts = (int(action) // 5, int(action) % 5)
    return "/".join("idle" if p == 0 else TARGETS[p - 1] for p in parts)


def predict(model, obs, mask):
    a, _ = model.predict(obs, deterministic=True, action_masks=mask)
    result = int(a)
    if not mask[result]:
        raise AssertionError(f"Model returned masked-out action {result}")
    return result


def run_one(seed, policy, models, horizon, deadlines, fuel, compare_models):
    env = HierarchicalResourceEnv(option_horizon=horizon, deadlines=deadlines, fuel_budgets=fuel)
    obs, _ = env.reset(seed=seed)
    base = env.base
    # Per-target information is initialized before the urgent objective spawns.
    completed_at = [None] * 4
    expired_at = [None] * 4
    spawned_at = None
    feasible_at_first_visible = [None] * 4
    # The final primitive-step state at which an expiring objective could be saved
    # if an agent immediately followed an obstacle-aware shortest path.
    last_feasible_step = [None] * 4
    feasible_just_before_expiry = [None] * 4
    fuel_just_before_expiry = [None] * 4
    assigned_count = [[0] * 4 for _ in range(2)]
    first_assigned_at = [[None] * 4 for _ in range(2)]
    decisions = []
    primitive_steps = 0
    obstacle_changes = 0
    total_return = 0.0
    decision_index = 0
    original_step = base.step

    def hooked_step(action):
        nonlocal spawned_at, obstacle_changes
        before_completed = base.completed.copy()
        before_expired = base.expired.copy()
        before_fuel = tuple(map(int, base.fuel))
        # Record feasibility strictly using PRE-step state (no hindsight).
        pre_feasible = [feasible_agents(base, t) if base.active[t] and
                        not base.completed[t] and not base.expired[t] else [] for t in range(4)]
        for t in range(4):
            if base.active[t] and not base.completed[t] and not base.expired[t]:
                if feasible_at_first_visible[t] is None:
                    feasible_at_first_visible[t] = int(bool(pre_feasible[t]))
                if pre_feasible[t]:
                    last_feasible_step[t] = int(base.steps)
        result = original_step(action)
        info = result[-1]
        if info.get("obstacles_changed"):
            obstacle_changes += 1
        if info.get("target_spawned"):
            spawned_at = int(base.steps)
            # The urgent objective becomes visible *after* this primitive step.
            feasible_at_first_visible[3] = int(bool(feasible_agents(base, 3)))
        for t in range(4):
            if not before_completed[t] and base.completed[t]:
                completed_at[t] = int(base.steps)
            if not before_expired[t] and base.expired[t]:
                expired_at[t] = int(base.steps)
                feasible_just_before_expiry[t] = int(bool(pre_feasible[t]))
                fuel_just_before_expiry[t] = before_fuel
        return result

    base.step = hooked_step
    try:
        while True:
            decision_index += 1
            mask = env.action_masks()
            # All models see exactly the same live observation for disagreement analysis.
            predictions = [predict(m, obs, mask) for m in models] if compare_models else []
            if policy == "heuristic":
                action = resource_heuristic_action(env)
            else:
                action = predictions[int(policy.split("_")[-1])] if compare_models else predict(
                    models[int(policy.split("_")[-1])], obs, mask)
            if not mask[action]:
                raise AssertionError("Controller issued an invalid assignment")
            choices = (action // 5, action % 5)
            row = dict(policy=policy, seed=seed, decision=decision_index, step=int(base.steps),
                       action=int(action), assignment=label_action(action),
                       agent0_fuel=int(base.fuel[0]), agent1_fuel=int(base.fuel[1]),
                       a_active=int(base.active[0] and not base.completed[0] and not base.expired[0]),
                       a_feasible_agents=",".join(map(str, feasible_agents(base, 0))),
                       model0_action=predictions[0] if compare_models else "",
                       model1_action=predictions[1] if compare_models else "",
                       model2_action=predictions[2] if compare_models else "",
                       models_all_agree=int(len(set(predictions)) == 1) if compare_models else "")
            decisions.append(row)
            for agent, choice in enumerate(choices):
                if choice:
                    t = choice - 1
                    assigned_count[agent][t] += 1
                    if first_assigned_at[agent][t] is None:
                        first_assigned_at[agent][t] = int(base.steps)
            obs, reward, done, truncated, info = env.step(action)
            total_return += float(reward)
            primitive_steps += int(info["low_level_steps"])
            if done or truncated:
                break
        if primitive_steps != int(base.steps):
            raise AssertionError("Primitive-step accounting mismatch")
        ep = dict(policy=policy, seed=seed, success=int(info["success"]),
                  weighted_fraction=float(info["weighted_fraction"]),
                  weighted_completed=int(info["weighted_completed"]),
                  expired_targets=int(info["expired_targets"]),
                  primitive_steps=primitive_steps, decisions=decision_index,
                  return_value=total_return, obstacles_changed=obstacle_changes,
                  urgent_spawned=int(spawned_at is not None),
                  agent0_fuel_remaining=int(base.fuel[0]),
                  agent1_fuel_remaining=int(base.fuel[1]),
                  model_agreement_rate=(float(np.mean([r["models_all_agree"] for r in decisions]))
                                        if compare_models else ""))
        target_rows = []
        for t, name in enumerate(TARGETS):
            expiry_fuel = fuel_just_before_expiry[t]
            target_rows.append(dict(
                policy=policy, seed=seed, target=name,
                deadline=int(base.deadlines[t]), priority=int(base.priorities[t]),
                capable_agents=",".join(str(i) for i in range(2) if base.capabilities[i,t]),
                completed=int(base.completed[t]), expired=int(base.expired[t]),
                completion_step=completed_at[t] if completed_at[t] is not None else "",
                expiry_step=expired_at[t] if expired_at[t] is not None else "",
                first_visible_feasible=feasible_at_first_visible[t]
                    if feasible_at_first_visible[t] is not None else "",
                last_feasible_step=last_feasible_step[t]
                    if last_feasible_step[t] is not None else "",
                feasible_just_before_expiry=feasible_just_before_expiry[t]
                    if feasible_just_before_expiry[t] is not None else "",
                fuel0_just_before_expiry=expiry_fuel[0] if expiry_fuel is not None else "",
                fuel1_just_before_expiry=expiry_fuel[1] if expiry_fuel is not None else "",
                first_assigned_agent0=first_assigned_at[0][t] if first_assigned_at[0][t] is not None else "",
                first_assigned_agent1=first_assigned_at[1][t] if first_assigned_at[1][t] is not None else "",
                assignments_agent0=assigned_count[0][t],
                assignments_agent1=assigned_count[1][t],
            ))
        return ep, target_rows, decisions
    finally:
        base.step = original_step
        env.close()


def summarize(episodes, targets):
    policies = list(dict.fromkeys(x["policy"] for x in episodes))
    print("\nPolicy                Success  Weighted  Expired  Steps  Fuel left (0,1)")
    for p in policies:
        r = [x for x in episodes if x["policy"] == p]
        avg = lambda k: float(np.mean([x[k] for x in r]))
        print(f'{p:<22}{avg("success"):>6.1%}    {avg("weighted_fraction"):>6.1%}'
              f'    {avg("expired_targets"):>5.2f}    {avg("primitive_steps"):>5.1f}'
              f'    ({avg("agent0_fuel_remaining"):.1f}, {avg("agent1_fuel_remaining"):.1f})')
    print("\nTarget expiry rate by policy (%)")
    print(f'{"Target":<12}' + ''.join(f'{p:>14}' for p in policies))
    for name in TARGETS:
        line = f'{name:<12}'
        for p in policies:
            subset = [x for x in targets if x["policy"] == p and x["target"] == name]
            line += f'{np.mean([x["expired"] for x in subset]):>13.1%} '
        print(line)
    print("\nExpired targets: reachable at the primitive step immediately before expiry")
    for p in policies:
        lost = [x for x in targets if x["policy"] == p and x["expired"]]
        near = sum(x["feasible_just_before_expiry"] == 1 for x in lost)
        initial = sum(x["first_visible_feasible"] == 1 for x in lost)
        print(f'{p}: {len(lost)} expirations; {near} feasible just before expiry; '
              f'{initial} feasible when first visible')
    keys = sorted({x["seed"] for x in episodes})
    outcome = {(r["seed"], r["policy"]): r for r in episodes}
    if "heuristic" in policies:
        print("\nPaired weighted completion vs heuristic (worse / better / tied):")
        for p in policies:
            if p == "heuristic":
                continue
            diffs = [outcome[(s, p)]["weighted_fraction"] - outcome[(s, "heuristic")]["weighted_fraction"] for s in keys]
            print(f'{p}: {sum(v < -1e-9 for v in diffs)} / '
                  f'{sum(v > 1e-9 for v in diffs)} / {sum(abs(v) <= 1e-9 for v in diffs)}')
    ppos = [p for p in policies if p.startswith("ppo_")]
    if len(ppos) > 1:
        print("\nPPO pairwise success/outcome disagreements across matched seeds:")
        for i in range(len(ppos)):
            for j in range(i+1, len(ppos)):
                a,b=ppos[i],ppos[j]
                sd=sum(outcome[(s,a)]["success"] != outcome[(s,b)]["success"] for s in keys)
                wd=sum(abs(outcome[(s,a)]["weighted_fraction"]-outcome[(s,b)]["weighted_fraction"])>1e-9 for s in keys)
                print(f"{a} vs {b}: success differs on {sd}/{len(keys)}, weighted value on {wd}/{len(keys)}")
        agreed = [x["model_agreement_rate"] for x in episodes if x["policy"] == "ppo_0"]
        print(f"All three PPO actions agree on {np.mean(agreed):.1%} of decisions "
              "along PPO_0 trajectories (same observations/masks).")
    print("\nA 'feasible' target is only a current-state BFS lower-bound check; "
          "it ignores other-agent interference and future obstacle changes.")
    print("Initial seeds are paired, but policies may encounter different obstacle histories.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", required=True, nargs=3, metavar="MODEL",
                   help="three V4 MaskablePPO ZIP files (training seeds 42,123,456)")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed-start", type=int, default=2000)
    p.add_argument("--option-horizon", type=int, default=3)
    p.add_argument("--deadlines", type=int, nargs=4, default=(12,18,24,23))
    p.add_argument("--fuel", type=int, nargs=2, default=(16,18))
    p.add_argument("--output-prefix", default="results/v4_failure_diagnostic")
    args = p.parse_args()
    if args.episodes < 1 or args.option_horizon < 1:
        p.error("episodes and option-horizon must be positive")
    models = [MaskablePPO.load(path, device="cpu") for path in args.models]
    episodes, targets, decisions = [], [], []
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        for policy in ("ppo_0", "ppo_1", "ppo_2", "heuristic"):
            ep, ts, ds = run_one(seed, policy, models, args.option_horizon,
                                 tuple(args.deadlines), tuple(args.fuel),
                                 compare_models=(policy == "ppo_0"))
            episodes.append(ep)
            targets.extend(ts)
            decisions.extend(ds)
    csv_write(args.output_prefix + "_episodes.csv", episodes)
    csv_write(args.output_prefix + "_targets.csv", targets)
    csv_write(args.output_prefix + "_decisions.csv", decisions)
    summarize(episodes, targets)


if __name__ == "__main__":
    main()
