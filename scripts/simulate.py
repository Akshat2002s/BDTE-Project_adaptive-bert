"""Serve bursty traffic with fixed operating points, a greedy SLA rule and the MDP.

    python scripts/simulate.py                 # uses results/elastic_table.json + _examples.npz
    python scripts/simulate.py --name elastic

Controller inputs come from the val split (accuracy, exit fractions); the
simulation draws real test queries, so the reported accuracy is on test.
Writes results/sim_<name>.json and results/sim_<name>.png.
"""
from __future__ import annotations

import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np

from abert.controller import Action, Traffic, pareto_actions, sla_greedy, solve_mdp
from abert.sim import ServedPoint, mmpp_arrivals, simulate
from abert.utils import load_config, resolve


class Fixed:
    def __init__(self, action):
        self.action = action

    def __call__(self, q):
        return self.action


def load_points(name: str):
    table = json.loads((resolve("results") / f"{name}_table.json").read_text(encoding="utf-8"))
    ex = np.load(resolve("results") / f"{name}_examples.npz")
    cost = {(c["width"], c["exit"]): c for c in table["per_exit_cost"]}
    actions, served = {}, {}
    for row in table["points"]:
        w, pname = row["width"], row["name"]
        forced = "exit" in row
        if row["split"] == "val":
            fr = {int(e): f for e, f in row["exit_fractions"].items()}
            actions[pname] = Action(
                pname, row["accuracy"] / 100.0,
                np.array([cost[w, e]["latency_ms_p50"] / 1000.0 for e in fr]),
                np.array(list(fr.values())), row["gflops"])
        else:
            key = "gflops" if forced else "gflops_thr"
            served[pname] = ServedPoint(
                pname, ex[f"test/{pname}/exit"], ex[f"test/{pname}/correct"],
                {e: cost[w, e]["latency_ms_p50"] / 1000.0 for e in table["exits"]},
                {e: max(0.0, math.log(cost[w, e]["latency_ms_p95"] / cost[w, e]["latency_ms_p50"]) / 1.645)
                 for e in table["exits"]},
                {e: cost[w, e][key] for e in table["exits"]})
    return table, actions, served


def main() -> None:
    cfg = load_config()
    sc = cfg["sim"]
    p = argparse.ArgumentParser()
    p.add_argument("--name", default=cfg["elastic"]["out_dir"].rstrip("/").split("/")[-1])
    args = p.parse_args()

    table, actions, served = load_points(args.name)
    widths, exits = table["widths"], table["exits"]
    full = actions[f"{args.name}-w{max(widths)}-exit{max(exits)}"]
    mu = 1.0 / full.mean_s
    calm, burst = sc["calm_load"] * mu, sc["burst_load"] * mu
    traffic = Traffic.two_state(calm, burst, sc["calm_mean_s"], sc["burst_mean_s"])
    sla_s, cap = sc["sla_ms"] / 1000.0, sc["queue_capacity"]
    print(f"full model serves {mu:.2f} q/s; calm {calm:.2f} q/s, burst {burst:.2f} q/s, "
          f"mean load {traffic.mean_rate / mu:.2f}x of full-model capacity")

    candidates = pareto_actions(list(actions.values()))
    print(f"{len(candidates)} Pareto-optimal operating points on val (of {len(actions)}):")
    for a in candidates:
        print(f"  {a.name:32s} val acc {100 * a.accuracy:6.2f}%  mean service {1000 * a.mean_s:6.1f} ms")

    policies = {f"fixed: {a.name}": Fixed(a) for a in candidates}
    if full.name not in {a.name for a in candidates}:
        policies[f"fixed: {full.name}"] = Fixed(full)
    policies["greedy SLA rule"] = sla_greedy(candidates, sla_s)
    mdps = {}
    for beta in sc["betas"]:
        pol = solve_mdp(candidates, traffic, cap, sla_s, beta, sc["drop_penalty"])
        mdps[beta] = pol
        policies[f"MDP beta={beta}"] = pol

    results = {}
    for label, pol in policies.items():
        runs = []
        for seed in sc["seeds"]:
            rng = np.random.default_rng(seed)
            arr = mmpp_arrivals((calm, burst), (sc["calm_mean_s"], sc["burst_mean_s"]),
                                sc["duration_s"], rng)
            runs.append(simulate(pol, served, arr, cap, sla_s, np.random.default_rng(seed + 1000)))
        keys = [k for k in runs[0] if k != "choices"]
        results[label] = {k: float(np.mean([r[k] for r in runs])) for k in keys}
        results[label].update({f"{k}_std": float(np.std([r[k] for r in runs])) for k in keys})
        mix = {}
        for r in runs:
            for k, v in r["choices"].items():
                mix[k] = mix.get(k, 0) + v
        tot = sum(mix.values())
        results[label]["choice_mix"] = {k: v / tot for k, v in sorted(mix.items(), key=lambda kv: -kv[1])}

    print(f"\n{'policy':42s} {'acc':>6s} {'eff.acc':>7s} {'drop%':>6s} {'SLA%':>6s} "
          f"{'p95 ms':>8s} {'GFLOPs':>7s}")
    for label, r in results.items():
        print(f"{label:42s} {r['accuracy']:6.2f} {r['effective_accuracy']:7.2f} {r['drop_rate']:6.2f} "
              f"{r['sla_violation']:6.2f} {r['resp_ms_p95']:8.0f} {r['gflops_mean']:7.2f}")

    out = resolve("results")
    policy_tables = {str(b): [pol.actions[c].name for c in pol.choice[1:]] for b, pol in mdps.items()}
    (out / f"sim_{args.name}.json").write_text(json.dumps({
        "config": sc, "full_model_rate": mu, "candidates": [a.name for a in candidates],
        "results": results, "mdp_policies": policy_tables}, indent=1), encoding="utf-8")

    # figure: (left) effective accuracy vs SLA violations, (right) MDP policy
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fixed = [(r["sla_violation"], r["effective_accuracy"], l) for l, r in results.items() if l.startswith("fixed")]
    ax.scatter([f[0] for f in fixed], [f[1] for f in fixed], marker="s", color="#7a7a7a", label="fixed point")
    for x, y, l in fixed:
        ax.annotate(l.split(f"{args.name}-")[-1], (x, y), fontsize=6, xytext=(3, 2), textcoords="offset points")
    g = results["greedy SLA rule"]
    ax.scatter(g["sla_violation"], g["effective_accuracy"], marker="^", s=60, color="#d08a1e", label="greedy SLA rule")
    m = [(results[f"MDP beta={b}"]["sla_violation"], results[f"MDP beta={b}"]["effective_accuracy"], b) for b in mdps]
    ax.plot([x for x, _, _ in m], [y for _, y, _ in m], "o-", color="#2a6fdb", label="MDP (beta sweep)")
    for x, y, b in m:
        ax.annotate(f"b={b}", (x, y), fontsize=7, xytext=(3, -8), textcoords="offset points", color="#2a6fdb")
    ax.set_xlabel(f"SLA violations incl. drops (% of queries, SLA {sc['sla_ms']} ms)")
    ax.set_ylabel("Effective test accuracy (%, drops count as wrong)")
    ax.set_title("Bursty traffic: accuracy vs SLA")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    mid = sorted(mdps)[len(mdps) // 2]
    pol = mdps[mid]
    qs = np.arange(1, cap + 1)
    ax2.step(qs, [100 * pol.actions[c].accuracy for c in pol.choice[1:]], where="mid", color="#2a6fdb")
    ax2.set_xlabel("queries in system")
    ax2.set_ylabel("val accuracy of chosen point (%)", color="#2a6fdb")
    ax3 = ax2.twinx()
    ax3.step(qs, [1000 * pol.actions[c].mean_s for c in pol.choice[1:]], where="mid", color="#d08a1e")
    ax3.set_ylabel("mean service time (ms)", color="#d08a1e")
    ax2.set_title(f"MDP policy (beta={mid})")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"sim_{args.name}.png", dpi=200)
    print(f"wrote {out / f'sim_{args.name}.json'} and {out / f'sim_{args.name}.png'}")


if __name__ == "__main__":
    main()