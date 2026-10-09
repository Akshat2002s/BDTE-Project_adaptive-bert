"""Accuracy vs GFLOPs from results/points.jsonl (test split by default).

    python scripts/plot_pareto.py            # -> results/pareto_test.png
    python scripts/plot_pareto.py --split val
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from abert import registry
from abert.utils import ROOT

STYLE = {"static": dict(marker="s", color="#7a7a7a"),
         "adaptive": dict(marker="o", color="#2a6fdb")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    df = registry.load(args.split)
    if df.empty:
        raise SystemExit("No points registered for this split yet")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for family, group in df.groupby("family"):
        ax.scatter(group["gflops"], group["accuracy"], label=family, s=40, **STYLE[family])
        for _, r in group.iterrows():
            ax.annotate(r["name"], (r["gflops"], r["accuracy"]), fontsize=7,
                        xytext=(4, 3), textcoords="offset points")

    adaptive = df[df["family"] == "adaptive"]
    if not adaptive.empty:
        front = adaptive[registry.pareto_mask(adaptive)].sort_values("gflops")
        ax.plot(front["gflops"], front["accuracy"], color=STYLE["adaptive"]["color"],
                lw=1.5, label="adaptive frontier")

    ax.set_xlabel("GFLOPs per query (seq len fixed, torch FlopCounterMode)")
    ax.set_ylabel(f"Accuracy on {args.split} (%)")
    ax.set_title("Banking77: accuracy vs compute")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = ROOT / "results" / f"pareto_{args.split}.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
