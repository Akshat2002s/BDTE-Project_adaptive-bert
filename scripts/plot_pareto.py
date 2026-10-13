"""Accuracy vs GFLOPs from results/points.jsonl (test split by default).

    python scripts/plot_pareto.py                     # -> results/pareto_test.png
    python scripts/plot_pareto.py --split val
    python scripts/plot_pareto.py --adaptive elastic-nokd

Static baselines and the adaptive frontier are labelled; every other operating
point of the chosen elastic run is drawn faintly (hollow = forced exit,
filled = confidence threshold).
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from abert import registry
from abert.utils import ROOT

STATIC = dict(marker="s", color="#7a7a7a", s=60, zorder=4)
BLUE = "#2a6fdb"
STATIC_NAMES = {"bert-base": "BERT-base", "distilbert": "DistilBERT"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--adaptive", default="elastic",
                        help="elastic run whose points to plot (point-name prefix)")
    args = parser.parse_args()

    df = registry.load(args.split)
    if df.empty:
        raise SystemExit("No points registered for this split yet")
    static = df[df["family"] == "static"]
    adaptive = df[(df["family"] == "adaptive") & df["name"].str.match(rf"{args.adaptive}-w[0-9.]+-")]

    fig, ax = plt.subplots(figsize=(8, 5))

    if not adaptive.empty:
        thr = adaptive["name"].str.contains("-thr")
        ax.scatter(adaptive.loc[~thr, "gflops"], adaptive.loc[~thr, "accuracy"], s=22,
                   facecolors="none", edgecolors=BLUE, alpha=0.45, label="forced exit (width, exit)")
        ax.scatter(adaptive.loc[thr, "gflops"], adaptive.loc[thr, "accuracy"], s=16,
                   color=BLUE, alpha=0.3, label="confidence threshold (width, threshold)")
        front = adaptive[registry.pareto_mask(adaptive)].sort_values("gflops")
        ax.plot(front["gflops"], front["accuracy"], "o-", color=BLUE, lw=1.6, ms=5,
                label="adaptive frontier", zorder=3)
        last_x, k = -1.0, 0
        for i, (_, r) in enumerate(front.iterrows()):
            # thin out labels where frontier points crowd together; always label both ends
            if r["gflops"] - last_x < 0.3 and i != len(front) - 1:
                continue
            short = r["name"].removeprefix(f"{args.adaptive}-")
            ax.annotate(short, (r["gflops"], r["accuracy"]), fontsize=7, color=BLUE,
                        xytext=(5, -11 if k % 2 else 5), textcoords="offset points")
            last_x, k = r["gflops"], k + 1

    ax.scatter(static["gflops"], static["accuracy"], label="static baseline", **STATIC)
    for _, r in static.iterrows():
        ax.annotate(f"{STATIC_NAMES.get(r['name'], r['name'])}\n{r['accuracy']:.2f}%",
                    (r["gflops"], r["accuracy"]), fontsize=8, color="#555555",
                    xytext=(-12, -26), textcoords="offset points")

    ax.set_xlabel("GFLOPs per query (seq len 64, torch FlopCounterMode)")
    ax.set_ylabel(f"Accuracy on {args.split} (%)")
    ax.set_title("Banking77: accuracy vs compute")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = ROOT / "results" / f"pareto_{args.split}.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
