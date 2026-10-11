"""Measure every operating point of a trained ElasticBert and register it.

    python scripts/measure_elastic.py                                 # checkpoints/elastic
    python scripts/measure_elastic.py checkpoints/elastic-nokd --name elastic-nokd

Operating points (all family "adaptive", val and test rows each):
  * forced:    every (width, exit)                      -> <name>-w<width>-exit<L>
  * threshold: every (width, threshold in config)       -> <name>-w<width>-thr<t>
    Cost is the expected cost over that split's own exit fractions
    (flops.expected_flops); a query leaving at exit L also paid for the exit heads
    before L.

Also writes, for the controller and the simulation:
  results/<name>_table.json     per point: accuracy, GFLOPs, exit fractions, latency
  results/<name>_examples.npz   per point and split: each query's exit layer and
                                whether it was answered correctly
Accuracies are computed on GPU if there is one; FLOPs and latency always on CPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from abert import registry
from abert.bench import count_parameters, cpu_latency_ms, host_info, model_size_mb
from abert.data import load_split, tokenize
from abert.elastic import ElasticBert, collect_all_logits, exit_head_gflops, threshold_exits
from abert.flops import expected_flops, model_gflops
from abert.models import build_tokenizer
from abert.utils import Tee, get_device, load_config, resolve, run_info


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path, nargs="?", default=Path(cfg["elastic"]["out_dir"]))
    p.add_argument("--name", default=None, help="point-name prefix (default: checkpoint folder name)")
    p.add_argument("--notes", default="")
    args = p.parse_args()
    name = args.name or args.checkpoint.name
    ckpt = resolve(args.checkpoint)

    Tee(resolve("logs") / f"measure_{name}.log")
    print(f"=== measure {json.dumps({'name': name, 'checkpoint': args.checkpoint.as_posix(), **run_info()})}")
    max_len, m = cfg["data"]["max_len"], cfg["measure"]
    widths, thresholds = cfg["elastic"]["widths"], cfg["thresholds"]

    model = ElasticBert.load(ckpt)
    exits = model.exit_layers
    tokenizer = build_tokenizer(ckpt)
    frames = load_split(resolve(cfg["data"]["split_file"]))
    loaders = {s: DataLoader(tokenize(frames[s], tokenizer, max_len), batch_size=256)
               for s in ("val", "test")}

    # accuracy inputs: logits of every exit, every width, both splits
    device = get_device()
    model.to(device)
    logits = {(w, s): collect_all_logits(model, loaders[s], device, w)
              for w in widths for s in ("val", "test")}

    # sanity check: the real threshold forward agrees with the replay used below
    ids, mask, _ = next(iter(loaders["test"]))
    model.set_width(widths[-1])
    with torch.no_grad():
        real = model(ids.to(device), mask.to(device), threshold=0.9)
    _, replay_exit = threshold_exits(
        {e: lg[: len(ids)] for e, lg in logits[widths[-1], "test"][0].items()}, 0.9)
    agree = (real.exit_layer.cpu() == replay_exit).float().mean().item()
    print(f"threshold replay agrees with real forward on {100 * agree:.1f}% of {len(ids)} queries")

    # cost per (width, exit) on CPU
    model.to("cpu")
    head = exit_head_gflops(model)
    cost = {}
    for w in widths:
        model.set_width(w)
        for i, e in enumerate(exits):
            g = model_gflops(model, max_len, force_exit=e)
            lat = cpu_latency_ms(model, max_len, m["latency_threads"], m["latency_warmup"],
                                 m["latency_runs"], force_exit=e)
            cost[w, e] = {"gflops": g, "gflops_thr": g + i * head, **lat}
            print(f"  w{w:<5} exit{e:<3d} {g:7.3f} GFLOPs  p50 {lat['latency_ms_p50']:6.1f} ms  "
                  f"p95 {lat['latency_ms_p95']:6.1f} ms")

    model.set_width(1.0)
    common = {"family": "adaptive", "params": count_parameters(model),
              "size_mb": model_size_mb(model), "host": host_info(m["latency_threads"]),
              "checkpoint": args.checkpoint.as_posix(), "notes": args.notes}

    table, examples = [], {}

    def add(point: str, w: float, kind: str, value, split: str, pred, exit_at, labels):
        fracs = {int(e): float((exit_at == e).float().mean()) for e in exits}
        fracs = {e: f for e, f in fracs.items() if f > 0}
        if kind == "exit":
            gflops = cost[w, value]["gflops"]
            lat50 = cost[w, value]["latency_ms_p50"]
            lat95 = cost[w, value]["latency_ms_p95"]
        else:
            gflops = expected_flops(fracs, {e: cost[w, e]["gflops_thr"] for e in exits})
            lat50 = sum(f * cost[w, e]["latency_ms_p50"] for e, f in fracs.items())
            lat95 = sum(f * cost[w, e]["latency_ms_p95"] for e, f in fracs.items())
        correct = (pred == labels)
        row = {"name": point, "width": w, kind: value, "split": split,
               "accuracy": 100.0 * correct.float().mean().item(), "gflops": gflops,
               "latency_ms_p50": lat50, "latency_ms_p95": lat95,
               "exit_fractions": {str(e): f for e, f in fracs.items()}}
        registry.record({**common, **row})
        table.append(row)
        examples[f"{split}/{point}/exit"] = exit_at.numpy().astype(np.int8)
        examples[f"{split}/{point}/correct"] = correct.numpy()

    for w in widths:
        for split in ("val", "test"):
            all_logits, labels = logits[w, split]
            for e in exits:
                pred = all_logits[e].argmax(-1)
                add(f"{name}-w{w}-exit{e}", w, "exit", e, split, pred,
                    torch.full_like(labels, e), labels)
            for t in thresholds:
                pred, exit_at = threshold_exits(all_logits, t)
                add(f"{name}-w{w}-thr{t}", w, "threshold", t, split, pred, exit_at, labels)

    print(f"\n{'point':32s} {'val':>6s} {'test':>6s} {'GFLOPs(test)':>12s} {'p50 ms':>7s}")
    by = {(r["name"], r["split"]): r for r in table}
    for pname in dict.fromkeys(r["name"] for r in table):
        v, t = by[pname, "val"], by[pname, "test"]
        print(f"{pname:32s} {v['accuracy']:6.2f} {t['accuracy']:6.2f} {t['gflops']:12.3f} "
              f"{t['latency_ms_p50']:7.1f}")

    out = resolve("results")
    (out / f"{name}_table.json").write_text(json.dumps({
        "name": name, "exits": list(exits), "widths": widths, "host": common["host"],
        "per_exit_cost": [{"width": w, "exit": e, **c} for (w, e), c in cost.items()],
        "points": table}, indent=1), encoding="utf-8")
    np.savez_compressed(out / f"{name}_examples.npz", **examples)
    print(f"wrote {out / f'{name}_table.json'} and {out / f'{name}_examples.npz'}")


if __name__ == "__main__":
    main()
