"""Measure a trained static checkpoint and register it in results/points.jsonl.

    python scripts/measure.py checkpoints/bert-base --name bert-base

Records accuracy on val and test, GFLOPs, params, size and CPU latency.
Run it on the machine whose latency you want to report, and keep that the
same machine for every point.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from abert import registry
from abert.bench import count_parameters, cpu_latency_ms, host_info, model_size_mb
from abert.data import get_loaders, num_labels
from abert.flops import model_gflops
from abert.models import build_classifier, build_tokenizer
from abert.train import evaluate
from abert.utils import ROOT, Tee, get_device, load_config, resolve, run_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    Tee(ROOT / "logs" / f"measure_{args.name}.log")
    print(f"=== measure {json.dumps({'name': args.name, 'checkpoint': str(args.checkpoint), **run_info()})}")
    cfg = load_config()
    max_len, m = cfg["data"]["max_len"], cfg["measure"]
    split_file = resolve(cfg["data"]["split_file"])

    tokenizer = build_tokenizer(args.checkpoint)
    model = build_classifier(args.checkpoint, num_labels(split_file))
    device = get_device()
    loaders = get_loaders(split_file, tokenizer, max_len, batch_size=256)
    acc = {s: evaluate(model.to(device), loaders[s], device) for s in ("val", "test")}

    model = model.to("cpu")
    common = {
        "name": args.name,
        "family": "static",
        "gflops": model_gflops(model, max_len),
        "params": count_parameters(model),
        "size_mb": model_size_mb(model),
        **cpu_latency_ms(model, max_len, m["latency_threads"],
                         m["latency_warmup"], m["latency_runs"]),
        "host": host_info(m["latency_threads"]),
        "checkpoint": str(args.checkpoint),
        "notes": args.notes,
    }
    for split in ("val", "test"):
        row = registry.record({**common, "split": split, "accuracy": acc[split]})
        print(f"{row['name']:20s} {split:4s} acc {row['accuracy']:6.2f}%  "
              f"{row['gflops']:.3f} GFLOPs  p50 {row['latency_ms_p50']:.1f} ms")


if __name__ == "__main__":
    main()
