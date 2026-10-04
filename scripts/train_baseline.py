"""Fine-tune one static baseline from configs/base.yaml.

    python scripts/train_baseline.py bert-base
    python scripts/train_baseline.py distilbert
    python scripts/train_baseline.py bert-base --amp   # fp16, only on tensor-core GPUs

Writes the best-on-val checkpoint to checkpoints/<name>/. Then run
scripts/measure.py on it to register the operating point.
"""
from __future__ import annotations

import argparse
import json

from abert.data import get_loaders, num_labels
from abert.models import build_classifier, build_tokenizer
from abert.train import fit
from abert.utils import ROOT, Tee, get_device, load_config, resolve, run_info, set_seed


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(cfg["baselines"]))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction,
                        default=cfg["train"]["amp"],
                        help="fp16 mixed precision (default from configs/base.yaml train.amp)")
    args = parser.parse_args()

    spec = cfg["baselines"][args.name]
    out_dir = ROOT / "checkpoints" / args.name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists. Baselines are frozen; delete it deliberately to retrain.")

    Tee(ROOT / "logs" / f"train_{args.name}.log")
    meta = {"name": args.name, **spec, "amp": args.amp, "seed": cfg["seed"],
            "max_len": cfg["data"]["max_len"], **run_info()}
    print(f"=== run {json.dumps(meta)}")
    set_seed(cfg["seed"])
    split_file = resolve(cfg["data"]["split_file"])

    tokenizer = build_tokenizer(spec["hf_name"])
    model = build_classifier(spec["hf_name"], num_labels(split_file))
    loaders = get_loaders(split_file, tokenizer, cfg["data"]["max_len"], spec["batch_size"])

    best = fit(model, loaders, get_device(), lr=spec["lr"], epochs=spec["epochs"],
               out_dir=out_dir, tokenizer=tokenizer, amp=args.amp)
    print(f"best val acc {best:.2f}% -> {out_dir}")
    (out_dir / "train_meta.json").write_text(
        json.dumps({**meta, "best_val_acc": best}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
