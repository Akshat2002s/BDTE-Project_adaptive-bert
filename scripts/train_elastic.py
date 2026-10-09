"""Train the elastic (width x depth) BERT.  YOU WRITE THE TODO PARTS: guide Step 4.

Provided: arguments, logging, data loading, saving. Yours: teacher_logits, exit_loss,
and the sandwich training step inside main().


    python scripts/train_elastic.py                       # main model -> checkpoints/elastic
    python scripts/train_elastic.py --no-kd --out checkpoints/elastic-nokd          # ablation
    python scripts/train_elastic.py --random-order --out checkpoints/elastic-random # ablation

Steps:
  1. Teacher logits: run the fine-tuned bert-base over the train set once and keep
     the logits, so the teacher never has to sit in GPU memory during training.
  2. Importance reordering: rank every layer's heads and FFN neurons by
     |d loss / d gate| (Michel et al. 2019) and sort them, so width w = prefix slice.
  3. Sandwich rule: each step trains the largest width, the smallest width and
     `sandwich_random` random middle widths; every width trains all exits with
     alpha * KD(teacher) + (1 - alpha) * CE. Gradients of all widths are summed,
     then one optimizer step.
  4. Each epoch: val accuracy of every (width, exit); keep the epoch with the best
     mean. The test set is never touched here.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from abert.data import load_split, num_labels, tokenize
from abert.elastic import ElasticBert, collect_all_logits
from abert.models import build_classifier, build_tokenizer
from abert.train import has_tensor_cores
from abert.utils import Tee, get_device, load_config, resolve, run_info, set_seed


@torch.no_grad()
def teacher_logits(teacher, dataset: TensorDataset, device, batch_size: int = 256) -> torch.Tensor:
    """Run the teacher over the train set once and keep the logits."""
    teacher.eval().to(device)
    teach_logits = []
    for ids, mask, _ in DataLoader(dataset, batch_size=batch_size):
        teach_logits.append(teacher(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float().cpu())
    return torch.cat(teach_logits)

def exit_loss(all_logits: dict[int, torch.Tensor], labels, t_logits, alpha: float, temp: float):
    """Mean over exits of alpha * KD + (1 - alpha) * CE."""
    losses = []
    for logits in all_logits.values():
        ce = F.cross_entropy(logits, labels)
        if alpha > 0:
            kd = F.kl_div(F.log_softmax(logits / temp, -1),      # student: log-probabilities
                          F.softmax(t_logits / temp, -1),        # teacher: probabilities
                          reduction="batchmean") * temp ** 2
            losses.append(alpha * kd + (1 - alpha) * ce)
        else:
            losses.append(ce)                                   # --no-kd ablation
    return torch.stack(losses).mean()


def val_table(model, loader, device, widths) -> dict[str, float]:
    table = {}
    for w in widths:
        logits, labels = collect_all_logits(model, loader, device, w)
        for e, lg in logits.items():
            table[f"w{w}-exit{e}"] = 100.0 * (lg.argmax(-1) == labels).float().mean().item()
    return table


def print_table(table: dict[str, float], widths, exits) -> None:
    print("        " + "".join(f"exit{e:<6d}" for e in exits))
    for w in widths:
        print(f"  w{w:<5}" + "".join(f"{table[f'w{w}-exit{e}']:6.2f}    " for e in exits))


def main() -> None:
    cfg = load_config()
    ec = cfg["elastic"]
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default=ec["teacher"])
    p.add_argument("--out", default=ec["out_dir"])
    p.add_argument("--epochs", type=int, default=ec["epochs"])
    p.add_argument("--no-kd", action="store_true", help="ablation: cross-entropy only")
    p.add_argument("--random-order", action="store_true",
                   help="ablation: random instead of importance-based head/neuron order")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=cfg["train"]["amp"])
    args = p.parse_args()

    out_dir, teacher_dir = resolve(args.out), resolve(args.teacher)
    if (out_dir / "elastic.pt").exists():
        raise SystemExit(f"{out_dir} already has a model; delete it deliberately to retrain.")
    Tee(resolve("logs") / f"train_{out_dir.name}.log")
    widths, exits = list(ec["widths"]), tuple(ec["exit_layers"])
    alpha = 0.0 if args.no_kd else ec["kd_alpha"]
    meta = {"name": out_dir.name, "teacher": str(args.teacher), **ec, "epochs": args.epochs,
            "kd_alpha": alpha, "random_order": args.random_order, "amp": args.amp,
            "seed": cfg["seed"], "max_len": cfg["data"]["max_len"], **run_info()}
    print(f"=== run {json.dumps(meta)}")
    set_seed(cfg["seed"])
    device = get_device()
    split_file = resolve(cfg["data"]["split_file"])
    max_len = cfg["data"]["max_len"]

    tokenizer = build_tokenizer(teacher_dir)
    frames = load_split(split_file)
    train_ds = tokenize(frames["train"], tokenizer, max_len)
    val_loader = DataLoader(tokenize(frames["val"], tokenizer, max_len), batch_size=256)

    # 1. teacher logits, then the student starts as a copy of the teacher
    teacher = build_classifier(teacher_dir, num_labels(split_file))
    start = time.perf_counter()
    t_logits = teacher_logits(teacher, train_ds, device)
    print(f"teacher logits for {len(t_logits)} train queries in {time.perf_counter() - start:.0f}s")
    model = ElasticBert.from_classifier(teacher.cpu(), exits)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.to(device)
    train_loader = DataLoader(TensorDataset(*train_ds.tensors, t_logits),
                              batch_size=ec["batch_size"], shuffle=True)

    # 2. importance reordering
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.random_order:
        g = torch.Generator().manual_seed(cfg["seed"])
        L, H, FF = len(model.layers), model.config.num_attention_heads, model.config.intermediate_size
        head_imp, ff_imp = torch.rand(L, H, generator=g), torch.rand(L, FF, generator=g)
        print("ordering: RANDOM (ablation)")
    else:
        start = time.perf_counter()
        batches = [b[:3] for _, b in zip(range(ec["importance_batches"]), train_loader)]
        head_imp, ff_imp = model.compute_importance(batches, device)
        print(f"importance from {len(batches)} batches in {time.perf_counter() - start:.0f}s")
    model.reorder(head_imp, ff_imp)
    torch.save({"head": head_imp, "ffn": ff_imp}, out_dir / "importance.pt")

    before = val_table(model, val_loader, device, widths)
    print("val accuracy after reordering, before training:")
    print_table(before, widths, exits)

    # 3. sandwich training
    opt = torch.optim.AdamW(model.parameters(), lr=ec["lr"], weight_decay=0.01)
    steps = args.epochs * len(train_loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_amp = args.amp and device.type == "cuda"
    if use_amp and not has_tensor_cores(device):
        print("WARNING: fp16 AMP on a GPU without tensor cores is several times slower (--no-amp).")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    largest, smallest = max(widths), min(widths)
    middle = [w for w in widths if w not in (largest, smallest)]
    rng = random.Random(cfg["seed"])

    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        for ids, mask, labels, t_lg in tqdm(train_loader, desc=f"epoch {epoch}"):
            ids, mask, labels, t_lg = (t.to(device) for t in (ids, mask, labels, t_lg))
            step_widths = [largest, smallest] + rng.sample(middle, min(ec["sandwich_random"], len(middle)))
            opt.zero_grad(set_to_none=True)
            for w in step_widths:
                model.set_width(w)
                with torch.autocast(device.type, enabled=use_amp):
                    out = model(ids, mask, return_all=True)
                    loss = exit_loss(out.all_logits, labels, t_lg, alpha, ec["kd_temperature"])
                scaler.scale(loss / len(step_widths)).backward()   # gradients add up across widths
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)        # one update for all widths together
            scaler.update()
            sched.step()

        table = val_table(model, val_loader, device, widths)
        mean = sum(table.values()) / len(table)
        mem = (f"  peak_mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
               if device.type == "cuda" else "")
        print(f"epoch {epoch}/{args.epochs}  mean val_acc {mean:.2f}%  "
              f"time {time.perf_counter() - start:.0f}s{mem}")
        print_table(table, widths, exits)
        history.append({"epoch": epoch, "mean_val_acc": mean, "val": table})
        if mean > best:
            best = mean
            model.set_width(1.0)
            model.save(out_dir)
            tokenizer.save_pretrained(out_dir)

    print(f"best mean val acc {best:.2f}% -> {out_dir}")
    (out_dir / "train_meta.json").write_text(json.dumps(
        {**meta, "best_mean_val_acc": best, "val_before_training": before, "history": history},
        indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
