"""Plain fine-tuning loop for the static baselines (best epoch chosen on val)."""
from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


@torch.no_grad()
def evaluate(model, loader: DataLoader, device) -> float:
    """Accuracy in percent."""
    model.eval()
    correct = total = 0
    for input_ids, attention_mask, labels in loader:
        logits = model(input_ids=input_ids.to(device),
                       attention_mask=attention_mask.to(device)).logits
        correct += (logits.argmax(-1).cpu() == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / total


def has_tensor_cores(device) -> bool:
    """Best-effort check: compute capability >= 7.0, minus the Turing parts that
    report 7.5 but ship without tensor cores (GTX 16xx, MX 4xx/5xx)."""
    if device.type != "cuda":
        return False
    props = torch.cuda.get_device_properties(device)
    no_tc = ("GTX 16" in props.name) or (" MX" in props.name)
    return props.major >= 7 and not no_tc


def fit(model, loaders: dict[str, DataLoader], device, *, lr: float, epochs: int,
        out_dir: Path, tokenizer=None, amp: bool = False, warmup_frac: float = 0.1,
        weight_decay: float = 0.01) -> float:
    """Train, keep the checkpoint with the best val accuracy in `out_dir`.

    Returns that best val accuracy. The test set is never touched here.
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * len(loaders["train"])
    sched = get_linear_schedule_with_warmup(opt, int(warmup_frac * steps), steps)
    use_amp = amp and device.type == "cuda"
    if use_amp and not has_tensor_cores(device):
        print(f"WARNING: fp16 AMP on {torch.cuda.get_device_name(device)}, which has no "
              "tensor cores. Expect it to be several times slower than fp32 (--no-amp).")
    print(f"precision: {'fp16 autocast' if use_amp else 'fp32'} on {device}")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = -1.0
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        for input_ids, attention_mask, labels in tqdm(loaders["train"], desc=f"epoch {epoch}"):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = model(input_ids=input_ids.to(device),
                             attention_mask=attention_mask.to(device),
                             labels=labels.to(device)).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

        val_acc = evaluate(model, loaders["val"], device)
        mem = (f"  peak_mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
               if device.type == "cuda" else "")
        print(f"epoch {epoch}/{epochs}  val_acc {val_acc:.2f}%  "
              f"time {time.perf_counter() - start:.0f}s{mem}")
        if val_acc > best:
            best = val_acc
            model.save_pretrained(out_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(out_dir)
    return best
