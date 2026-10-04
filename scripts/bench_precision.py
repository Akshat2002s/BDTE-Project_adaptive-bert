"""Why is training slow? fp32 vs fp16-autocast, measured at three levels.

    python scripts/bench_precision.py

Level 1  raw GEMMs with BERT's shapes, fp32 vs fp16 -> achieved TFLOPS
Level 2  full BERT-base training steps on random tokens, one factor at a time
         (precision x batch size, plus a fwd / fwd+bwd / full-step breakdown),
         variants interleaved A,B,A,B so thermal drift can't favour one of them
Level 3  torch.profiler: top GPU kernels, and GPU-busy time vs wall time per step

Random weights and random tokens: nothing is downloaded, and speed does not
depend on the values. Results are also written to logs/bench_precision.json.
"""
from __future__ import annotations

import json
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import BertConfig, BertForSequenceClassification

from abert.utils import ROOT

SEQ, LABELS, VOCAB = 64, 77, 30522
DEV = torch.device("cuda")


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


# --------------------------------------------------------------------------
# Level 1: GEMMs
# --------------------------------------------------------------------------

def gemm_tflops(m: int, k: int, n: int, dtype: torch.dtype, reps: int = 50) -> float:
    a = torch.randn(m, k, device=DEV, dtype=dtype)
    b = torch.randn(k, n, device=DEV, dtype=dtype)
    for _ in range(10):
        a @ b
    start = sync_time()
    for _ in range(reps):
        a @ b
    elapsed = sync_time() - start
    return 2 * m * k * n * reps / elapsed / 1e12


def level1() -> dict:
    print("\n=== Level 1: raw GEMM throughput (BERT FFN shapes, batch 32 x seq 64) ===")
    shapes = {"up   2048x768 @ 768x3072": (2048, 768, 3072),
              "down 2048x3072 @ 3072x768": (2048, 3072, 768)}
    out = {}
    for label, (m, k, n) in shapes.items():
        row = {name: gemm_tflops(m, k, n, dt)
               for name, dt in (("fp32", torch.float32), ("fp16", torch.float16))}
        out[label] = row
        print(f"  {label}:  fp32 {row['fp32']:5.2f} TFLOPS   fp16 {row['fp16']:5.2f} TFLOPS"
              f"   (fp16/fp32 = {row['fp16'] / row['fp32']:.2f}x)")
    return out


# --------------------------------------------------------------------------
# Level 2: training steps
# --------------------------------------------------------------------------

def make_model():
    cfg = BertConfig(num_labels=LABELS, attn_implementation="eager")  # bert-base sizes
    return BertForSequenceClassification(cfg).to(DEV).train()


def make_batch(bs: int):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, VOCAB, (bs, SEQ), generator=g).to(DEV)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "labels": torch.randint(0, LABELS, (bs,), generator=g).to(DEV)}


def run_step(model, opt, scaler, batch, amp: bool, phase: str = "step") -> None:
    """phase: 'fwd' | 'fwd+bwd' | 'step' (fwd + bwd + optimizer)."""
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        loss = model(**batch).loss
    if phase == "fwd":
        return
    scaler.scale(loss).backward()
    if phase == "fwd+bwd":
        return
    scaler.step(opt)
    scaler.update()


def time_steps(model, opt, scalers, batch, amp, phase, warmup, steps, label) -> list[float]:
    scaler = scalers[amp]
    for _ in range(warmup):
        run_step(model, opt, scaler, batch, amp, phase)
    times = []
    for i in range(steps):
        start = sync_time()
        run_step(model, opt, scaler, batch, amp, phase)
        times.append(sync_time() - start)
        print(f"    {label:28s} step {i + 1:2d}/{steps}  {times[-1]:.3f} s", flush=True)
    return times


def level2(model, opt, scalers) -> dict:
    print("\n=== Level 2a: full training step, precision x batch (interleaved A,B,A,B) ===")
    variants = [("fp32", False, 32), ("fp16-autocast", True, 32),
                ("fp32", False, 16), ("fp16-autocast", True, 16)]
    batches = {bs: make_batch(bs) for bs in (16, 32)}
    times: dict[str, list[float]] = {}
    for rnd in (1, 2):
        print(f"  round {rnd}/2")
        for name, amp, bs in variants:
            key = f"{name} bs{bs}"
            times.setdefault(key, []).extend(
                time_steps(model, opt, scalers, batches[bs], amp, "step", 2, 6, key))

    summary = {}
    print("\n  median s/step   ms/sample")
    for key, ts in times.items():
        med = statistics.median(ts)
        bs = int(key.split("bs")[1])
        summary[key] = {"median_s": med, "ms_per_sample": 1000 * med / bs}
        print(f"  {key:22s} {med:7.3f}   {1000 * med / bs:7.1f}")

    print("\n=== Level 2b: where the time goes (batch 32) ===")
    phases = {}
    for name, amp in (("fp32", False), ("fp16-autocast", True)):
        phases[name] = {}
        for phase in ("fwd", "fwd+bwd", "step"):
            ts = time_steps(model, opt, scalers, batches[32], amp, phase, 1, 4, f"{name} {phase}")
            phases[name][phase] = statistics.median(ts)
        p = phases[name]
        print(f"  {name:14s} fwd {p['fwd']:.3f}s   bwd {p['fwd+bwd'] - p['fwd']:.3f}s   "
              f"optimizer {p['step'] - p['fwd+bwd']:.3f}s   total {p['step']:.3f}s")
    return {"steps": summary, "phases": phases}


# --------------------------------------------------------------------------
# Level 3: profiler
# --------------------------------------------------------------------------

def _device_time(evt) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(evt, attr):
            return getattr(evt, attr)
    return 0.0


def level3(model, opt, scalers, steps: int = 3) -> dict:
    print("\n=== Level 3: profiler, batch 32 ===")
    batch = make_batch(32)
    out = {}
    for name, amp in (("fp32", False), ("fp16-autocast", True)):
        for _ in range(2):
            run_step(model, opt, scalers[amp], batch, amp)
        torch.cuda.synchronize()
        try:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                start = time.perf_counter()
                for _ in range(steps):
                    run_step(model, opt, scalers[amp], batch, amp)
                torch.cuda.synchronize()
                wall = (time.perf_counter() - start) / steps
        except Exception as exc:  # CUPTI is not always available on Windows
            print(f"  {name}: profiler failed ({exc!r})")
            continue

        kernels = [e for e in prof.key_averages()
                   if str(getattr(e, "device_type", "")).endswith("CUDA") and _device_time(e) > 0]
        busy = sum(_device_time(e) for e in kernels) / steps / 1e6   # us -> s
        top = sorted(kernels, key=_device_time, reverse=True)[:8]
        print(f"\n  {name}: wall {wall:.3f} s/step, GPU kernels busy {busy:.3f} s/step "
              f"({100 * busy / wall:.0f}% of wall)")
        for e in top:
            share = _device_time(e) / steps / 1e6 / busy * 100 if busy else 0
            print(f"    {share:5.1f}%  {e.key[:90]}")
        out[name] = {"wall_s": wall, "gpu_busy_s": busy,
                     "top_kernels": [(e.key, _device_time(e) / steps / 1e6) for e in top]}
    return out


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / 2**30:.1f} GiB, "
          f"compute capability {props.major}.{props.minor}, torch {torch.__version__}")
    torch.backends.cuda.matmul.allow_tf32 = False  # irrelevant on Turing, explicit anyway

    results = {"gpu": props.name, "torch": torch.__version__, "level1": level1()}
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scalers = {False: torch.amp.GradScaler("cuda", enabled=False),
               True: torch.amp.GradScaler("cuda", enabled=True)}
    results["level2"] = level2(model, opt, scalers)
    results["level3"] = level3(model, opt, scalers)
    results["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30

    out = ROOT / "logs" / "bench_precision.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\npeak memory {results['peak_mem_gib']:.2f} GiB, results -> {out}")


if __name__ == "__main__":
    main()
