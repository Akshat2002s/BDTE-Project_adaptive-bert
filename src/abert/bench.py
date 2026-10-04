"""Latency, size and parameter count. Supporting evidence for the Pareto
plot, and the service-time model the MDP controller is built on."""
from __future__ import annotations

import io
import platform
import statistics
import time

import torch

from .flops import dummy_inputs


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: torch.nn.Module) -> float:
    """Serialized state_dict size in MB (same definition as the labs)."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / (1024 ** 2)


@torch.no_grad()
def cpu_latency_ms(model: torch.nn.Module, max_len: int, threads: int,
                   warmup: int = 10, runs: int = 100,
                   **forward_kwargs) -> dict[str, float]:
    """Single-query CPU latency with a fixed thread count.

    The deployment target is an on-premise CPU box, so latency is always
    measured on CPU with batch size 1, regardless of where training ran.
    """
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    model = model.eval().to("cpu")
    vocab = getattr(getattr(model, "config", None), "vocab_size", 30522)
    inputs = dummy_inputs(max_len, vocab_size=vocab)
    try:
        for _ in range(warmup):
            model(**inputs, **forward_kwargs)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            model(**inputs, **forward_kwargs)
            times.append((time.perf_counter() - start) * 1000.0)
    finally:
        torch.set_num_threads(prev_threads)
    times.sort()
    return {
        "latency_ms_p50": statistics.median(times),
        "latency_ms_p95": times[int(0.95 * (len(times) - 1))],
    }


def host_info(threads: int) -> dict[str, str | int]:
    return {"cpu": platform.processor() or platform.machine(),
            "node": platform.node(), "threads": threads}
