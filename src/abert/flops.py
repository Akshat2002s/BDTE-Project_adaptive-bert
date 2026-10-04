"""The ONE FLOPs methodology for every operating point in the project.

Tool: torch.utils.flop_counter.FlopCounterMode (built into PyTorch).
  * Counts at the aten-op level, so it follows Python control flow. That
    matters for early exit: run the forward with a forced exit layer and only
    the layers that actually executed are counted. Trace-based tools (fvcore)
    record a single static graph instead.
  * Counts matmul-family ops (linear, matmul, bmm, attention) as 2 * MACs.
    LayerNorm, softmax, GELU and embedding lookups are not counted. That is
    the same for every model, so comparisons stay fair; state it in the report.

Conventions (fixed for the whole project):
  * batch size 1, sequence length = configs/base.yaml data.max_len
  * models are built with attn_implementation="eager" so attention is plain
    matmuls on every backend (see models.py)
  * the reported number is FLOPs per query, in GFLOPs

For early exit with a confidence threshold, the cost of a configuration is the
expected FLOPs: sum over exits of (fraction of test queries leaving at that
exit) * (FLOPs of running up to that exit). Use `expected_flops` for that so
everyone computes it the same way.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch.utils.flop_counter import FlopCounterMode


def dummy_inputs(max_len: int, batch_size: int = 1, vocab_size: int = 30522,
                 device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Fixed-length, fully unpadded input. Padding does not change dense FLOPs."""
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, vocab_size, (batch_size, max_len), generator=g)
    return {
        "input_ids": ids.to(device),
        "attention_mask": torch.ones_like(ids).to(device),
    }


@torch.no_grad()
def count_flops(forward: Callable[[], object]) -> int:
    """FLOPs executed by `forward()` (a zero-arg closure around one model call)."""
    with FlopCounterMode(display=False) as counter:
        forward()
    return counter.get_total_flops()


def model_gflops(model: torch.nn.Module, max_len: int, **forward_kwargs) -> float:
    """GFLOPs for one query through `model`, with optional extra forward kwargs
    (e.g. a forced exit layer for the elastic model)."""
    model.eval()
    device = next(model.parameters()).device
    vocab = getattr(getattr(model, "config", None), "vocab_size", 30522)
    inputs = dummy_inputs(max_len, vocab_size=vocab, device=device)
    return count_flops(lambda: model(**inputs, **forward_kwargs)) / 1e9


def expected_flops(exit_fractions: Mapping[int, float],
                   flops_per_exit: Mapping[int, float]) -> float:
    """Average per-query cost of an early-exit policy.

    exit_fractions: {exit_layer: fraction of queries that exited there}, sums to 1
    flops_per_exit: {exit_layer: FLOPs of running the trunk up to that exit + its head}
    """
    total = sum(exit_fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"exit fractions sum to {total}, expected 1.0")
    return sum(frac * flops_per_exit[layer] for layer, frac in exit_fractions.items())
