"""Elastic (width x depth) BERT: the agreed interface, not yet implemented.

This file pins down the contract so the controller/benchmarking side can be
built against it in parallel with the model side.

One weight file, many operating points:
  * width: fraction of attention heads AND FFN neurons kept per layer, after
    importance-based reordering, so a smaller width is a prefix slice of a
    larger one (nested by construction). The hidden size (768) never changes,
    so exit heads can be shared across widths.
  * depth: exit heads after layers EXIT_LAYERS. With a threshold, a query
    leaves at the first exit whose confidence clears it; with `force_exit`,
    it always runs to exactly that exit (used for per-exit FLOPs).

For FLOPs to drop, the active width must run on physically sliced weight
views (weight[:k]), not masks. tests/test_elastic.py should assert that
model_gflops falls as width falls, and that the slices share storage with the
full weights (torch.equal on every slice).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

WIDTHS = (1.0, 0.75, 0.5, 0.25)
EXIT_LAYERS = (4, 8, 12)


@dataclass(frozen=True)
class OperatingPoint:
    width: float                 # one of WIDTHS
    exit_layer: int | None       # forced exit (one of EXIT_LAYERS), or None = use threshold
    threshold: float | None = None  # confidence threshold when exit_layer is None

    @property
    def name(self) -> str:
        depth = f"exit{self.exit_layer}" if self.exit_layer else f"thr{self.threshold}"
        return f"elastic-w{self.width}-{depth}"


@dataclass
class ElasticOutput:
    logits: torch.Tensor         # (batch, num_labels) from the exit each query used
    exit_layer: torch.Tensor     # (batch,) layer index each query exited at


class ElasticBert(nn.Module):
    """TODO(model owner). Suggested build: own encoder layers that load the
    bert-base-uncased weights, rather than patching transformers internals
    (they changed a lot in 5.x and slicing is easier on code we control)."""

    def __init__(self, num_labels: int):
        super().__init__()
        raise NotImplementedError

    def set_width(self, width: float) -> None:
        """Activate the first `width` fraction of heads/neurons in every layer."""
        raise NotImplementedError

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                force_exit: int | None = None,
                threshold: float | None = None) -> ElasticOutput:
        raise NotImplementedError
