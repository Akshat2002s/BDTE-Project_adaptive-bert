"""Elastic (width x depth) BERT: one weight file, many operating points.

  * width: fraction of attention heads and FFN neurons kept per layer. Heads and
    neurons are sorted by importance first, so a smaller width is a prefix slice
    (a view) of the full weights: nested by construction.
  * depth: exit heads after EXIT_LAYERS, shared across widths (the hidden size
    never changes). A query leaves at a forced exit, or at the first exit whose
    confidence clears a threshold.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertConfig, BertForSequenceClassification

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
    all_logits: dict[int, torch.Tensor] = field(default_factory=dict)  # {exit: logits}, return_all only


class ElasticLayer(nn.Module):
    """One BERT encoder layer whose heads and FFN neurons can be sliced."""

    def __init__(self, hidden: int, heads: int, ff: int, eps: float, dropout: float):
        super().__init__()
        self.query, self.key, self.value, self.attn_out = [nn.Linear(hidden, hidden) for _ in range(4)] # query, key, value, attention output
        self.ln_attn = nn.LayerNorm(hidden, eps=eps)  # layer norm after attention
        self.ln_ffn = nn.LayerNorm(hidden, eps=eps)   # layer norm after the feed-forward network
        self.ffn_up = nn.Linear(hidden, ff)           # feed-forward up projection
        self.ffn_down = nn.Linear(ff, hidden)         # feed-forward down projection
        self.dropout = nn.Dropout(dropout)  # dropout layer for reg
        self.num_heads = heads               # number of attention heads
        self.head_dim = hidden // heads      # dimension per head
        self.active_heads = heads            # active heads (full width to start)
        self.active_ff = ff                  # active FFN neurons (full width to start)
        self.head_gate = None                # gates, only set while scoring importance
        self.ff_gate = None

    def set_width(self, width: float) -> None:
        active_heads = round(width * self.num_heads) # calculate the number of active heads based on the width
        active_ff = round(width * self.ffn_up.out_features) # active FFN neurons based on the width
        self.active_heads = max(1, active_heads) # ensure at least one head is active
        self.active_ff = max(1, active_ff) # ensure at least one FFN neuron is active

    def forward(self, x: torch.Tensor, mask_bias: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, hidden). mask_bias: (batch, 1, 1, seq), 0 for real tokens and a
        huge negative number for padding. Returns (batch, seq, hidden).
        """
        b, t, _ = x.shape # get batch size and sequence length
        h = self.active_heads # number of active attention heads
        d = h * self.head_dim # total dimension for the active heads
        f = self.active_ff # number of active FFN neurons

        q, k, v = [F.linear(x, layer.weight[:d], layer.bias[:d]).view(b, \
                     t, h, self.head_dim).transpose(1, 2) for layer in (self.query, self.key, self.value)] # compute query, key, and value projections for multi-head attention
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) + mask_bias # compute attention scores with scaling and add mask bias
        probs = self.dropout(F.softmax(scores, dim=-1)) # compute attention probabilities with dropout
        ctx = probs @ v # compute context vectors
        if self.head_gate is not None: 
            ctx = ctx * self.head_gate[:h].view(1, h, 1, 1) # apply head gates to context vectors for importance scoring
        ctx = ctx.transpose(1, 2).contiguous().view(b, t, d) # reshape context vectors back to (batch, seq, hidden)
        attn_out = F.linear(ctx, self.attn_out.weight[:, :d],   self.attn_out.bias) # compute attention output projection
        x = self.ln_attn(x + self.dropout(attn_out)) # apply layer normalization to the sum of input and attention output
        mid = F.gelu(F.linear(x, self.ffn_up.weight[:f], self.ffn_up.bias[:f])) # compute FFN intermediate representation with GELU activation
        if self.ff_gate is not None:
            mid = mid * self.ff_gate[:f].view(1, 1, f) # apply FFN gates to mid representation for importance scoring
        out = F.linear(mid, self.ffn_down.weight[:, :f], self.ffn_down.bias) # compute FFN output projection
        return self.ln_ffn(x + self.dropout(out)) 
    
    @torch.no_grad()
    def reorder(self, head_order: torch.Tensor, ff_order: torch.Tensor) -> None:
        dh = self.head_dim # dimension per head
        
        # head order [2, 0, 1] -> rows [128..191, 0..63, 64..127]: each head is a block of dh rows
        rows = torch.cat([torch.arange(i * dh, (i + 1) * dh) for i in head_order.tolist()])
        for lin in (self.query, self.key, self.value):
            lin.weight.copy_(lin.weight[rows])        # outputs = heads -> permute rows
            lin.bias.copy_(lin.bias[rows])
        self.attn_out.weight.copy_(self.attn_out.weight[:, rows])   # input = heads -> columns
        self.ffn_up.weight.copy_(self.ffn_up.weight[ff_order])      # outputs = neurons -> rows
        self.ffn_up.bias.copy_(self.ffn_up.bias[ff_order])
        self.ffn_down.weight.copy_(self.ffn_down.weight[:, ff_order])  # input = neurons -> columns


class ExitHead(nn.Module):
    """BERT's own head: pooler ([CLS] -> dense -> tanh) + classifier."""

    def __init__(self, hidden: int, num_labels: int, dropout: float):
        super().__init__()
        self.pooler = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = torch.tanh(self.pooler(hidden_states[:, 0]))
        return self.classifier(self.dropout(pooled))


class ElasticBert(nn.Module):

    def __init__(self, num_labels: int, config: BertConfig | None = None,
                 exit_layers: tuple[int, ...] = EXIT_LAYERS):
        super().__init__()
        self.config = config or BertConfig()   # flops.py / bench.py read config.vocab_size
        c = self.config
        if max(exit_layers) != c.num_hidden_layers:
            raise ValueError("the last exit must be the last layer")
        self.num_labels = num_labels
        self.exit_layers = tuple(exit_layers)
        self.width = 1.0
        self.word_emb = nn.Embedding(c.vocab_size, c.hidden_size, padding_idx=c.pad_token_id)
        self.pos_emb = nn.Embedding(c.max_position_embeddings, c.hidden_size)
        self.type_emb = nn.Embedding(c.type_vocab_size, c.hidden_size)
        self.emb_ln = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.emb_dropout = nn.Dropout(c.hidden_dropout_prob)
        self.layers = nn.ModuleList(
            ElasticLayer(c.hidden_size, c.num_attention_heads, c.intermediate_size,
                         c.layer_norm_eps, c.hidden_dropout_prob)
            for _ in range(c.num_hidden_layers))
        # ModuleDict keys must be strings: "4", "8", "12"
        self.exits = nn.ModuleDict(
            {str(layer): ExitHead(c.hidden_size, num_labels, c.hidden_dropout_prob)
             for layer in self.exit_layers})

    def set_width(self, width: float) -> None:
        """Activate the first `width` fraction of heads/neurons in every layer."""
        if not 0.0 < width <= 1.0:
            raise ValueError(f"width must be in (0, 1], got {width}")
        self.width = width
        for layer in self.layers:
            layer.set_width(width)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Word + position + token type (always 0: single-sentence queries), LayerNorm, dropout."""
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = (self.word_emb(input_ids) + self.pos_emb(pos)[None]
             + self.type_emb(torch.zeros_like(input_ids)))
        return self.emb_dropout(self.emb_ln(x))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                force_exit: int | None = None, threshold: float | None = None,
                return_all: bool = False) -> ElasticOutput:
        """Three modes:
          * return_all=True: run all layers, logits from every exit (training/eval).
          * force_exit=L:    run layers 1..L, logits from exit L only.
          * threshold=t:     per query, leave at the first exit with max prob >= t
                             (finished queries are dropped from the batch, so the
                             compute really stops). Last exit always answers.
        With none of them, it behaves like force_exit=<last layer>.
        """
        if force_exit is not None and force_exit not in self.exit_layers:
            raise ValueError(f"force_exit must be one of {self.exit_layers}")
        x = self._embed(input_ids)
        # 0 for real tokens, a huge negative number for padding; shape (batch, 1, 1, seq)
        mask_bias = ((1.0 - attention_mask[:, None, None, :].to(x.dtype))
                     * torch.finfo(x.dtype).min)
        batch = input_ids.shape[0]

        if threshold is None:
            # forced exit (or return_all): run up to last, compute the exits we need
            last = self.exit_layers[-1] if (return_all or force_exit is None) else force_exit
            all_logits = {}
            for i, layer in enumerate(self.layers[:last], start=1):
                x = layer(x, mask_bias)
                if str(i) in self.exits and (return_all or i == last):
                    all_logits[i] = self.exits[str(i)](x)
            exit_t = torch.full((batch,), last, dtype=torch.long, device=x.device)
            return ElasticOutput(all_logits[last], exit_t, all_logits if return_all else {})

        # shrink the batch as queries leave
        logits = torch.zeros(batch, self.num_labels, dtype=x.dtype, device=x.device)
        exit_t = torch.zeros(batch, dtype=torch.long, device=x.device)
        alive = torch.arange(batch, device=x.device)   # original row index of each remaining query
        for i, layer in enumerate(self.layers, start=1):
            x = layer(x, mask_bias)
            if str(i) not in self.exits:
                continue
            out = self.exits[str(i)](x)
            if i == self.exit_layers[-1]:
                done = torch.ones_like(alive, dtype=torch.bool)       # last exit answers everything left
            else:
                done = out.softmax(-1).max(-1).values >= threshold
            logits[alive[done]] = out[done]
            exit_t[alive[done]] = i
            keep = ~done
            if not keep.any():
                break
            alive, x, mask_bias = alive[keep], x[keep], mask_bias[keep]
        return ElasticOutput(logits, exit_t)

    # -- importance ordering ---------------------------------------------------
    def compute_importance(self, batches, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Head and FFN-neuron importance: |d loss / d gate| summed over `batches`
        (Michel et al. 2019), at full width with dropout off."""
        self.set_width(1.0)
        was_training = self.training
        self.eval()  # no dropout noise in the scores
        for layer in self.layers:
            layer.head_gate = torch.ones(layer.num_heads, device=device, requires_grad=True)
            layer.ff_gate = torch.ones(layer.ffn_up.out_features, device=device, requires_grad=True)
        head_imp = torch.zeros(len(self.layers), self.layers[0].num_heads)
        ff_imp = torch.zeros(len(self.layers), self.layers[0].ffn_up.out_features)
        try:
            for input_ids, attention_mask, labels in batches:
                out = self(input_ids.to(device), attention_mask.to(device))
                loss = F.cross_entropy(out.logits, labels.to(device))
                gates = [g for l in self.layers for g in (l.head_gate, l.ff_gate)]
                grads = torch.autograd.grad(loss, gates)
                for i in range(len(self.layers)):
                    head_imp[i] += grads[2 * i].abs().cpu()
                    ff_imp[i] += grads[2 * i + 1].abs().cpu()
        finally:
            for layer in self.layers:
                layer.head_gate = layer.ff_gate = None
            self.train(was_training)
        return head_imp, ff_imp

    def reorder(self, head_imp: torch.Tensor, ff_imp: torch.Tensor) -> None:
        """Sort every layer's heads and FFN neurons by descending importance."""
        for i, layer in enumerate(self.layers):
            layer.reorder(head_imp[i].argsort(descending=True),
                          ff_imp[i].argsort(descending=True))

    @classmethod
    def from_classifier(cls, source: str | Path | BertForSequenceClassification,
                        exit_layers: tuple[int, ...] = EXIT_LAYERS) -> "ElasticBert":
        """Build from a (fine-tuned) BertForSequenceClassification. Every exit head
        starts as a copy of its pooler + classifier."""
        hf = (source if isinstance(source, BertForSequenceClassification)
              else BertForSequenceClassification.from_pretrained(source))
        model = cls(hf.config.num_labels, hf.config, exit_layers)
        sd = hf.state_dict()
        p = "bert.embeddings."
        new = {"word_emb.weight": sd[p + "word_embeddings.weight"],
               "pos_emb.weight": sd[p + "position_embeddings.weight"],
               "type_emb.weight": sd[p + "token_type_embeddings.weight"],
               "emb_ln.weight": sd[p + "LayerNorm.weight"],
               "emb_ln.bias": sd[p + "LayerNorm.bias"]}
        names = {"query": "attention.self.query", "key": "attention.self.key",
                 "value": "attention.self.value", "attn_out": "attention.output.dense",
                 "ln_attn": "attention.output.LayerNorm", "ffn_up": "intermediate.dense",
                 "ffn_down": "output.dense", "ln_ffn": "output.LayerNorm"}
        for i in range(hf.config.num_hidden_layers):
            for ours, theirs in names.items():
                for wb in ("weight", "bias"):
                    new[f"layers.{i}.{ours}.{wb}"] = sd[f"bert.encoder.layer.{i}.{theirs}.{wb}"]
        for layer in model.exit_layers:
            for wb in ("weight", "bias"):
                new[f"exits.{layer}.pooler.{wb}"] = sd[f"bert.pooler.dense.{wb}"]
                new[f"exits.{layer}.classifier.{wb}"] = sd[f"classifier.{wb}"]
        model.load_state_dict({k: v.clone() for k, v in new.items()})   # errors on any missing or misnamed key
        return model

    def save(self, out_dir: str | Path) -> None:
        """Write elastic.pt (weights) + elastic_config.json (architecture)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out_dir / "elastic.pt")
        (out_dir / "elastic_config.json").write_text(json.dumps(
            {"bert": self.config.to_dict(), "num_labels": self.num_labels,
             "exit_layers": list(self.exit_layers)}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, out_dir: str | Path, map_location="cpu") -> "ElasticBert":
        """Inverse of save()."""
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "elastic_config.json").read_text(encoding="utf-8"))
        model = cls(meta["num_labels"], BertConfig.from_dict(meta["bert"]),
                    tuple(meta["exit_layers"]))
        model.load_state_dict(torch.load(out_dir / "elastic.pt", map_location=map_location))
        return model


@torch.no_grad()
def collect_all_logits(model: ElasticBert, loader, device, width: float
                       ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Logits of every exit at one width over a loader of (ids, mask, labels, ...).
    Returns ({exit_layer: (n, num_labels) cpu logits}, (n,) labels)."""
    model.eval()
    model.set_width(width)
    chunks: dict[int, list[torch.Tensor]] = {e: [] for e in model.exit_layers}
    labels = []
    for input_ids, attention_mask, y, *_ in loader:
        out = model(input_ids.to(device), attention_mask.to(device), return_all=True)
        for e, logits in out.all_logits.items():
            chunks[e].append(logits.float().cpu())
        labels.append(y)
    return {e: torch.cat(c) for e, c in chunks.items()}, torch.cat(labels)


def exit_head_gflops(model: ElasticBert) -> float:
    """GFLOPs of one exit head for one query (pooler + classifier, 2 x MACs)."""
    h = model.config.hidden_size
    return (2 * h * h + 2 * h * model.num_labels) / 1e9


def threshold_exits(all_logits: dict[int, torch.Tensor], threshold: float
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay threshold early exit from cached per-exit logits
    Returns (predictions, exit_layer)."""
    exits = sorted(all_logits)
    n = all_logits[exits[0]].shape[0]
    pred = torch.full((n,), -1, dtype=torch.long)
    exit_at = torch.zeros(n, dtype=torch.long)
    undecided = torch.ones(n, dtype=torch.bool)
    for layer in exits:
        conf, p = all_logits[layer].float().softmax(-1).max(-1)
        if layer == exits[-1]:
            take = undecided                        # the last exit answers everything left
        else:
            take = undecided & (conf >= threshold)
        pred[take], exit_at[take] = p[take], layer
        undecided &= ~take
    return pred, exit_at