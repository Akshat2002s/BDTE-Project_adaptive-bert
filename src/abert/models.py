"""Model builders for the static baselines.

Every model is built with attn_implementation="eager" so attention runs as
plain matmuls everywhere. That keeps FLOPs counting and CPU latency
comparable across baselines and the elastic model.
"""
from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_classifier(hf_name_or_path: str | Path, num_labels: int):
    return AutoModelForSequenceClassification.from_pretrained(
        hf_name_or_path, num_labels=num_labels, attn_implementation="eager",
    )


def build_tokenizer(hf_name_or_path: str | Path):
    return AutoTokenizer.from_pretrained(hf_name_or_path)


# TODO(baselines): statically head-pruned BERTs. Plan: take the trained
# bert-base checkpoint, rank heads by importance (same scoring the elastic
# model will use), physically remove the bottom k% of heads and the matching
# FFN neurons, fine-tune, then register each ratio as its own static point.
# Check what head pruning transformers 5.x supports before writing it.
