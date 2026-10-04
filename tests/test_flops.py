"""Checks the FLOPs methodology against a hand calculation on a tiny BERT.

No downloads: the model is randomly initialised from a small config.
"""
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification

from abert.flops import count_flops, dummy_inputs, expected_flops, model_gflops

D, FF, LAYERS, HEADS, LABELS, SEQ = 64, 128, 2, 4, 7, 16


def tiny_bert():
    cfg = BertConfig(vocab_size=2000, hidden_size=D, num_hidden_layers=LAYERS,
                     num_attention_heads=HEADS, intermediate_size=FF,
                     max_position_embeddings=64, num_labels=LABELS,
                     attn_implementation="eager")
    return BertForSequenceClassification(cfg).eval()


def analytic_flops() -> int:
    """2 * MACs of every matmul in one forward of a single sequence."""
    per_layer = (
        2 * SEQ * D * D * 4        # Q, K, V and output projections
        + 2 * SEQ * SEQ * D * 2    # QK^T and attn @ V (summed over heads)
        + 2 * SEQ * D * FF * 2     # FFN up and down
    )
    pooler = 2 * D * D             # [CLS] dense layer
    classifier = 2 * D * LABELS
    return LAYERS * per_layer + pooler + classifier


def test_matches_hand_calculation():
    model = tiny_bert()
    inputs = dummy_inputs(SEQ, vocab_size=2000)
    assert count_flops(lambda: model(**inputs)) == analytic_flops()


def test_model_gflops_wrapper():
    assert model_gflops(tiny_bert(), SEQ) == pytest.approx(analytic_flops() / 1e9)


def test_counts_only_executed_layers():
    """Dynamic control flow: running half the layers costs half the encoder FLOPs."""
    model = tiny_bert()
    inputs = dummy_inputs(SEQ, vocab_size=2000)
    full = count_flops(lambda: model(**inputs))
    model.bert.encoder.layer = model.bert.encoder.layer[:1]
    half = count_flops(lambda: model(**inputs))
    head = 2 * D * D + 2 * D * LABELS
    assert (full - head) == 2 * (half - head)


def test_expected_flops():
    assert expected_flops({4: 0.5, 12: 0.5}, {4: 10.0, 12: 30.0}) == 20.0
    with pytest.raises(ValueError):
        expected_flops({4: 0.5}, {4: 10.0})


def test_padding_does_not_change_flops():
    model = tiny_bert()
    inputs = dummy_inputs(SEQ, vocab_size=2000)
    padded = {**inputs, "attention_mask": inputs["attention_mask"].clone()}
    padded["attention_mask"][:, SEQ // 2:] = 0
    with torch.no_grad():
        assert count_flops(lambda: model(**inputs)) == count_flops(lambda: model(**padded))
