"""Contract tests for ElasticBert on a tiny random BERT (no downloads)."""
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification

from abert.elastic import ElasticBert, exit_head_gflops, threshold_exits
from abert.flops import dummy_inputs, model_gflops

D, FF, LAYERS, HEADS, LABELS, SEQ, VOCAB = 64, 128, 6, 4, 7, 16, 2000
EXITS = (2, 4, 6)


def tiny_hf():
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=VOCAB, hidden_size=D, num_hidden_layers=LAYERS,
                     num_attention_heads=HEADS, intermediate_size=FF,
                     max_position_embeddings=64, num_labels=LABELS,
                     attn_implementation="eager")
    return BertForSequenceClassification(cfg).eval()


def batch(n=5, pad=True):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(1000, VOCAB, (n, SEQ), generator=g)
    mask = torch.ones_like(ids)
    if pad:
        mask[1:, SEQ - 5:] = 0
    return ids, mask


@pytest.fixture
def pair():
    hf = tiny_hf()
    return hf, ElasticBert.from_classifier(hf, exit_layers=EXITS).eval()


def test_full_width_last_exit_matches_hf(pair):
    hf, el = pair
    ids, mask = batch()
    with torch.no_grad():
        ref = hf(input_ids=ids, attention_mask=mask).logits
        out = el(ids, mask).logits
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_flops_fall_with_width_and_depth(pair):
    _, el = pair
    grid = {}
    for w in (1.0, 0.75, 0.5, 0.25):
        el.set_width(w)
        for e in EXITS:
            grid[w, e] = model_gflops(el, SEQ, force_exit=e)
    for e in EXITS:
        assert grid[1.0, e] > grid[0.75, e] > grid[0.5, e] > grid[0.25, e]
    for w in (1.0, 0.5):
        assert grid[w, 2] < grid[w, 4] < grid[w, 6]


def test_flops_match_hf_at_full_size(pair):
    hf, el = pair
    assert model_gflops(el, SEQ) == pytest.approx(model_gflops(hf, SEQ))


def test_slices_are_views_of_full_weights(pair):
    _, el = pair
    el.set_width(0.5)
    layer = el.layers[0]
    d = layer.active_heads * layer.head_dim
    s = layer.query.weight[:d]
    assert s.data_ptr() == layer.query.weight.data_ptr()
    assert torch.equal(s, layer.query.weight[:d])
    assert layer.ffn_down.weight[:, :layer.active_ff].untyped_storage().data_ptr() \
        == layer.ffn_down.weight.untyped_storage().data_ptr()


def test_reorder_keeps_full_width_function(pair):
    _, el = pair
    ids, mask = batch()
    with torch.no_grad():
        before = el(ids, mask, return_all=True).all_logits
    labels = torch.randint(0, LABELS, (ids.shape[0],))
    head_imp, ff_imp = el.compute_importance([(ids, mask, labels)], "cpu")
    assert head_imp.shape == (LAYERS, HEADS) and ff_imp.shape == (LAYERS, FF)
    assert (head_imp > 0).all()
    el.reorder(head_imp, ff_imp)
    el.eval()
    with torch.no_grad():
        after = el(ids, mask, return_all=True).all_logits
    for e in EXITS:
        torch.testing.assert_close(after[e], before[e], atol=1e-5, rtol=1e-5)
    # most important head is now head 0 in every layer
    head_imp2, _ = el.compute_importance([(ids, mask, labels)], "cpu")
    assert (head_imp2.argmax(1) == 0).all()


def test_forced_exit_matches_return_all(pair):
    _, el = pair
    ids, mask = batch()
    el.set_width(0.75)
    with torch.no_grad():
        all_logits = el(ids, mask, return_all=True).all_logits
        for e in EXITS:
            out = el(ids, mask, force_exit=e)
            torch.testing.assert_close(out.logits, all_logits[e])
            assert (out.exit_layer == e).all()


@pytest.mark.parametrize("thr", [0.0, 0.15, 0.2, 1.01])
def test_threshold_forward_matches_replay(pair, thr):
    _, el = pair
    ids, mask = batch(n=32, pad=False)
    with torch.no_grad():
        all_logits = el(ids, mask, return_all=True).all_logits
        out = el(ids, mask, threshold=thr)
    pred, exit_at = threshold_exits(all_logits, thr)
    assert torch.equal(out.exit_layer, exit_at)
    assert torch.equal(out.logits.argmax(-1), pred)
    if thr == 0.0:
        assert (exit_at == EXITS[0]).all()
    if thr > 1:
        assert (exit_at == EXITS[-1]).all()


def test_save_load_roundtrip(pair, tmp_path):
    _, el = pair
    el.save(tmp_path)
    back = ElasticBert.load(tmp_path).eval()
    ids, mask = batch()
    with torch.no_grad():
        torch.testing.assert_close(back(ids, mask).logits, el(ids, mask).logits)


def test_exit_head_flops(pair):
    _, el = pair
    head = el.exits[str(EXITS[0])]
    from abert.flops import count_flops
    h = torch.zeros(1, SEQ, D)
    assert count_flops(lambda: head(h)) / 1e9 == pytest.approx(exit_head_gflops(el))


def test_layer_matches_hf_layer():
    """Step 1 on its own: one ElasticLayer at full width = one Hugging Face BertLayer."""
    from abert.elastic import ElasticLayer
    hf = tiny_hf()
    ref = hf.bert.encoder.layer[0]
    layer = ElasticLayer(D, HEADS, FF, eps=1e-12, dropout=0.0).eval()
    pairs = {"query": ref.attention.self.query, "key": ref.attention.self.key,
             "value": ref.attention.self.value, "attn_out": ref.attention.output.dense,
             "ln_attn": ref.attention.output.LayerNorm, "ffn_up": ref.intermediate.dense,
             "ffn_down": ref.output.dense, "ln_ffn": ref.output.LayerNorm}
    with torch.no_grad():
        for ours, theirs in pairs.items():
            getattr(layer, ours).weight.copy_(theirs.weight)
            getattr(layer, ours).bias.copy_(theirs.bias)
    x = torch.randn(3, SEQ, D)
    no_mask = torch.zeros(3, 1, 1, SEQ)
    with torch.no_grad():
        torch.testing.assert_close(layer(x, no_mask), ref(x), atol=1e-5, rtol=1e-5)
        layer.set_width(0.5)
        assert layer.active_heads == HEADS // 2 and layer.active_ff == FF // 2
        assert layer(x, no_mask).shape == x.shape
