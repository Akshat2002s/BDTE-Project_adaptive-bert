"""Sanity checks on the frozen split. Skipped until prepare_data.py has run."""
import json

import pytest

from abert.utils import load_config, resolve

SPLIT = resolve(load_config()["data"]["split_file"])
pytestmark = pytest.mark.skipif(not SPLIT.exists(), reason="run scripts/prepare_data.py first")


def test_split_is_disjoint_and_complete():
    from abert.data import load_split
    frames = load_split(SPLIT)
    assert len(frames["train"]) + len(frames["val"]) == 10003   # official train size
    assert len(frames["test"]) == 3080
    for name in ("train", "val", "test"):
        assert frames[name]["label"].nunique() == 77


def test_split_file_metadata():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    assert len(split["labels"]) == 77
    assert split["val_indices"] == sorted(set(split["val_indices"]))
