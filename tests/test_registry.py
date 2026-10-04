import pandas as pd
import pytest

from abert import registry


def test_pareto_mask_drops_dominated_points():
    df = pd.DataFrame({
        "name": ["a", "b", "c", "d"],
        "gflops": [1.0, 2.0, 3.0, 2.5],
        "accuracy": [80.0, 85.0, 90.0, 84.0],   # d is dominated by b
    })
    assert df[registry.pareto_mask(df)]["name"].tolist() == ["a", "b", "c"]


def test_record_validates(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REGISTRY", tmp_path / "points.jsonl")
    with pytest.raises(ValueError):
        registry.record({"name": "x", "family": "static", "split": "val"})
    with pytest.raises(ValueError):
        registry.record({"name": "x", "family": "nope", "split": "val",
                         "accuracy": 1.0, "gflops": 1.0})
    registry.record({"name": "x", "family": "static", "split": "val",
                     "accuracy": 1.0, "gflops": 1.0})
    registry.record({"name": "x", "family": "static", "split": "val",
                     "accuracy": 2.0, "gflops": 1.0})
    df = registry.load("val")
    assert len(df) == 1 and df["accuracy"].item() == 2.0
