"""results/points.jsonl: the single shared record of every operating point.

One JSON line per (operating point, split). Append-only: re-measuring a point
appends a new line, and readers keep the latest line per (name, split). The
Pareto plot and the report tables are generated from this file only.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import pandas as pd

from .utils import ROOT

REGISTRY = ROOT / "results" / "points.jsonl"

REQUIRED = {"name", "family", "split", "accuracy", "gflops"}
FAMILIES = {"static", "adaptive"}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def record(point: dict) -> dict:
    """Append one operating point. Required keys: name, family, split, accuracy, gflops."""
    missing = REQUIRED - point.keys()
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    if point["family"] not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if point["split"] not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    row = {**point,
           "git_commit": _git_commit(),
           "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def load(split: str | None = None) -> pd.DataFrame:
    """Latest entry per (name, split), optionally filtered to one split."""
    if not REGISTRY.exists():
        return pd.DataFrame()
    df = pd.read_json(REGISTRY, lines=True)
    df = df.drop_duplicates(subset=["name", "split"], keep="last")
    return df[df["split"] == split] if split else df


def pareto_mask(df: pd.DataFrame) -> pd.Series:
    """True for points not dominated in (lower gflops, higher accuracy)."""
    order = df.sort_values(["gflops", "accuracy"], ascending=[True, False])
    best, keep = -float("inf"), set()
    for idx, acc in order["accuracy"].items():
        if acc > best:
            keep.add(idx)
            best = acc
    return df.index.to_series().isin(keep)
