"""Banking77 loading with a frozen, shared train/val/test split.

The official dataset only ships train and test. We carve a stratified
validation split out of train ONCE (scripts/prepare_data.py), write the row
indices to data/splits/banking77_split.json and commit that file. Exit
thresholds and the MDP controller are tuned on `val`; the Pareto frontier is
reported on `test` only.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .utils import ROOT

# Official source (PolyAI). The HF hub copy is a loading script that newer
# `datasets` versions refuse to run, so we read the original CSVs directly.
SOURCE_URLS = {
    "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
}
RAW_DIR = ROOT / "data" / "raw"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_raw() -> dict[str, Path]:
    """Fetch the two official CSVs into data/raw/ if they are not there yet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, url in SOURCE_URLS.items():
        path = RAW_DIR / f"{split}.csv"
        if not path.exists():
            # Save the bytes untouched so the checksum is identical on every OS.
            with urllib.request.urlopen(url) as response:
                path.write_bytes(response.read())
        paths[split] = path
    return paths


def make_split(split_file: Path, val_frac: float, seed: int) -> dict:
    """Create the frozen split file. Refuses to overwrite an existing one."""
    if split_file.exists():
        raise FileExistsError(f"{split_file} already exists; the split is frozen.")
    paths = download_raw()
    train = pd.read_csv(paths["train"])
    labels = sorted(train["category"].unique())
    _, val_idx = train_test_split(
        range(len(train)), test_size=val_frac, random_state=seed,
        stratify=train["category"],
    )
    split = {
        "labels": labels,
        "val_indices": sorted(int(i) for i in val_idx),
        "seed": seed,
        "val_frac": val_frac,
        "sha256": {k: _sha256(p) for k, p in paths.items()},
    }
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(split, indent=1), encoding="utf-8")
    return split


def load_split(split_file: Path) -> dict[str, pd.DataFrame]:
    """Return {'train', 'val', 'test'} DataFrames with `text` and `label` columns."""
    split = json.loads(Path(split_file).read_text(encoding="utf-8"))
    paths = download_raw()
    for k, p in paths.items():
        if _sha256(p) != split["sha256"][k]:
            raise ValueError(f"{p} differs from the file the split was made from.")

    label_to_id = {name: i for i, name in enumerate(split["labels"])}
    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
    is_val = train.index.isin(split["val_indices"])

    out = {"train": train[~is_val], "val": train[is_val], "test": test}
    return {
        k: pd.DataFrame({"text": df["text"].tolist(),
                         "label": df["category"].map(label_to_id).tolist()})
        for k, df in out.items()
    }


def num_labels(split_file: Path) -> int:
    return len(json.loads(Path(split_file).read_text(encoding="utf-8"))["labels"])


def tokenize(df: pd.DataFrame, tokenizer, max_len: int) -> TensorDataset:
    """Pad/truncate every query to exactly `max_len` so cost is input-independent."""
    enc = tokenizer(
        df["text"].tolist(), padding="max_length", truncation=True,
        max_length=max_len, return_tensors="pt",
    )
    return TensorDataset(
        enc["input_ids"], enc["attention_mask"],
        torch.tensor(df["label"].tolist(), dtype=torch.long),
    )


def get_loaders(split_file: Path, tokenizer, max_len: int, batch_size: int,
                eval_batch_size: int = 256) -> dict[str, DataLoader]:
    frames = load_split(split_file)
    return {
        k: DataLoader(
            tokenize(df, tokenizer, max_len),
            batch_size=batch_size if k == "train" else eval_batch_size,
            shuffle=(k == "train"),
        )
        for k, df in frames.items()
    }
