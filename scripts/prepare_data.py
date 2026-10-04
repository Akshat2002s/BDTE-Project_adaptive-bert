"""Create the frozen Banking77 train/val/test split (run once, commit the JSON).

    python scripts/prepare_data.py
"""
from __future__ import annotations

from abert.data import load_split, make_split
from abert.models import build_tokenizer
from abert.utils import load_config, resolve


def main() -> None:
    cfg = load_config()
    split_file = resolve(cfg["data"]["split_file"])
    if split_file.exists():
        print(f"{split_file} already exists, using it as-is.")
    else:
        make_split(split_file, cfg["data"]["val_frac"], cfg["seed"])
        print(f"wrote {split_file}")

    frames = load_split(split_file)
    for k, df in frames.items():
        print(f"{k:5s} {len(df):6d} queries, {df['label'].nunique()} labels")

    # How much does the fixed max_len truncate? Report it once, quote it in the report.
    max_len = cfg["data"]["max_len"]
    tok = build_tokenizer("bert-base-uncased")
    lengths = [len(ids) for ids in tok(frames["train"]["text"].tolist())["input_ids"]]
    over = sum(n > max_len for n in lengths) / len(lengths)
    print(f"max_len={max_len}: {100 * over:.2f}% of train queries truncated "
          f"(longest is {max(lengths)} tokens)")


if __name__ == "__main__":
    main()
