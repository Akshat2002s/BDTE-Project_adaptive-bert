"""Small shared helpers: paths, config loading, seeding, device."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = ROOT / "configs" / "base.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path (as written in configs) to an absolute one."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"using GPU {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("using MPS")
        return torch.device("mps")
    else:
        print("using CPU")
        return torch.device("cpu")

def run_info() -> dict:
    """What produced a result: time, machine, library versions, GPU, git commit."""
    import platform
    import subprocess
    from datetime import datetime

    import transformers

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "uncommitted"
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit,
    }


class Tee:
    """Mirror stdout into a log file (progress bars go to stderr and stay console-only)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, text: str) -> None:
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def __getattr__(self, name):
        # isatty(), encoding, fileno() etc. behave like the real console
        return getattr(self.stdout, name)
