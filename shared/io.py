"""
shared/io.py
Parquet cache I/O and model artifact save/load.
All engines use these helpers — never call pd.read_csv on the raw CSV after Phase 0.
"""
from __future__ import annotations
import pickle
from pathlib import Path

import pandas as pd

from .config import cfg


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the Phase-0 unified parquet. Raises FileNotFoundError if not built yet."""
    p = path or cfg.phase0_parquet
    if not p.exists():
        raise FileNotFoundError(
            f"Phase-0 dataset not found at {p}. Run scripts/build_phase0.py first."
        )
    return pd.read_parquet(p)


def save_artifact(obj: object, name: str) -> Path:
    """Pickle an object to data/artifacts/<name>.pkl. Returns the written path."""
    out = cfg.artifacts_dir / f"{name}.pkl"
    with open(out, "wb") as f:
        pickle.dump(obj, f, protocol=5)
    return out


def load_artifact(name: str) -> object:
    """Load a previously saved artifact. Raises FileNotFoundError if missing."""
    p = cfg.artifacts_dir / f"{name}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"Artifact '{name}' not found at {p}.")
    with open(p, "rb") as f:
        return pickle.load(f)
