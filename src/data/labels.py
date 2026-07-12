"""SCP-ECG label handling for PTB-XL.

PTB-XL annotates each record with a dict of SCP-ECG statements -> likelihood, e.g.
``{"NORM": 100.0, "SR": 0.0}``. There are 71 distinct SCP statements across the
diagnostic, form, and rhythm categories. We treat presence of a statement (any
nonzero likelihood, by default) as a positive multi-label target.

`scp_statements.csv` (shipped with PTB-XL) maps each code to human-readable
descriptions and to coarser diagnostic superclasses, which the explanation layer
uses for plain-English phrasing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

from src.config import PTBXL_DIR


def load_scp_statements(ptbxl_dir: Path = PTBXL_DIR):
    """Return the scp_statements.csv table (code -> description/superclass)."""
    return pd.read_csv(ptbxl_dir / "scp_statements.csv", index_col=0)


def load_database(ptbxl_dir: Path = PTBXL_DIR):
    """Return ptbxl_database.csv with `scp_codes` parsed from string to dict."""
    df = pd.read_csv(ptbxl_dir / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def build_label_space(ptbxl_dir: Path = PTBXL_DIR) -> list[str]:
    """The canonical, sorted list of the 71 SCP codes (stable column order)."""
    scp = load_scp_statements(ptbxl_dir)
    return sorted(scp.index.tolist())


def encode(scp_codes: dict[str, float], label_space: list[str], min_likelihood: float = 0.0) -> np.ndarray:
    """One record's scp_codes dict -> binary vector over `label_space`."""
    idx = {code: i for i, code in enumerate(label_space)}
    y = np.zeros(len(label_space), dtype=np.float32)
    for code, likelihood in scp_codes.items():
        if code in idx and likelihood > min_likelihood:
            y[idx[code]] = 1.0
    return y
