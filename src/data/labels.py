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


def present_codes(scp_codes: dict[str, float]) -> list[str]:
    """The SCP codes considered *present* for a record.

    PTB-XL convention (and the Strodthoff et al. benchmark): a statement is
    present if it is a *key* of ``scp_codes``, regardless of the likelihood
    value. A likelihood of ``0.0`` means "assigned, likelihood unstated" — it is
    still a positive label (e.g. ``{'NORM': 100.0, 'SR': 0.0}`` has sinus rhythm
    present). Do **not** filter on ``likelihood > 0`` or you silently drop labels.
    """
    return sorted(scp_codes.keys())


def encode(scp_codes: dict[str, float], label_space: list[str]) -> np.ndarray:
    """One record's scp_codes dict -> binary presence vector over `label_space`."""
    idx = {code: i for i, code in enumerate(label_space)}
    y = np.zeros(len(label_space), dtype=np.float32)
    for code in present_codes(scp_codes):
        if code in idx:
            y[idx[code]] = 1.0
    return y


# --- Superclass / category structure (from scp_statements.csv) --------------
# The 5 coarse diagnostic superclasses PTB-XL groups its diagnostic codes into.
DIAGNOSTIC_SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def diagnostic_superclass_map(scp) -> dict[str, str]:
    """Map each *diagnostic* SCP code -> its diagnostic superclass (NORM/MI/...)."""
    diag = scp[scp["diagnostic"] == 1.0]
    return diag["diagnostic_class"].dropna().to_dict()


def category_members(scp) -> dict[str, set[str]]:
    """Sets of SCP codes belonging to each category: diagnostic / form / rhythm."""
    return {
        "diagnostic": set(scp.index[scp["diagnostic"] == 1.0]),
        "form": set(scp.index[scp["form"] == 1.0]),
        "rhythm": set(scp.index[scp["rhythm"] == 1.0]),
    }


def aggregate_superclasses(scp_codes: dict[str, float], superclass_map: dict[str, str]) -> list[str]:
    """A record's scp_codes -> sorted list of its diagnostic superclasses."""
    out = {superclass_map[c] for c in scp_codes if c in superclass_map}
    return sorted(out)
