"""Build train/val/test manifests for PTB-XL, split by patient.

Splits follow PTB-XL's official ``strat_fold`` (1-10), which is assigned at the
**patient** level — every record from a given patient shares a fold, so using the
folds directly prevents patient leakage. We verify that invariant here rather than
trusting it.

    folds 1-8 -> train,  fold 9 -> val,  fold 10 -> test

Each manifest row carries the identifiers, demographics, signal file paths, and the
labels in two forms: the raw list of present SCP codes and the aggregated diagnostic
superclasses. Downstream code multi-hots these with ``src.data.labels.encode``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import MANIFEST_DIR, TEST_FOLD, TRAIN_FOLDS, VAL_FOLD
from src.data import labels as L

# Columns carried straight through from ptbxl_database.csv.
_PASSTHROUGH = [
    "patient_id", "age", "sex", "height", "weight",
    "device", "site", "recording_date", "strat_fold",
    "filename_lr", "filename_hr",
]


def split_of(fold: int) -> str:
    if fold in TRAIN_FOLDS:
        return "train"
    if fold == VAL_FOLD:
        return "val"
    if fold == TEST_FOLD:
        return "test"
    raise ValueError(f"unexpected strat_fold {fold!r}")


def build_manifest(df: pd.DataFrame, scp: pd.DataFrame) -> pd.DataFrame:
    """Return one manifest DataFrame (indexed by ecg_id) with labels + split."""
    smap = L.diagnostic_superclass_map(scp)

    out = df[_PASSTHROUGH].copy()
    out.insert(0, "ecg_id", df.index)
    out["split"] = out["strat_fold"].map(split_of)
    # Labels: JSON-encoded lists keep the CSV readable and lossless.
    out["scp_codes"] = df["scp_codes"].apply(lambda d: json.dumps(L.present_codes(d)))
    out["diagnostic_superclasses"] = df["scp_codes"].apply(
        lambda d: json.dumps(L.aggregate_superclasses(d, smap))
    )
    # One-hot the 5 diagnostic superclasses for convenient stratified analysis.
    for sc in L.DIAGNOSTIC_SUPERCLASSES:
        out[f"sc_{sc}"] = df["scp_codes"].apply(
            lambda d, sc=sc: int(sc in L.aggregate_superclasses(d, smap))
        )
    return out


def assert_no_patient_leakage(manifest: pd.DataFrame) -> None:
    """Raise if any patient's records land in more than one split."""
    per_patient = manifest.groupby("patient_id")["split"].nunique()
    bad = per_patient[per_patient > 1]
    if len(bad):
        raise AssertionError(
            f"{len(bad)} patient(s) span multiple splits, e.g. {bad.index[:5].tolist()}"
        )


def write_manifests(out_dir: Path = MANIFEST_DIR) -> dict[str, Path]:
    """Build and write train/val/test manifest CSVs. Returns the paths written."""
    df = L.load_database()
    scp = L.load_scp_statements()
    manifest = build_manifest(df, scp)
    assert_no_patient_leakage(manifest)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        sub = manifest[manifest["split"] == split]
        path = out_dir / f"{split}.csv"
        sub.to_csv(path, index=False)
        paths[split] = path
    # Also write the full manifest for convenience.
    full = out_dir / "manifest_full.csv"
    manifest.to_csv(full, index=False)
    paths["full"] = full
    return paths


if __name__ == "__main__":
    written = write_manifests()
    for split, path in written.items():
        n = len(pd.read_csv(path))
        print(f"{split:6s} {n:6d} rows -> {path}")
