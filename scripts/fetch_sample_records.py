#!/usr/bin/env python3
"""Fetch a small, curated set of PTB-XL waveforms for the Phase 2 sanity notebook.

Downloading the full waveform archive is ~3 GB; the preprocessing sanity checks only
need a handful of records. This grabs individual ``.hea``/``.dat`` files straight from
PhysioNet (a few MB total) into the real ``records100/`` layout, so ``PTBXLDataset``
and ``wfdb.rdsamp`` read them transparently. Run ``make data`` for the full set when you
start training (Phase 3+).

    python scripts/fetch_sample_records.py           # 100 Hz curated set (+ one 500 Hz)
    python scripts/fetch_sample_records.py --force    # re-download
"""

from __future__ import annotations

import argparse
import urllib.request

import pandas as pd

from src.config import PTBXL_DIR

PTBXL_VERSION = "1.0.3"
META_BASE = f"https://physionet.org/files/ptb-xl/{PTBXL_VERSION}"

# Curated across all six diagnostic groups + artifact examples for the filter demo.
SAMPLES: dict[int, str] = {
    3: "NORM - clean normal sinus rhythm",
    17: "AFIB - atrial fibrillation (rhythm)",
    8: "IMI - inferior MI (also baseline-drift + noise flagged)",
    180: "CLBBB - complete left bundle branch block (CD)",
    30: "LVH - left ventricular hypertrophy (HYP), baseline drift",
    22: "NDT - non-diagnostic T abnormalities (STTC)",
    1: "NORM w/ low voltage + noise flag - filter demo",
}
# Also fetch this record at 500 Hz to demonstrate 500 -> 100 Hz resampling.
HR_DEMO_ID = 3


def _download(rel_path: str, force: bool) -> None:
    """Download <META_BASE>/<rel_path>.hea and .dat into PTBXL_DIR/<rel_path>.*"""
    for ext in (".hea", ".dat"):
        out = PTBXL_DIR / (rel_path + ext)
        if out.exists() and not force:
            print(f"  [skip] {rel_path}{ext}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        url = f"{META_BASE}/{rel_path}{ext}"
        urllib.request.urlretrieve(url, out)  # noqa: S310 (trusted host)
        print(f"  [get ] {rel_path}{ext}  ({out.stat().st_size / 1e3:.0f} KB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    db = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    print(f"Fetching {len(SAMPLES)} sample records (100 Hz) + 1 at 500 Hz into {PTBXL_DIR}")
    for eid, note in SAMPLES.items():
        print(f"ecg_id {eid}: {note}")
        _download(db.loc[eid, "filename_lr"], args.force)
    print(f"ecg_id {HR_DEMO_ID}: 500 Hz version (resample demo)")
    _download(db.loc[HR_DEMO_ID, "filename_hr"], args.force)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
