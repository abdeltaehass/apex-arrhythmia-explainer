"""Preprocess a PTB-XL split once and cache it as arrays.

The preprocessing chain (wfdb read -> resample -> band-pass -> z-score) is the same
every epoch, so we run it a single time and cache ``(X, Y)`` to ``data/processed/``.
Training then reads from RAM instead of hitting ~17k wfdb files per epoch, which turns
a 20-epoch run from hours into minutes. Cache files are keyed by split + sampling rate.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader

from src.config import PROCESSED_DIR, PTBXL_DIR
from src.detection.dataset import PTBXLDataset


def build_split_cache(
    split: str,
    sampling_rate: int = 100,
    ptbxl_dir=PTBXL_DIR,
    num_workers: int = 4,
    force: bool = False,
    ecg_ids: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X[N,12,T], Y[N,71])`` for a split, building/loading the disk cache.

    ``ecg_ids`` (a debugging/smoke subset) bypasses the cache entirely.
    """
    cache = PROCESSED_DIR / f"{split}_{sampling_rate}hz.npz"
    if ecg_ids is None and cache.exists() and not force:
        d = np.load(cache)
        return d["X"], d["Y"]

    ds = PTBXLDataset(split, sampling_rate=sampling_rate, ptbxl_dir=ptbxl_dir, ecg_ids=ecg_ids)
    if len(ds) == 0:
        raise RuntimeError(f"no records for split={split!r} (is the dataset downloaded?)")
    loader = DataLoader(ds, batch_size=64, num_workers=num_workers)
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb.numpy().astype(np.float32))
        ys.append(yb.numpy().astype(np.float32))
    X, Y = np.concatenate(xs), np.concatenate(ys)

    if ecg_ids is None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=X, Y=Y)
    return X, Y
