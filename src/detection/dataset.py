"""PTB-XL torch Dataset.

Loads waveforms via `wfdb`, applies the Phase 2 preprocessing pipeline (resample ->
band-pass 0.5-40 Hz -> per-lead z-score), and returns (signal[12, T], label[71]) pairs.
Split selection uses PTB-XL's `strat_fold`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import DEFAULT_SAMPLING_RATE, PTBXL_DIR, TEST_FOLD, TRAIN_FOLDS, VAL_FOLD
from src.data.labels import build_label_space, encode, load_database
from src.preprocessing.pipeline import Preprocessor

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # keep importable before torch is installed
    torch = None  # type: ignore
    Dataset = object  # type: ignore


def _fold_filter(split: str):
    if split == "train":
        return lambda f: f in TRAIN_FOLDS
    if split == "val":
        return lambda f: f == VAL_FOLD
    if split == "test":
        return lambda f: f == TEST_FOLD
    raise ValueError(f"unknown split: {split}")


class PTBXLDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        sampling_rate: int = DEFAULT_SAMPLING_RATE,
        ptbxl_dir: Path = PTBXL_DIR,
        transform="default",
        target_rate: int = DEFAULT_SAMPLING_RATE,
        ecg_ids: list[int] | None = None,
    ):
        """`transform="default"` applies the standard preprocessing chain (resample to
        `target_rate` -> band-pass -> z-score). Pass `None` for raw signals, or a
        callable for a custom chain. `ecg_ids` restricts the split to specific records
        (handy for a downloaded sample subset or debugging)."""
        self.sampling_rate = sampling_rate
        self.target_rate = target_rate
        self.ptbxl_dir = ptbxl_dir
        if transform == "default":
            transform = Preprocessor(fs_in=sampling_rate, fs_out=target_rate)
        self.transform = transform
        self.label_space = build_label_space(ptbxl_dir)

        df = load_database(ptbxl_dir)
        keep = df["strat_fold"].apply(_fold_filter(split))
        if ecg_ids is not None:
            keep &= df.index.isin(ecg_ids)
        self.df = df[keep].reset_index()
        col = "filename_lr" if sampling_rate == 100 else "filename_hr"
        self.records = self.df[col].tolist()
        self.labels = np.stack(
            [encode(codes, self.label_space) for codes in self.df["scp_codes"]]
        ) if len(self.df) else np.empty((0, len(self.label_space)), np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        import wfdb  # local import so the module loads without wfdb installed

        signal, _ = wfdb.rdsamp(str(self.ptbxl_dir / self.records[i]))
        signal = signal.T.astype(np.float32)  # (12, T)
        if self.transform is not None:
            signal = self.transform(signal)
        label = self.labels[i]
        if torch is not None:
            return torch.from_numpy(signal), torch.from_numpy(label)
        return signal, label
