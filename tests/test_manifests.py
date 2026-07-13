"""Tests for the PTB-XL manifest builder.

The split/label logic is checked on a tiny synthetic frame (no data needed). A
second test runs against the real metadata CSVs when they are present, and is
skipped otherwise so CI without the dataset still passes.
"""

import json

import pandas as pd
import pytest

from src.config import PTBXL_DIR
from src.data import manifests as M


def test_split_of_folds():
    assert [M.split_of(f) for f in (1, 8)] == ["train", "train"]
    assert M.split_of(9) == "val"
    assert M.split_of(10) == "test"
    with pytest.raises(ValueError):
        M.split_of(0)


def _tiny_frame():
    # patient 1 has two records (folds 1 & 8 -> both train); patient 2 is in test.
    df = pd.DataFrame(
        {
            "patient_id": [1.0, 1.0, 2.0],
            "age": [60, 61, 70], "sex": [0, 0, 1],
            "height": [None, None, None], "weight": [None, None, None],
            "device": ["d", "d", "d"], "site": [0, 0, 1],
            "recording_date": ["x", "x", "x"], "strat_fold": [1, 8, 10],
            "filename_lr": ["a", "b", "c"], "filename_hr": ["a", "b", "c"],
            "scp_codes": [{"NORM": 100.0, "SR": 0.0}, {"AFIB": 100.0}, {"IMI": 0.0}],
        },
        index=pd.Index([1, 2, 3], name="ecg_id"),
    )
    scp = pd.DataFrame(
        {"diagnostic": [1.0, 1.0, 1.0, None], "form": [None, None, None, None],
         "rhythm": [None, None, 1.0, 1.0], "diagnostic_class": ["NORM", "MI", "STTC", None]},
        index=["NORM", "IMI", "AFIB", "SR"],
    )
    return df, scp


def test_present_codes_keeps_zero_likelihood():
    from src.data import labels as L
    # SR at 0.0 must still count as present.
    assert L.present_codes({"NORM": 100.0, "SR": 0.0}) == ["NORM", "SR"]


def test_build_manifest_labels_and_splits():
    df, scp = _tiny_frame()
    man = M.build_manifest(df, scp)
    assert list(man["split"]) == ["train", "train", "test"]
    # record 1: NORM(diagnostic->NORM) present; SR present but not diagnostic
    assert json.loads(man.loc[man["ecg_id"] == 1, "scp_codes"].iloc[0]) == ["NORM", "SR"]
    assert json.loads(man.loc[man["ecg_id"] == 1, "diagnostic_superclasses"].iloc[0]) == ["NORM"]
    assert man.loc[man["ecg_id"] == 1, "sc_NORM"].iloc[0] == 1
    # no patient split leakage on the synthetic frame
    M.assert_no_patient_leakage(man)


def test_leakage_detector_trips():
    df, scp = _tiny_frame()
    man = M.build_manifest(df, scp)
    man.loc[man["ecg_id"] == 2, "split"] = "test"  # patient 1 now spans train+test
    with pytest.raises(AssertionError):
        M.assert_no_patient_leakage(man)


@pytest.mark.skipif(
    not (PTBXL_DIR / "ptbxl_database.csv").exists(),
    reason="PTB-XL metadata not downloaded",
)
def test_real_manifest_splits_and_no_leakage():
    from src.data import labels as L

    df = L.load_database()
    scp = L.load_scp_statements()
    man = M.build_manifest(df, scp)
    assert len(man) == len(df)
    M.assert_no_patient_leakage(man)
    # every record lands in exactly one split
    assert set(man["split"]) == {"train", "val", "test"}
    assert man["split"].isna().sum() == 0
