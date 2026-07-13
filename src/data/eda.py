"""Exploratory data analysis for PTB-XL.

Pure-ish analysis helpers (return DataFrames/Series) plus plotting helpers that
save PNGs. Kept import-light so the EDA notebook and a headless script can share
them. Nothing here needs the raw signal files — only the two metadata CSVs.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

# NB: we deliberately do NOT force a matplotlib backend here — that would override
# `%matplotlib inline` in the notebook. Headless callers (scripts/run_eda.py) select
# the Agg backend themselves before importing this module.
import matplotlib.pyplot as plt
import pandas as pd

from src.data import labels as L

SEX_LABELS = {0: "male", 1: "female"}  # PTB-XL encoding


# --- Tabular analyses -------------------------------------------------------
def code_prevalence(df: pd.DataFrame, scp: pd.DataFrame) -> pd.DataFrame:
    """Per-SCP-code record counts + human-readable description + category."""
    counts = Counter()
    for codes in df["scp_codes"]:
        counts.update(L.present_codes(codes))
    cats = L.category_members(scp)
    rows = []
    for code in L.build_label_space():
        n = counts.get(code, 0)
        rows.append({
            "code": code,
            "count": n,
            "prevalence_%": round(100 * n / len(df), 3),
            "description": scp.loc[code, "description"] if code in scp.index else "",
            "diagnostic": code in cats["diagnostic"],
            "form": code in cats["form"],
            "rhythm": code in cats["rhythm"],
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def superclass_prevalence(df: pd.DataFrame, scp: pd.DataFrame) -> pd.DataFrame:
    smap = L.diagnostic_superclass_map(scp)
    counts = Counter()
    for codes in df["scp_codes"]:
        counts.update(L.aggregate_superclasses(codes, smap))
    rows = [{"superclass": sc, "count": counts.get(sc, 0),
             "prevalence_%": round(100 * counts.get(sc, 0) / len(df), 2)}
            for sc in L.DIAGNOSTIC_SUPERCLASSES]
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def labels_per_record(df: pd.DataFrame) -> pd.Series:
    return df["scp_codes"].apply(lambda d: len(L.present_codes(d)))


def records_per_patient(df: pd.DataFrame) -> pd.Series:
    return df.groupby("patient_id").size()


# PTB-XL anonymizes patients older than 89 by recording age as 300.
ANON_AGE = 300


def clean_age(df: pd.DataFrame) -> pd.Series:
    """Age with the anonymization sentinel (300) removed, for stats/plots."""
    age = pd.to_numeric(df["age"], errors="coerce")
    return age.where(age != ANON_AGE)


def demographics_summary(df: pd.DataFrame) -> dict:
    age = clean_age(df)
    return {
        "n_records": len(df),
        "n_patients": int(df["patient_id"].nunique()),
        "age_median": float(age.median()),
        "age_mean": round(float(age.mean()), 1),
        "age_anonymized_300": int((pd.to_numeric(df["age"], errors="coerce") == ANON_AGE).sum()),
        "age_missing": int(age.isna().sum()),
        "sex_male": int((df["sex"] == 0).sum()),
        "sex_female": int((df["sex"] == 1).sum()),
        "n_devices": int(df["device"].nunique()),
        "n_sites": int(df["site"].nunique()),
    }


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().mul(100).round(2).sort_values(ascending=False)
    miss = miss[miss > 0]
    return pd.DataFrame({"column": miss.index, "missing_%": miss.values})


# --- Plots ------------------------------------------------------------------
def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_code_prevalence(prev: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 13))
    ax.barh(prev["code"], prev["count"], color="#c0392b")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("record count (log scale)")
    ax.set_title(f"PTB-XL: prevalence of all {len(prev)} SCP-ECG statements")
    ax.tick_params(axis="y", labelsize=6)
    return _save(fig, out)


def plot_superclass(sc: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sc["superclass"], sc["count"], color="#2c3e50")
    ax.set_ylabel("record count")
    ax.set_title("Diagnostic superclass distribution")
    for i, v in enumerate(sc["count"]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    return _save(fig, out)


def plot_demographics(df: pd.DataFrame, out: Path) -> Path:
    age = clean_age(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(age.dropna(), bins=40, color="#2980b9")
    axes[0].set_title("Age distribution (excl. anonymized >89)")
    axes[0].set_xlabel("age (years)")
    sex = df["sex"].map(SEX_LABELS).value_counts()
    axes[1].bar(sex.index.astype(str), sex.values, color=["#2980b9", "#c0392b"])
    axes[1].set_title("Sex distribution")
    fig.tight_layout()
    return _save(fig, out)


def plot_labels_per_record(counts: pd.Series, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    vc = counts.value_counts().sort_index()
    ax.bar(vc.index.astype(int), vc.values, color="#16a085")
    ax.set_xlabel("number of SCP statements per record")
    ax.set_ylabel("record count")
    ax.set_title("Labels per record (multi-label density)")
    return _save(fig, out)
