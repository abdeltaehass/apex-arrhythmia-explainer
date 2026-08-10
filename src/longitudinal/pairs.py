"""Phase 22 — building the longitudinal cohort out of PTB-XL.

PTB-XL is usually treated as 21,799 independent recordings, but 2,111 of its patients
were recorded more than once: 5,041 records that form **2,930 consecutive pairs**. This
module turns the flat database into that pair cohort and carves out the three sub-cohorts
the rest of the phase depends on.

**The pairing rule.** Records are sorted by ``recording_date`` within a patient and paired
*consecutively* (1->2, 2->3, ...), not all-against-all. A clinician reading a follow-up ECG
compares it to the most recent prior study, so consecutive pairing is the clinically
faithful choice; all-pairs would also inflate n by counting the same patient's drift
repeatedly.

**No fold leakage, for free.** PTB-XL's ``strat_fold`` is assigned per *patient*, so every
record of a patient shares a fold — verified in :func:`assert_no_fold_leakage`. A pair can
therefore never straddle the train/test boundary, and the 294 test-fold pairs are held out
by exactly the same partition the detector was trained under. Nothing extra to do.

**The same-day null cohort.** 327 pairs were recorded less than 24 h apart (median 13.7 h,
102 of them under an hour). Over minutes-to-hours a PR interval does not
meaningfully remodel, so the spread of *measured* differences across these pairs is
dominated by measurement error plus beat-to-beat physiology rather than by disease. That
makes them a test-retest null: the noise floor every reported change must clear
(:mod:`src.longitudinal.delta` fits its thresholds on this cohort).

The null is *not* perfectly clean and is not treated as such. Same-day pairs do contain
genuine fast change — arrhythmia termination is the obvious one (16404 -> 16408 is atrial
fibrillation reverting to sinus rhythm 53 minutes later). :func:`same_day_null` therefore
takes a ``stable_only`` flag that additionally requires the two records to carry an
identical SCP code set, which drops 327 -> 91 pairs but removes the pairs that visibly
changed. Both variants are reported; the difference between them is itself informative.

**The gold cohort.** 110 PTB-XL reports contain a clinician's own comparison sentence
("compared with tracing of 30:7:92. there is now st segment elevation in v5,6"). For 12 of
them the referenced prior tracing is itself in PTB-XL and recoverable, either by parsing
the cited date or via "earlier today". Those 12 pairs carry a *cardiologist-written change
statement* — the only real reference standard available for this task, and what the Phase-22
examples are graded against instead of my own adjudication.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PTBXL_DIR

# "compared with tracing of 30:7:92" / "compared with tracings of 4:5,92" — the separator
# is inconsistent in the source data (':' , '.' , ',' , '/') and the year is 2-digit.
_CITED_DATE = re.compile(r"tracing[s]?\s+of\s+(\d{1,2})\s*[:.,/]\s*(\d{1,2})\s*[:.,/]\s*(\d{2,4})")
_EARLIER_TODAY = re.compile(r"earlier\s+today|tracing\s+of\s+today", re.I)
_COMPARISON = re.compile(r"compared\s+(?:with|to)", re.I)


@dataclass(frozen=True)
class ECGPair:
    """One prior -> current comparison."""

    patient_id: int
    prior_id: int          # ecg_id of the earlier recording
    current_id: int        # ecg_id of the later recording
    prior_date: pd.Timestamp
    current_date: pd.Timestamp
    fold: int              # PTB-XL strat_fold (shared by both — see assert_no_fold_leakage)
    prior_codes: frozenset[str]
    current_codes: frozenset[str]
    prior_report: str = ""
    current_report: str = ""

    @property
    def interval_days(self) -> int:
        return int((self.current_date - self.prior_date).days)

    @property
    def interval_hours(self) -> float:
        return (self.current_date - self.prior_date).total_seconds() / 3600.0

    @property
    def same_day(self) -> bool:
        """Recorded less than 24 h apart.

        Deliberately *elapsed hours*, not calendar-date equality. 168 of the 327 rapid
        repeats cross midnight (10:28 -> 08:20 the next morning is a 21.9 h next-day
        follow-up), and a date-equality test would throw away more than half of the null
        cohort for a reason — where the clock happened to roll over — that has nothing to
        do with whether the heart had time to change.
        """
        return self.interval_hours < 24.0

    @property
    def new_codes(self) -> frozenset[str]:
        """SCP codes present now but not before (ground-truth 'new onset')."""
        return self.current_codes - self.prior_codes

    @property
    def resolved_codes(self) -> frozenset[str]:
        """SCP codes present before but not now (ground-truth 'resolved')."""
        return self.prior_codes - self.current_codes

    @property
    def persistent_codes(self) -> frozenset[str]:
        return self.prior_codes & self.current_codes

    @property
    def label_stable(self) -> bool:
        """True iff the annotated code set is byte-identical between the two studies."""
        return self.prior_codes == self.current_codes

    def gap_bucket(self) -> str:
        """Coarse elapsed-time bucket used for stratified reporting."""
        if self.same_day:
            return "<24h"
        d = self.interval_days
        if d <= 7:
            return "1-7d"
        if d <= 30:
            return "8-30d"
        if d <= 365:
            return "31-365d"
        return ">1y"

    def describe_gap(self) -> str:
        """Human phrasing of the elapsed time, for the report header."""
        def plural(n: int, unit: str) -> str:
            return f"{n} {unit} earlier" if n == 1 else f"{n} {unit}s earlier"

        h = self.interval_hours
        if h < 24:
            return plural(round(h), "hour") if h >= 1 else plural(round(h * 60), "minute")
        d = self.interval_days
        if d < 60:
            return plural(d, "day")
        if d < 730:
            return plural(d // 30, "month")
        return plural(d // 365, "year")


def load_longitudinal_db(ptbxl_dir: Path = PTBXL_DIR) -> pd.DataFrame:
    """PTB-XL database restricted to patients with 2+ recordings, date-sorted."""
    df = pd.read_csv(ptbxl_dir / "ptbxl_database.csv")
    df["recording_date"] = pd.to_datetime(df["recording_date"])
    df["report"] = df["report"].fillna("")
    df["code_set"] = df["scp_codes"].apply(lambda s: frozenset(ast.literal_eval(s).keys()))
    counts = df.groupby("patient_id").size()
    repeat_ids = counts[counts >= 2].index
    out = df[df["patient_id"].isin(repeat_ids)].copy()
    return out.sort_values(["patient_id", "recording_date"]).reset_index(drop=True)


def assert_no_fold_leakage(df: pd.DataFrame) -> None:
    """Fail loudly if any patient's records span more than one ``strat_fold``.

    The whole held-out story of this phase rests on PTB-XL assigning folds per patient. It
    does — but it is asserted rather than assumed, because a pair that straddled folds
    would silently evaluate the detector on its own training data.
    """
    spans = df.groupby("patient_id")["strat_fold"].nunique()
    offenders = spans[spans > 1]
    if len(offenders):
        raise AssertionError(
            f"{len(offenders)} patient(s) span multiple folds, e.g. {offenders.index[:5].tolist()} "
            "— consecutive pairs would leak across the train/test split"
        )


def build_pairs(df: pd.DataFrame | None = None, folds: tuple[int, ...] | None = None,
                ptbxl_dir: Path = PTBXL_DIR) -> list[ECGPair]:
    """All consecutive prior->current pairs, optionally restricted to ``folds``.

    Pass ``folds=(10,)`` for the held-out test cohort (294 pairs), ``(9,)`` for validation
    (241). ``None`` returns all 2,930.
    """
    if df is None:
        df = load_longitudinal_db(ptbxl_dir)
    assert_no_fold_leakage(df)
    if folds is not None:
        df = df[df["strat_fold"].isin(folds)]

    pairs: list[ECGPair] = []
    for patient_id, group in df.groupby("patient_id"):
        g = group.sort_values("recording_date").reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.loc[i], g.loc[i + 1]
            pairs.append(ECGPair(
                patient_id=int(patient_id),
                prior_id=int(a["ecg_id"]), current_id=int(b["ecg_id"]),
                prior_date=a["recording_date"], current_date=b["recording_date"],
                fold=int(a["strat_fold"]),
                prior_codes=a["code_set"], current_codes=b["code_set"],
                prior_report=str(a["report"]), current_report=str(b["report"]),
            ))
    return pairs


def same_day_null(pairs: list[ECGPair], stable_only: bool = False,
                  max_hours: float | None = None) -> list[ECGPair]:
    """The test-retest cohort used to fit the change-detection noise floor.

    ``stable_only`` additionally requires an identical SCP code set (327 -> 91 pairs),
    excluding same-day pairs that genuinely changed — arrhythmia termination being the
    real one. ``max_hours`` tightens the window further (102 pairs are <1 h apart).

    Both the loose and strict variants are reported in ``docs/longitudinal/report.md``: the
    loose one over-states the noise floor by absorbing real change, the strict one risks
    understating it by conditioning on labels that themselves came from reading the ECG.
    Neither is unimpeachable, so the phase quotes the gap between them.
    """
    out = [p for p in pairs if p.same_day]
    if max_hours is not None:
        out = [p for p in out if p.interval_hours <= max_hours]
    if stable_only:
        out = [p for p in out if p.label_stable]
    return out


@dataclass(frozen=True)
class GoldPair:
    """A pair whose *current* report contains the reading cardiologist's own comparison."""

    pair: ECGPair
    statement: str      # the clinician's comparison sentence(s), verbatim
    match: str          # how the prior was recovered: "exact-date" | "earlier-today"


def _comparison_sentences(report: str) -> str:
    """The portion of a PTB-XL report from the comparison clause onward, verbatim."""
    m = _COMPARISON.search(report)
    return report[m.start():].strip() if m else report.strip()


def gold_comparison_pairs(df: pd.DataFrame | None = None,
                          ptbxl_dir: Path = PTBXL_DIR) -> list[GoldPair]:
    """Pairs where PTB-XL's own report states how the ECG changed since the prior study.

    Recovers the prior record two ways:

    - **exact-date** — the report cites a date ("tracing of 30:7:92") that resolves to a
      real earlier record of the same patient.
    - **earlier-today** — the report says "earlier today" and the patient does have an
      earlier record on that calendar day.

    Yields 12 pairs. Most of the 110 comparison reports are *not* recoverable, because the
    tracing the cardiologist held was never included in PTB-XL — so this is a small,
    precious set, not a benchmark. It is the reference standard for the Phase-22 examples.
    """
    if df is None:
        df = load_longitudinal_db(ptbxl_dir)
    by_patient = {pid: g.sort_values("recording_date").reset_index(drop=True)
                  for pid, g in df.groupby("patient_id")}

    gold: list[GoldPair] = []
    for _, row in df[df["report"].str.contains(_COMPARISON, na=False)].iterrows():
        g = by_patient[row["patient_id"]]
        earlier = g[g["recording_date"] < row["recording_date"]]
        if earlier.empty:
            continue

        prior_row = None
        match = ""
        m = _CITED_DATE.search(row["report"])
        if m:
            day, month, year = (int(x) for x in m.groups())
            year += 1900 if year < 100 else 0
            try:
                cited = pd.Timestamp(year=year, month=month, day=day).date()
            except ValueError:
                continue
            hits = earlier[earlier["recording_date"].dt.date == cited]
            if not hits.empty:
                prior_row, match = hits.iloc[-1], "exact-date"
        elif _EARLIER_TODAY.search(row["report"]):
            candidate = earlier.iloc[-1]
            if candidate["recording_date"].date() == row["recording_date"].date():
                prior_row, match = candidate, "earlier-today"

        if prior_row is None:
            continue
        gold.append(GoldPair(
            pair=ECGPair(
                patient_id=int(row["patient_id"]),
                prior_id=int(prior_row["ecg_id"]), current_id=int(row["ecg_id"]),
                prior_date=prior_row["recording_date"], current_date=row["recording_date"],
                fold=int(row["strat_fold"]),
                prior_codes=prior_row["code_set"], current_codes=row["code_set"],
                prior_report=str(prior_row["report"]), current_report=str(row["report"]),
            ),
            statement=_comparison_sentences(str(row["report"])),
            match=match,
        ))
    return sorted(gold, key=lambda gp: gp.pair.current_id)


def cohort_summary(pairs: list[ECGPair]) -> dict:
    """Counts used in the report header and asserted by the tests."""
    buckets: dict[str, int] = {}
    for p in pairs:
        buckets[p.gap_bucket()] = buckets.get(p.gap_bucket(), 0) + 1
    return {
        "n_pairs": len(pairs),
        "n_patients": len({p.patient_id for p in pairs}),
        "n_same_day": sum(p.same_day for p in pairs),
        "n_label_stable": sum(p.label_stable for p in pairs),
        "label_change_rate": ((sum(not p.label_stable for p in pairs) / len(pairs))
                             if pairs else float("nan")),
        "by_gap_bucket": buckets,
        "median_gap_days": (float(pd.Series([p.interval_days for p in pairs]).median())
                            if pairs else float("nan")),
    }


def load_signal(ecg_id: int, df: pd.DataFrame | None = None, sampling_rate: int = 100,
                ptbxl_dir: Path = PTBXL_DIR) -> tuple[np.ndarray, int]:
    """Read one PTB-XL record as a raw ``(12, T)`` array in mV, plus its sampling rate.

    Deliberately *raw*: interval and ST measurement need physical millivolts, and the
    detector's z-scoring would destroy the amplitudes.
    """
    import wfdb

    if df is None:
        df = pd.read_csv(ptbxl_dir / "ptbxl_database.csv")
    row = df.loc[df["ecg_id"] == ecg_id]
    if row.empty:
        raise KeyError(f"ecg_id {ecg_id} not in the PTB-XL database")
    col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    signal, _ = wfdb.rdsamp(str(ptbxl_dir / row.iloc[0][col]))
    return np.asarray(signal, dtype=np.float32).T, sampling_rate
