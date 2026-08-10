"""Phase 22 — the orchestrator: two recordings in, one change report out.

Ties the phase together — measure both studies, run the detector on both, difference them,
render the narrative, audit it — behind :func:`compare_records` (by PTB-XL ecg_id) and
:func:`compare_signals` (by raw array, for callers outside the dataset).

**Calibrated probabilities are used here even though serving still does not.** Phase 17
fitted a per-label vector scaler that cut ECE from 0.0793 to 0.0020, and it has been
sitting unused in the inference path. Serial comparison is the place where that matters
most: a new-onset call is a *difference of two threshold crossings*, so a label whose
probability is systematically inflated toward 0.5 will flicker across the line between two
studies of an unchanged patient and manufacture new findings out of nothing. The calibrator
is applied when ``outputs/calibration.json`` is present and silently skipped when it is
not, with :attr:`ComparisonResult.calibrated` recording which happened so a report can
never be quoted without knowing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import CFG, ROOT
from src.longitudinal.delta import LongitudinalDelta, build_delta, load_mdc
from src.longitudinal.intervals import IntervalSet, measure
from src.longitudinal.pairs import ECGPair, load_signal
from src.longitudinal.report import (
    ChangeConsistency,
    ChangeReport,
    check_change_consistency,
    render_change_report,
)

CALIBRATION_PATH = ROOT / "outputs" / "calibration.json"


@dataclass
class ComparisonResult:
    """The full Phase-22 output for one prior/current pair."""

    delta: LongitudinalDelta
    report: ChangeReport
    consistency: ChangeConsistency
    prior_intervals: IntervalSet
    current_intervals: IntervalSet
    prior_probs: dict[str, float]
    current_probs: dict[str, float]
    calibrated: bool

    def as_dict(self) -> dict:
        return {
            "delta": self.delta.as_dict(),
            "report": self.report.as_dict(),
            "consistency": self.consistency.as_dict(),
            "prior_intervals": self.prior_intervals.as_dict(),
            "current_intervals": self.current_intervals.as_dict(),
            "calibrated": self.calibrated,
        }


def _load_calibrator(path: Path = CALIBRATION_PATH):
    """The Phase-17 scaler chosen by ``calibration.json['best']``, or ``None``."""
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text())
        from src.eval.calibration import load_scaler

        return load_scaler(blob[blob["best"]])
    except Exception:      # a malformed artefact must not take down comparison
        return None


def _predict(signals: list[np.ndarray], sampling_rate: int, checkpoint=None,
             device: str = "cpu", calibrate: bool = True
             ) -> tuple[list[dict[str, float]], list[str], bool]:
    """Per-label probabilities for a batch of raw recordings.

    Both studies go through the model in a single forward pass so they cannot pick up
    different numerical treatment.
    """
    import torch

    from src.preprocessing.pipeline import preprocess
    from src.serving.model_cache import get_detector

    model, label_space, _ = get_detector(checkpoint, device=device)
    batch = np.stack([preprocess(np.asarray(s, dtype=np.float32), fs_in=sampling_rate,
                                 fs_out=CFG.sampling_rate, detect_rpeaks=False)[0]
                      for s in signals])
    with torch.no_grad():
        logits = model(torch.from_numpy(batch).to(device)).cpu().numpy()

    calibrator = _load_calibrator() if calibrate else None
    if calibrator is not None:
        probs = calibrator.transform(logits)
        used = True
    else:
        probs = 1.0 / (1.0 + np.exp(-logits))
        used = False
    return ([{c: float(p) for c, p in zip(label_space, row, strict=True)} for row in probs],
            label_space, used)


def compare_signals(prior_signal, current_signal, sampling_rate: int = 100,
                    pair: ECGPair | None = None, checkpoint=None, device: str = "cpu",
                    with_detector: bool = True, calibrate: bool = True,
                    mdc: dict[str, float] | None = None) -> ComparisonResult:
    """Compare two raw ``(12, T)`` recordings in mV.

    ``with_detector=False`` runs the measurement channel alone — no torch import, no
    checkpoint — which is what the interval-only evaluations and the unit tests use.
    """
    prior_signal = np.asarray(prior_signal, dtype=np.float32)
    current_signal = np.asarray(current_signal, dtype=np.float32)
    prior_iv = measure(prior_signal, sampling_rate)
    current_iv = measure(current_signal, sampling_rate)

    prior_probs: dict[str, float] = {}
    current_probs: dict[str, float] = {}
    descriptions: dict[str, str] = {}
    calibrated = False
    if with_detector:
        (prior_probs, current_probs), label_space, calibrated = _predict(
            [prior_signal, current_signal], sampling_rate, checkpoint, device, calibrate)
        try:
            from src.serving.model_cache import get_scp_statements

            scp = get_scp_statements()
            descriptions = {c: (str(scp.loc[c, "description"]) if c in scp.index else "")
                            for c in label_space}
        except Exception:
            descriptions = {}

    delta = build_delta(
        prior_iv, current_iv,
        prior_probs=prior_probs if with_detector else None,
        current_probs=current_probs if with_detector else None,
        descriptions=descriptions, pair=pair, mdc=mdc or load_mdc(),
        decision_threshold=CFG.review_threshold,
    )
    report = render_change_report(delta)
    consistency = check_change_consistency(report.text, delta)
    return ComparisonResult(delta, report, consistency, prior_iv, current_iv,
                            prior_probs, current_probs, calibrated)


def compare_records(prior_id: int, current_id: int, df=None, sampling_rate: int = 100,
                    pair: ECGPair | None = None, **kwargs) -> ComparisonResult:
    """Compare two PTB-XL records by ``ecg_id``.

    When ``pair`` is omitted the dates are read from the database so the report can still
    say *when* the prior study was — a serial comparison without an elapsed time is close
    to useless clinically.
    """
    import pandas as pd

    from src.config import PTBXL_DIR

    if df is None:
        df = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv")
        df["recording_date"] = pd.to_datetime(df["recording_date"])
    if pair is None:
        import ast

        rows = {int(r["ecg_id"]): r for _, r in df[df["ecg_id"].isin([prior_id, current_id])].iterrows()}
        if prior_id in rows and current_id in rows:
            a, b = rows[prior_id], rows[current_id]
            codes = lambda r: frozenset(  # noqa: E731
                r["code_set"] if "code_set" in r else ast.literal_eval(r["scp_codes"]).keys())
            pair = ECGPair(
                patient_id=int(a["patient_id"]), prior_id=prior_id, current_id=current_id,
                prior_date=a["recording_date"], current_date=b["recording_date"],
                fold=int(a["strat_fold"]), prior_codes=codes(a), current_codes=codes(b),
                prior_report=str(a.get("report", "")), current_report=str(b.get("report", "")),
            )

    prior_signal, fs = load_signal(prior_id, df, sampling_rate)
    current_signal, _ = load_signal(current_id, df, sampling_rate)
    return compare_signals(prior_signal, current_signal, fs, pair=pair, **kwargs)
