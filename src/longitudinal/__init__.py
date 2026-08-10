"""Phase 22 — longitudinal (serial) ECG comparison.

Single-record analysis answers "what is wrong with this ECG". Serial comparison answers a
different and harder question — "what is *different* about this ECG" — and in practice the
second one drives more decisions than the first: new ST depression since yesterday means
something that ST depression alone does not.

    from src.longitudinal import compare_records

    result = compare_records(prior_id=8636, current_id=8641)
    print(result.report.text)

The pieces, in the order the data flows through them:

- :mod:`~src.longitudinal.pairs`     build the patient pair cohort out of PTB-XL
- :mod:`~src.longitudinal.intervals` measure PR / QRS / QT / QTc and per-lead ST levels
- :mod:`~src.longitudinal.delta`     difference two studies, gated by a measured noise floor
- :mod:`~src.longitudinal.report`    render the narrative, and audit it for fabricated change
"""

from src.longitudinal.compare import ComparisonResult, compare_records, compare_signals
from src.longitudinal.delta import (
    DEFAULT_MDC,
    FindingChange,
    IntervalChange,
    LongitudinalDelta,
    STChange,
    build_delta,
    load_mdc,
)
from src.longitudinal.intervals import IntervalSet, measure, split_half
from src.longitudinal.pairs import (
    ECGPair,
    GoldPair,
    build_pairs,
    cohort_summary,
    gold_comparison_pairs,
    load_longitudinal_db,
    load_signal,
    same_day_null,
)
from src.longitudinal.report import (
    ChangeConsistency,
    ChangeReport,
    check_change_consistency,
    render_change_report,
)

__all__ = [
    "DEFAULT_MDC", "ChangeConsistency", "ChangeReport", "ComparisonResult", "ECGPair",
    "FindingChange", "GoldPair", "IntervalChange", "IntervalSet", "LongitudinalDelta",
    "STChange", "build_delta", "build_pairs", "check_change_consistency", "cohort_summary",
    "compare_records", "compare_signals", "gold_comparison_pairs", "load_longitudinal_db",
    "load_mdc", "load_signal", "measure", "render_change_report", "same_day_null",
    "split_half",
]
