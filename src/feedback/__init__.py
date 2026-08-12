"""Phase 24 — human-in-the-loop feedback.

Collect reviewer verdicts on what APEX claimed, store them, and use them to re-tune
per-label decision thresholds — the part of a deployed system that improves after launch
rather than only at training time.

    from src.feedback import FeedbackStore, RatedFinding, update_thresholds

    with FeedbackStore() as store:
        store.log_review([RatedFinding("AFIB", 0.93, "correct")], reviewer_id="dr_a",
                         missed=["1AVB"])
        thresholds = update_thresholds(store)
        thresholds.save()

Then pass the thresholds back into inference:

    analyze_signal(signal, 100, thresholds=ThresholdSet.load())

The pieces:

- :mod:`~src.feedback.store`    SQLite schema and logging
- :mod:`~src.feedback.policy`   the update rule, and the failure modes it is shaped around
- :mod:`~src.feedback.select`   applying per-label thresholds + exploration at inference
- :mod:`~src.feedback.simulate` a synthetic reviewer, for measuring whether the loop works

The short version of what the simulation found: feedback on surfaced findings alone can
only ever raise thresholds (9 of 9 moves upward), letting reviewers report missed findings
does *not* fix that (10 of 10 still upward), and deliberately sampling below the threshold
does (1 up, 18 down) — worth +0.022 macro-F1. See ``docs/feedback/report.md``.
"""

from src.feedback.policy import (
    PolicyConfig,
    ThresholdSet,
    exploration_rates,
    precision_lcb,
    update_thresholds,
)
from src.feedback.select import select_findings
from src.feedback.store import (
    VERDICT_CORRECT,
    VERDICT_INCORRECT,
    VERDICT_UNCERTAIN,
    VERDICTS,
    FeedbackStore,
    LabelCounts,
    RatedFinding,
)

__all__ = [
    "VERDICTS", "VERDICT_CORRECT", "VERDICT_INCORRECT", "VERDICT_UNCERTAIN",
    "FeedbackStore", "LabelCounts", "PolicyConfig", "RatedFinding", "ThresholdSet",
    "exploration_rates", "precision_lcb", "select_findings", "update_thresholds",
]
