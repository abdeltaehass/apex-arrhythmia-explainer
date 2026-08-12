"""Phase 24 — validating the feedback loop with a simulated reviewer.

Infrastructure that has never been exercised is a hypothesis, not a system. Real clinical
feedback is not available here, but PTB-XL's labels can stand in for a reviewer: show the
simulated clinician what APEX surfaced, let them mark each finding against the annotation,
and run the loop for real. That converts "we built a feedback pipeline" into a measurable
claim about whether the pipeline helps, and — more usefully — into a measurement of the
conditions under which it *hurts*.

The reviewer model has three knobs, each corresponding to something that goes wrong with
real reviewers:

``error_rate``
    The reviewer is wrong. Ratings are opinion, not ground truth (Phase 22 measured PTB-XL's
    own annotators disagreeing 72% of the time at zero elapsed time), so the loop must
    survive being fed mistakes.

``uncertain_rate``
    The reviewer declines to call it, more often near the decision boundary where the model
    is also least sure. Those are precisely the ratings the loop would most like to have, and
    precisely the ones a real clinician is least willing to give.

``miss_report_rate``
    The reviewer notices only *some* of what the model missed. Nobody exhaustively lists
    every absent finding, so recall evidence arrives sparse and biased toward the obvious.

The loop is run in arms that switch the mitigations on one at a time
(:func:`run_arm`), which is what isolates the ratchet from its fixes. Feedback is streamed
from the **validation** fold and every reported metric is computed on the **test** fold, so
no threshold is ever chosen and scored on the same records.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.feedback.policy import PolicyConfig, ThresholdSet, exploration_rates, update_thresholds
from src.feedback.store import (
    VERDICT_CORRECT,
    VERDICT_INCORRECT,
    VERDICT_UNCERTAIN,
    FeedbackStore,
    RatedFinding,
)


@dataclass
class ReviewerModel:
    """A synthetic clinician."""

    error_rate: float = 0.0          # probability of flipping a correct/incorrect verdict
    uncertain_rate: float = 0.0      # baseline probability of declining to call it
    miss_report_rate: float = 0.0    # probability of reporting any given missed finding
    # Uncertainty concentrates near the decision boundary; this scales that effect.
    boundary_uncertainty: float = 0.0

    def verdict(self, is_true: bool, confidence: float, threshold: float,
                rng: np.random.Generator) -> str:
        p_unc = self.uncertain_rate
        if self.boundary_uncertainty:
            # closest to the threshold -> most likely to be called uncertain
            nearness = max(0.0, 1.0 - abs(confidence - threshold) / 0.3)
            p_unc = min(0.9, p_unc + self.boundary_uncertainty * nearness)
        if rng.random() < p_unc:
            return VERDICT_UNCERTAIN
        if rng.random() < self.error_rate:
            is_true = not is_true
        return VERDICT_CORRECT if is_true else VERDICT_INCORRECT


@dataclass
class ArmConfig:
    """One experimental condition."""

    name: str
    collect_missed: bool = False     # reviewer may report false negatives
    explore: bool = False            # surface some sub-threshold findings for review
    reviewer: ReviewerModel = field(default_factory=ReviewerModel)
    policy: PolicyConfig = field(default_factory=PolicyConfig)


@dataclass
class Snapshot:
    """Held-out performance at one point in the stream."""

    n_reports: int
    n_ratings: int
    macro_f1: float
    macro_precision: float
    macro_recall: float
    mean_threshold: float
    n_moved: int
    # The ratchet is a claim about *direction*, so direction is measured directly rather
    # than inferred from the mean: an unaided loop should raise thresholds and essentially
    # never lower one.
    moved_up: int = 0
    moved_down: int = 0


def macro_scores(probs: np.ndarray, y: np.ndarray, label_space: list[str],
                 thresholds: ThresholdSet) -> tuple[float, float, float]:
    """Macro precision / recall / F1 over labels that occur in ``y``.

    Macro rather than micro on purpose: the point of per-label thresholds is the rare
    labels, and micro-averaging would let the common ones drown them out.

    **Precision is averaged only over labels the model actually predicts.** At a global 0.5
    threshold, 32 of these 71 labels are never predicted at all, and scoring an undefined
    precision as 0.0 (numpy's and sklearn's default) does something perverse here: pushing a
    label's threshold up until it goes silent would *lower* reported precision, so the
    metric would punish the very behaviour it is meant to detect. A label that makes no
    predictions makes no mistakes; it is recall and F1 that must carry the cost, and they
    do — both are averaged over every present label, scoring a silent one as 0.
    """
    t = np.array([thresholds.get(c) for c in label_space], dtype=float)
    pred = probs >= t[None, :]
    present = y.sum(axis=0) > 0
    tp = (pred & (y > 0)).sum(axis=0).astype(float)
    fp = (pred & (y == 0)).sum(axis=0).astype(float)
    fn = ((~pred) & (y > 0)).sum(axis=0).astype(float)
    predicts = (tp + fp) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(predicts, tp / np.maximum(tp + fp, 1), 0.0)
        rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    scored = present & predicts
    macro_p = float(prec[scored].mean()) if scored.any() else 0.0
    return (macro_p, float(rec[present].mean()), float(f1[present].mean()))


def run_arm(arm: ArmConfig, val_probs: np.ndarray, val_y: np.ndarray,
            test_probs: np.ndarray, test_y: np.ndarray, label_space: list[str],
            store: FeedbackStore, batch_size: int = 100, seed: int = 0,
            initial: ThresholdSet | None = None) -> list[Snapshot]:
    """Stream the validation fold through the loop, scoring on test after each batch."""
    rng = np.random.default_rng(seed)
    thresholds = initial or ThresholdSet()
    order = rng.permutation(len(val_probs))

    baseline = {c: thresholds.get(c) for c in label_space}
    p, r, f1 = macro_scores(test_probs, test_y, label_space, thresholds)
    history = [Snapshot(0, 0, f1, p, r,
                        float(np.mean([thresholds.get(c) for c in label_space])), 0)]

    explore_rates: dict[str, float] = {}
    n_ratings = 0
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        for i in batch:
            probs_i = val_probs[i]
            truth = {label_space[j] for j in np.flatnonzero(val_y[i])}

            from src.feedback.select import select_findings

            surfaced, exploratory = select_findings(
                probs_i, label_space, thresholds,
                explore_rates if arm.explore else None,
                exploration_floor=arm.policy.exploration_floor, rng=rng)

            rated: list[RatedFinding] = []
            for label in surfaced:
                j = label_space.index(label)
                t = thresholds.get(label)
                verdict = arm.reviewer.verdict(label in truth, float(probs_i[j]), t, rng)
                is_probe = label in exploratory
                rated.append(RatedFinding(
                    label=label, confidence=float(probs_i[j]), verdict=verdict, threshold=t,
                    exploratory=is_probe,
                    sampling_rate=(explore_rates.get(label, arm.policy.base_exploration)
                                   if is_probe else 1.0)))

            missed: list[str] = []
            missed_conf: dict[str, float] = {}
            if arm.collect_missed:
                for label in truth - set(surfaced):
                    if rng.random() < arm.reviewer.miss_report_rate:
                        missed.append(label)
                        missed_conf[label] = float(probs_i[label_space.index(label)])

            n_ratings += len(rated)
            store.log_review(rated, reviewer_id="sim", record_ref=f"val-{i}",
                             thresholds=None, missed=missed, missed_confidences=missed_conf)

        thresholds = update_thresholds(store, thresholds, arm.policy)
        if arm.explore:
            explore_rates = exploration_rates(store, arm.policy)
        p, r, f1 = macro_scores(test_probs, test_y, label_space, thresholds)
        up = sum(1 for c in label_space if thresholds.get(c) > baseline[c] + 1e-9)
        down = sum(1 for c in label_space if thresholds.get(c) < baseline[c] - 1e-9)
        history.append(Snapshot(
            n_reports=int(start + len(batch)), n_ratings=n_ratings,
            macro_f1=f1, macro_precision=p, macro_recall=r,
            mean_threshold=float(np.mean([thresholds.get(c) for c in label_space])),
            n_moved=up + down, moved_up=up, moved_down=down))
    return history
