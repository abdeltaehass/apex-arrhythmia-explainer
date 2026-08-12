"""Phase 24 — turning reviewer feedback into per-label decision thresholds.

This is the online-learning half of the loop, and it is mostly an exercise in not fooling
yourself. Feedback is cheap to collect and easy to misuse; three failure modes below are
specific enough to be worth naming, because each of them produces a system that looks like
it is improving while getting worse.

**1. The ratchet.** A reviewer can only rate findings APEX actually showed them. So every
rating is drawn from the region *above* the current threshold, and the only errors visible
in that region are false positives. Raising a threshold removes false positives and is
always supported by the data; lowering it would admit findings nobody has ever rated and is
never supported. A loop built on this data alone therefore moves in one direction forever.
Precision climbs, the dashboard looks better every week, and recall quietly collapses.

This is verification bias, and it is not fixed by being careful with the statistics — the
data genuinely does not contain the answer. It is fixed by changing what gets collected, in
two ways: reviewers can report findings APEX **missed** (:mod:`src.feedback.store`), and a
small fraction of *sub-threshold* findings are deliberately surfaced for review
(:func:`exploration_rates`). ``scripts/feedback_sim.py`` measures how badly the loop
degrades without them.

**2. Small-n whiplash.** Three ratings on a rare label is not evidence, but it will happily
produce a precision of 0.33 and a large threshold move. Rather than bolting a minimum count
onto a point estimate, the decision rule uses the **lower bound of a Beta posterior**: the
threshold moves only if precision is credibly above target, so a small sample fails the
test by construction because its lower bound is far beneath its mean. The prior is anchored
to the model's own claim — a calibrated confidence of 0.8 *is* a prediction that 80% of
such findings are correct (Phase 17), which makes it the natural null the feedback has to
overturn.

**3. Treating reviewers as ground truth.** They are not. Phase 22 found PTB-XL's own
annotators disagreeing 72% of the time at zero elapsed time. Ratings are opinion, so the
loop is deliberately sluggish: every threshold is capped in how far it can move per update
and clamped to a sane band, and :meth:`~src.feedback.store.FeedbackStore.agreement` reports
inter-rater agreement so a deployment can notice when the feedback is measuring its
reviewers rather than its model.

**Uncertain is not half-correct.** An `uncertain` verdict is dropped from the precision
estimate. It says the reviewer could not tell, which is a fact about the case; scoring it
in either direction would manufacture an opinion nobody gave.

**Missed findings never enter the precision estimate.** It is tempting — a missed finding
is a known true positive with a known confidence, apparently free data below threshold. But
reviewers report the misses they happen to notice and never report the sub-threshold *false*
positives sitting next to them, so folding them in would bias precision upward exactly where
the data is thinnest, and produce the mirror image of the ratchet. They are used only as a
signal that recall is being lost for that label, which raises its exploration rate so the
region gets sampled properly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

from src.config import CFG, ROOT
from src.feedback.store import FeedbackStore, LabelCounts

THRESHOLD_PATH = ROOT / "outputs" / "feedback_thresholds.json"

# Candidate thresholds considered per label.
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


@dataclass
class PolicyConfig:
    """Knobs for the update rule. The defaults are deliberately conservative."""

    # "f1" chooses the threshold maximising estimated F1; "precision" holds a precision
    # floor and takes the most recall available under it. F1 is the default because a
    # precision target silently assumes recall is already adequate, and here it is not —
    # macro recall at the shipped 0.5 threshold is 0.185.
    objective: str = "f1"
    target_precision: float = 0.70
    # Quantile of the Beta posterior the target must clear. 0.05 = "we are 95% sure
    # precision is at least the target". Lower is bolder.
    credible_level: float = 0.05
    # Prior strength in pseudo-observations, centred on the model's own confidence.
    # 20 means roughly 20 ratings are needed before feedback outweighs the model.
    prior_strength: float = 20.0
    min_ratings: int = 20          # hard floor: no move at all below this
    max_step: float = 0.10         # largest single-update change in a threshold
    floor: float = 0.05
    ceiling: float = 0.95
    # Exploration: fraction of sub-threshold findings surfaced for review anyway.
    base_exploration: float = 0.05
    max_exploration: float = 0.25
    exploration_floor: float = 0.20   # never solicit review below this confidence


@dataclass
class LabelDecision:
    """What happened to one label's threshold, and why."""

    label: str
    old_threshold: float
    new_threshold: float
    n_rated: int
    n_uncertain: int
    n_missed: int
    observed_precision: float | None
    precision_lcb: float | None
    reason: str
    # True when no threshold on the grid reaches the target: a signal to retrain this
    # label, not to keep tightening it.
    target_unreachable: bool = False

    @property
    def moved(self) -> bool:
        return abs(self.new_threshold - self.old_threshold) > 1e-9


@dataclass
class ThresholdSet:
    """Per-label decision thresholds, with provenance."""

    thresholds: dict[str, float] = field(default_factory=dict)
    default: float = CFG.review_threshold
    updated_at: str = ""
    n_ratings: int = 0
    decisions: list[LabelDecision] = field(default_factory=list)

    def get(self, label: str) -> float:
        return self.thresholds.get(label, self.default)

    def as_dict(self) -> dict:
        return {
            "default": self.default,
            "thresholds": self.thresholds,
            "updated_at": self.updated_at,
            "n_ratings": self.n_ratings,
            "decisions": [
                {"label": d.label, "old": d.old_threshold, "new": d.new_threshold,
                 "n_rated": d.n_rated, "n_uncertain": d.n_uncertain, "n_missed": d.n_missed,
                 "observed_precision": d.observed_precision,
                 "precision_lcb": d.precision_lcb, "reason": d.reason}
                for d in self.decisions if d.moved
            ],
            "needs_model_work": sorted(d.label for d in self.decisions
                                       if d.target_unreachable),
        }

    def save(self, path: Path = THRESHOLD_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: Path = THRESHOLD_PATH) -> ThresholdSet:
        if not path.exists():
            return cls()
        blob = json.loads(path.read_text())
        return cls(thresholds={k: float(v) for k, v in blob.get("thresholds", {}).items()},
                   default=float(blob.get("default", CFG.review_threshold)),
                   updated_at=blob.get("updated_at", ""),
                   n_ratings=int(blob.get("n_ratings", 0)))


def precision_lcb(correct: int, incorrect: int, prior_mean: float,
                  cfg: PolicyConfig) -> float:
    """Lower credible bound on precision, Beta posterior with a model-anchored prior.

    ``prior_mean`` is what the model itself predicts precision to be in this region — the
    mean calibrated confidence of the findings involved. Feedback has to out-weigh that
    prior to move anything, which is the intended behaviour: the model's calibrated opinion
    is not nothing, and overturning it should take evidence.
    """
    prior_mean = float(np.clip(prior_mean, 1e-3, 1 - 1e-3))
    a = cfg.prior_strength * prior_mean + correct
    b = cfg.prior_strength * (1.0 - prior_mean) + incorrect
    return float(beta_dist.ppf(cfg.credible_level, a, b))


def _estimated_f1(pairs, t: float) -> tuple[float, float]:
    """Importance-weighted F1 at threshold ``t``, and the weighted evidence behind it.

    Each rating carries ``1 / P(shown)``. Supra-threshold findings were shown with
    certainty (weight 1); exploratory probes were shown with probability epsilon and so
    stand in for ``1/epsilon`` similar findings that were never reviewed. Without that
    correction a 5%-sampled region contributes a twentieth of its true influence, and the
    estimate stays anchored wherever the threshold already is.

    ``FN`` counts only *observed* positives that fall below ``t`` — true positives sitting
    below the exploration floor are invisible, so recall is over-estimated and the chosen
    threshold is biased slightly high. That residual bias is why the exploration floor is a
    documented parameter rather than an implementation detail.
    """
    tp = sum(w for c, v, w in pairs if c >= t and v == "correct")
    fp = sum(w for c, v, w in pairs if c >= t and v == "incorrect")
    fn = sum(w for c, v, w in pairs if c < t and v == "correct")
    denom = 2 * tp + fp + fn
    return ((2 * tp / denom) if denom > 0 else 0.0, tp + fp)


def _decide_label(counts: LabelCounts, current: float, cfg: PolicyConfig) -> LabelDecision:
    """Choose a threshold for one label under the configured objective."""
    pairs = [(c, v, w) for c, v, w in counts.confidences if v in ("correct", "incorrect")]
    n_rated = len(pairs)
    base = LabelDecision(counts.label, current, current, n_rated, counts.uncertain,
                         counts.missed, counts.observed_precision, None, "")

    if n_rated < cfg.min_ratings:
        base.reason = f"held: {n_rated} informative ratings < min_ratings={cfg.min_ratings}"
        return base

    if cfg.objective == "f1":
        scored = [(t, *_estimated_f1(pairs, float(t))) for t in GRID]
        viable = [(t, f1) for t, f1, evidence in scored if evidence >= 1.0]
        if not viable:
            base.reason = "held: no threshold has weighted evidence behind it"
            return base
        best_t, best_f1 = max(viable, key=lambda x: x[1])
        current_f1, _ = _estimated_f1(pairs, current)
        if best_f1 <= current_f1 + 1e-6:
            base.reason = (f"held: estimated F1 {current_f1:.3f} already best available "
                           f"(candidate {best_t:.2f} -> {best_f1:.3f})")
            return base
        step = float(np.clip(float(best_t) - current, -cfg.max_step, cfg.max_step))
        base.new_threshold = round(float(np.clip(current + step, cfg.floor, cfg.ceiling)), 3)
        base.reason = (f"estimated F1 {current_f1:.3f} -> {best_f1:.3f} at {best_t:.2f}"
                       + ("" if abs(float(best_t) - current) <= cfg.max_step
                          else f"; move capped at {cfg.max_step}"))
        return base

    best_t = None
    best_lcb = None
    for t in GRID:
        above = [(c, v, w) for c, v, w in pairs if c >= t]
        if len(above) < cfg.min_ratings:
            continue                      # not enough evidence about this region
        correct = sum(1 for _, v, _ in above if v == "correct")
        incorrect = len(above) - correct
        prior_mean = float(np.mean([c for c, _, _ in above]))
        lcb = precision_lcb(correct, incorrect, prior_mean, cfg)
        if lcb >= cfg.target_precision:
            best_t, best_lcb = float(t), lcb
            break                          # GRID ascends: the first hit is the lowest

    if best_t is None:
        # Nothing on the grid clears the bar, and the correct response is to do nothing.
        #
        # The tempting fallback — "precision is too low, so raise the threshold a notch" —
        # is wrong, and measurably so. On this model only 21 of 71 labels can reach
        # precision 0.70 at *any* threshold. For the other 50 that rule fires on every
        # update and ratchets the threshold upward forever, buying no precision and
        # destroying recall; it cost 3.4 points of macro-F1 before it was removed.
        #
        # A threshold trades recall for precision along the model's existing ROC curve. It
        # cannot manufacture precision the curve does not contain. When the target is
        # unreachable the finding is that *the model* needs work on this label — more data,
        # better features, a different loss — and the honest output is to say so and leave
        # the threshold alone.
        base.precision_lcb = precision_lcb(
            sum(1 for _, v, _ in pairs if v == "correct"),
            sum(1 for _, v, _ in pairs if v == "incorrect"),
            float(np.mean([c for c, _, _ in pairs])), cfg)
        base.target_unreachable = True
        base.reason = (f"held: no threshold reaches precision {cfg.target_precision} "
                       f"(best LCB {base.precision_lcb:.2f}) — this label needs model work, "
                       "not threshold tuning")
        return base

    step = float(np.clip(best_t - current, -cfg.max_step, cfg.max_step))
    base.new_threshold = round(float(np.clip(current + step, cfg.floor, cfg.ceiling)), 3)
    base.precision_lcb = best_lcb
    base.reason = (f"lowest threshold with precision LCB {best_lcb:.2f} >= "
                   f"{cfg.target_precision} is {best_t:.2f}"
                   + ("" if abs(best_t - current) <= cfg.max_step
                      else f"; move capped at {cfg.max_step}"))
    return base


def update_thresholds(store: FeedbackStore, current: ThresholdSet | None = None,
                      cfg: PolicyConfig | None = None,
                      labels: list[str] | None = None) -> ThresholdSet:
    """Recompute per-label thresholds from everything in the store.

    Labels with no feedback keep whatever they had — the absence of ratings is not evidence
    of anything, and a loop that drifts thresholds for unrated labels is just adding noise.
    """
    cfg = cfg or PolicyConfig()
    current = current or ThresholdSet()
    counts = store.counts_by_label()
    from datetime import UTC, datetime

    out = ThresholdSet(thresholds=dict(current.thresholds), default=current.default,
                       updated_at=datetime.now(UTC).isoformat())
    total = 0
    for label in (labels or sorted(counts)):
        c = counts.get(label)
        if c is None:
            continue
        total += c.rated
        decision = _decide_label(c, current.get(label), cfg)
        out.decisions.append(decision)
        if decision.moved:
            out.thresholds[label] = decision.new_threshold
    out.n_ratings = total
    return out


def exploration_rates(store: FeedbackStore, cfg: PolicyConfig | None = None
                      ) -> dict[str, float]:
    """Per-label probability of surfacing a sub-threshold finding for review.

    Exploration is the price of being able to *lower* a threshold ever again, and it is not
    free — every exploratory finding is a clinician's attention spent on something the model
    did not believe. So it is spent where there is reason to think recall is being lost:
    labels reviewers have reported as missed get a higher rate, the rest sit at the base.
    """
    cfg = cfg or PolicyConfig()
    rates: dict[str, float] = {}
    for label, counts in store.counts_by_label().items():
        rate = cfg.base_exploration
        if counts.missed:
            # Each reported miss buys more sampling, saturating at max_exploration.
            rate = min(cfg.max_exploration,
                       cfg.base_exploration * (1.0 + counts.missed))
        rates[label] = round(rate, 4)
    return rates
