"""Phase 24 — tests for feedback collection, the update policy, and the UI panel.

Data-independent: every test builds its own temporary database. Nothing needs PTB-XL, a
checkpoint, or the network.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.feedback import (
    FeedbackStore,
    PolicyConfig,
    RatedFinding,
    ThresholdSet,
    exploration_rates,
    precision_lcb,
    select_findings,
    update_thresholds,
)
from src.feedback.simulate import ReviewerModel, macro_scores


@pytest.fixture
def store(tmp_path):
    with FeedbackStore(tmp_path / "fb.db") as s:
        yield s


def fill(store, label: str, n: int, precision: float, confidence: float = 0.8,
         seed: int = 0, **kw) -> None:
    """Log ``n`` ratings for ``label`` at a given true precision."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        verdict = "correct" if rng.random() < precision else "incorrect"
        store.log_review([RatedFinding(label, confidence, verdict, threshold=0.5, **kw)],
                         reviewer_id="r1")


# --- store -------------------------------------------------------------------
def test_round_trip(store):
    store.log_review([RatedFinding("AFIB", 0.9, "correct", threshold=0.5)],
                     reviewer_id="dr_a", record_ref="rec-1", missed=["1AVB"],
                     missed_confidences={"1AVB": 0.42})
    s = store.summary()
    assert s["ratings"] == 1 and s["missed"] == 1 and s["reviewers"] == 1
    assert store.missed_confidences("1AVB") == [0.42]


def test_unknown_verdict_is_rejected(store):
    with pytest.raises(ValueError, match="unknown verdict"):
        store.log_review([RatedFinding("AFIB", 0.9, "probably")], reviewer_id="r")


def test_uncertain_is_counted_but_kept_out_of_precision(store):
    for verdict in ("correct", "correct", "uncertain", "uncertain", "incorrect"):
        store.log_review([RatedFinding("AFIB", 0.8, verdict)], reviewer_id="r")
    counts = store.counts_by_label()["AFIB"]
    assert counts.uncertain == 2
    assert counts.rated == 3                      # 2 correct + 1 incorrect
    assert counts.observed_precision == pytest.approx(2 / 3)


def test_exploratory_rows_are_distinguishable(store):
    store.log_review([RatedFinding("AFIB", 0.9, "correct"),
                      RatedFinding("STD_", 0.3, "incorrect", exploratory=True,
                                   sampling_rate=0.05)], reviewer_id="r")
    assert store.summary()["exploratory"] == 1
    weights = {c: w for _, _, w in store.counts_by_label()["STD_"].confidences
               for c in ["w"]}
    assert weights["w"] == pytest.approx(20.0)    # 1 / 0.05


def test_agreement_is_empty_until_a_finding_is_rated_twice(store):
    store.log_review([RatedFinding("AFIB", 0.9, "correct")], reviewer_id="a")
    assert store.agreement() == {}


def test_rollback_leaves_no_partial_review(store):
    with pytest.raises(ValueError):
        store.log_review([RatedFinding("AFIB", 0.9, "correct"),
                          RatedFinding("STD_", 0.8, "nonsense")], reviewer_id="r")
    assert store.summary()["ratings"] == 0, "a rejected review must not half-commit"


# --- policy ------------------------------------------------------------------
def test_small_samples_do_not_move_a_threshold(store):
    fill(store, "RVH", 4, precision=0.0)
    decision = update_thresholds(store).decisions[0]
    assert not decision.moved and "min_ratings" in decision.reason


def test_lower_bound_tightens_as_evidence_accumulates():
    few = precision_lcb(2, 1, 0.8, PolicyConfig())
    many = precision_lcb(55, 5, 0.8, PolicyConfig())
    assert few < many
    assert few < 0.7 <= many, "3 ratings must not clear a bar that 60 ratings does"


def test_prior_is_anchored_on_the_models_own_confidence():
    """With no observations the posterior is the prior, so confidence sets the bar."""
    assert precision_lcb(0, 0, 0.95, PolicyConfig()) > precision_lcb(0, 0, 0.55, PolicyConfig())


def test_moves_are_capped(store):
    cfg = PolicyConfig(objective="precision", min_ratings=10, max_step=0.05)
    fill(store, "STD_", 60, precision=0.1, confidence=0.9)
    decision = update_thresholds(store, ThresholdSet(), cfg).decisions[0]
    assert abs(decision.new_threshold - 0.5) <= 0.05 + 1e-9


def test_unreachable_precision_target_holds_and_flags_for_model_work(store):
    """A threshold cannot manufacture precision the ROC curve does not contain."""
    cfg = PolicyConfig(objective="precision", min_ratings=10, target_precision=0.9)
    fill(store, "NDT", 80, precision=0.15, confidence=0.7)
    decision = update_thresholds(store, ThresholdSet(), cfg).decisions[0]
    assert decision.target_unreachable
    assert not decision.moved, "blind ratcheting cost 3.4 points of macro-F1; do not restore it"
    assert "model work" in decision.reason


def test_labels_without_feedback_are_left_alone(store):
    fill(store, "AFIB", 40, precision=0.95)
    out = update_thresholds(store, ThresholdSet(thresholds={"RVH": 0.5}))
    assert out.get("RVH") == 0.5
    assert "STD_" not in out.thresholds


def test_f1_objective_lowers_a_threshold_when_recall_is_being_lost(store):
    """The behaviour the whole phase turns on: the loop must be able to go *down*."""
    cfg = PolicyConfig(objective="f1", min_ratings=10, max_step=0.2)
    rng = np.random.default_rng(0)
    # A label that is accurate well below the current 0.5 threshold, sampled by exploration.
    for _ in range(60):
        conf = float(rng.uniform(0.25, 0.5))
        store.log_review([RatedFinding("AFIB", conf, "correct", threshold=0.5,
                                       exploratory=True, sampling_rate=0.1)],
                         reviewer_id="r")
    for _ in range(40):
        conf = float(rng.uniform(0.5, 1.0))
        store.log_review([RatedFinding("AFIB", conf, "correct", threshold=0.5)],
                         reviewer_id="r")
    decision = update_thresholds(store, ThresholdSet(), cfg).decisions[0]
    assert decision.new_threshold < 0.5, decision.reason


def test_threshold_set_persists(tmp_path):
    ts = ThresholdSet(thresholds={"AFIB": 0.42}, default=0.5, n_ratings=17)
    path = ts.save(tmp_path / "t.json")
    back = ThresholdSet.load(path)
    assert back.get("AFIB") == 0.42 and back.get("UNSEEN") == 0.5


def test_missing_threshold_file_yields_defaults(tmp_path):
    assert ThresholdSet.load(tmp_path / "nope.json").get("AFIB") == 0.5


def test_exploration_targets_labels_with_reported_misses(store):
    store.log_review([RatedFinding("AFIB", 0.9, "correct")], reviewer_id="r",
                     missed=["1AVB", "1AVB", "1AVB"])
    rates = exploration_rates(store, PolicyConfig())
    assert rates["1AVB"] > rates["AFIB"]
    assert rates["1AVB"] <= PolicyConfig().max_exploration


# --- selection ---------------------------------------------------------------
def test_default_selection_matches_the_global_threshold():
    surfaced, exploratory = select_findings([0.9, 0.45], ["A", "B"])
    assert surfaced == ["A"] and exploratory == set()


def test_per_label_thresholds_are_applied():
    ts = ThresholdSet(thresholds={"B": 0.4}, default=0.5)
    surfaced, _ = select_findings([0.9, 0.45], ["A", "B"], thresholds=ts)
    assert surfaced == ["A", "B"]


def test_exploration_surfaces_below_threshold_and_marks_it():
    surfaced, exploratory = select_findings(
        [0.9, 0.30], ["A", "B"], exploration={"B": 1.0},
        rng=np.random.default_rng(0))
    assert surfaced == ["A", "B"] and exploratory == {"B"}


def test_exploration_respects_its_floor():
    """Very low-confidence findings are not worth a clinician's attention."""
    surfaced, _ = select_findings([0.9, 0.05], ["A", "B"], exploration={"B": 1.0},
                                  exploration_floor=0.2, rng=np.random.default_rng(0))
    assert surfaced == ["A"]


# --- simulation --------------------------------------------------------------
def test_reviewer_error_rate_flips_verdicts():
    rng = np.random.default_rng(0)
    always_wrong = ReviewerModel(error_rate=1.0)
    assert always_wrong.verdict(True, 0.9, 0.5, rng) == "incorrect"


def test_uncertainty_concentrates_at_the_decision_boundary():
    rng = np.random.default_rng(0)
    model = ReviewerModel(boundary_uncertainty=0.9)
    near = [model.verdict(True, 0.51, 0.5, rng) for _ in range(200)]
    far = [model.verdict(True, 0.99, 0.5, rng) for _ in range(200)]
    assert near.count("uncertain") > far.count("uncertain")


def test_macro_precision_ignores_labels_the_model_never_predicts():
    """Scoring a silent label's precision as 0 would punish the ratchet's symptom."""
    probs = np.array([[0.9, 0.1], [0.8, 0.05]])
    y = np.array([[1, 1], [1, 1]])
    p, r, f1 = macro_scores(probs, y, ["A", "B"], ThresholdSet(default=0.5))
    assert p == pytest.approx(1.0)      # A is perfect; B is never predicted, not scored
    assert r == pytest.approx(0.5)      # B's recall of 0 is counted
