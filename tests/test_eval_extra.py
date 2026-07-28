"""Tests for Phase-12 eval helpers: superclass aggregation + BLEU/ROUGE (no model/data)."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from src.eval import text_metrics as tm  # noqa: E402
from src.eval.superclass import SUPERCLASSES, to_superclass  # noqa: E402

_LABEL_SPACE = ["NORM", "IMI", "AMI", "NDT", "AFIB"]


def _scp():
    """Minimal scp_statements.csv frame: codes with a diagnostic_class (AFIB has none)."""
    return pd.DataFrame(
        {"diagnostic": [1.0, 1.0, 1.0, 1.0, np.nan],
         "diagnostic_class": ["NORM", "MI", "MI", "STTC", np.nan]},
        index=_LABEL_SPACE,
    )


# --- superclass aggregation --------------------------------------------------
def test_to_superclass_any_member_rule():
    # record 0: IMI present -> MI super; record 1: NORM present -> NORM super
    y = np.array([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=float)
    sup = to_superclass(y, _LABEL_SPACE, _scp(), reduce="max")
    assert sup.shape == (2, len(SUPERCLASSES))
    mi, norm = SUPERCLASSES.index("MI"), SUPERCLASSES.index("NORM")
    assert sup[0, mi] == 1.0 and sup[0, norm] == 0.0
    assert sup[1, norm] == 1.0 and sup[1, mi] == 0.0


def test_to_superclass_prob_takes_max_over_members():
    p = np.array([[0.1, 0.3, 0.8, 0.2, 0.9]], dtype=float)  # MI members IMI=0.3, AMI=0.8
    sup = to_superclass(p, _LABEL_SPACE, _scp(), reduce="max")
    assert sup[0, SUPERCLASSES.index("MI")] == pytest.approx(0.8)  # max(0.3, 0.8)


def test_rhythm_code_not_mapped_to_any_superclass():
    # AFIB (no diagnostic_class) contributes to no superclass column
    y = np.array([[0, 0, 0, 0, 1]], dtype=float)  # only AFIB present
    sup = to_superclass(y, _LABEL_SPACE, _scp(), reduce="max")
    assert sup.sum() == 0.0


# --- BLEU / ROUGE ------------------------------------------------------------
def test_bleu_identical_is_high():
    s = "atrial fibrillation with rapid ventricular response"
    assert tm.bleu(s, s) > 0.9


def test_bleu_disjoint_far_below_identical():
    ref = "atrial fibrillation with rapid ventricular response and no acute changes"
    disjoint = tm.bleu("completely unrelated vocabulary describing something else entirely here", ref)
    assert disjoint < 0.15                      # low-n add-1 floor drops with length
    assert tm.bleu(ref, ref) > disjoint + 0.6   # identical scores far higher


def test_bleu_empty_hypothesis():
    assert tm.bleu("", "something") == 0.0


def test_rouge_n_and_l():
    hyp, ref = "sinus rhythm normal ecg", "normal sinus rhythm"
    assert 0.0 < tm.rouge_n(hyp, ref, 1) <= 1.0
    assert tm.rouge_l(hyp, ref) > 0.0
    assert tm.rouge_l("", "x") == 0.0


def test_rouge1_perfect_overlap():
    assert tm.rouge_n("a b c", "c b a", 1) == pytest.approx(1.0)  # same unigrams


def test_score_and_corpus_keys():
    s = tm.score("atrial fibrillation", "atrial fibrillation")
    assert set(s) == {"bleu4", "rouge1", "rouge2", "rougeL"}
    assert s["rouge1"] == pytest.approx(1.0)
    corpus = tm.corpus_score([("a b", "a b"), ("c d", "c e")])
    assert set(corpus) == {"bleu4", "rouge1", "rouge2", "rougeL"}


def test_corpus_score_empty():
    assert tm.corpus_score([])["bleu4"] == 0.0
