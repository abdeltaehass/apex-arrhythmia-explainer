"""Smoke tests for the eval logic that needs no heavy deps."""

from src.eval.consistency import check, consistency_rate
from src.eval.hallucination import evaluate, hallucination_rate


def test_consistency_pass():
    r = check(asserted_findings={"AFIB"}, surfaced_labels={"AFIB", "LBBB"})
    assert r.consistent
    assert r.unsupported == set()


def test_consistency_detects_fabrication():
    r = check(asserted_findings={"AFIB", "STEMI"}, surfaced_labels={"AFIB"})
    assert not r.consistent
    assert r.unsupported == {"STEMI"}


def test_consistency_rate():
    good = check({"AFIB"}, {"AFIB"})
    bad = check({"STEMI"}, {"AFIB"})
    assert consistency_rate([good, bad]) == 0.5


def test_hallucination_flag_on_fabrication():
    r = check({"AFIB", "STEMI"}, {"AFIB"})
    flag = evaluate("rec1", r, grounded_findings={"AFIB"})
    assert flag.flagged
    assert flag.unsupported_findings == {"STEMI"}


def test_hallucination_flag_on_ungrounded():
    r = check({"AFIB"}, {"AFIB"})  # consistent...
    flag = evaluate("rec2", r, grounded_findings=set())  # ...but not grounded
    assert flag.flagged
    assert flag.ungrounded_findings == {"AFIB"}
    assert flag.unsupported_findings == set()


def test_hallucination_rate():
    r1 = check({"STEMI"}, {"AFIB"})   # fabricated
    r2 = check({"AFIB"}, {"AFIB"})    # clean
    f1 = evaluate("a", r1, {"AFIB"})
    f2 = evaluate("b", r2, {"AFIB"})
    assert hallucination_rate([f1, f2]) == 0.5
