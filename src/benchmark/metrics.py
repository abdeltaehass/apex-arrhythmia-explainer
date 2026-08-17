"""Phase 28 — scoring the comparison, on every axis that matters.

Accuracy is one column. A specialist model that loses on AUROC can still be the right
choice on latency, cost, privacy, or rare-class behaviour, and a comparison that reports
only accuracy hides exactly the trade-off the decision turns on.

Four things are measured here.

**Discrimination** — per-superclass AUROC. Records where a system declined to answer are
excluded from its AUROC and counted separately as *coverage*, because scoring an abstention
as 0.5 (or as 0) silently rewards hedging and confuses "wrong" with "did not answer".

**Self-consistency** — does the system's free text agree with its own scores? APEX's
template is consistent by construction: it renders from the detections and cannot say
anything else. A language model can, and does — in the first smoke run of this benchmark the
model answered "MI: 9/10" directly above the sentence "There is no evidence of myocardial
ischemia". That is not a hallucination against ground truth; it is the model contradicting
itself in the same reply, and it is invisible to any accuracy metric.

**Latency** — p50/p95 wall clock around the work each system does, network included for
API-backed systems, because that is a real property of the architecture.

**Cost** — token-based for hosted models, amortized-compute for local ones. Prices are
inputs, not facts: they are dated, sourced, and easy to override, since a benchmark that
hardcodes a vendor's price list is wrong the moment the list changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

from src.benchmark.features import SUPERCLASSES

# --- pricing assumptions ------------------------------------------------------
# INPUTS, not measurements. Verify against current vendor pricing before quoting.
PRICING_AS_OF = "2026-08 (list prices; verify before citing)"
API_PRICING_PER_MTOK = {
    # model -> (input $/1M tokens, output $/1M tokens)
    "gpt-4o": (2.50, 10.00),
    "claude-fable-5": (3.00, 15.00),
}
# Local cost is amortized hardware+power. Expressed as an hourly rate so cost per
# inference falls straight out of measured latency; the default is a commodity cloud
# CPU/GPU-lite instance and is meant to be overridden.
LOCAL_HOURLY_USD = 0.10


@dataclass
class SystemScores:
    name: str
    kind: str = "specialist"
    hosting: str = "local"
    auroc: dict[str, float] = field(default_factory=dict)
    macro_auroc: float = float("nan")
    coverage: float = float("nan")          # fraction of records answered
    n_records: int = 0
    n_errors: int = 0
    latency_p50: float = float("nan")
    latency_p95: float = float("nan")
    self_contradiction: float = float("nan")
    self_contradiction_n: int = 0        # records whose text asserted anything at all
    tokens_in: int = 0
    tokens_out: int = 0
    usd_per_1k: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(vars(self))


def per_superclass_auroc(y_true: np.ndarray, scores: list[dict[str, float | None]]
                         ) -> tuple[dict[str, float], float]:
    """AUROC per superclass over the records where the system supplied a score."""
    out: dict[str, float] = {}
    for j, label in enumerate(SUPERCLASSES):
        pairs = [(y_true[i, j], s[label]) for i, s in enumerate(scores)
                 if s.get(label) is not None]
        if len(pairs) < 20:
            continue
        y = np.array([p[0] for p in pairs])
        v = np.array([p[1] for p in pairs], dtype=float)
        if 0 < y.sum() < len(y):
            out[label] = float(roc_auc_score(y, v))
    macro = float(np.mean(list(out.values()))) if out else float("nan")
    return out, macro


def coverage(scores: list[dict[str, float | None]]) -> float:
    """Fraction of (record, superclass) cells the system actually answered."""
    if not scores:
        return float("nan")
    total = len(scores) * len(SUPERCLASSES)
    answered = sum(1 for s in scores for label in SUPERCLASSES if s.get(label) is not None)
    return answered / total


def self_contradiction_rate(outputs) -> tuple[float, int]:
    """``(rate, n_considered)``: how often free text asserts a category the scores deny.

    The text is parsed with the Phase-7 machinery (SCP impression terms), each asserted code
    is mapped to its superclass, and a contradiction is counted when the text names a
    category the scores put below 0.5. Records whose text asserts nothing are skipped rather
    than counted as clean — silence is not agreement.

    **The denominator is returned, not just the rate**, because it is usually small and the
    rate is meaningless without it. Most replies from a small model are hedged prose that
    names no category at all: in a 12-record probe only one reply asserted anything, so a
    reported "100% self-contradiction" rested on a single record. A rate printed alone in a
    comparison table would read as a damning, well-supported number. It is not one.
    """
    from src.data.labels import diagnostic_superclass_map, load_scp_statements
    from src.generation.parse import asserted_findings

    code_to_super = diagnostic_superclass_map(load_scp_statements())
    considered = contradicted = 0
    for out in outputs:
        if not out.ok or not out.explanation:
            continue
        asserted = {code_to_super.get(c) for c in asserted_findings(out.explanation)}
        asserted.discard(None)
        if not asserted:
            continue
        considered += 1
        if any((out.scores.get(s) is not None and out.scores[s] < 0.5) for s in asserted):
            contradicted += 1
    return ((contradicted / considered) if considered else float("nan")), considered


def latency_percentiles(outputs) -> tuple[float, float]:
    values = np.array([o.latency_s for o in outputs if o.ok and np.isfinite(o.latency_s)])
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(values, 50)), float(np.percentile(values, 95))


def cost_per_1k(system, outputs, hourly_usd: float = LOCAL_HOURLY_USD) -> float:
    """USD per 1,000 inferences.

    Hosted models are priced on measured tokens; local models on measured wall clock at an
    assumed hourly rate. The two are not the same kind of number — one is a bill, the other
    an amortization — and the report says so rather than presenting them as interchangeable.
    """
    ok = [o for o in outputs if o.ok]
    if not ok:
        return float("nan")
    if getattr(system, "hosting", "local") == "api":
        model = getattr(system, "model", "")
        price = API_PRICING_PER_MTOK.get(model)
        if price is None:
            return float("nan")
        tin = sum(o.tokens_in for o in ok) / len(ok)
        tout = sum(o.tokens_out for o in ok) / len(ok)
        return (tin * price[0] + tout * price[1]) / 1e6 * 1000
    mean_latency = float(np.mean([o.latency_s for o in ok]))
    return mean_latency * 1000 / 3600 * hourly_usd


def summarize(system, outputs, y_true: np.ndarray,
              hourly_usd: float = LOCAL_HOURLY_USD) -> SystemScores:
    ok = [o for o in outputs if o.ok]
    scores = [o.scores for o in ok]
    keep = np.array([i for i, o in enumerate(outputs) if o.ok])
    auroc, macro = (per_superclass_auroc(y_true[keep], scores) if len(keep)
                    else ({}, float("nan")))
    p50, p95 = latency_percentiles(outputs)
    contradiction, contradiction_n = self_contradiction_rate(ok)
    desc = system.describe()
    return SystemScores(
        name=desc["name"], kind=desc.get("kind", "specialist"),
        hosting=desc.get("hosting", "local"),
        auroc=auroc, macro_auroc=macro, coverage=coverage(scores),
        n_records=len(outputs), n_errors=len(outputs) - len(ok),
        latency_p50=p50, latency_p95=p95,
        self_contradiction=contradiction, self_contradiction_n=contradiction_n,
        tokens_in=sum(o.tokens_in for o in ok), tokens_out=sum(o.tokens_out for o in ok),
        usd_per_1k=cost_per_1k(system, outputs, hourly_usd),
    )
