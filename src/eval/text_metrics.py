"""Compact BLEU / ROUGE for explanation-quality scoring — no NLTK / rouge-score dep.

Used to score a free-text ECG interpretation (e.g. from a generalist multimodal LLM)
against APEX's templated clinical reference. Standard sentence-level BLEU-4 (with
brevity penalty + add-1 smoothing so a single short report doesn't collapse to 0) and
ROUGE-1/2/L (F1). These are lexical-overlap metrics: a *low* score against the clinical
template means the wording diverges, not necessarily that the reading is wrong — read
alongside the superclass hit-rate, not on their own.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """Sentence BLEU-``max_n`` with brevity penalty and add-1 smoothing."""
    hyp, ref = tokenize(hypothesis), tokenize(reference)
    if not hyp:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        hyp_ng, ref_ng = _ngrams(hyp, n), _ngrams(ref, n)
        overlap = sum((hyp_ng & ref_ng).values())
        total = max(1, len(hyp) - n + 1)
        precisions.append((overlap + 1) / (total + 1))  # add-1 smoothing
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref) / max(1, len(hyp)))
    return bp * geo_mean


def _f1(overlap: int, hyp_total: int, ref_total: int) -> float:
    if overlap == 0 or hyp_total == 0 or ref_total == 0:
        return 0.0
    prec, rec = overlap / hyp_total, overlap / ref_total
    return 2 * prec * rec / (prec + rec)


def rouge_n(hypothesis: str, reference: str, n: int = 1) -> float:
    """ROUGE-``n`` F1 (n-gram overlap)."""
    hyp, ref = tokenize(hypothesis), tokenize(reference)
    hyp_ng, ref_ng = _ngrams(hyp, n), _ngrams(ref, n)
    overlap = sum((hyp_ng & ref_ng).values())
    return _f1(overlap, sum(hyp_ng.values()), sum(ref_ng.values()))


def _lcs_length(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            prev, dp[j] = dp[j], (prev + 1 if x == y else max(dp[j], dp[j - 1]))
    return dp[-1]


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 (longest common subsequence)."""
    hyp, ref = tokenize(hypothesis), tokenize(reference)
    if not hyp or not ref:
        return 0.0
    return _f1(_lcs_length(hyp, ref), len(hyp), len(ref))


def score(hypothesis: str, reference: str) -> dict[str, float]:
    """All metrics at once: ``bleu4``, ``rouge1``, ``rouge2``, ``rougeL``."""
    return {
        "bleu4": round(bleu(hypothesis, reference, 4), 4),
        "rouge1": round(rouge_n(hypothesis, reference, 1), 4),
        "rouge2": round(rouge_n(hypothesis, reference, 2), 4),
        "rougeL": round(rouge_l(hypothesis, reference), 4),
    }


def corpus_score(pairs: list[tuple[str, str]]) -> dict[str, float]:
    """Mean of :func:`score` over ``(hypothesis, reference)`` pairs."""
    if not pairs:
        return {"bleu4": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = [score(h, r) for h, r in pairs]
    return {k: round(sum(s[k] for s in scores) / len(scores), 4) for k in scores[0]}
