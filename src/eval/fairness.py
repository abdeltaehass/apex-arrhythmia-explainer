"""Phase 14 — demographic subgroup performance analysis.

Answers "does AUROC differ by patient age or sex?" with enough statistical care that the
answer is trustworthy in either direction. Two methodological points drive the design:

**Compare on a common label set.** Macro-AUROC skips labels with only one class present
(AUROC is undefined there). Different subgroups have different label coverage, so a naive
per-subgroup macro-AUROC averages over *different* label sets and the comparison is
apples-to-oranges. :func:`common_evaluable_labels` restricts every subgroup in a
comparison to the labels evaluable in *all* of them, so the macro numbers are comparable.

**Report uncertainty, not just point estimates.** Subgroups differ in size by an order of
magnitude (PTB-XL's test split has 13 under-18 records against 721 aged 60–75). A raw gap
between two point estimates says nothing without a confidence interval, so subgroup AUROCs
and their differences come with bootstrap CIs; a gap whose CI straddles zero is reported
as "no detectable difference", not as a finding.

Model-free: everything operates on already-computed ``(N, L)`` label/probability matrices
plus a per-record demographic frame, so it is testable without torch or the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# PTB-XL anonymizes patients older than 89 by recording age as 300 (see src/data/eda.py).
ANON_AGE = 300

# PTB-XL encodes sex as 0 = male, 1 = female.
SEX_LABELS = {0: "male", 1: "female"}

# Age bands used for the model-card breakdown. The <18 band exists to *measure* how thin
# pediatric coverage is, not because the model is intended for it — see the model card's
# out-of-scope section.
AGE_BANDS: list[tuple[str, float, float]] = [
    ("<18", 0, 18),
    ("18-39", 18, 40),
    ("40-59", 40, 60),
    ("60-74", 60, 75),
    ("75+", 75, np.inf),
]


def age_band(age: float) -> str | None:
    """Band name for one age, or ``None`` if unusable (missing / the 300 sentinel).

    The sentinel is excluded rather than bucketed into ``75+``: it means ">89", so
    treating it as a real age would be inventing precision the dataset deliberately
    removed.
    """
    if age is None:
        return None
    a = float(age)
    if not np.isfinite(a) or a == ANON_AGE or a <= 0:
        return None
    for name, lo, hi in AGE_BANDS:
        if lo <= a < hi:
            return name
    return None


def evaluable_labels(y_true: np.ndarray) -> set[int]:
    """Label indices with both classes present — the ones AUROC is defined for."""
    return {j for j in range(y_true.shape[1])
            if 0 < int(y_true[:, j].sum()) < y_true.shape[0]}


def common_evaluable_labels(y_true: np.ndarray, masks: dict[str, np.ndarray]) -> set[int]:
    """Label indices evaluable in *every* subgroup — the fair common ground for macro."""
    sets = [evaluable_labels(y_true[m]) for m in masks.values() if m.any()]
    if not sets:
        return set()
    return set.intersection(*sets)


def macro_auroc_on(y_true: np.ndarray, y_prob: np.ndarray, labels: set[int]) -> float:
    """Macro-AUROC over an explicit label subset (NaN if none are evaluable here)."""
    from sklearn.metrics import roc_auc_score

    vals = []
    for j in sorted(labels):
        col = y_true[:, j]
        if col.min() == col.max():
            continue
        vals.append(float(roc_auc_score(col, y_prob[:, j])))
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_auroc_ci(
    y_true: np.ndarray, y_prob: np.ndarray, labels: set[int],
    n_boot: int = 200, alpha: float = 0.05, seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a subgroup's macro-AUROC (resampling records)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 2 or not labels:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = macro_auroc_on(y_true[idx], y_prob[idx], labels)
        if not np.isnan(v):
            stats.append(v)
    if not stats:
        return (float("nan"), float("nan"))
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


def bootstrap_gap_ci(
    y_true: np.ndarray, y_prob: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray,
    labels: set[int], n_boot: int = 200, alpha: float = 0.05, seed: int = 42,
) -> tuple[float, float, float]:
    """CI for the macro-AUROC *gap* (a − b), resampling within each subgroup.

    Returns ``(gap, lo, hi)``. A CI containing 0 means the observed gap is not
    distinguishable from sampling noise at this sample size.
    """
    rng = np.random.default_rng(seed)
    ia, ib = np.flatnonzero(mask_a), np.flatnonzero(mask_b)
    gap = macro_auroc_on(y_true[ia], y_prob[ia], labels) - macro_auroc_on(y_true[ib], y_prob[ib], labels)
    if len(ia) < 2 or len(ib) < 2 or not labels:
        return (gap, float("nan"), float("nan"))
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(ia, len(ia), replace=True)
        sb = rng.choice(ib, len(ib), replace=True)
        va = macro_auroc_on(y_true[sa], y_prob[sa], labels)
        vb = macro_auroc_on(y_true[sb], y_prob[sb], labels)
        if not (np.isnan(va) or np.isnan(vb)):
            diffs.append(va - vb)
    if not diffs:
        return (gap, float("nan"), float("nan"))
    return (gap, float(np.percentile(diffs, 100 * alpha / 2)),
            float(np.percentile(diffs, 100 * (1 - alpha / 2))))


@dataclass
class SubgroupResult:
    """One subgroup's size and macro-AUROC with a bootstrap CI."""

    name: str
    n: int
    macro_auroc: float
    ci_low: float
    ci_high: float
    n_labels: int

    @property
    def reliable(self) -> bool:
        """Whether the subgroup is large enough for its number to mean much.

        A deliberately blunt floor: below ~50 records the CI is so wide that the point
        estimate should not be quoted without its interval.
        """
        return self.n >= 50


def subgroup_breakdown(
    y_true: np.ndarray, y_prob: np.ndarray, masks: dict[str, np.ndarray],
    n_boot: int = 200, seed: int = 42, labels: set[int] | None = None,
) -> tuple[list[SubgroupResult], set[int]]:
    """Macro-AUROC + CI for each subgroup, all on one shared label set.

    By default the shared set is the labels evaluable in *every* subgroup. Pass ``labels``
    to supply it instead — needed when one tiny subgroup would otherwise collapse the
    intersection (a 13-record band makes almost no label evaluable, which would drag the
    whole comparison down to a handful of labels). Deriving the set from the larger
    subgroups and scoring the small one on it keeps every number comparable *and*
    meaningful; the small subgroup is still flagged via ``SubgroupResult.reliable``.

    Returns ``(results, label_set)``. Subgroups with no records are skipped.
    """
    common = common_evaluable_labels(y_true, masks) if labels is None else labels
    results = []
    for name, m in masks.items():
        if not m.any():
            continue
        yt, yp = y_true[m], y_prob[m]
        auc = macro_auroc_on(yt, yp, common)
        lo, hi = bootstrap_auroc_ci(yt, yp, common, n_boot=n_boot, seed=seed)
        results.append(SubgroupResult(name=name, n=int(m.sum()), macro_auroc=auc,
                                      ci_low=lo, ci_high=hi, n_labels=len(common)))
    return results, common


def max_gap(results: list[SubgroupResult], reliable_only: bool = True) -> float:
    """Largest macro-AUROC spread across subgroups (optionally only reliable ones)."""
    vals = [r.macro_auroc for r in results
            if (r.reliable or not reliable_only) and not np.isnan(r.macro_auroc)]
    return float(max(vals) - min(vals)) if len(vals) >= 2 else float("nan")
