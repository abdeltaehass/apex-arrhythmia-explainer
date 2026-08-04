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


# --- per-label subgroup analysis (Phase 18) ----------------------------------
# Going label-by-label rather than macro raises two problems the macro view hides:
# statistical power (a label needs enough positives *in each subgroup*, and most of the
# 71 do not have that) and multiple comparisons (testing 71 labels at alpha=0.05 yields
# ~3.5 false positives by chance alone). Both are handled explicitly below; a per-label
# fairness table without them is a machine for generating spurious disparities.

MIN_POSITIVES_PER_SUBGROUP = 10


def label_auroc(y_col: np.ndarray, p_col: np.ndarray) -> float:
    """AUROC for one label on one subgroup (NaN if only one class is present)."""
    from sklearn.metrics import roc_auc_score

    if y_col.min() == y_col.max():
        return float("nan")
    return float(roc_auc_score(y_col, p_col))


@dataclass
class LabelGap:
    """One label's performance in two subgroups, with the gap and its uncertainty."""

    label: str
    auroc_a: float
    auroc_b: float
    n_pos_a: int
    n_pos_b: int
    gap: float                 # auroc_a - auroc_b
    ci_low: float
    ci_high: float
    p_value: float
    q_value: float = float("nan")   # Benjamini-Hochberg adjusted
    powered: bool = True            # enough positives in both subgroups

    @property
    def significant(self) -> bool:
        """Significant after FDR correction — the only claim worth making."""
        return self.powered and np.isfinite(self.q_value) and self.q_value < 0.05


def bootstrap_label_gap(
    y: np.ndarray, p: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, j: int,
    n_boot: int = 400, alpha: float = 0.05, seed: int = 42,
) -> tuple[float, float, float, float]:
    """Gap, CI and a two-sided bootstrap p-value for one label between two subgroups.

    The p-value is the standard bootstrap two-sided proportion: how often the resampled
    difference lands on the other side of zero, doubled. Returns ``(gap, lo, hi, p)``.
    """
    rng = np.random.default_rng(seed)
    ia, ib = np.flatnonzero(mask_a), np.flatnonzero(mask_b)
    ya, pa = y[ia, j], p[ia, j]
    yb, pb = y[ib, j], p[ib, j]
    gap = label_auroc(ya, pa) - label_auroc(yb, pb)
    if not np.isfinite(gap):
        return (gap, float("nan"), float("nan"), float("nan"))

    diffs = []
    for _ in range(n_boot):
        sa = rng.integers(0, len(ia), len(ia))
        sb = rng.integers(0, len(ib), len(ib))
        va = label_auroc(ya[sa], pa[sa])
        vb = label_auroc(yb[sb], pb[sb])
        if np.isfinite(va) and np.isfinite(vb):
            diffs.append(va - vb)
    if len(diffs) < 20:
        return (gap, float("nan"), float("nan"), float("nan"))
    d = np.asarray(diffs)
    lo = float(np.percentile(d, 100 * alpha / 2))
    hi = float(np.percentile(d, 100 * (1 - alpha / 2)))
    frac_le = float((d <= 0).mean())
    pval = float(min(1.0, 2 * min(frac_le, 1 - frac_le)))
    return (gap, lo, hi, pval)


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (q-values), NaNs preserved.

    Controls the false discovery rate across the label-wise tests. Without this, testing
    every label independently guarantees a handful of "disparities" that are pure noise.
    Returns q-values rather than a reject/accept mask so the caller picks its own
    threshold (``LabelGap.significant`` uses q < 0.05).
    """
    arr = np.asarray(pvals, dtype=float)
    finite = np.flatnonzero(np.isfinite(arr))
    q = np.full(arr.shape, np.nan)
    if finite.size == 0:
        return q.tolist()
    order = finite[np.argsort(arr[finite])]
    m = len(order)
    prev = 1.0
    # step-up from the largest p-value, enforcing monotonicity
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        val = min(prev, arr[idx] * m / rank)
        q[idx] = prev = val
    return q.tolist()


def per_label_gaps(
    y: np.ndarray, p: np.ndarray, label_space: list[str],
    mask_a: np.ndarray, mask_b: np.ndarray,
    min_positives: int = MIN_POSITIVES_PER_SUBGROUP,
    n_boot: int = 400, seed: int = 42,
) -> list[LabelGap]:
    """Per-label AUROC gap between two subgroups, FDR-corrected across labels.

    Labels without ``min_positives`` positives in *both* subgroups are still returned but
    marked ``powered=False`` and excluded from the FDR correction — reporting an AUROC
    computed on three positives as if it were a finding is worse than reporting nothing.
    """
    out: list[LabelGap] = []
    for j, code in enumerate(label_space):
        na, nb = int(y[mask_a, j].sum()), int(y[mask_b, j].sum())
        powered = na >= min_positives and nb >= min_positives
        if not powered:
            out.append(LabelGap(code, label_auroc(y[mask_a, j], p[mask_a, j]),
                                label_auroc(y[mask_b, j], p[mask_b, j]), na, nb,
                                float("nan"), float("nan"), float("nan"), float("nan"),
                                powered=False))
            continue
        gap, lo, hi, pv = bootstrap_label_gap(y, p, mask_a, mask_b, j, n_boot, seed=seed)
        out.append(LabelGap(code, label_auroc(y[mask_a, j], p[mask_a, j]),
                            label_auroc(y[mask_b, j], p[mask_b, j]), na, nb,
                            gap, lo, hi, pv, powered=True))

    powered_idx = [i for i, g in enumerate(out) if g.powered]
    qs = benjamini_hochberg([out[i].p_value for i in powered_idx])
    for i, q in zip(powered_idx, qs, strict=True):
        out[i].q_value = q
    return out
