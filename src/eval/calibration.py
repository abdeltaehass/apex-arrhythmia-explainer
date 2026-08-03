"""Phase 17 — multi-label confidence calibration.

A network can rank cases well and still lie about *how sure* it is. Calibration asks the
question a clinician actually cares about: **when the model says 0.8, is it right 80% of
the time?** That is a different question from AUROC, and post-hoc temperature scaling fixes
it without touching the ranking at all.

Two things in here deserve care.

**1. Defining ECE correctly for multi-label.** For a *multi-class* softmax model the
standard formulation bins by the top-class confidence ``max_k p_k`` and compares it to
*accuracy*. Reusing that formula for independent per-label sigmoids is a category error:
here each of the 71 outputs is its own binary problem, and the quantity being calibrated is
the probability of the positive class, so the bin's ``mean(p)`` must be compared to the
bin's **empirical positive rate** ``mean(y)`` — not to "how often the thresholded decision
was correct".

The distinction is not academic. In this dataset 78% of all label-instances sit at
``p < 0.1``, almost all of them true negatives. Comparing *accuracy* (≈0.998, since
predicting "absent" is right) against *mean probability* (≈0.009) charges ≈0.99 of error to
predictions that are in fact well calibrated — which is precisely how APEX's earlier
"ECE ≈ 0.90" arose. See ``docs/calibration/report.md``; :func:`ece` implements the correct
definition and :func:`accuracy_style_ece` is kept only to reproduce the old number.

**2. Fitting on the right split.** Temperature is fitted on **validation** logits and
applied unchanged to test. Fitting it on test would be leakage, and a calibration metric
tuned on its own evaluation set means nothing.

Because every scaler here is a *monotonic* per-label transform of the logit, AUROC is
mathematically unchanged — calibration is free in discrimination terms. The script asserts
this rather than assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-7


# --- basic scores ------------------------------------------------------------
def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of the probabilities (proper scoring rule, lower is better)."""
    return float(np.mean((np.asarray(y_prob, float) - np.asarray(y_true, float)) ** 2))


def nll(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean binary cross-entropy in nats (proper scoring rule, lower is better)."""
    p = np.clip(np.asarray(y_prob, float), EPS, 1 - EPS)
    y = np.asarray(y_true, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# --- reliability / ECE -------------------------------------------------------
def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15,
                      strategy: str = "uniform") -> dict:
    """Bin predictions and return mean predicted vs observed frequency per bin.

    ``strategy="uniform"`` uses equal-width bins (the standard for ECE).
    ``strategy="quantile"`` uses equal-count bins, which is far more readable as a diagram
    when the probability mass is piled up near zero — as it is here.
    """
    p = np.asarray(y_prob, float).ravel()
    y = np.asarray(y_true, float).ravel()
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    conf, obs, counts, lows, highs = [], [], [], [], []
    for lo, hi, last in zip(edges[:-1], edges[1:],
                            [False] * (len(edges) - 2) + [True], strict=True):
        m = (p >= lo) & (p <= hi) if last else (p >= lo) & (p < hi)
        if not m.any():
            continue
        conf.append(float(p[m].mean()))
        obs.append(float(y[m].mean()))
        counts.append(int(m.sum()))
        lows.append(float(lo))
        highs.append(float(hi))
    return {"confidence": np.array(conf), "observed": np.array(obs),
            "count": np.array(counts), "bin_low": np.array(lows),
            "bin_high": np.array(highs), "n": int(p.size)}


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15,
        strategy: str = "uniform") -> float:
    """Expected Calibration Error: count-weighted mean ``|mean(p) − mean(y)|`` per bin.

    This is the "if it says 0.8 it should be right 80% of the time" definition, pooled over
    every (record, label) prediction.
    """
    c = reliability_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    if c["count"].size == 0:
        return float("nan")
    w = c["count"] / c["count"].sum()
    return float(np.sum(w * np.abs(c["confidence"] - c["observed"])))


def mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Maximum Calibration Error — the worst single bin (ignores how rare that bin is)."""
    c = reliability_curve(y_true, y_prob, n_bins=n_bins)
    if c["count"].size == 0:
        return float("nan")
    return float(np.max(np.abs(c["confidence"] - c["observed"])))


def classwise_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15,
                  min_positives: int = 10) -> tuple[float, dict[int, float]]:
    """Mean per-label ECE, so common labels can't drown out rare ones.

    Pooled ECE is dominated by whichever labels contribute the most predictions; a
    per-label average weights each of the 71 diagnostic questions equally. Labels with
    fewer than ``min_positives`` positives are skipped — their ECE is mostly sampling
    noise. Returns ``(mean, per_label)``.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    per: dict[int, float] = {}
    for j in range(y.shape[1]):
        if int(y[:, j].sum()) < min_positives:
            continue
        per[j] = ece(y[:, j], p[:, j], n_bins=n_bins)
    mean = float(np.mean(list(per.values()))) if per else float("nan")
    return mean, per


def accuracy_style_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """The **mis-specified** multi-class-style ECE, kept only to reproduce the old number.

    Compares each bin's mean probability against the *accuracy* of the thresholded
    decision. For independent sigmoids that charges near-1.0 error to the (correct,
    well-calibrated) low-probability negatives that make up most of the predictions. Do not
    use it to report calibration — see the module docstring.
    """
    p = np.asarray(y_prob, float).ravel()
    y = np.asarray(y_true, float).ravel()
    correct = (y == (p >= 0.5)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out, n = 0.0, p.size
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        out += (m.sum() / n) * abs(correct[m].mean() - p[m].mean())
    return float(out)


# --- scalers -----------------------------------------------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _fit_scalar_temperature(logits: np.ndarray, y: np.ndarray,
                            grid: np.ndarray | None = None) -> float:
    """Temperature minimizing BCE, by coarse grid then local refinement.

    A grid search rather than a gradient optimizer: the objective is 1-D, smooth and
    strictly unimodal in log T, so this is both simpler and immune to the
    initialization/step-size failures LBFGS can hit on near-flat regions.
    """
    grid = np.geomspace(0.25, 12.0, 60) if grid is None else grid
    best_t, best = 1.0, np.inf
    for t in grid:
        loss = nll(y, _sigmoid(logits / t))
        if loss < best:
            best, best_t = loss, float(t)
    lo, hi = best_t * 0.8, best_t * 1.25
    for t in np.linspace(lo, hi, 40):
        loss = nll(y, _sigmoid(logits / t))
        if loss < best:
            best, best_t = loss, float(t)
    return best_t


@dataclass
class TemperatureScaler:
    """Single global temperature: ``p = sigmoid(logit / T)``.

    The classic Guo et al. method and the spec's ask. One parameter for all 71 labels, so
    it cannot overfit, but it also cannot correct labels that are miscalibrated in
    *different* directions — which per-label scaling can.
    """

    temperature: float = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> TemperatureScaler:
        self.temperature = _fit_scalar_temperature(np.asarray(logits, float).ravel(),
                                                   np.asarray(y_true, float).ravel())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return _sigmoid(np.asarray(logits, float) / self.temperature)

    def to_dict(self) -> dict:
        return {"method": "temperature", "temperature": self.temperature}


@dataclass
class PerLabelTemperatureScaler:
    """One temperature per label — the natural multi-label extension.

    PTB-XL's 71 labels differ hugely in base rate, and class-weighted training distorts
    each of them differently, so a single global T is a compromise. Labels with too few
    validation positives fall back to the global temperature rather than fitting noise.
    """

    temperatures: np.ndarray | None = None
    fallback: float = 1.0
    min_positives: int = 10
    n_fallback: int = 0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> PerLabelTemperatureScaler:
        logits = np.asarray(logits, float)
        y = np.asarray(y_true, float)
        self.fallback = _fit_scalar_temperature(logits.ravel(), y.ravel())
        temps = np.full(logits.shape[1], self.fallback, dtype=float)
        self.n_fallback = 0
        for j in range(logits.shape[1]):
            pos = int(y[:, j].sum())
            if pos < self.min_positives or pos == len(y):
                self.n_fallback += 1
                continue
            temps[j] = _fit_scalar_temperature(logits[:, j], y[:, j])
        self.temperatures = temps
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if self.temperatures is None:
            raise RuntimeError("fit() first")
        return _sigmoid(np.asarray(logits, float) / self.temperatures[None, :])

    def to_dict(self) -> dict:
        return {"method": "per_label_temperature",
                "temperatures": None if self.temperatures is None
                else [round(float(t), 5) for t in self.temperatures],
                "fallback": self.fallback, "n_fallback": self.n_fallback}


@dataclass
class VectorScaler:
    """Per-label affine rescaling of the logit: ``p = sigmoid(a_j · logit + b_j)``.

    Temperature scaling can only *sharpen or soften*; it cannot shift. Class-weighted BCE
    inflates the positive class, which is largely a **bias** error, so allowing a per-label
    intercept is the transform that actually matches how this model was broken. Still
    monotonic per label (``a_j > 0``), so per-label AUROC is preserved.
    """

    a: np.ndarray | None = None
    b: np.ndarray | None = None
    min_positives: int = 10
    n_fallback: int = 0
    _global: tuple[float, float] = field(default=(1.0, 0.0))

    @staticmethod
    def _fit_ab(z: np.ndarray, y: np.ndarray, a0: float, b0: float) -> tuple[float, float]:
        """Coordinate descent over (a, b) — few parameters, very cheap, no deps."""
        a, b = a0, b0
        best = nll(y, _sigmoid(a * z + b))
        for scale in (1.0, 0.3, 0.1, 0.03):
            improved = True
            while improved:
                improved = False
                for da in (scale, -scale):
                    cand = max(a + da, 1e-3)
                    loss = nll(y, _sigmoid(cand * z + b))
                    if loss < best - 1e-9:
                        a, best, improved = cand, loss, True
                for db in (scale, -scale):
                    loss = nll(y, _sigmoid(a * z + (b + db)))
                    if loss < best - 1e-9:
                        b, best, improved = b + db, loss, True
        return a, b

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> VectorScaler:
        logits = np.asarray(logits, float)
        y = np.asarray(y_true, float)
        ga, gb = self._fit_ab(logits.ravel(), y.ravel(), 1.0, 0.0)
        self._global = (ga, gb)
        a = np.full(logits.shape[1], ga)
        b = np.full(logits.shape[1], gb)
        self.n_fallback = 0
        for j in range(logits.shape[1]):
            pos = int(y[:, j].sum())
            if pos < self.min_positives or pos == len(y):
                self.n_fallback += 1
                continue
            a[j], b[j] = self._fit_ab(logits[:, j], y[:, j], ga, gb)
        self.a, self.b = a, b
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if self.a is None:
            raise RuntimeError("fit() first")
        return _sigmoid(np.asarray(logits, float) * self.a[None, :] + self.b[None, :])

    def to_dict(self) -> dict:
        return {"method": "vector_scaling",
                "a": None if self.a is None else [round(float(v), 5) for v in self.a],
                "b": None if self.b is None else [round(float(v), 5) for v in self.b],
                "n_fallback": self.n_fallback}


def load_scaler(d: dict):
    """Rebuild a fitted scaler from :meth:`to_dict` output."""
    method = d.get("method")
    if method == "temperature":
        return TemperatureScaler(temperature=float(d["temperature"]))
    if method == "per_label_temperature":
        s = PerLabelTemperatureScaler(fallback=float(d.get("fallback", 1.0)))
        s.temperatures = np.asarray(d["temperatures"], dtype=float)
        return s
    if method == "vector_scaling":
        s = VectorScaler()
        s.a = np.asarray(d["a"], dtype=float)
        s.b = np.asarray(d["b"], dtype=float)
        return s
    raise ValueError(f"unknown calibration method: {method!r}")
