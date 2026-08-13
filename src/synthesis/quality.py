"""Phase 25 — is the synthetic data any good, and is it actually *new*?

Downstream AUROC is the outcome that matters, but it cannot distinguish a generator that
learned the class from one that memorized its training examples: both raise the effective
sample count and only one of them adds information. So generated data is checked directly,
three ways.

**Memorization.** For every synthetic recording, the distance to its nearest *training*
neighbour is compared with the same distance computed for genuinely held-out real
recordings. Held-out real data sets the scale of "as close as an independent sample of this
class gets". Synthetic samples that sit systematically closer than that are copies, and
augmenting with them is duplication wearing a hat. The statistic is the ratio of medians:
around 1.0 is healthy, well below 1.0 is memorization.

**Physiology.** Reuses Phase 22's interval measurement, which is the sharpest fidelity test
available here and costs nothing extra: a synthetic recording that is really an ECG has a
measurable PR interval, a QRS duration in the right range, and a plausible heart rate.
Spectral or distributional similarity scores can be satisfied by signals no heart could
produce; a delineator that refuses to find a QRS complex cannot.

**Diversity.** Mean pairwise distance among synthetic samples against the same quantity
among real ones. Mode collapse shows up here as a diversity ratio far below 1 even when
memorization looks fine — the generator producing one plausible ECG over and over.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _flat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(len(x), -1)


def _nn_distance(query: np.ndarray, reference: np.ndarray, batch: int = 64) -> np.ndarray:
    """Euclidean distance from each row of ``query`` to its nearest row in ``reference``."""
    q, r = _flat(query), _flat(reference)
    if len(r) == 0 or len(q) == 0:
        return np.array([])
    out = np.empty(len(q), dtype=np.float32)
    r_sq = (r ** 2).sum(axis=1)
    for start in range(0, len(q), batch):
        chunk = q[start:start + batch]
        d2 = (chunk ** 2).sum(axis=1)[:, None] - 2.0 * chunk @ r.T + r_sq[None, :]
        out[start:start + batch] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


@dataclass
class QualityReport:
    n_synthetic: int = 0
    memorization_ratio: float = float("nan")   # median NN(synth->train) / NN(heldout->train)
    nn_synth: float = float("nan")
    nn_heldout: float = float("nan")
    diversity_ratio: float = float("nan")
    measurable_rate: float = float("nan")      # fraction with a delineable QRS
    p_detected_rate: float = float("nan")
    intervals: dict = field(default_factory=dict)   # measure -> {synthetic, real}
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


def memorization(synthetic: np.ndarray, train: np.ndarray,
                 heldout: np.ndarray) -> tuple[float, float, float]:
    """``(ratio, median NN synth->train, median NN heldout->train)``.

    ``heldout`` must be real recordings of the same class that the generator never saw —
    they calibrate how close an *honest* new example of this class lands.
    """
    d_syn = _nn_distance(synthetic, train)
    d_real = _nn_distance(heldout, train)
    if not len(d_syn) or not len(d_real):
        return float("nan"), float("nan"), float("nan")
    m_syn, m_real = float(np.median(d_syn)), float(np.median(d_real))
    return (m_syn / m_real if m_real > 0 else float("nan")), m_syn, m_real


def diversity(synthetic: np.ndarray, real: np.ndarray, max_n: int = 128,
              seed: int = 0) -> float:
    """Mean pairwise distance among synthetic samples, over the same among real ones."""
    rng = np.random.default_rng(seed)

    def mean_pairwise(x: np.ndarray) -> float:
        x = _flat(x)
        if len(x) < 2:
            return float("nan")
        if len(x) > max_n:
            x = x[rng.choice(len(x), max_n, replace=False)]
        d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
        iu = np.triu_indices(len(x), k=1)
        return float(np.sqrt(np.maximum(d2[iu], 0.0)).mean())

    a, b = mean_pairwise(synthetic), mean_pairwise(real)
    return a / b if b and np.isfinite(b) and b > 0 else float("nan")


def physiology(synthetic: np.ndarray, real: np.ndarray, fs: int = 100,
               max_n: int = 64) -> dict:
    """Compare measured intervals between synthetic and real recordings.

    Uses :func:`src.longitudinal.intervals.measure`. Note the signals here are z-scored
    model-space tensors, not millivolts, so *absolute* ST amplitudes are not comparable —
    the durations (PR, QRS, QT) and rate are, and those are what this reports.
    """
    from src.longitudinal.intervals import measure

    def stats(batch: np.ndarray) -> dict:
        out: dict[str, list[float]] = {"heart_rate": [], "pr": [], "qrs": [], "qt": []}
        measurable = p_found = 0
        for x in batch[:max_n]:
            m = measure(np.asarray(x, dtype=float), fs)
            measurable += int(m.measurable)
            p_found += int(m.p_detected)
            for key in out:
                v = getattr(m, key)
                if v is not None:
                    out[key].append(float(v))
        n = max(1, min(len(batch), max_n))
        return {"measurable_rate": measurable / n, "p_detected_rate": p_found / n,
                **{k: (float(np.median(v)) if v else float("nan")) for k, v in out.items()},
                **{f"{k}_n": len(v) for k, v in out.items()}}

    return {"synthetic": stats(synthetic), "real": stats(real)}


def assess(synthetic: np.ndarray, train: np.ndarray, heldout: np.ndarray,
           fs: int = 100) -> QualityReport:
    """Run all three checks and package the result."""
    ratio, m_syn, m_real = memorization(synthetic, train, heldout)
    report = QualityReport(
        n_synthetic=len(synthetic), memorization_ratio=ratio, nn_synth=m_syn,
        nn_heldout=m_real, diversity_ratio=diversity(synthetic, train),
    )
    phys = physiology(synthetic, train, fs)
    report.intervals = phys
    report.measurable_rate = phys["synthetic"]["measurable_rate"]
    report.p_detected_rate = phys["synthetic"]["p_detected_rate"]

    if np.isfinite(ratio) and ratio < 0.6:
        report.notes.append(f"memorization ratio {ratio:.2f} — synthetic samples sit much "
                            "closer to training data than held-out real data does")
    if np.isfinite(report.diversity_ratio) and report.diversity_ratio < 0.5:
        report.notes.append(f"diversity ratio {report.diversity_ratio:.2f} — possible mode "
                            "collapse")
    if report.measurable_rate < 0.5:
        report.notes.append(f"only {report.measurable_rate:.0%} of synthetic recordings have "
                            "a delineable QRS complex")
    return report
