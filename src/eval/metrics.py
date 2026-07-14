"""Detection metrics for APEX: per-label AUROC, macro/micro F1, calibration.

All functions take numpy arrays:
    y_true : (N, L) binary label matrix
    y_prob : (N, L) predicted probabilities in [0, 1]
where L == number of SCP-ECG labels (71).
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import f1_score, roc_auc_score
except ImportError:  # keep the module importable before deps are installed
    roc_auc_score = f1_score = None  # type: ignore


def per_label_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> dict[int, float]:
    """AUROC for each label. Labels with a single class present are skipped (NaN)."""
    out: dict[int, float] = {}
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if col.min() == col.max():  # only one class -> AUROC undefined
            out[j] = float("nan")
            continue
        out[j] = float(roc_auc_score(col, y_prob[:, j]))
    return out


def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    vals = [v for v in per_label_auroc(y_true, y_prob).values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def f1_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Macro and micro F1 given binarized predictions."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Per-label thresholds that maximize F1 on the given (validation) split."""
    thresholds = np.full(y_true.shape[1], 0.5)
    grid = np.linspace(0.05, 0.95, 19)
    for j in range(y_true.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = f1_score(y_true[:, j], (y_prob[:, j] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[j] = best_t
    return thresholds


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """ECE over all label predictions pooled together (15-bin default)."""
    probs = y_prob.ravel()
    correct = (y_true.ravel() == (probs >= 0.5)).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        conf = probs[mask].mean()
        acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def summary(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray | None = None) -> dict:
    """One call to produce the headline metrics for a W&B log."""
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5)
    y_pred = (y_prob >= thresholds).astype(int)
    out = {"macro_auroc": macro_auroc(y_true, y_prob), "ece": expected_calibration_error(y_true, y_prob)}
    out.update(f1_scores(y_true, y_pred))
    return out
