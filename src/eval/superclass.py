"""Aggregate the 71 SCP-code outputs into PTB-XL's 5 diagnostic superclasses.

APEX's detector is trained on the full 71-code ("all") task. The published PTB-XL
benchmark (Strodthoff et al. 2021) also reports a *superdiagnostic* task over the five
coarse diagnostic superclasses — NORM, MI (myocardial infarction), STTC (ST/T change),
CD (conduction disturbance), HYP (hypertrophy). To compare on that task we pool the
71-code predictions down to the 5 superclasses:

    a superclass is *present* if any of its member diagnostic codes is present, and its
    *score* is the max probability over its members (the "at least one" rule).

Only the ~44 diagnostic codes map to a superclass (rhythm/form codes don't); the mapping
comes from ``scp_statements.csv``'s ``diagnostic_class`` column.
"""

from __future__ import annotations

import numpy as np

from src.data.labels import DIAGNOSTIC_SUPERCLASSES, diagnostic_superclass_map

SUPERCLASSES = DIAGNOSTIC_SUPERCLASSES  # ("NORM", "MI", "STTC", "CD", "HYP")


def superclass_member_indices(label_space: list[str], scp) -> dict[str, list[int]]:
    """Superclass -> indices into ``label_space`` of its member diagnostic codes."""
    code_to_super = diagnostic_superclass_map(scp)
    members: dict[str, list[int]] = {s: [] for s in SUPERCLASSES}
    for i, code in enumerate(label_space):
        s = code_to_super.get(code)
        if s in members:
            members[s].append(i)
    return members


def to_superclass(matrix: np.ndarray, label_space: list[str], scp, reduce: str = "max") -> np.ndarray:
    """Pool an ``(N, 71)`` matrix to ``(N, 5)`` over the superclasses.

    ``reduce="max"`` (default) implements the "at least one member" rule — correct for
    both the binary target (any member present) and the score (strongest member).
    """
    members = superclass_member_indices(label_space, scp)
    out = np.zeros((matrix.shape[0], len(SUPERCLASSES)), dtype=matrix.dtype)
    op = np.max if reduce == "max" else np.mean
    for j, s in enumerate(SUPERCLASSES):
        idx = members[s]
        if idx:
            out[:, j] = op(matrix[:, idx], axis=1)
    return out


def superclass_auroc(y_true: np.ndarray, y_prob: np.ndarray, label_space: list[str], scp) -> dict[str, float]:
    """Per-superclass AUROC (+ ``macro``) from 71-code true/prob matrices."""
    from src.eval.metrics import per_label_auroc

    yt = to_superclass(y_true, label_space, scp, reduce="max")
    yp = to_superclass(y_prob, label_space, scp, reduce="max")
    per = per_label_auroc(yt, yp)  # keyed by column index
    out = {SUPERCLASSES[j]: per[j] for j in range(len(SUPERCLASSES))}
    valid = [v for v in out.values() if not np.isnan(v)]
    out["macro"] = float(np.mean(valid)) if valid else float("nan")
    return out
