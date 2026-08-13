"""Phase 25 — constructing a rare-class experiment that can actually be measured.

PTB-XL's genuinely rare labels cannot support the experiment this phase is nominally about.
The 17 labels with fewer than 50 training examples have **1 to 5 positives in the test
fold** — ``PRC(S)`` has one, ``2AVB`` has one, ``3AVB`` has two. AUROC computed against two
positives takes a handful of discrete values and has a standard error wider than any effect
augmentation could plausibly produce. Reporting "AUROC improved from 0.71 to 0.83 on 3AVB"
from two positive cases would be reporting noise with a decimal point on it.

So rarity is *induced* instead, on labels that do have enough test support to measure:
:func:`make_rare` keeps ``n_keep`` positive annotations for a target label in the training
set and masks the rest to zero. The label becomes rare in exactly the way that matters — the
model sees few examples of it — while the test fold is untouched and keeps its 40 to 112
positives. That buys enough statistical power to detect an effect if one exists.

Masking rather than deleting records is deliberate. Deleting every excess positive would
shrink the training set and remove those recordings' *other* labels too, confounding the
comparison with a dataset-size change. Masking holds everything else fixed and mimics what
real rarity looks like in practice: an under-annotated condition, present in the data but
rarely written down. It does introduce false negatives, and that cost is real — but it is
identical across every arm, so it cannot explain a difference between them.

The genuinely rare labels are still reported in ``docs/synthesis/report.md``, flagged as
underpowered, because pretending they do not exist would be its own kind of dishonesty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Targets: rare enough that a few dozen examples is a plausible regime, common enough in
# the test fold that AUROC means something. Test positives in parentheses.
DEFAULT_TARGETS = (
    "CLBBB",    # 54
    "CRBBB",    # 54
    "ILMI",     # 48
    "LAO/LAE",  # 42
    "PAC",      # 40
    "LOWT",     # 44
    "ISCAL",    # 66
    "QWAVE",    # 55
)


@dataclass
class RarityScenario:
    """One induced-rarity training set."""

    y_masked: np.ndarray                     # (N, L) labels after masking
    kept: dict[str, np.ndarray] = field(default_factory=dict)   # label -> kept row indices
    masked: dict[str, int] = field(default_factory=dict)        # label -> n annotations removed
    n_keep: int = 0
    targets: tuple[str, ...] = ()

    def positive_rows(self, label: str) -> np.ndarray:
        """Row indices still annotated positive for ``label`` — the few real examples."""
        return self.kept[label]


def make_rare(y: np.ndarray, label_space: list[str], targets=DEFAULT_TARGETS,
              n_keep: int = 50, seed: int = 0) -> RarityScenario:
    """Mask all but ``n_keep`` positive annotations for each target label.

    Returns a scenario carrying the masked label matrix and, per target, which rows kept
    their annotation — the handful of real examples every arm is allowed to learn from and
    the *only* data a generator may be trained on.
    """
    rng = np.random.default_rng(seed)
    y_masked = np.asarray(y, dtype=np.float32).copy()
    kept: dict[str, np.ndarray] = {}
    masked: dict[str, int] = {}

    for label in targets:
        if label not in label_space:
            raise KeyError(f"{label!r} is not in the label space")
        j = label_space.index(label)
        positives = np.flatnonzero(y_masked[:, j] > 0)
        if len(positives) <= n_keep:
            kept[label] = positives
            masked[label] = 0
            continue
        keep_idx = rng.choice(positives, size=n_keep, replace=False)
        drop = np.setdiff1d(positives, keep_idx, assume_unique=False)
        y_masked[drop, j] = 0.0
        kept[label] = np.sort(keep_idx)
        masked[label] = int(len(drop))

    return RarityScenario(y_masked=y_masked, kept=kept, masked=masked, n_keep=n_keep,
                          targets=tuple(targets))


def oversample_indices(scenario: RarityScenario, n_train: int, factor: int = 8,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Row indices for an epoch in which rare positives are repeated ``factor`` times.

    The cheapest possible intervention, and the one a synthetic-data method has to beat to
    be worth its complexity: it costs nothing, cannot go wrong, and adds no information —
    which makes it exactly the right yardstick for measuring how much information synthetic
    samples really add.
    """
    rng = rng or np.random.default_rng()
    extra = [np.repeat(rows, factor - 1) for rows in scenario.kept.values() if len(rows)]
    idx = np.concatenate([np.arange(n_train), *extra]) if extra else np.arange(n_train)
    rng.shuffle(idx)
    return idx
