"""Phase 24 — which findings to surface, given per-label thresholds and exploration.

Two jobs the default path does not have:

**Per-label thresholds.** A single global 0.5 treats a label the model is excellent at and
one it is hopeless at identically. Feedback produces per-label thresholds
(:mod:`src.feedback.policy`) and this is where they are applied.

**Exploration.** Occasionally surfacing a finding the model scored *below* its threshold,
flagged as such, so a reviewer can say whether it was right. Without this the loop has no
data below threshold and can never lower one — see the ratchet discussion in
:mod:`src.feedback.policy`. It is a genuine cost (a clinician looks at something the model
did not believe) so the rate is small, per-label, and targeted at labels where misses have
actually been reported.

Exploration is **off unless explicitly asked for**. A clinical read should not be sprinkled
with low-confidence guesses because the vendor wants training data; it is a mode a
deployment opts into, with the reviewer told which findings are exploratory.
"""

from __future__ import annotations

import numpy as np

from src.config import CFG


def select_findings(probs, label_space: list[str],
                    thresholds=None,
                    exploration: dict[str, float] | None = None,
                    exploration_floor: float = 0.20,
                    rng: np.random.Generator | None = None
                    ) -> tuple[list[str], set[str]]:
    """Return ``(surfaced_labels, exploratory_labels)``.

    ``thresholds`` is a :class:`~src.feedback.policy.ThresholdSet` (or ``None`` for the
    global default). ``exploration`` maps label -> probability of surfacing that label when
    it falls below its threshold but above ``exploration_floor``; ``None`` disables it.

    ``exploratory_labels`` is always a subset of ``surfaced_labels`` — they are surfaced, but
    the caller must mark them so the reviewer knows the model did not clear its own bar.
    """
    probs = np.asarray(probs, dtype=float)
    rng = rng or np.random.default_rng()
    surfaced: list[str] = []
    exploratory: set[str] = set()

    for j, label in enumerate(label_space):
        t = thresholds.get(label) if thresholds is not None else CFG.review_threshold
        p = float(probs[j])
        if p >= t:
            surfaced.append(label)
            continue
        if not exploration:
            continue
        rate = exploration.get(label, 0.0)
        if rate > 0.0 and p >= exploration_floor and rng.random() < rate:
            surfaced.append(label)
            exploratory.add(label)
    return surfaced, exploratory
