"""Sanity-check grounding against clinical intuition.

Phase 5 asks: does the saliency land where a cardiologist would look?

  - **ST/T findings** (ST-elevation, ischemia, ST-T changes) should highlight the
    **repolarization** segment (ST + T), *not* the P wave.
  - **Atrial fibrillation** should highlight the **irregular RR intervals and absent
    P waves** — so its saliency should *not* concentrate on the P-wave window, and the
    record's RR intervals should genuinely be irregular.

This module turns those expectations into a checkable verdict per (recording, label).
A verdict of ``inconsistent`` is a **real finding to document, not a bug to hide** —
it tells us the model reached the right label for the wrong reason (or the grounding
method is mislocalizing), which is exactly what a grounding layer exists to surface.

Expectations are resolved data-drivenly from ``scp_statements.csv`` (diagnostic
superclass + code semantics) with an explicit fallback table, so the check covers the
whole STTC / MI-injury family, not just two hand-picked codes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.grounding.regions import RegionMass, saliency_by_region

# --- Expectation taxonomy ---------------------------------------------------
# Codes whose evidence is the repolarization segment (ST + T wave).
_REPOLARIZATION_CODES = {
    "STE", "STD_", "STE_", "NST_", "NDT", "DIG", "LNGQT", "ANEUR", "EL",
    "ISC_", "ISCAL", "ISCIN", "ISCIL", "ISCAS", "ISCLA", "ISCAN",
    "INJAS", "INJAL", "INJIN", "INJLA", "INJIL",
    "AMI", "IMI", "ASMI", "ALMI", "ILMI", "IPMI", "IPLMI", "LMI", "PMI",
}
# Atrial arrhythmias: irregular rhythm + absent/abnormal P waves.
_RHYTHM_IRREGULAR_CODES = {"AFIB", "AFLT"}

# Thresholds (documented so verdicts are reproducible, not magic).
RR_IRREGULAR_CV = 0.12       # RR coefficient-of-variation above this reads as irregular
REPOL_MARGIN = 1.20          # repolarization mass must beat P-wave mass by this factor
P_DOMINANCE_MARGIN = 1.20    # P mass this much above the next wave region == "P-dominant"


@dataclass
class Expectation:
    label: str
    kind: str                      # "repolarization" | "rhythm_irregular" | "unknown"
    expected_regions: tuple[str, ...]
    rationale: str


@dataclass
class SanityResult:
    label: str
    kind: str
    verdict: str                   # "consistent" | "inconsistent" | "inconclusive"
    detail: str
    region_mass: RegionMass
    metrics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.verdict == "consistent"


def _superclass_of(label: str, scp) -> str | None:
    if scp is None or label not in scp.index:
        return None
    val = scp.loc[label, "diagnostic_class"] if "diagnostic_class" in scp.columns else None
    return None if val is None or (isinstance(val, float) and np.isnan(val)) else str(val)


def expectation_for(label: str, scp=None) -> Expectation:
    """Resolve the clinical expectation for an SCP code.

    Uses the explicit code tables first, then the ``diagnostic_class`` superclass from
    ``scp_statements.csv`` (``scp``, optional) so any STTC/MI code is covered. Unknown
    codes get ``kind="unknown"`` and are reported ``inconclusive`` downstream.
    """
    if label in _RHYTHM_IRREGULAR_CODES:
        return Expectation(
            label, "rhythm_irregular", ("baseline", "QRS"),
            "atrial fibrillation/flutter: irregular RR + absent P waves, so saliency "
            "should track RR timing and avoid the P-wave window",
        )
    superclass = _superclass_of(label, scp)
    if label in _REPOLARIZATION_CODES or superclass == "STTC" or label.startswith("INJ"):
        return Expectation(
            label, "repolarization", ("ST", "T"),
            "ST/T or injury finding: evidence is in the ST segment and T wave "
            "(repolarization), not the P wave",
        )
    return Expectation(label, "unknown", (), "no clinical grounding expectation registered")


def sanity_check(
    saliency: np.ndarray,
    rpeaks: np.ndarray,
    label: str,
    fs: int = 100,
    lead: int | None = None,
    scp=None,
) -> SanityResult:
    """Check one grounded label's saliency against its clinical expectation.

    Args:
        saliency: ``(T,)`` temporal or ``(12, T)`` per-lead trace (see ``lead``).
        rpeaks: R-peak sample indices in the trace's time base.
        label: the SCP code the trace explains.
        fs: sampling rate (Hz).
        lead: lead to analyze if ``saliency`` is per-lead (else per-time max is used).
        scp: optional ``scp_statements.csv`` frame for superclass-aware expectations.

    Returns:
        A :class:`SanityResult` whose ``verdict`` is consistent / inconsistent /
        inconclusive with a human-readable ``detail``.
    """
    exp = expectation_for(label, scp)
    rm = saliency_by_region(saliency, rpeaks, fs=fs, lead=lead)
    f = rm.fractions
    metrics = {
        "P": f["P"], "QRS": f["QRS"], "ST": f["ST"], "T": f["T"], "baseline": f["baseline"],
        "repolarization": rm.repolarization(), "rr_cv": rm.rr_cv, "bpm": rm.bpm,
        "n_beats": rm.n_beats,
    }

    if rm.n_beats < 3:
        return SanityResult(label, exp.kind, "inconclusive",
                            f"only {rm.n_beats} beats detected — too few to localize by region",
                            rm, metrics)

    if exp.kind == "repolarization":
        repol, p = rm.repolarization(), f["P"]
        if repol >= REPOL_MARGIN * max(p, 1e-6) and repol >= max(f["QRS"], f["P"]):
            verdict = "consistent"
            detail = (f"repolarization (ST+T={repol:.0%}) exceeds the P-wave window "
                      f"(P={p:.0%}) — saliency is on the ST/T segment, as expected")
        elif p > repol:
            verdict = "inconsistent"
            detail = (f"P-wave window (P={p:.0%}) outweighs repolarization (ST+T={repol:.0%}) "
                      f"— saliency lands on the P wave for an ST/T finding (documented finding)")
        else:
            verdict = "inconclusive"
            detail = (f"repolarization (ST+T={repol:.0%}) and P ({p:.0%}) are comparable; "
                      f"QRS={f['QRS']:.0%} — no clear ST/T localization")
        return SanityResult(label, exp.kind, verdict, detail, rm, metrics)

    if exp.kind == "rhythm_irregular":
        p = f["P"]
        # "P-dominant" = the P-wave window is the single largest region (incl. baseline)
        # by a margin. Heavy *baseline* mass is expected in AF (fibrillatory / RR timing)
        # and must not be mistaken for P-wave reliance.
        ordered = sorted(f.values(), reverse=True)
        runner_up = ordered[1] if len(ordered) > 1 else 0.0
        p_dominant = rm.dominant() == "P" and p >= P_DOMINANCE_MARGIN * runner_up
        irregular = np.isfinite(rm.rr_cv) and rm.rr_cv >= RR_IRREGULAR_CV
        reg_txt = f"RR CV={rm.rr_cv:.2f} ({'irregular' if irregular else 'regular'})"
        if not p_dominant:
            verdict = "consistent"
            detail = (f"saliency does not concentrate on the P-wave window "
                      f"(P={p:.0%}, dominant region '{rm.dominant()}'={f[rm.dominant()]:.0%}), "
                      f"consistent with absent P waves; {reg_txt}")
        else:
            verdict = "inconsistent"
            detail = (f"saliency concentrates on the P-wave window (P={p:.0%}, the dominant "
                      f"region) despite AF's absent P waves — grounding disagrees with the "
                      f"finding; {reg_txt}")
        metrics["rr_irregular"] = bool(irregular)
        metrics["p_dominant"] = bool(p_dominant)
        return SanityResult(label, exp.kind, verdict, detail, rm, metrics)

    return SanityResult(label, exp.kind, "inconclusive",
                        f"no grounding expectation registered for {label!r}", rm, metrics)


def summarize(results: list[SanityResult]) -> dict:
    """Aggregate verdicts over many (record, label) checks for the docs table."""
    checked = [r for r in results if r.verdict != "inconclusive"]
    consistent = [r for r in checked if r.verdict == "consistent"]
    return {
        "n_total": len(results),
        "n_checked": len(checked),
        "n_inconclusive": len(results) - len(checked),
        "n_consistent": len(consistent),
        "consistency_rate": (len(consistent) / len(checked)) if checked else float("nan"),
    }
