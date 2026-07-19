"""Grounding layer: per-label, per-lead saliency over the raw ECG (Phase 5).

Public API:

    from src.grounding import load_detector, ground, sanity_check

    model, label_space, _ = load_detector("outputs/final_best.pt")
    saliency = ground(model, clean_signal, "AFIB", label_space=label_space)["AFIB"]
    result = sanity_check(saliency.per_lead, rpeaks, "AFIB")

``ground`` returns, per label, a :class:`LeadSaliency` with a ``(12, T)`` per-lead
trace aligned to the raw signal. ``sanity_check`` grades that trace against clinical
intuition (ST findings -> ST/T segment, AF -> irregular RR + absent P waves).
"""

from src.grounding.loader import default_cam_target, load_detector
from src.grounding.regions import WAVE_WINDOWS, RegionMass, saliency_by_region
from src.grounding.saliency import (
    LEAD_NAMES,
    LeadSaliency,
    grad_cam_1d,
    ground,
    is_grounded,
    lead_saliency,
)
from src.grounding.sanity import SanityResult, expectation_for, sanity_check, summarize

__all__ = [
    "load_detector",
    "default_cam_target",
    "ground",
    "lead_saliency",
    "grad_cam_1d",
    "is_grounded",
    "LeadSaliency",
    "LEAD_NAMES",
    "saliency_by_region",
    "RegionMass",
    "WAVE_WINDOWS",
    "sanity_check",
    "expectation_for",
    "summarize",
    "SanityResult",
]
