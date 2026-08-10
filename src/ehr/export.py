"""Phase 23 — the single call that turns an APEX report into something an EHR can ingest.

:func:`to_ehr_export` bundles the three deliverables — the pasteable sentence, the ICD-10-CM
suggestions, and the FHIR R4 bundle — into one object, because in practice an integration
wants all three at once and wants them consistent with each other.

The ICD-10 suggestions are deliberately shaped as *suggestions*: each carries its tier, and
for ECG-suggestive findings the more specific code a clinician might reach for is included
alongside the evidence that would justify it. Nothing here is presented as a coding
decision. That is a human's call, and the structure makes it awkward to pretend otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.ehr.codes import ICDMapping, suggestions
from src.ehr.fhir import to_fhir_bundle, validate_bundle
from src.ehr.impression import one_line_impression
from src.serving.schema import APEXReport


@dataclass
class EHRExport:
    """Everything an EHR integration needs from one recording."""

    impression: str
    icd10: list[ICDMapping] = field(default_factory=list)
    fhir_bundle: dict = field(default_factory=dict)
    review_required: bool = False
    validation_errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.validation_errors

    def icd10_codes(self) -> list[str]:
        """Just the codes, for a claim line."""
        return [m.icd10 for m in self.icd10 if m.icd10]

    def as_dict(self) -> dict:
        return {
            "impression": self.impression,
            "icd10": [
                {"scp": m.scp, "code": m.icd10, "display": m.display, "tier": m.tier,
                 "candidate": m.candidate, "candidate_display": m.candidate_display,
                 "requires": m.requires, "note": m.note}
                for m in self.icd10
            ],
            "fhir_bundle": self.fhir_bundle,
            "review_required": self.review_required,
            "fhir_valid": self.valid,
            "validation_errors": self.validation_errors,
        }


def to_ehr_export(report: APEXReport, patient_reference: str | None = None,
                  record_identifier: str | None = None, intervals=None,
                  validate: bool = True) -> EHRExport:
    """Condense an :class:`APEXReport` into its EHR-ready forms.

    ``intervals`` is an optional Phase-22 ``IntervalSet``; supplying it adds the rate to the
    impression and LOINC-coded measurement observations to the bundle.
    """
    heart_rate = getattr(intervals, "heart_rate", None) if intervals is not None else None
    bundle = to_fhir_bundle(report, patient_reference=patient_reference,
                            record_identifier=record_identifier, intervals=intervals)
    return EHRExport(
        impression=one_line_impression(report, heart_rate=heart_rate),
        icd10=suggestions([f.label for f in report.findings]),
        fhir_bundle=bundle,
        review_required=report.review_recommended,
        validation_errors=validate_bundle(bundle) if validate else [],
    )
