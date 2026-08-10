"""Phase 23 — EHR integration layer.

Turns an :class:`~src.serving.schema.APEXReport` into the three things a hospital system
needs: a sentence a clinician can paste into a note, ICD-10-CM codes a billing system can
read, and an HL7 FHIR R4 bundle an interface engine can ingest.

    from src.ehr import to_ehr_export

    export = to_ehr_export(report, record_identifier="ptbxl-00123")
    export.impression        # one pasteable sentence, with mandatory attribution
    export.icd10_codes()     # ['I48.91']
    export.fhir_bundle       # a validated FHIR R4 Bundle
    export.valid             # False if the bundle failed validation

The pieces:

- :mod:`~src.ehr.codes`       SCP -> ICD-10-CM / LOINC, split by what an ECG can establish
- :mod:`~src.ehr.impression`  the one-sentence condenser
- :mod:`~src.ehr.fhir`        the R4 bundle emitter, plus a validator with teeth
- :mod:`~src.ehr.export`      the single call that produces all three
"""

from src.ehr.codes import (
    ABNORMAL_ECG,
    ICD10_MAP,
    LOINC_MEASUREMENTS,
    TIER_DEFINITIONAL,
    TIER_NOT_CODABLE,
    TIER_SUGGESTIVE,
    ICDMapping,
    icd10_for,
    suggestions,
)
from src.ehr.export import EHRExport, to_ehr_export
from src.ehr.fhir import check_bindings, to_fhir_bundle, validate_bundle
from src.ehr.impression import ATTRIBUTION, impression_components, one_line_impression

__all__ = [
    "ABNORMAL_ECG", "ATTRIBUTION", "EHRExport", "ICD10_MAP", "ICDMapping",
    "LOINC_MEASUREMENTS", "TIER_DEFINITIONAL", "TIER_NOT_CODABLE", "TIER_SUGGESTIVE",
    "check_bindings", "icd10_for", "impression_components", "one_line_impression",
    "suggestions", "to_ehr_export", "to_fhir_bundle", "validate_bundle",
]
