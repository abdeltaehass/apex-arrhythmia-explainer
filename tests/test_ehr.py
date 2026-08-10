"""Phase 23 — tests for the EHR integration layer.

Data-independent: reports are constructed directly, so nothing here needs PTB-XL, a
checkpoint, or the network. The FHIR tests validate against the real R4B
StructureDefinitions when ``fhir.resources`` is installed and fall back to the
binding/reference checks (which are pure Python) when it is not.
"""

from __future__ import annotations

import copy
import re

import pytest

from src.ehr import (
    ATTRIBUTION,
    ICD10_MAP,
    TIER_DEFINITIONAL,
    TIER_NOT_CODABLE,
    TIER_SUGGESTIVE,
    icd10_for,
    one_line_impression,
    suggestions,
    to_ehr_export,
    to_fhir_bundle,
    validate_bundle,
)
from src.ehr.codes import ABNORMAL_ECG, ICD10CM_PATTERN, LOINC_MEASUREMENTS, LOINC_PATTERN
from src.ehr.impression import MAX_CHARS
from src.generation.vocab import VOCAB
from src.serving.schema import APEXReport, ConsistencyOut, FindingOut

try:
    import fhir.resources  # noqa: F401

    HAS_FHIR = True
except ImportError:                                          # pragma: no cover
    HAS_FHIR = False


def make_report(codes, review: bool = False, confidence: float = 0.9,
                needs_review: bool = False, leads=None) -> APEXReport:
    return APEXReport(
        findings=[FindingOut(label=c, description=(VOCAB[c].impression or VOCAB[c].finding)
                             if c in VOCAB else c,
                             confidence=confidence, needs_review=needs_review,
                             leads=list(leads or []))
                  for c in codes],
        consistency=ConsistencyOut(consistent=True),
        review_recommended=review,
    )


# --- terminology --------------------------------------------------------------
def test_every_scp_code_is_mapped():
    assert set(ICD10_MAP) == set(VOCAB), "the mapping must cover the whole label space"


def test_all_codes_are_well_formed():
    for scp, m in ICD10_MAP.items():
        if m.icd10 is not None:
            assert re.match(ICD10CM_PATTERN, m.icd10), f"{scp}: {m.icd10}"
        if m.candidate is not None:
            assert re.match(ICD10CM_PATTERN, m.candidate), f"{scp}: {m.candidate}"
    for lc in LOINC_MEASUREMENTS.values():
        assert re.match(LOINC_PATTERN, lc.code)


def test_suggestive_findings_all_code_to_abnormal_ecg():
    """An ECG that only *suggests* a diagnosis may not assert one."""
    for scp, m in ICD10_MAP.items():
        if m.tier == TIER_SUGGESTIVE:
            assert m.icd10 == ABNORMAL_ECG[0], f"{scp} escaped R94.31 with {m.icd10}"


def test_infarction_never_auto_codes_a_disease():
    """Q waves are not an MI. I21/I25 may appear only as a candidate with stated evidence."""
    for scp in ("ASMI", "AMI", "IMI", "ALMI", "QWAVE", "STE_", "INJAS"):
        m = icd10_for(scp)
        assert m.tier == TIER_SUGGESTIVE
        assert m.icd10 == "R94.31"
        if m.candidate:
            assert m.candidate.startswith(("I21", "I25"))
            assert m.requires, f"{scp}: a specific code must say what would justify it"


def test_atrial_fibrillation_is_unspecified_not_paroxysmal():
    """A 10-second recording cannot establish chronicity, so I48.91 is the only honest code.

    I48.0 (paroxysmal), I48.11/I48.19 (persistent) and I48.20/I48.21 (chronic/permanent) all
    encode duration or treatment history that is nowhere in the signal. Suggesting one would
    be upcoding.
    """
    m = icd10_for("AFIB")
    assert m.icd10 == "I48.91"
    assert m.tier == TIER_DEFINITIONAL
    assert m.icd10 not in {"I48.0", "I48.11", "I48.19", "I48.20", "I48.21"}


def test_hypertrophy_is_not_coded_as_cardiomegaly():
    m = icd10_for("LVH")
    assert m.icd10 == "R94.31" and m.candidate == "I51.7"
    assert "echocardiograph" in (m.requires or "").lower()


def test_normal_and_physiological_variants_are_not_codable():
    for scp in ("NORM", "SR", "SARRH"):
        m = icd10_for(scp)
        assert m.tier == TIER_NOT_CODABLE and m.icd10 is None


def test_pacemaker_is_a_status_code():
    assert icd10_for("PACE").icd10 == "Z95.0"


def test_suggestions_deduplicate_shared_codes():
    """Five ischemic territories are one R94.31, not five."""
    out = suggestions(["ISCAL", "ISCIN", "ISCAS", "ISCIL", "ISCLA"])
    assert [m.icd10 for m in out] == ["R94.31"]


def test_suggestions_put_definitional_codes_first():
    out = suggestions(["ASMI", "AFIB", "1AVB"])
    assert out[0].tier == TIER_DEFINITIONAL and out[-1].icd10 == "R94.31"


def test_suggestions_skip_non_codable():
    assert suggestions(["NORM", "SR"]) == []


# --- impression ---------------------------------------------------------------
def test_impression_is_one_sentence_with_attribution():
    s = one_line_impression(make_report(["AFIB"]), heart_rate=110)
    assert s.endswith(".") and s.count(".") == 1
    assert ATTRIBUTION in s


def test_impression_always_carries_attribution_even_when_empty():
    assert ATTRIBUTION in one_line_impression(make_report([]))


def test_impression_leads_with_rhythm_and_rate():
    s = one_line_impression(make_report(["SR", "STD_"]), heart_rate=72)
    assert s.startswith("Sinus rhythm at 72 bpm")


def test_impression_omits_rate_when_unknown():
    s = one_line_impression(make_report(["SR", "STD_"]))
    assert "bpm" not in s


def test_urgent_finding_leads_and_is_not_repeated():
    s = one_line_impression(make_report(["SR", "INJAS", "STD_"]), heart_rate=90)
    assert s.startswith("URGENT")
    assert s.count("subendocardial injury") == 1


def test_territories_collapse_into_one_clause():
    s = one_line_impression(make_report(["SR", "ISCAL", "ISCIN", "ISCAS"]), heart_rate=70)
    assert "multi-territory ischemia" in s
    assert "anterolateral ischemia" not in s


def test_single_territory_is_not_collapsed():
    s = one_line_impression(make_report(["SR", "ISCAL"]), heart_rate=70)
    assert "anterolateral ischemia" in s and "multi-territory" not in s


def test_review_clause_appears_only_when_review_recommended():
    assert "flagged for review" in one_line_impression(make_report(["AFIB"], review=True))
    assert "flagged for review" not in one_line_impression(make_report(["AFIB"]))


def test_impression_respects_the_length_cap():
    many = ["SR", "AFIB", "1AVB", "LAFB", "CRBBB", "LVH", "STD_", "INVT", "PVC", "PAC",
            "ASMI", "IMI", "LNGQT", "NDT"]
    s = one_line_impression(make_report(many), heart_rate=88)
    assert len(s) <= MAX_CHARS
    assert "further finding" in s
    assert ATTRIBUTION in s          # never truncated away


def test_normal_study_reads_as_normal():
    s = one_line_impression(make_report(["NORM", "SR"]), heart_rate=65)
    assert "normal ecg" in s.lower()


# --- FHIR ---------------------------------------------------------------------
@pytest.fixture
def bundle():
    report = make_report(["AFIB", "STD_"], review=True, leads=["V4", "V5"])
    return to_fhir_bundle(report, record_identifier="ptbxl-00001")


def _resources(bundle: dict, rtype: str) -> list[dict]:
    return [e["resource"] for e in bundle["entry"]
            if e["resource"]["resourceType"] == rtype]


def test_bundle_validates(bundle):
    assert validate_bundle(bundle) == []


def test_bundle_is_a_collection_with_expected_resources(bundle):
    assert bundle["resourceType"] == "Bundle" and bundle["type"] == "collection"
    assert len(_resources(bundle, "DiagnosticReport")) == 1
    assert len(_resources(bundle, "Device")) == 1
    assert len(_resources(bundle, "Observation")) == 2


def test_status_is_preliminary_when_review_recommended():
    flagged = to_fhir_bundle(make_report(["AFIB"], review=True))
    clean = to_fhir_bundle(make_report(["AFIB"], review=False))
    assert _resources(flagged, "DiagnosticReport")[0]["status"] == "preliminary"
    assert _resources(clean, "DiagnosticReport")[0]["status"] == "final"


def test_no_patient_identifiers_are_invented(bundle):
    patient = _resources(bundle, "Patient")[0]
    assert set(patient) <= {"resourceType", "identifier"}
    for banned in ("name", "birthDate", "address", "telecom", "gender"):
        assert banned not in patient


def test_patient_resource_is_omitted_when_caller_supplies_a_reference():
    b = to_fhir_bundle(make_report(["AFIB"]), patient_reference="Patient/real-mrn")
    assert _resources(b, "Patient") == []
    assert _resources(b, "DiagnosticReport")[0]["subject"]["reference"] == "Patient/real-mrn"
    assert validate_bundle(b) == []


def test_findings_carry_both_scp_and_icd10_codings(bundle):
    obs = [o for o in _resources(bundle, "Observation")
           if o.get("valueCodeableConcept")]
    systems = {c["system"] for o in obs for c in o["valueCodeableConcept"]["coding"]}
    assert "http://loinc.org" not in systems           # LOINC is on `code`, not `value`
    assert any("icd-10-cm" in s for s in systems)
    assert any("scp-ecg" in s for s in systems)


def test_conclusion_codes_are_icd10(bundle):
    dr = _resources(bundle, "DiagnosticReport")[0]
    codes = {c["coding"][0]["code"] for c in dr["conclusionCode"]}
    assert "I48.91" in codes


def test_confidence_travels_as_an_extension(bundle):
    obs = _resources(bundle, "Observation")[0]
    urls = {e["url"] for e in obs["extension"]}
    assert any(u.endswith("apex-model-confidence") for u in urls)
    values = [e["valueDecimal"] for e in obs["extension"] if "confidence" in e["url"]]
    assert 0.0 <= values[0] <= 1.0


def test_every_observation_names_the_generating_device(bundle):
    device_urls = {e["fullUrl"] for e in bundle["entry"]
                   if e["resource"]["resourceType"] == "Device"}
    for obs in _resources(bundle, "Observation"):
        assert obs["device"]["reference"] in device_urls


def test_intervals_become_loinc_quantity_observations():
    class Intervals:
        heart_rate, pr, qrs, qt, qtc_fridericia = 72.0, 168.0, 92.0, 380.0, 396.0

    b = to_fhir_bundle(make_report(["SR"]), intervals=Intervals())
    quantities = {o["code"]["coding"][0]["code"]: o["valueQuantity"]
                  for o in _resources(b, "Observation") if "valueQuantity" in o}
    assert quantities["8867-4"]["code"] == "/min"          # heart rate, UCUM
    assert quantities["8625-6"]["value"] == 168.0          # PR interval, ms
    assert all(q["system"] == "http://unitsofmeasure.org" for q in quantities.values())
    assert validate_bundle(b) == []


def test_missing_intervals_emit_no_observation():
    class Sparse:
        heart_rate, pr, qrs, qt, qtc_fridericia = 72.0, None, None, None, None

    b = to_fhir_bundle(make_report(["AFIB"]), intervals=Sparse())
    codes = {o["code"]["coding"][0]["code"] for o in _resources(b, "Observation")
             if "valueQuantity" in o}
    assert codes == {"8867-4"}, "an unmeasurable PR must be absent, not zero"


# --- the validator itself has to have teeth -----------------------------------
@pytest.fixture
def clean(bundle):
    return copy.deepcopy(bundle)


def _first(bundle: dict, rtype: str) -> dict:
    return _resources(bundle, rtype)[0]


def test_validator_rejects_device_as_report_performer(clean):
    device_url = next(e["fullUrl"] for e in clean["entry"]
                      if e["resource"]["resourceType"] == "Device")
    _first(clean, "DiagnosticReport")["performer"] = [{"reference": device_url}]
    assert any("performer" in e for e in validate_bundle(clean))


def test_validator_rejects_out_of_valueset_status(clean):
    _first(clean, "DiagnosticReport")["status"] = "definitely-final"
    assert any("status" in e for e in validate_bundle(clean))


def test_validator_rejects_bad_bundle_type(clean):
    clean["type"] = "not-a-bundle-type"
    assert any("Bundle.type" in e for e in validate_bundle(clean))


def test_validator_rejects_dangling_references(clean):
    _first(clean, "Observation")["device"] = {
        "reference": "urn:uuid:00000000-0000-0000-0000-000000000000"}
    assert any("does not resolve" in e for e in validate_bundle(clean))


def test_validator_rejects_wrong_result_target(clean):
    device_url = next(e["fullUrl"] for e in clean["entry"]
                      if e["resource"]["resourceType"] == "Device")
    _first(clean, "DiagnosticReport")["result"] = [{"reference": device_url}]
    assert any("result" in e for e in validate_bundle(clean))


@pytest.mark.skipif(not HAS_FHIR, reason="fhir.resources not installed")
def test_schema_validator_rejects_missing_required_element(clean):
    del _first(clean, "Observation")["code"]
    assert validate_bundle(clean)


# --- end to end ---------------------------------------------------------------
def test_export_produces_all_three_deliverables():
    export = to_ehr_export(make_report(["AFIB", "ASMI"], review=True),
                           record_identifier="ptbxl-00042")
    assert ATTRIBUTION in export.impression
    assert export.icd10_codes() == ["I48.91", "R94.31"]
    assert export.fhir_bundle["resourceType"] == "Bundle"
    assert export.valid and export.validation_errors == []
    assert export.review_required


def test_export_is_json_serializable():
    import json

    export = to_ehr_export(make_report(["AFIB"]), record_identifier="x")
    assert json.loads(json.dumps(export.as_dict()))["icd10"][0]["code"] == "I48.91"


def test_normal_study_yields_no_billing_codes():
    export = to_ehr_export(make_report(["NORM", "SR"]))
    assert export.icd10_codes() == []
    assert export.valid
