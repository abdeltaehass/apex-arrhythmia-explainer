"""Phase 23 — emitting the report as an HL7 FHIR R4 Bundle.

FHIR is what clinical systems actually exchange, so this is the layer that decides whether
APEX is integrable or just interesting. The output is a ``Bundle`` of type ``collection``
holding a ``DiagnosticReport``, one ``Observation`` per finding, one per measured interval,
a ``Device`` describing the software, and a de-identified ``Patient``.

Plain ``dict`` is emitted rather than a typed model so nothing new is imported into the
serving path. Validity is not asserted, it is *checked*: ``tests/test_ehr.py`` and
``scripts/ehr_examples.py`` run every bundle through the ``fhir.resources`` R4B pydantic
models, which are generated from the published StructureDefinitions. "Looks like FHIR" and
"is FHIR" are different claims and only the second one is worth making.

That library alone is not sufficient, which is worth stating because it is the sort of
thing a project quietly relies on. Probing it with deliberately broken bundles shows it
catches structural faults — missing required elements, wrong primitive types, malformed
``dateTime``, two ``value[x]`` on one extension — but it accepts ``"status":
"definitely-final"`` and ``"type": "not-a-bundle-type"``, because those elements are
``code`` primitives whose required ValueSet binding lives outside the StructureDefinition's
type system. It also cannot see that a reference points at the wrong kind of resource. So
:func:`check_bindings` adds what the schema check misses: the required bindings for every
coded element emitted here, plus resolution of every internal reference against the
allowed target types. :func:`validate_bundle` runs both.

Decisions in here that are easy to get wrong:

**The Device does not go on ``DiagnosticReport.performer``.** In R4 that element is
``Reference(Practitioner | PractitionerRole | Organization | CareTeam)`` — software is not
an allowed performer, and a bundle that puts it there fails validation. The spec-correct
home for the generating device is ``Observation.device``, which every finding and
measurement carries. Report-level attribution additionally rides on an extension, so a
consuming system can tell mechanically that this report was machine-generated rather than
having to parse the conclusion prose for the word "computer-assisted".

**The report is ``preliminary`` whenever review is recommended.** ``DiagnosticReport.status``
is not decoration: downstream systems gate on it, and ``final`` asserts a result fit to act
on. APEX flags low-confidence findings precisely because they are not that, so the status
follows the review gate rather than always claiming ``final``.

**Findings are values, not codes.** Each finding Observation uses LOINC 8601-7 (*EKG
impression*) as its ``code`` and puts the finding in ``valueCodeableConcept``, carrying the
SCP label and its ICD-10-CM mapping as two codings of the same concept — which is exactly
what a ``CodeableConcept`` with several ``coding`` entries means. Inventing a code system
for the ``code`` element and leaving ``value`` empty would be the easier thing to write and
would not round-trip through anything.

**No PHI is emitted, ever.** APEX never receives patient identity — PTB-XL is de-identified
— so the ``Patient`` resource carries a caller-supplied identifier and nothing else: no
name, no birth date, no address. If the caller has a real patient in their own system they
pass ``patient_reference`` and no Patient resource is generated at all, so identity stays
where it belongs.

**Model confidence is an extension, not a repurposed standard field.** There is no element
in R4 meaning "the classifier's probability". Rather than smuggling it into
``Observation.value`` or a component with an invented LOINC code, it gets a named extension
under the project's own canonical URL. That is what extensions are for, and it keeps the
number out of fields that mean something else.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from src.ehr.codes import (
    LOINC_ECG_IMPRESSION,
    LOINC_ECG_STUDY,
    LOINC_MEASUREMENTS,
    SYSTEM_ICD10CM,
    SYSTEM_LOINC,
    SYSTEM_SCP,
    SYSTEM_UCUM,
    icd10_for,
    suggestions,
)
from src.ehr.impression import one_line_impression
from src.serving.schema import SCHEMA_VERSION, APEXReport

BASE_URL = "https://github.com/abdeltaehass/apex-arrhythmia-explainer"
EXT_CONFIDENCE = f"{BASE_URL}/StructureDefinition/apex-model-confidence"
EXT_GENERATED_BY = f"{BASE_URL}/StructureDefinition/apex-generated-by"
EXT_NEEDS_REVIEW = f"{BASE_URL}/StructureDefinition/apex-needs-review"

SYSTEM_OBS_CATEGORY = "http://terminology.hl7.org/CodeSystem/observation-category"
SYSTEM_DIAG_SERVICE = "http://terminology.hl7.org/CodeSystem/v2-0074"
SYSTEM_INTERPRETATION = "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"

DEVICE_NAME = "APEX — Arrhythmia Pattern Explainer"
DEVICE_NOTE = ("Clinical decision support software. Not a regulated diagnostic device. "
               "All findings are machine-generated and require physician confirmation.")


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _urn() -> str:
    return f"urn:uuid:{uuid.uuid4()}"


def _codeable(system: str, code: str, display: str, text: str | None = None) -> dict:
    return {"coding": [{"system": system, "code": code, "display": display}],
            "text": text or display}


def _device_resource(version: str) -> dict:
    return {
        "resourceType": "Device",
        "status": "active",
        "deviceName": [{"name": DEVICE_NAME, "type": "user-friendly-name"}],
        "type": {"text": "ECG interpretation decision-support software"},
        "version": [{"value": version}],
        "note": [{"text": DEVICE_NOTE}],
    }


def _patient_resource(identifier: str | None) -> dict:
    """A deliberately empty Patient: an identifier and nothing that could identify anyone."""
    resource: dict = {"resourceType": "Patient"}
    if identifier:
        resource["identifier"] = [{"system": f"{BASE_URL}/identifier/record", "value": identifier}]
    return resource


def _finding_observation(finding, subject_ref: str, device_ref: str, effective: str) -> dict:
    mapping = icd10_for(finding.label)
    codings = [{"system": SYSTEM_SCP, "code": finding.label,
                "display": finding.description or finding.label}]
    if mapping is not None and mapping.codable:
        codings.append({"system": SYSTEM_ICD10CM, "code": mapping.icd10,
                        "display": mapping.display})

    obs: dict = {
        "resourceType": "Observation",
        "status": "final",
        "category": [{"coding": [{"system": SYSTEM_OBS_CATEGORY, "code": "procedure",
                                  "display": "Procedure"}]}],
        "code": {"coding": [{"system": SYSTEM_LOINC, "code": LOINC_ECG_IMPRESSION.code,
                             "display": LOINC_ECG_IMPRESSION.display}],
                 "text": LOINC_ECG_IMPRESSION.display},
        "subject": {"reference": subject_ref},
        "effectiveDateTime": effective,
        "valueCodeableConcept": {"coding": codings,
                                 "text": finding.description or finding.label},
        "interpretation": [{"coding": [{"system": SYSTEM_INTERPRETATION, "code": "A",
                                        "display": "Abnormal"}]}],
        "device": {"reference": device_ref},
        "extension": [
            {"url": EXT_CONFIDENCE, "valueDecimal": round(float(finding.confidence), 4)},
            {"url": EXT_NEEDS_REVIEW, "valueBoolean": bool(finding.needs_review)},
        ],
    }
    if finding.leads:
        # Which leads the finding localises to, as a human-readable note rather than a
        # coded body-site: FHIR's bodySite expects an anatomical concept, and an ECG lead
        # is a recording vector, not a body part.
        obs["note"] = [{"text": "Implicated leads: " + ", ".join(finding.leads)}]
    if finding.flags:
        obs.setdefault("note", []).append(
            {"text": "Flags: " + "; ".join(f"{f.type.value}: {f.message}" for f in finding.flags)})
    return obs


def _measurement_observation(key: str, value: float, subject_ref: str, device_ref: str,
                             effective: str) -> dict | None:
    loinc = LOINC_MEASUREMENTS.get(key)
    if loinc is None or value is None:
        return None
    return {
        "resourceType": "Observation",
        "status": "final",
        "category": [{"coding": [{"system": SYSTEM_OBS_CATEGORY, "code": "procedure",
                                  "display": "Procedure"}]}],
        "code": {"coding": [{"system": SYSTEM_LOINC, "code": loinc.code,
                             "display": loinc.display}],
                 "text": loinc.display},
        "subject": {"reference": subject_ref},
        "effectiveDateTime": effective,
        "valueQuantity": {"value": round(float(value), 1), "unit": loinc.unit,
                          "system": SYSTEM_UCUM, "code": loinc.ucum},
        "device": {"reference": device_ref},
    }


def to_fhir_bundle(report: APEXReport, patient_reference: str | None = None,
                   record_identifier: str | None = None, intervals=None,
                   effective: str | None = None, version: str = SCHEMA_VERSION) -> dict:
    """Render an :class:`APEXReport` as a FHIR R4 ``Bundle``.

    ``patient_reference`` is the caller's own patient reference (e.g. ``"Patient/1234"``);
    supply it and no Patient resource is created, so identity never enters this system. With
    it omitted a de-identified Patient carrying only ``record_identifier`` is generated so
    the bundle is self-contained for testing.

    ``intervals`` is an optional Phase-22 :class:`~src.longitudinal.intervals.IntervalSet`;
    when present its measurements become LOINC-coded quantity Observations, which is what
    lets a receiving system trend a PR interval over time instead of re-deriving it.
    """
    effective = effective or _now()
    entries: list[dict] = []

    device_url = _urn()
    entries.append({"fullUrl": device_url, "resource": _device_resource(version)})

    if patient_reference:
        subject_ref = patient_reference
    else:
        patient_url = _urn()
        entries.append({"fullUrl": patient_url,
                        "resource": _patient_resource(record_identifier)})
        subject_ref = patient_url

    result_refs: list[dict] = []
    for finding in report.findings:
        url = _urn()
        entries.append({"fullUrl": url,
                        "resource": _finding_observation(finding, subject_ref, device_url,
                                                         effective)})
        result_refs.append({"reference": url})

    if intervals is not None:
        for key in LOINC_MEASUREMENTS:
            value = getattr(intervals, key, None)
            obs = _measurement_observation(key, value, subject_ref, device_url, effective)
            if obs is None:
                continue
            url = _urn()
            entries.append({"fullUrl": url, "resource": obs})
            result_refs.append({"reference": url})

    codes = [f.label for f in report.findings]
    conclusion_codes = [
        {"coding": [{"system": SYSTEM_ICD10CM, "code": m.icd10, "display": m.display}],
         "text": m.display}
        for m in suggestions(codes)
    ]

    diagnostic_report = {
        "resourceType": "DiagnosticReport",
        # `final` asserts a result fit to act on; APEX says otherwise whenever it has
        # flagged anything for review, so the status follows that gate.
        "status": "preliminary" if report.review_recommended else "final",
        "category": [{"coding": [{"system": SYSTEM_DIAG_SERVICE, "code": "EC",
                                  "display": "Electrocardiac"}]}],
        "code": {"coding": [{"system": SYSTEM_LOINC, "code": LOINC_ECG_STUDY.code,
                             "display": LOINC_ECG_STUDY.display}],
                 "text": LOINC_ECG_STUDY.display},
        "subject": {"reference": subject_ref},
        "effectiveDateTime": effective,
        "issued": effective,
        "result": result_refs,
        "conclusion": one_line_impression(
            report, heart_rate=getattr(intervals, "heart_rate", None) if intervals else None),
        "extension": [{"url": EXT_GENERATED_BY, "valueReference": {"reference": device_url}}],
    }
    if conclusion_codes:
        diagnostic_report["conclusionCode"] = conclusion_codes
    entries.insert(1, {"fullUrl": _urn(), "resource": diagnostic_report})

    return {"resourceType": "Bundle", "type": "collection", "timestamp": effective,
            "entry": entries}


# --- required ValueSet bindings (R4) -----------------------------------------
# Only the elements this module actually emits. Each is a `code` primitive with a
# *required* binding, which means a value outside the set makes the resource invalid.
REQUIRED_BINDINGS: dict[tuple[str, str], frozenset[str]] = {
    ("Bundle", "type"): frozenset({
        "document", "message", "transaction", "transaction-response", "batch",
        "batch-response", "history", "searchset", "collection"}),
    ("DiagnosticReport", "status"): frozenset({
        "registered", "partial", "preliminary", "final", "amended", "corrected",
        "appended", "cancelled", "entered-in-error", "unknown"}),
    ("Observation", "status"): frozenset({
        "registered", "preliminary", "final", "amended", "corrected", "cancelled",
        "entered-in-error", "unknown"}),
    ("Device", "status"): frozenset({"active", "inactive", "entered-in-error", "unknown"}),
}

DEVICE_NAME_TYPES = frozenset({"udi-label-name", "user-friendly-name",
                               "patient-reported-name", "manufacturer-name", "model-name",
                               "other"})

# Allowed reference targets in R4, for the references this module emits. Note
# DiagnosticReport.performer: software is deliberately absent from it.
REFERENCE_TARGETS: dict[tuple[str, str], frozenset[str]] = {
    ("DiagnosticReport", "subject"): frozenset({"Patient", "Group", "Device", "Location"}),
    ("DiagnosticReport", "performer"): frozenset({"Practitioner", "PractitionerRole",
                                                  "Organization", "CareTeam"}),
    ("DiagnosticReport", "result"): frozenset({"Observation"}),
    ("Observation", "subject"): frozenset({"Patient", "Group", "Device", "Location"}),
    ("Observation", "device"): frozenset({"Device", "DeviceMetric"}),
}


def check_bindings(bundle: dict) -> list[str]:
    """Required-binding and reference-target checks the schema validator does not do."""
    problems: list[str] = []

    btype = bundle.get("type")
    if btype not in REQUIRED_BINDINGS[("Bundle", "type")]:
        problems.append(f"Bundle.type {btype!r} is not in the required value set")

    by_url: dict[str, str] = {}
    for entry in bundle.get("entry", []):
        url = entry.get("fullUrl")
        rtype = entry.get("resource", {}).get("resourceType")
        if url and rtype:
            by_url[url] = rtype

    def check_ref(rtype: str, field: str, ref: dict) -> None:
        allowed = REFERENCE_TARGETS.get((rtype, field))
        target = ref.get("reference", "")
        if not target:
            return
        if target.startswith("urn:uuid:"):
            resolved = by_url.get(target)
            if resolved is None:
                problems.append(f"{rtype}.{field} -> {target} does not resolve in the bundle")
                return
        elif "/" in target:
            resolved = target.split("/")[0]
        else:
            return
        if allowed and resolved not in allowed:
            problems.append(f"{rtype}.{field} may not reference a {resolved} "
                            f"(allowed: {sorted(allowed)})")

    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        rtype = res.get("resourceType", "?")
        expected = REQUIRED_BINDINGS.get((rtype, "status"))
        if expected is not None and res.get("status") not in expected:
            problems.append(f"{rtype}.status {res.get('status')!r} is not in the required "
                            "value set")
        for name in res.get("deviceName", []) or []:
            if name.get("type") not in DEVICE_NAME_TYPES:
                problems.append(f"Device.deviceName.type {name.get('type')!r} is not in the "
                                "required value set")
        for field in ("subject", "device"):
            if isinstance(res.get(field), dict):
                check_ref(rtype, field, res[field])
        for field in ("result", "performer"):
            for ref in res.get(field, []) or []:
                if isinstance(ref, dict):
                    check_ref(rtype, field, ref)
    return problems


def validate_bundle(bundle: dict) -> list[str]:
    """Full validation: R4B StructureDefinitions plus bindings. ``[]`` means valid.

    Returns error strings rather than raising, so a caller can report several problems at
    once. The schema half requires ``fhir.resources`` (a dev/test dependency — the emitter
    itself has none); the binding half is pure Python and always runs.
    """
    problems: list[str] = []
    try:
        from fhir.resources.R4B.bundle import Bundle
    except ImportError:                                     # pragma: no cover
        problems.append("fhir.resources is not installed — schema validation skipped")
    else:
        try:
            Bundle.model_validate(bundle)
        except Exception as e:                              # noqa: BLE001
            problems.extend(line for line in str(e).splitlines() if line.strip())
    return problems + check_bindings(bundle)
