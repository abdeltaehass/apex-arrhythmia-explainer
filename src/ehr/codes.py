"""Phase 23 — terminology: SCP-ECG findings to ICD-10-CM and LOINC.

An EHR does not speak SCP-ECG. To hand a hospital system anything it can store, bill, or
trend, APEX's 71 internal labels have to be expressed in the code systems clinical software
actually uses: **ICD-10-CM** for diagnoses and **LOINC** for observations.

The mapping is not a lookup table with obvious contents. Two things make it a design
problem with a right and a wrong answer.

**1. An ECG finding is usually not a billable diagnosis.** This is the central distinction
here and everything else follows from it. For some findings the ECG *is* the diagnostic
test: complete AV block, atrial fibrillation, right bundle branch block are defined by what
the tracing shows, and a 12-lead recording is sufficient to assert them. For most others
the ECG only raises a suspicion that some other test has to settle. Myocardial infarction
requires a troponin rise and fall under the Fourth Universal Definition — Q waves on an ECG
are not an MI. Left ventricular hypertrophy on voltage criteria has poor agreement with
echocardiography, and ICD-10-CM's nearest code (I51.7 *Cardiomegaly*) is an anatomical
diagnosis that imaging, not an ECG, establishes.

So findings are split into two tiers. :data:`TIER_DEFINITIONAL` findings get their specific
ICD-10-CM code. :data:`TIER_SUGGESTIVE` findings get **R94.31 — Abnormal electrocardiogram
[ECG] [EKG]**, which is the honest code for "the tracing is abnormal and something else
must establish why". The more specific code a clinician *might* reach for is still carried,
in :attr:`ICDMapping.candidate`, together with the evidence it would need
(:attr:`ICDMapping.requires`) — visible for review, never auto-suggested. A system that
emitted I21.9 off a Q wave would be generating a fraudulent claim, not a helpful one.

**2. ICD-10-CM encodes information a single ECG cannot carry.** Atrial fibrillation is the
worked example, and it is worth being precise because the obvious mapping is wrong. ICD-10
distinguishes paroxysmal (I48.0), persistent (I48.11 / I48.19), chronic (I48.20),
permanent (I48.21) and unspecified (I48.91) atrial fibrillation. Those distinctions are
about *duration and treatment history* — paroxysmal means terminating within seven days —
and none of them are visible in ten seconds of signal. A recording showing AF supports
exactly one code: **I48.91, unspecified**. Choosing I48.0 instead is upcoding: a more
specific code than the documentation supports.

Phase 22's serial comparison can supply genuine supporting evidence here (AF present now,
absent on a prior study, is real evidence of an intermittent pattern), but it is still not
sufficient on its own, so the tier does not change. See ``docs/ehr/report.md``.

**Provenance.** Every code below was verified against the U.S. National Library of
Medicine's Clinical Table Search Service on 2026-08-10 — ICD-10-CM via the ``icd10cm/v3``
endpoint, LOINC via ``loinc_items/v3`` — and the ``display`` strings are the official
descriptions returned by it, not paraphrases. ``scripts/verify_terminology.py`` re-runs
that check against the live service so the table cannot silently rot: ICD-10-CM is revised
annually and does move underneath you. It moved here — FY2024 subdivided I47.1 into
I47.10/I47.11/I47.19, which turned the previously billable I47.1 into a non-billable
category header.

**Licensing.** ICD-10-CM is published by CDC/NCHS and is in the public domain, so the codes
are reproduced here directly. LOINC is © Regenstrief Institute, Inc., made available under
the LOINC License; using individual codes as identifiers is permitted, and no part of the
LOINC database is redistributed here — only the six codes this module emits. SNOMED CT
would require an affiliate licence and is deliberately not used.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- code system URIs (FHIR canonical) ---------------------------------------
SYSTEM_ICD10CM = "http://hl7.org/fhir/sid/icd-10-cm"
SYSTEM_LOINC = "http://loinc.org"
SYSTEM_UCUM = "http://unitsofmeasure.org"
# APEX's own labels have no standard code system, so they get a local namespace rather
# than being forced into a standard one they do not fit.
SYSTEM_SCP = "https://github.com/abdeltaehass/apex-arrhythmia-explainer/CodeSystem/scp-ecg"

TERMINOLOGY_VERIFIED_ON = "2026-08-10"
TERMINOLOGY_SOURCE = "NLM Clinical Table Search Service (icd10cm/v3, loinc_items/v3)"

TIER_DEFINITIONAL = "ecg-definitional"   # the ECG establishes the diagnosis
TIER_SUGGESTIVE = "ecg-suggestive"       # the ECG suggests it; other evidence must confirm
TIER_NOT_CODABLE = "not-codable"         # normal / physiological variant — no diagnosis

# The code for "this tracing is abnormal, and what it means is a clinical question".
ABNORMAL_ECG = ("R94.31", "Abnormal electrocardiogram [ECG] [EKG]")


@dataclass(frozen=True)
class ICDMapping:
    """One SCP code's billing representation."""

    scp: str
    icd10: str | None                    # what APEX will suggest (None = no diagnosis)
    display: str                         # official ICD-10-CM description, verbatim
    tier: str
    candidate: str | None = None         # the specific code, IF other evidence existed
    candidate_display: str | None = None
    requires: str | None = None          # the evidence that candidate would need
    note: str = ""

    @property
    def codable(self) -> bool:
        return self.icd10 is not None


def _definitional(scp: str, code: str, display: str, note: str = "") -> ICDMapping:
    return ICDMapping(scp, code, display, TIER_DEFINITIONAL, note=note)


def _suggestive(scp: str, candidate: str | None = None, candidate_display: str | None = None,
                requires: str | None = None, note: str = "") -> ICDMapping:
    return ICDMapping(scp, ABNORMAL_ECG[0], ABNORMAL_ECG[1], TIER_SUGGESTIVE,
                      candidate=candidate, candidate_display=candidate_display,
                      requires=requires, note=note)


def _not_codable(scp: str, note: str) -> ICDMapping:
    return ICDMapping(scp, None, "", TIER_NOT_CODABLE, note=note)


_MI_CANDIDATE = ("I25.2", "Old myocardial infarction")
_MI_REQUIRES = ("documented infarction history; an ECG pattern alone does not establish "
                "infarction (Fourth Universal Definition requires troponin rise/fall)")
_ACUTE_REQUIRES = ("serial troponin and clinical correlation; ECG injury pattern alone does "
                   "not establish acute infarction")
_HYP_CANDIDATE = ("I51.7", "Cardiomegaly")
_HYP_REQUIRES = "echocardiography or other imaging; ECG voltage criteria are not sufficient"

# --- the map ------------------------------------------------------------------
ICD10_MAP: dict[str, ICDMapping] = {}


def _add(m: ICDMapping) -> None:
    ICD10_MAP[m.scp] = m


# rhythm ----------------------------------------------------------------------
_add(_not_codable("SR", "normal sinus rhythm is not a diagnosis"))
_add(_not_codable("SARRH", "sinus arrhythmia is a physiological (usually respiratory) "
                           "variant, not a codable arrhythmia"))
_add(_definitional("SBRAD", "R00.1", "Bradycardia, unspecified",
                   "R00.1 explicitly includes sinus bradycardia"))
_add(_definitional("STACH", "R00.0", "Tachycardia, unspecified",
                   "R00.0 explicitly includes sinus tachycardia"))
_add(_definitional("AFIB", "I48.91", "Unspecified atrial fibrillation",
                   "paroxysmal/persistent/permanent (I48.0/I48.1-/I48.2-) all encode "
                   "duration or treatment history, which a single recording cannot show"))
_add(_definitional("AFLT", "I48.92", "Unspecified atrial flutter",
                   "typical vs atypical (I48.3/I48.4) needs the flutter circuit, which "
                   "surface ECG alone does not reliably establish"))
_add(_definitional("PSVT", "I47.10", "Supraventricular tachycardia, unspecified",
                   "I47.1 became a non-billable header when FY2024 subdivided it"))
_add(_definitional("SVTAC", "I47.10", "Supraventricular tachycardia, unspecified"))
_add(_definitional("SVARR", "I49.8", "Other specified cardiac arrhythmias"))

# ectopy ----------------------------------------------------------------------
_add(_definitional("PAC", "I49.1", "Atrial premature depolarization"))
_add(_definitional("PVC", "I49.3", "Ventricular premature depolarization"))
_add(_definitional("PRC(S)", "I49.40", "Unspecified premature depolarization",
                   "PTB-XL's PRC(S) does not say atrial or ventricular"))
_add(_definitional("BIGU", "I49.8", "Other specified cardiac arrhythmias",
                   "bigeminy has no dedicated code; the underlying ectopy carries it"))
_add(_definitional("TRIGU", "I49.8", "Other specified cardiac arrhythmias",
                   "trigeminy has no dedicated code"))

# pacing ----------------------------------------------------------------------
_add(_definitional("PACE", "Z95.0", "Presence of cardiac pacemaker",
                   "a status code, not a diagnosis — it describes the patient, not a disease"))

# conduction ------------------------------------------------------------------
_add(_definitional("1AVB", "I44.0", "Atrioventricular block, first degree"))
_add(_definitional("2AVB", "I44.1", "Atrioventricular block, second degree",
                   "Mobitz I vs II is not distinguished by PTB-XL's label"))
_add(_definitional("3AVB", "I44.2", "Atrioventricular block, complete"))
_add(_definitional("LPR", "I44.0", "Atrioventricular block, first degree",
                   "a PR interval over 200 ms is first-degree AV block by definition"))
_add(_definitional("CLBBB", "I44.7", "Left bundle-branch block, unspecified"))
_add(_definitional("ILBBB", "I44.7", "Left bundle-branch block, unspecified",
                   "ICD-10-CM has no incomplete-LBBB code; this is less specific than the "
                   "finding"))
_add(_definitional("CRBBB", "I45.10", "Unspecified right bundle-branch block"))
_add(_definitional("IRBBB", "I45.19", "Other right bundle-branch block",
                   "incomplete RBBB conventionally codes to I45.19"))
_add(_definitional("IVCD", "I45.4", "Nonspecific intraventricular block"))
_add(_definitional("LAFB", "I44.4", "Left anterior fascicular block"))
_add(_definitional("LPFB", "I44.5", "Left posterior fascicular block"))
_add(_definitional("WPW", "I45.6", "Pre-excitation syndrome"))

# chamber / hypertrophy — all suggestive --------------------------------------
for _scp in ("LVH", "VCLVH", "SEHYP"):
    _add(_suggestive(_scp, *_HYP_CANDIDATE, requires=_HYP_REQUIRES,
                     note="ECG voltage criteria for hypertrophy are specific but insensitive "
                          "and are not an anatomical diagnosis"))
_add(_suggestive("RVH", *_HYP_CANDIDATE, requires=_HYP_REQUIRES))
for _scp in ("LAO/LAE", "RAO/RAE"):
    _add(_suggestive(_scp, requires="echocardiography for chamber size",
                     note="atrial overload/enlargement pattern"))
for _scp in ("HVOLT", "LVOLT"):
    _add(_suggestive(_scp, note="a voltage observation, not a diagnosis; low voltage has "
                                "many causes (effusion, obesity, COPD, infiltration)"))

# repolarization — all suggestive ---------------------------------------------
for _scp in ("ISCAL", "ISCAN", "ISCAS", "ISCIL", "ISCIN", "ISCLA", "ISC_"):
    _add(_suggestive(_scp, "I25.10",
                     "Atherosclerotic heart disease of native coronary artery without "
                     "angina pectoris",
                     requires="clinical correlation and ischemia testing; repolarization "
                              "change is not specific for coronary disease",
                     note="ischemic-appearing repolarization change"))
_add(_suggestive("STE_", "I21.9", "Acute myocardial infarction, unspecified",
                 requires=_ACUTE_REQUIRES,
                 note="ST elevation is the highest-acuity ECG finding but is not itself an "
                      "infarction diagnosis; see src/serving/severity.py for triage"))
_add(_suggestive("STD_", requires="clinical correlation", note="ST depression"))
_add(_suggestive("ANEUR", "I25.3", "Aneurysm of heart",
                 requires="echocardiography or ventriculography",
                 note="persistent ST elevation with aneurysm morphology"))
_add(_suggestive("LNGQT", "I45.81", "Long QT syndrome",
                 requires="repeat ECGs, drug and electrolyte review, and clinical or genetic "
                          "assessment — a single prolonged QTc is not the syndrome",
                 note="acquired QT prolongation (drugs, electrolytes) is far commoner than "
                      "congenital long QT syndrome"))
_add(_suggestive("EL", requires="serum electrolytes — the diagnosis is a laboratory value",
                 note="ECG changes suggesting electrolyte disturbance"))
_add(_suggestive("DIG", requires="medication history and digoxin level",
                 note="digitalis effect; an expected drug effect, not toxicity"))
for _scp in ("INVT", "LOWT", "NDT", "NST_", "NT_", "TAB_"):
    _add(_suggestive(_scp, requires="clinical correlation",
                     note="non-specific T-wave / ST-T abnormality"))

# infarction — all suggestive --------------------------------------------------
for _scp in ("AMI",):
    _add(_suggestive(_scp, "I21.9", "Acute myocardial infarction, unspecified",
                     requires=_ACUTE_REQUIRES))
for _scp in ("INJAL", "INJAS", "INJIL", "INJIN", "INJLA"):
    _add(_suggestive(_scp, "I21.9", "Acute myocardial infarction, unspecified",
                     requires=_ACUTE_REQUIRES, note="subendocardial injury pattern"))
for _scp in ("ALMI", "ASMI", "ILMI", "IMI", "IPLMI", "IPMI", "LMI", "PMI"):
    _add(_suggestive(_scp, *_MI_CANDIDATE, requires=_MI_REQUIRES,
                     note="infarction pattern by territory; age of infarct is not "
                          "determinable from one tracing"))
for _scp in ("ABQRS", "QWAVE"):
    _add(_suggestive(_scp, *_MI_CANDIDATE, requires=_MI_REQUIRES,
                     note="Q waves may be positional or normal-variant"))

# normal ------------------------------------------------------------------------
_add(_not_codable("NORM", "a normal ECG is not a diagnosis; code the encounter reason "
                          "instead"))


def icd10_for(scp: str) -> ICDMapping | None:
    """The billing mapping for one SCP code, or ``None`` if the code is unknown."""
    return ICD10_MAP.get(scp)


def suggestions(codes) -> list[ICDMapping]:
    """Codable, de-duplicated ICD-10-CM suggestions for a set of SCP findings.

    De-duplication matters: a report naming five ischemic territories yields five findings
    that all map to R94.31, and a claim carrying R94.31 five times is malformed. Ordered
    ECG-definitional first, since those are the codes that stand on the ECG alone.
    """
    seen: set[str] = set()
    out: list[ICDMapping] = []
    for scp in sorted(codes):
        m = ICD10_MAP.get(scp)
        if m is None or not m.codable or m.icd10 in seen:
            continue
        seen.add(m.icd10)
        out.append(m)
    out.sort(key=lambda m: (m.tier != TIER_DEFINITIONAL, m.icd10 or ""))
    return out


# --- LOINC: the observation side ----------------------------------------------
@dataclass(frozen=True)
class LoincCode:
    code: str
    display: str          # official LOINC Long Common Name, verbatim
    unit: str | None = None       # UCUM display
    ucum: str | None = None       # UCUM code


LOINC_ECG_STUDY = LoincCode("11524-6", "EKG study")
LOINC_ECG_IMPRESSION = LoincCode("8601-7", "EKG impression")

# Phase-22 interval measurements, keyed by IntervalSet attribute.
LOINC_MEASUREMENTS: dict[str, LoincCode] = {
    "heart_rate": LoincCode("8867-4", "Heart rate", "beats/minute", "/min"),
    "pr": LoincCode("8625-6", "P-R Interval", "milliseconds", "ms"),
    "qrs": LoincCode("8633-0", "QRS duration", "milliseconds", "ms"),
    "qt": LoincCode("8634-8", "Q-T interval", "milliseconds", "ms"),
    "qtc_fridericia": LoincCode("8636-3", "Q-T interval corrected", "milliseconds", "ms"),
}

# ICD-10-CM codes are letter + 2 digits, optionally '.' + up to 4 alphanumerics.
ICD10CM_PATTERN = r"^[A-TV-Z][0-9][0-9AB](\.[0-9A-TV-Z]{1,4})?$"
LOINC_PATTERN = r"^\d{1,5}-\d$"
