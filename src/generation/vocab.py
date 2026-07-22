"""Clinical vocabulary: SCP-ECG code -> cardiologist-register phrasing.

Each of PTB-XL's 71 SCP statements maps to a :class:`Entry` with

- ``group``   — where the statement belongs in a report (rhythm, conduction, …),
                used to order and cluster the Findings/Impression sections.
- ``finding`` — a *factual, morphological* observation ("ST-segment depression"),
                the kind of sentence that goes under **Findings**.
- ``impression`` — the *interpretive* named entity ("subendocardial ischemia"), the
                kind of phrase that goes under **Impression**. ``None`` means the
                statement only contributes a finding (e.g. an isolated Q wave).
- ``territory`` — for infarction/ischemia codes, the coronary territory whose leads
                the finding localizes to; drives the "in the inferior leads (II, III,
                aVF)" clause. ``None`` for non-localizing statements.

This is deliberately a hand-authored lookup, not learned: it defines the *target*
clinical language the generator is fine-tuned to reproduce, and the reverse map
(:func:`impression_terms`) lets `src/eval/consistency.py` check that generated text
only asserts findings the detector actually surfaced.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Lead territories -------------------------------------------------------
# PTB-XL lead order: I, II, III, aVR, aVL, aVF, V1..V6.
TERRITORIES: dict[str, list[str]] = {
    "inferior": ["II", "III", "aVF"],
    "lateral": ["I", "aVL", "V5", "V6"],
    "high lateral": ["I", "aVL"],
    "anterior": ["V2", "V3", "V4"],
    "anteroseptal": ["V1", "V2", "V3"],
    "septal": ["V1", "V2"],
    "anterolateral": ["I", "aVL", "V4", "V5", "V6"],
    "posterior": ["V1", "V2", "V3"],  # posterior read reciprocally in the septal leads
    "inferolateral": ["II", "III", "aVF", "V5", "V6"],
    "inferoposterior": ["II", "III", "aVF", "V1", "V2", "V3"],
    "inferoposterolateral": ["II", "III", "aVF", "V5", "V6", "V1", "V2", "V3"],
}

# Group order in a report (Findings section follows this sequence).
GROUP_ORDER = (
    "rhythm", "ectopy", "pacing", "conduction", "chamber",
    "repolarization", "infarction", "normal", "technical",
)


@dataclass(frozen=True)
class Entry:
    group: str
    finding: str
    impression: str | None = None
    territory: str | None = None


# --- The 71-code vocabulary -------------------------------------------------
VOCAB: dict[str, Entry] = {
    # -- rhythm ------------------------------------------------------------
    "SR": Entry("rhythm", "Regular rhythm with normal P waves preceding each QRS", "sinus rhythm"),
    "STACH": Entry("rhythm", "Regular rhythm with normal P waves at a rate above 100 bpm",
                   "sinus tachycardia"),
    "SBRAD": Entry("rhythm", "Regular rhythm with normal P waves at a rate below 60 bpm",
                   "sinus bradycardia"),
    "SARRH": Entry("rhythm", "Sinus rhythm with beat-to-beat variation in the P-P interval",
                   "sinus arrhythmia"),
    "AFIB": Entry("rhythm", "Irregularly irregular rhythm without discernible P waves",
                  "atrial fibrillation"),
    "AFLT": Entry("rhythm", "Regular atrial activity with a sawtooth flutter-wave baseline",
                  "atrial flutter"),
    "SVTAC": Entry("rhythm", "Narrow-complex tachycardia of supraventricular origin",
                   "supraventricular tachycardia"),
    "PSVT": Entry("rhythm", "Abrupt-onset narrow-complex tachycardia",
                  "paroxysmal supraventricular tachycardia"),
    "SVARR": Entry("rhythm", "Irregular supraventricular rhythm", "supraventricular arrhythmia"),
    "PACE": Entry("pacing", "Pacing spikes preceding paced complexes",
                  "normally functioning artificial pacemaker"),
    # -- ectopy ------------------------------------------------------------
    "PAC": Entry("ectopy", "Premature, abnormally shaped P waves interrupting the rhythm",
                 "atrial premature complexes"),
    "PVC": Entry("ectopy", "Premature wide QRS complexes without a preceding P wave",
                 "ventricular premature complexes"),
    "PRC(S)": Entry("ectopy", "Premature complexes interrupting the underlying rhythm",
                    "premature complexes"),
    "BIGU": Entry("ectopy", "Every other beat is a premature complex", "bigeminy"),
    "TRIGU": Entry("ectopy", "Every third beat is a premature complex", "trigeminy"),
    # -- conduction --------------------------------------------------------
    "1AVB": Entry("conduction", "Prolonged PR interval beyond 200 ms with every P conducted",
                  "first-degree AV block"),
    "2AVB": Entry("conduction", "Intermittent failure of P-wave conduction to the ventricles",
                  "second-degree AV block"),
    "3AVB": Entry("conduction", "AV dissociation with independent atrial and ventricular rates",
                  "third-degree (complete) AV block"),
    "CLBBB": Entry("conduction", "Broad QRS with a monophasic R wave in the lateral leads and "
                   "absent septal Q waves", "complete left bundle branch block"),
    "CRBBB": Entry("conduction", "Broad QRS with an rSR' pattern in V1 and wide S waves laterally",
                   "complete right bundle branch block"),
    "ILBBB": Entry("conduction", "Left bundle branch block morphology with QRS under 120 ms",
                   "incomplete left bundle branch block"),
    "IRBBB": Entry("conduction", "rSR' pattern in V1 with QRS under 120 ms",
                   "incomplete right bundle branch block"),
    "IVCD": Entry("conduction", "Widened QRS not meeting criteria for a specific bundle branch block",
                  "non-specific intraventricular conduction delay"),
    "LAFB": Entry("conduction", "Left axis deviation with small q in aVL and rS in the inferior leads",
                  "left anterior fascicular block"),
    "LPFB": Entry("conduction", "Right axis deviation with a fascicular conduction pattern",
                  "left posterior fascicular block"),
    "WPW": Entry("conduction", "Short PR interval with a delta wave slurring the QRS upstroke",
                 "ventricular pre-excitation (Wolff-Parkinson-White)"),
    "LPR": Entry("conduction", "Prolonged PR interval", "prolonged AV conduction"),
    # -- chamber enlargement / voltage ------------------------------------
    "LVH": Entry("chamber", "Increased left-sided QRS voltage meeting hypertrophy criteria",
                 "left ventricular hypertrophy"),
    "RVH": Entry("chamber", "Right axis deviation with a dominant R wave in V1",
                 "right ventricular hypertrophy"),
    "LAO/LAE": Entry("chamber", "Broad, notched P waves", "left atrial enlargement"),
    "RAO/RAE": Entry("chamber", "Tall, peaked P waves", "right atrial enlargement"),
    "SEHYP": Entry("chamber", "Prominent septal forces", "septal hypertrophy"),
    "VCLVH": Entry("chamber", "QRS voltages meeting left ventricular hypertrophy criteria",
                   "voltage criteria for left ventricular hypertrophy"),
    "HVOLT": Entry("chamber", "Increased QRS voltage", None),
    "LVOLT": Entry("chamber", "Low QRS voltage in the frontal and precordial leads", None),
    # -- repolarization / ST-T --------------------------------------------
    "NDT": Entry("repolarization", "Non-diagnostic T-wave changes", "non-diagnostic T-wave abnormality"),
    "NST_": Entry("repolarization", "Non-specific ST-segment changes", "non-specific ST changes"),
    "NT_": Entry("repolarization", "Non-specific T-wave changes", "non-specific T-wave changes"),
    "TAB_": Entry("repolarization", "T-wave abnormality", "T-wave abnormality"),
    "INVT": Entry("repolarization", "Inverted T waves", "T-wave inversion"),
    "LOWT": Entry("repolarization", "Low-amplitude T waves", "low T-wave amplitude"),
    "STD_": Entry("repolarization", "Non-specific ST-segment depression", "ST-segment depression"),
    "STE_": Entry("repolarization", "Non-specific ST-segment elevation", "ST-segment elevation"),
    "LNGQT": Entry("repolarization", "Prolonged QT interval", "long QT interval"),
    "DIG": Entry("repolarization", "Scooped ST-segment depression consistent with digitalis effect",
                 "digitalis effect"),
    "EL": Entry("repolarization", "Repolarization changes suggesting an electrolyte or drug effect",
                "electrolyte/drug effect"),
    "ANEUR": Entry("repolarization", "Persistent ST-segment elevation with the morphology of a "
                   "ventricular aneurysm", "ST-T changes compatible with ventricular aneurysm"),
    "ISC_": Entry("repolarization", "Ischemic-type repolarization changes", "non-specific ischemia"),
    "ISCAN": Entry("repolarization", "T-wave inversion", "anterior ischemia", "anterior"),
    "ISCAS": Entry("repolarization", "T-wave inversion", "anteroseptal ischemia", "anteroseptal"),
    "ISCAL": Entry("repolarization", "T-wave inversion", "anterolateral ischemia", "anterolateral"),
    "ISCLA": Entry("repolarization", "T-wave inversion", "lateral ischemia", "lateral"),
    "ISCIN": Entry("repolarization", "T-wave inversion", "inferior ischemia", "inferior"),
    "ISCIL": Entry("repolarization", "T-wave inversion", "inferolateral ischemia", "inferolateral"),
    # -- infarction / injury ----------------------------------------------
    "INJAS": Entry("infarction", "ST-segment depression", "subendocardial injury, anteroseptal",
                   "anteroseptal"),
    "INJAL": Entry("infarction", "ST-segment depression", "subendocardial injury, anterolateral",
                   "anterolateral"),
    "INJIN": Entry("infarction", "ST-segment depression", "subendocardial injury, inferior", "inferior"),
    "INJIL": Entry("infarction", "ST-segment depression", "subendocardial injury, inferolateral",
                   "inferolateral"),
    "INJLA": Entry("infarction", "ST-segment depression", "subendocardial injury, lateral", "lateral"),
    "AMI": Entry("infarction", "Pathological Q waves", "anterior myocardial infarction", "anterior"),
    "ASMI": Entry("infarction", "Pathological Q waves", "anteroseptal myocardial infarction",
                  "anteroseptal"),
    "ALMI": Entry("infarction", "Pathological Q waves", "anterolateral myocardial infarction",
                  "anterolateral"),
    "IMI": Entry("infarction", "Pathological Q waves", "inferior myocardial infarction", "inferior"),
    "ILMI": Entry("infarction", "Pathological Q waves", "inferolateral myocardial infarction",
                  "inferolateral"),
    "LMI": Entry("infarction", "Pathological Q waves", "lateral myocardial infarction", "lateral"),
    "IPMI": Entry("infarction", "Pathological Q waves with reciprocal septal changes",
                  "inferoposterior myocardial infarction", "inferoposterior"),
    "IPLMI": Entry("infarction", "Pathological Q waves with reciprocal septal changes",
                   "inferoposterolateral myocardial infarction", "inferoposterolateral"),
    "PMI": Entry("infarction", "Tall R waves with reciprocal ST-segment depression",
                 "posterior myocardial infarction", "posterior"),
    "QWAVE": Entry("infarction", "Pathological Q waves", None),
    "ABQRS": Entry("infarction", "Abnormal QRS morphology", None),
    # -- normal ------------------------------------------------------------
    "NORM": Entry("normal", "Normal P-wave morphology, axis, intervals, and R-wave progression "
                  "with no significant ST-segment or T-wave abnormality", "normal ECG"),
}

# Codes that represent acute/ischemic repolarization or infarction — their absence is
# what "no acute ischemic changes" asserts.
ISCHEMIC_GROUPS = ("infarction",)
ISCHEMIC_CODES = frozenset(
    c for c, e in VOCAB.items()
    if e.group in ISCHEMIC_GROUPS or c.startswith(("ISC", "INJ")) or c in {"STE_", "ANEUR"}
)


def entry(code: str) -> Entry | None:
    return VOCAB.get(code)


def leads_for(code: str) -> list[str]:
    """Default lead set a localized finding cites (empty if the code doesn't localize)."""
    e = VOCAB.get(code)
    if e is None or e.territory is None:
        return []
    return TERRITORIES.get(e.territory, [])


def impression_terms() -> dict[str, str]:
    """Reverse lookup: impression phrase (lowercased) -> SCP code, for consistency checks."""
    return {e.impression.lower(): c for c, e in VOCAB.items() if e.impression}
