"""Phase 27 — the Spanish clinical vocabulary.

A one-for-one Spanish counterpart to :data:`src.generation.vocab.VOCAB`: for each of the 71
SCP-ECG statements, the factual morphological sentence that belongs under *Hallazgos* and
the interpretive term that belongs under *Impresión*.

**Hand-authored, not machine-translated, and the distinction is the point.** Clinical
Spanish is not English with Spanish words. "Bundle branch block" is *bloqueo de rama*, not
*bloqueo del haz*; a fascicular block is conventionally *hemibloqueo* in Spanish cardiology
even though "hemiblock" is archaic in English; atrial enlargement reads *crecimiento
auricular*, not the literal *agrandamiento*. A translation layer bolted onto English output
gets these subtly wrong in ways that read as foreign to a Spanish-speaking clinician while
remaining comprehensible enough that nobody files a bug.

Every impression term here is checked against real Spanish-language cardiology prose by
``scripts/i18n_eval.py``, which is what keeps this file honest rather than merely
plausible-looking.

**Accents are written properly** (*fibrilación*, *isquemia*, *Impresión*). Dropping them
would make matching easier and the text wrong. :mod:`src.i18n.parse` normalizes accents when
*reading* text instead, so a model or a clinician who types ``fibrilacion`` is still
understood — the tolerance belongs in the parser, not in the vocabulary.
"""

from __future__ import annotations

from src.generation.vocab import Entry

# Coronary territory names, for the "en las derivaciones ..." clause.
TERRITORIES_ES: dict[str, str] = {
    "inferior": "inferiores",
    "lateral": "laterales",
    "high lateral": "laterales altas",
    "anterior": "anteriores",
    "anteroseptal": "anteroseptales",
    "septal": "septales",
    "anterolateral": "anterolaterales",
    "posterior": "posteriores",
    "inferolateral": "inferolaterales",
    "inferoposterior": "inferoposteriores",
    "inferoposterolateral": "inferoposterolaterales",
}

VOCAB_ES: dict[str, Entry] = {
    # -- ritmo ---------------------------------------------------------------
    "SR": Entry("rhythm", "Ritmo regular con ondas P normales precediendo a cada QRS",
                "ritmo sinusal"),
    "STACH": Entry("rhythm", "Ritmo regular con ondas P normales a una frecuencia superior a "
                             "100 lpm", "taquicardia sinusal"),
    "SBRAD": Entry("rhythm", "Ritmo regular con ondas P normales a una frecuencia inferior a "
                             "60 lpm", "bradicardia sinusal"),
    "SARRH": Entry("rhythm", "Ritmo sinusal con variación latido a latido del intervalo P-P",
                   "arritmia sinusal"),
    "AFIB": Entry("rhythm", "Ritmo irregularmente irregular sin ondas P identificables",
                  "fibrilación auricular"),
    "AFLT": Entry("rhythm", "Actividad auricular regular con ondas de flutter en dientes de "
                            "sierra", "flutter auricular"),
    "SVTAC": Entry("rhythm", "Taquicardia de complejo estrecho de origen supraventricular",
                   "taquicardia supraventricular"),
    "PSVT": Entry("rhythm", "Taquicardia de complejo estrecho de inicio súbito",
                  "taquicardia supraventricular paroxística"),
    "SVARR": Entry("rhythm", "Ritmo supraventricular irregular", "arritmia supraventricular"),

    # -- estimulación --------------------------------------------------------
    "PACE": Entry("pacing", "Espículas de estimulación precediendo a los complejos estimulados",
                  "marcapasos artificial con funcionamiento normal"),

    # -- extrasistolia -------------------------------------------------------
    "PAC": Entry("ectopy", "Ondas P prematuras de morfología anómala que interrumpen el ritmo",
                 "extrasístoles auriculares"),
    "PVC": Entry("ectopy", "Complejos QRS anchos prematuros sin onda P precedente",
                 "extrasístoles ventriculares"),
    "PRC(S)": Entry("ectopy", "Complejos prematuros que interrumpen el ritmo de base",
                    "extrasístoles"),
    "BIGU": Entry("ectopy", "Cada segundo latido es un complejo prematuro", "bigeminismo"),
    "TRIGU": Entry("ectopy", "Cada tercer latido es un complejo prematuro", "trigeminismo"),

    # -- conducción ----------------------------------------------------------
    "1AVB": Entry("conduction", "Intervalo PR prolongado por encima de 200 ms con conducción "
                                "de todas las ondas P", "bloqueo AV de primer grado"),
    "2AVB": Entry("conduction", "Fallo intermitente de la conducción de la onda P a los "
                                "ventrículos", "bloqueo AV de segundo grado"),
    "3AVB": Entry("conduction", "Disociación AV con frecuencias auricular y ventricular "
                                "independientes", "bloqueo AV completo"),
    "CLBBB": Entry("conduction", "QRS ancho con onda R monofásica en las derivaciones "
                                 "laterales y ausencia de ondas Q septales",
                   "bloqueo completo de rama izquierda"),
    "CRBBB": Entry("conduction", "QRS ancho con patrón rSR' en V1 y ondas S empastadas en "
                                 "las derivaciones laterales",
                   "bloqueo completo de rama derecha"),
    "ILBBB": Entry("conduction", "Morfología de bloqueo de rama izquierda con QRS inferior a "
                                 "120 ms", "bloqueo incompleto de rama izquierda"),
    "IRBBB": Entry("conduction", "Patrón rSR' en V1 con QRS inferior a 120 ms",
                   "bloqueo incompleto de rama derecha"),
    "IVCD": Entry("conduction", "QRS ensanchado que no cumple criterios de bloqueo de rama "
                                "específico", "trastorno inespecífico de la conducción "
                                              "intraventricular"),
    "LAFB": Entry("conduction", "Desviación del eje a la izquierda con q pequeña en aVL y rS "
                                "en las derivaciones inferiores",
                  "hemibloqueo anterior izquierdo"),
    "LPFB": Entry("conduction", "Desviación del eje a la derecha con patrón de conducción "
                                "fascicular", "hemibloqueo posterior izquierdo"),
    "WPW": Entry("conduction", "Intervalo PR corto con onda delta que empasta el ascenso del "
                               "QRS", "preexcitación ventricular (Wolff-Parkinson-White)"),
    "LPR": Entry("conduction", "Intervalo PR prolongado", "conducción AV prolongada"),

    # -- cavidades -----------------------------------------------------------
    "LVH": Entry("chamber", "Aumento del voltaje del QRS izquierdo que cumple criterios de "
                            "hipertrofia", "hipertrofia ventricular izquierda"),
    "RVH": Entry("chamber", "Desviación del eje a la derecha con onda R dominante en V1",
                 "hipertrofia ventricular derecha"),
    "LAO/LAE": Entry("chamber", "Ondas P anchas y melladas", "crecimiento auricular izquierdo"),
    "RAO/RAE": Entry("chamber", "Ondas P altas y picudas", "crecimiento auricular derecho"),
    "SEHYP": Entry("chamber", "Fuerzas septales prominentes", "hipertrofia septal"),
    "VCLVH": Entry("chamber", "Voltajes del QRS que cumplen criterios de hipertrofia "
                              "ventricular izquierda",
                   "criterios de voltaje para hipertrofia ventricular izquierda"),
    "HVOLT": Entry("chamber", "Aumento del voltaje del QRS", None),
    "LVOLT": Entry("chamber", "Bajo voltaje del QRS en las derivaciones de miembros y "
                              "precordiales", None),

    # -- repolarización ------------------------------------------------------
    "NDT": Entry("repolarization", "Cambios no diagnósticos de la onda T",
                 "alteración no diagnóstica de la onda T"),
    "NST_": Entry("repolarization", "Cambios inespecíficos del segmento ST",
                  "cambios inespecíficos del segmento ST"),
    "NT_": Entry("repolarization", "Cambios inespecíficos de la onda T",
                 "cambios inespecíficos de la onda T"),
    "TAB_": Entry("repolarization", "Alteración de la onda T", "alteración de la onda T"),
    "INVT": Entry("repolarization", "Ondas T invertidas", "inversión de la onda T"),
    "LOWT": Entry("repolarization", "Ondas T de baja amplitud", "onda T de baja amplitud"),
    "STD_": Entry("repolarization", "Descenso inespecífico del segmento ST",
                  "descenso del segmento ST"),
    "STE_": Entry("repolarization", "Elevación inespecífica del segmento ST",
                  "elevación del segmento ST"),
    "LNGQT": Entry("repolarization", "Intervalo QT prolongado", "intervalo QT largo"),
    "DIG": Entry("repolarization", "Descenso cóncavo del segmento ST compatible con efecto "
                                   "digitálico", "efecto digitálico"),
    "EL": Entry("repolarization", "Cambios de repolarización sugestivos de alteración "
                                  "electrolítica o farmacológica",
                "alteración electrolítica o farmacológica"),
    "ANEUR": Entry("repolarization", "Elevación persistente del segmento ST con morfología de "
                                     "aneurisma ventricular",
                   "cambios ST-T compatibles con aneurisma ventricular"),
    "ISC_": Entry("repolarization", "Cambios de repolarización de tipo isquémico",
                  "isquemia inespecífica"),
    "ISCAN": Entry("repolarization", "Inversión de la onda T", "isquemia anterior", "anterior"),
    "ISCAS": Entry("repolarization", "Inversión de la onda T", "isquemia anteroseptal",
                   "anteroseptal"),
    "ISCAL": Entry("repolarization", "Inversión de la onda T", "isquemia anterolateral",
                   "anterolateral"),
    "ISCLA": Entry("repolarization", "Inversión de la onda T", "isquemia lateral", "lateral"),
    "ISCIN": Entry("repolarization", "Inversión de la onda T", "isquemia inferior", "inferior"),
    "ISCIL": Entry("repolarization", "Inversión de la onda T", "isquemia inferolateral",
                   "inferolateral"),

    # -- infarto -------------------------------------------------------------
    "INJAS": Entry("infarction", "Descenso del segmento ST", "lesión subendocárdica anteroseptal",
                   "anteroseptal"),
    "INJAL": Entry("infarction", "Descenso del segmento ST",
                   "lesión subendocárdica anterolateral", "anterolateral"),
    "INJIN": Entry("infarction", "Descenso del segmento ST", "lesión subendocárdica inferior",
                   "inferior"),
    "INJIL": Entry("infarction", "Descenso del segmento ST",
                   "lesión subendocárdica inferolateral", "inferolateral"),
    "INJLA": Entry("infarction", "Descenso del segmento ST", "lesión subendocárdica lateral",
                   "lateral"),
    "AMI": Entry("infarction", "Ondas Q patológicas", "infarto de miocardio anterior",
                 "anterior"),
    "ASMI": Entry("infarction", "Ondas Q patológicas", "infarto de miocardio anteroseptal",
                  "anteroseptal"),
    "ALMI": Entry("infarction", "Ondas Q patológicas", "infarto de miocardio anterolateral",
                  "anterolateral"),
    "IMI": Entry("infarction", "Ondas Q patológicas", "infarto de miocardio inferior",
                 "inferior"),
    "ILMI": Entry("infarction", "Ondas Q patológicas", "infarto de miocardio inferolateral",
                  "inferolateral"),
    "LMI": Entry("infarction", "Ondas Q patológicas", "infarto de miocardio lateral", "lateral"),
    "IPMI": Entry("infarction", "Ondas Q patológicas con cambios recíprocos septales",
                  "infarto de miocardio inferoposterior", "inferoposterior"),
    "IPLMI": Entry("infarction", "Ondas Q patológicas con cambios recíprocos septales",
                   "infarto de miocardio inferoposterolateral", "inferoposterolateral"),
    "PMI": Entry("infarction", "Ondas R altas con descenso recíproco del segmento ST",
                 "infarto de miocardio posterior", "posterior"),
    "QWAVE": Entry("infarction", "Ondas Q patológicas", None),
    "ABQRS": Entry("infarction", "Morfología anormal del QRS", None),

    # -- normal --------------------------------------------------------------
    "NORM": Entry("normal", "Morfología, eje, intervalos y progresión de la onda R normales, "
                            "sin alteraciones significativas del segmento ST ni de la onda T",
                  "ECG normal"),
}
