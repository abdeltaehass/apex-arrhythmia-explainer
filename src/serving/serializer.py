"""Compose every stage's output into the Phase-8 structured report + validate input.

Three entry points:

- :func:`validate_signal` — the input gate. Rejects recordings with the wrong lead
  count; flags too-short recordings as potentially unreliable (still processed).
- :func:`build_report` — pure composition. Given the detection `StructuredInput`, the
  generated explanation text, and (optionally) a grounding map + reliability report,
  assembles an :class:`~src.serving.schema.APEXReport`. No torch, no model — this is
  what the schema-validation test suite exercises.
- :func:`analyze_signal` — the model-driven pipeline (validate -> preprocess -> detect
  -> generate -> optional grounding -> reliability -> serialize). Imports the heavy
  bits lazily so `build_report` stays usable without them.
"""

from __future__ import annotations

from src.config import CFG
from src.eval.reliability import ReliabilityReport, check_reliability
from src.generation.parse import asserted_findings, parse_report
from src.generation.templater import StructuredInput
from src.serving.schema import (
    DISCLAIMER,
    MIN_LEADS,
    MIN_RELIABLE_SECONDS,
    APEXReport,
    ConsistencyOut,
    FindingOut,
    Flag,
    FlagType,
    InputValidation,
    InputValidationError,
)


# --- Input validation --------------------------------------------------------
def _signal_shape(signal) -> tuple[int, int]:
    """Return ``(n_leads, n_samples)`` for a list-of-lists or an array-like ``(L, T)``.

    Raises ``InputValidationError`` for structurally unusable input (empty, or leads of
    unequal length) — those can't be meaningfully processed at all.
    """
    shape = getattr(signal, "shape", None)
    if shape is not None:  # numpy array / tensor
        if len(shape) != 2:
            raise InputValidationError(InputValidation(
                n_leads=0, duration_s=0.0, sampling_rate=0, ok=False, reliable=False,
                errors=[f"expected a 2-D (leads, samples) signal, got shape {tuple(shape)}"],
            ))
        return int(shape[0]), int(shape[1])
    # list of lists
    n_leads = len(signal)
    if n_leads == 0:
        raise InputValidationError(InputValidation(
            n_leads=0, duration_s=0.0, sampling_rate=0, ok=False, reliable=False,
            errors=["empty signal (no leads)"],
        ))
    lengths = {len(lead) for lead in signal}
    if len(lengths) != 1:
        raise InputValidationError(InputValidation(
            n_leads=n_leads, duration_s=0.0, sampling_rate=0, ok=False, reliable=False,
            errors=[f"leads have unequal sample counts: {sorted(lengths)}"],
        ))
    return n_leads, lengths.pop()


def validate_signal(signal, sampling_rate: int) -> InputValidation:
    """Validate a raw recording against the Phase-8 rules.

    Hard rejects (``ok=False``): fewer than ``MIN_LEADS`` (12) leads — or more, since
    the detector's input conv is fixed at 12 — and a non-positive sampling rate.
    Soft flag (``reliable=False``, still processed): a recording shorter than
    ``MIN_RELIABLE_SECONDS`` (5 s).
    """
    n_leads, n_samples = _signal_shape(signal)
    errors, warnings = [], []

    if sampling_rate <= 0:
        errors.append(f"sampling_rate must be positive, got {sampling_rate}")
        duration_s = 0.0
    else:
        duration_s = n_samples / sampling_rate

    if n_leads < MIN_LEADS:
        errors.append(f"recording has {n_leads} leads; a 12-lead ECG is required")
    elif n_leads > MIN_LEADS:
        errors.append(f"recording has {n_leads} leads; the detector expects exactly {MIN_LEADS}")

    if sampling_rate > 0 and duration_s < MIN_RELIABLE_SECONDS:
        warnings.append(
            f"recording is {duration_s:.1f}s (< {MIN_RELIABLE_SECONDS:.0f}s): "
            "too short to be reliable — interpret with caution"
        )

    ok = not errors
    return InputValidation(
        n_leads=n_leads, duration_s=round(duration_s, 3), sampling_rate=sampling_rate,
        ok=ok, reliable=ok and not warnings, errors=errors, warnings=warnings,
    )


# --- Composition -------------------------------------------------------------
def _flags_by_code(reliability: ReliabilityReport) -> dict[str, list[Flag]]:
    """Attribute each reliability flag to the finding code(s) it concerns."""
    out: dict[str, list[Flag]] = {}
    for f in reliability.low_confidence:
        out.setdefault(f.code, []).append(Flag(type=FlagType.LOW_CONFIDENCE, message=f.message))
    for g in reliability.grounding_conflicts:
        out.setdefault(g.code, []).append(Flag(type=FlagType.GROUNDING_CONFLICT, message=g.message))
    for m in reliability.mutual_exclusivity:
        for code in (m.code_a, m.code_b):
            out.setdefault(code, []).append(Flag(type=FlagType.MUTUAL_EXCLUSIVITY, message=m.message))
    return out


def build_report(
    structured_input: StructuredInput,
    explanation: str,
    reliability: ReliabilityReport | None = None,
    input_validation: InputValidation | None = None,
    record_id: str | None = None,
    review_threshold: float = CFG.review_threshold,
    low_confidence_threshold: float = CFG.low_confidence_threshold,
    grounded_leads: dict[str, list[str]] | None = None,
    saliency_by_code: dict | None = None,
) -> APEXReport:
    """Assemble the structured report. Pure composition — no model calls.

    ``structured_input``: the detector's surfaced findings (codes, confidences, leads).
    ``explanation``: the full generated two-section text.
    ``reliability``: a precomputed :class:`ReliabilityReport`; if ``None`` it is computed
    here from the structured input + parsed explanation (+ grounding, if given).
    ``input_validation``: attached to the report and, if the input was flagged
    unreliable, forces ``review_recommended`` and adds an ``unreliable_input`` flag.
    """
    surfaced = set(structured_input.codes())
    confidences = {f.code: f.confidence for f in structured_input.findings if f.confidence is not None}
    asserted = asserted_findings(explanation)
    parsed = parse_report(explanation)

    if reliability is None:
        rid = record_id or (
            str(structured_input.record_id) if structured_input.record_id is not None else "?"
        )
        reliability = check_reliability(
            rid, asserted, surfaced, confidences,
            leads_by_code=grounded_leads, saliency_by_code=saliency_by_code,
            low_confidence_threshold=low_confidence_threshold,
        )

    flags_by_code = _flags_by_code(reliability)
    findings_out: list[FindingOut] = []
    for f in structured_input.findings:
        flags = flags_by_code.get(f.code, [])
        conf = f.confidence if f.confidence is not None else 0.0
        needs_review = conf < review_threshold or bool(flags)
        findings_out.append(FindingOut(
            label=f.code, description=f.description, confidence=round(float(conf), 4),
            leads=list(f.leads), flags=flags, needs_review=needs_review,
        ))

    consistency = ConsistencyOut(
        consistent=reliability.consistency.consistent,
        asserted=sorted(reliability.consistency.asserted),
        surfaced=sorted(reliability.consistency.surfaced),
        unsupported=sorted(reliability.consistency.unsupported),
    )

    unreliable = input_validation is not None and not input_validation.reliable
    review_recommended = (
        any(f.needs_review for f in findings_out)
        or not consistency.consistent
        or reliability.any_flag
        or unreliable
    )

    report = APEXReport(
        findings=findings_out,
        impression=parsed.impression,
        explanation=explanation,
        consistency=consistency,
        review_recommended=review_recommended,
        input_validation=input_validation,
        disclaimer=DISCLAIMER,
    )
    # Surface an unreliable-input flag on the first finding (or none exist -> still gated
    # via review_recommended above) so the reason travels with the finding list too.
    if unreliable and report.findings:
        report.findings[0].flags.append(Flag(
            type=FlagType.UNRELIABLE_INPUT,
            message="; ".join(input_validation.warnings) or "recording flagged potentially unreliable",
        ))
        report.findings[0].needs_review = True
    return report


# --- Model-driven pipeline ---------------------------------------------------
def analyze_signal(
    signal,
    sampling_rate: int = 100,
    checkpoint=None,
    backend: str = "template",
    with_grounding: bool = False,
    device: str = "cpu",
) -> APEXReport:
    """Full pipeline: validate -> preprocess -> detect -> generate -> [ground] -> serialize.

    Raises :class:`InputValidationError` if the recording fails a hard rule. ``backend``
    picks the explanation generator (``"template"`` is deterministic and needs no LLM;
    ``"claude"``/``"local"`` need their deps). ``with_grounding`` runs the (more
    expensive) per-lead saliency so grounding-conflict flags are populated.
    """
    import numpy as np

    validation = validate_signal(signal, sampling_rate)
    if not validation.ok:
        raise InputValidationError(validation)

    from src.data.labels import load_scp_statements
    from src.generation.inference import generate_explanation
    from src.generation.templater import build_structured_input
    from src.grounding import load_detector
    from src.preprocessing.pipeline import preprocess

    raw = np.asarray(signal, dtype=np.float32)
    clean, _ = preprocess(raw, fs_in=sampling_rate, fs_out=CFG.sampling_rate)

    model, label_space, _ = load_detector(checkpoint, device=device) if checkpoint \
        else load_detector(device=device)

    import torch

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(clean).unsqueeze(0).to(device)))[0].cpu().numpy()

    surfaced = [label_space[j] for j in range(len(label_space)) if probs[j] >= CFG.review_threshold]
    confidences = {c: float(probs[label_space.index(c)]) for c in surfaced}
    scp = load_scp_statements()
    descriptions = {c: (scp.loc[c, "description"] if c in scp.index else "") for c in surfaced}

    si = build_structured_input(
        surfaced, confidences=confidences, descriptions=descriptions,
        review_threshold=CFG.review_threshold,
    )
    explanation = generate_explanation(si, backend=backend)

    grounded_leads = saliency_by_code = None
    if with_grounding:
        from src.generation.vocab import leads_for
        from src.grounding import ground

        localizing = [c for c in surfaced if leads_for(c)]
        if localizing:
            saliency_by_code = ground(model, clean, localizing, label_space=label_space,
                                     fs=CFG.sampling_rate)
            grounded_leads = {c: leads_for(c) for c in localizing}

    return build_report(
        si, explanation, input_validation=validation,
        grounded_leads=grounded_leads, saliency_by_code=saliency_by_code,
    )
