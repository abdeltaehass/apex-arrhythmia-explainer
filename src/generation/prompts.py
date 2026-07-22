"""Prompt templates for the explanation generator.

The generator is constrained: it may ONLY describe findings passed to it in the
structured input (each with its confidence and, once grounded, the lead/time region
supporting it). This constraint is what makes the consistency/hallucination checks in
`src/eval/` meaningful — the prompt forbids introducing any finding not in the
structured input — and it is what the fine-tuning data in `templater.py` teaches by
example, since every template target only ever states the findings it was given.

The exact two-section contract (``Findings:`` / ``Impression:``) is shared by every
backend — the Claude API path (`generation/inference.py`), the fine-tuned local model,
and the deterministic template — so `src/generation/parse.py` can parse all three the
same way.
"""

from __future__ import annotations

from src.generation.templater import StructuredInput

SYSTEM_PROMPT = """You are APEX, a clinical decision-support assistant that writes \
12-lead ECG reports for a clinician audience. You are given a structured list of \
detected findings, each with a confidence score and (when available) the lead(s) \
where the evidence appears. Write the report in exactly two sections, in this order, \
with no other text before or after:

Findings:
<factual, morphological observations only — what the tracing shows, and in which \
leads. One clause per finding.>

Impression:
<the interpretive summary — the named diagnosis/diagnoses the findings support, in \
cardiologist register.>

Strict rules:
- Only mention findings that appear in the provided list. Never introduce a diagnosis \
  that is not in the list, even if it seems clinically related.
- For each finding, state the supporting lead(s) if given.
- Append "(requires clinician confirmation)" to any finding whose confidence is below \
  the review threshold.
- If nothing in the list is an acute ischemic or infarction finding, end the \
  Impression with "No acute ischemic changes identified."
- Do not recommend treatment. Do not add a disclaimer — the caller appends it.
"""


def serialize_structured_input(si: StructuredInput) -> str:
    """The structured-detection block shared by every backend's user turn.

    Deterministic and information-preserving: this is the *sole* view of the case the
    generator sees, so anything not serialized here cannot legitimately appear in the
    output. Used verbatim as the SFT input (`dataset.py`) and the Claude user prompt.
    """
    lines = []
    demo = []
    if si.age is not None:
        demo.append(f"{int(si.age)}y")
    if si.sex:
        demo.append(str(si.sex))
    if demo:
        lines.append(f"Patient: {' '.join(demo)}")
    ctx = []
    if si.heart_rate_bpm is not None:
        ctx.append(f"rate {round(si.heart_rate_bpm)} bpm")
    if si.heart_axis and str(si.heart_axis).lower() not in ("nan", "none", ""):
        ctx.append(f"axis {si.heart_axis}")
    if ctx:
        lines.append(" | ".join(ctx))
    lines.append(f"Review threshold: {si.review_threshold:.2f}")
    lines.append("Detected findings:")
    if not si.findings:
        lines.append("- (none surfaced)")
    for f in si.findings:
        leads = f"leads={','.join(f.leads)}" if f.leads else "leads=n/a"
        conf = f"confidence={f.confidence:.2f}" if f.confidence is not None else "confidence=n/a"
        desc = f" ({f.description})" if f.description else ""
        lines.append(f"- {f.code}{desc}: {conf}, {leads}")
    return "\n".join(lines)


def build_user_prompt(si: StructuredInput) -> str:
    return serialize_structured_input(si) + "\n\nWrite the report now."


def target_text(findings: str, impression: str) -> str:
    """Render a report's two sections into the exact text the model is trained to emit."""
    return f"Findings:\n{findings}\n\nImpression:\n{impression}"


def build_chat_example(si: StructuredInput, findings: str, impression: str) -> list[dict]:
    """One supervised fine-tuning example as a chat-format message list.

    ``findings``/``impression`` are the target section texts (from
    `templater.render_report`, or a human reference). Handy for local inference
    (`generation/inference.py`) and inspection; `train_lora.py` trains from
    :func:`build_prompt_completion_example` instead (see its docstring for why).
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(si)},
        {"role": "assistant", "content": target_text(findings, impression)},
    ]


def build_prompt_completion_example(si: StructuredInput, findings: str, impression: str) -> dict:
    """One SFT example as TRL's "prompt-completion" pair: ``{"prompt": [...], "completion": [...]}``.

    Both are message lists (the system+user turns, and the assistant turn) rather than
    a flat string. This shape lets `SFTConfig(completion_only_loss=True)` mask the loss
    down to just the target report tokens by tokenizing ``prompt`` alone (with
    ``add_generation_prompt=True``) vs. ``prompt + completion`` and taking the length
    difference — no dependency on a chat template exposing TRL's
    ``{% generation %}`` masking markers, which many open instruct models' templates
    (including tiny local test models) don't define. `build_chat_example` gives the
    same conversation as one flat list for backends that just want to call the model.
    """
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(si)},
        ],
        "completion": [
            {"role": "assistant", "content": target_text(findings, impression)},
        ],
    }
