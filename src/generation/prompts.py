"""Prompt templates for the explanation generator.

The generator is constrained: it may ONLY describe findings passed to it in
`surfaced_findings` (each with its confidence and grounded lead/time region). This
constraint is what makes the consistency/hallucination checks meaningful — the
prompt forbids introducing any finding not in the structured input.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are APEX, a clinical decision-support assistant that explains \
12-lead ECG findings for a clinician audience. You are given a structured list of \
detected findings, each with a confidence score and the lead(s)/time window where \
the evidence appears. Write a concise, plain-English explanation.

Strict rules:
- Only mention findings that appear in the provided list. Never introduce a diagnosis \
  that is not in the list, even if it seems clinically related.
- For each finding, state the supporting lead(s) and what morphological feature drives it.
- Flag any finding whose confidence is below the review threshold as "requires \
  clinician confirmation".
- End with: "Decision support only — verify against the full clinical picture."
- Do not recommend treatment.
"""


def build_user_prompt(surfaced_findings: list[dict], review_threshold: float) -> str:
    """surfaced_findings: [{"label", "description", "confidence", "leads", "window"}]."""
    lines = [f"Review threshold: {review_threshold:.2f}", "Detected findings:"]
    for f in surfaced_findings:
        lines.append(
            f"- {f['label']} ({f['description']}): confidence={f['confidence']:.2f}, "
            f"leads={f.get('leads', 'n/a')}, window={f.get('window', 'n/a')}"
        )
    lines.append("\nWrite the explanation now.")
    return "\n".join(lines)
