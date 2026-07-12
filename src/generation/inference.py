"""Explanation generation inference.

Turns the detector's surfaced findings (+ grounding regions) into a plain-English
clinical explanation. v1 backend is the Anthropic Claude API; a fine-tuned local
model can be swapped behind the same `generate_explanation` interface.

The generated text is always passed through `src/eval/consistency.py` before being
shown, so a fabricated finding is caught even if the model ignores the prompt.
"""

from __future__ import annotations

from .prompts import SYSTEM_PROMPT, build_user_prompt


def generate_explanation(
    surfaced_findings: list[dict],
    review_threshold: float = 0.5,
    model: str = "claude-fable-5",
) -> str:
    """Generate an explanation. Requires ANTHROPIC_API_KEY in the environment.

    Kept intentionally thin — the interesting logic (constraint enforcement) lives
    in the prompt and the downstream consistency check, not here.
    """
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install anthropic to use the Claude backend") from e

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(surfaced_findings, review_threshold)}],
    )
    return resp.content[0].text
