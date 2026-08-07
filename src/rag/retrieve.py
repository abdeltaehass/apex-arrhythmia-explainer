"""Phase 21 — turn detected findings into retrieved clinical context for the prompt.

The retrieval unit is **one finding, not one record**. A record carrying atrial
fibrillation and an inferior infarct needs passages about both; a single merged query
embeds to the average of two unrelated ideas and reliably returns something about neither.
So each finding gets its own query and the results are fused, deduplicated, and capped.

:func:`format_context` renders the passages into the prompt block. It numbers them and
attaches source labels for two reasons: a clinician reading the output can check where a
statement came from, and the numbering gives the generator something to point at instead
of paraphrasing the passage as if it were its own knowledge.

**The instruction that has to travel with the context.** Retrieved cardiology text is full
of condition names the detector did not surface — an article about left bundle branch block
discusses infarction, an article on atrial fibrillation discusses stroke. Injecting that
into a prompt whose one hard rule is "only assert findings from the provided list" is
actively adversarial to the rule, so :data:`CONTEXT_INSTRUCTION` restates the boundary at
the point where it is under the most pressure. Whether that is enough is an empirical
question, and Phase 21's evaluation measures it rather than assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.generation.templater import StructuredInput
from src.rag.index import Hit

CONTEXT_INSTRUCTION = (
    "The reference material below is background on the conditions listed above. It is "
    "provided so your wording is clinically accurate. It is NOT a list of findings for "
    "this patient. Reference passages mention many conditions this patient does not have "
    "— never report a condition because it appears in the reference material. The "
    "detected-findings list above remains the only source of what this patient has."
)


@dataclass
class RetrievedContext:
    """The passages injected for one record, kept alongside the text for auditing."""

    hits: list[Hit]
    queries: list[str]

    @property
    def passage_ids(self) -> list[str]:
        return [h.passage.id for h in self.hits]

    def sources(self) -> list[dict]:
        return [{"id": h.passage.id, "title": h.passage.title, "source": h.passage.source,
                 "license": h.passage.license, "url": h.passage.url,
                 "score": round(h.score, 5)} for h in self.hits]


def build_query(code: str, description: str | None = None,
                leads: list[str] | None = None) -> str:
    """Query text for one finding: the code, its clinical name, and the leads involved.

    The code itself is kept in the query even though it is opaque jargon — it is exactly
    the token that matches the PTB-XL statement definition passage, which is the single
    most reliably correct thing the corpus can return for a finding.
    """
    parts = [code]
    if description:
        parts.append(description)
    if leads:
        parts.append("leads " + ", ".join(leads))
    return " ".join(parts)


def retrieve_for_findings(
    si: StructuredInput,
    index,
    k_per_finding: int = 2,
    max_passages: int = 6,
) -> RetrievedContext:
    """Retrieve context for every finding on a record, fused and capped.

    ``max_passages`` exists because prompt budget is finite and because more context is
    not monotonically better here: each extra passage is more unrelated condition names in
    front of a model that has been told not to mention them.
    """
    seen: dict[str, Hit] = {}
    queries: list[str] = []
    for f in si.findings:
        q = build_query(f.code, f.description, f.leads)
        queries.append(q)
        for hit in index.search(q, k=k_per_finding):
            prev = seen.get(hit.passage.id)
            if prev is None or hit.score > prev.score:
                seen[hit.passage.id] = hit
    ordered = sorted(seen.values(), key=lambda h: -h.score)[:max_passages]
    ordered = [Hit(h.passage, h.score, r) for r, h in enumerate(ordered)]
    return RetrievedContext(hits=ordered, queries=queries)


def format_context(ctx: RetrievedContext, max_chars_per_passage: int = 700) -> str:
    """Render retrieved passages into the prompt's reference block."""
    if not ctx.hits:
        return ""
    lines = ["Reference material (background only):"]
    for i, h in enumerate(ctx.hits, 1):
        text = h.passage.text
        if len(text) > max_chars_per_passage:
            text = text[:max_chars_per_passage].rsplit(" ", 1)[0] + "..."
        lines.append(f"[{i}] {h.passage.title} — {h.passage.source}\n{text}")
    lines.append("")
    lines.append(CONTEXT_INSTRUCTION)
    return "\n\n".join(lines)


def context_condition_names(ctx: RetrievedContext, vocab_terms: dict[str, str]) -> set[str]:
    """SCP codes whose clinical phrase literally appears in the retrieved passages.

    This is the measurement that makes the RAG failure mode visible: if the generator
    asserts a finding the detector never surfaced, was that condition sitting in the
    context window? Codes in this set are the ones RAG *put in front of the model*, so a
    hallucination among them is attributable to retrieval rather than to the model's
    priors.
    """
    blob = " ".join(h.passage.text.lower() for h in ctx.hits)
    return {code for code, term in vocab_terms.items() if term and term.lower() in blob}
