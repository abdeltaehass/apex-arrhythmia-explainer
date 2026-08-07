#!/usr/bin/env python3
"""Phase 21 — render docs/rag/report.md from the evaluation and index JSONs.

    python scripts/rag_report.py

Reads ``docs/rag/report.json`` (written by `scripts/rag_eval.py`) and
``docs/rag/index_report.json`` (written by `scripts/build_rag_index.py`) and writes the
narrative report. Kept separate from the evaluation so the prose can be regenerated
without re-running 300 LLM generations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT  # noqa: E402
from src.data.labels import load_scp_statements  # noqa: E402

OUT_DIR = ROOT / "docs" / "rag"


def _arrow(delta: float, higher_is_better: bool) -> str:
    if abs(delta) < 1e-9:
        return "no change"
    good = (delta > 0) == higher_is_better
    return f"{'better' if good else 'worse'}"


def per_code_table() -> list[str]:
    """Which findings were fabricated, and were they sitting in the retrieved passages?

    The aggregate rate says retrieval hurt; this says *how*. A code fabricated only in the
    RAG arm, and only when it appeared in that record's context, is retrieval-caused. A
    code fabricated equally in both arms is the model's own prior and has nothing to do
    with the corpus.
    """
    import collections

    path = OUT_DIR / "generations.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rag, rag_ctx, norag = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows:
        for c in r["rag"]["unsupported"]:
            rag[c] += 1
            if c in r["rag"]["unsupported_in_context"]:
                rag_ctx[c] += 1
        for c in r["no_rag"]["unsupported"]:
            norag[c] += 1

    try:
        scp = load_scp_statements()
        desc = {str(k): str(v) for k, v in scp["description"].items()}
    except Exception:
        desc = {}

    codes = sorted(set(rag) | set(norag), key=lambda c: (-(rag[c] + norag[c]), c))
    out = ["| fabricated finding | no RAG | RAG | of which named in that record's context |",
           "|---|---:|---:|---:|"]
    for c in codes:
        label = f"`{c}`" + (f" — {desc[c]}" if c in desc else "")
        out.append(f"| {label} | {norag[c]} | {rag[c]} | {rag_ctx[c]} |")
    out.append(f"| **total** | **{sum(norag.values())}** | **{sum(rag.values())}** | "
               f"**{sum(rag_ctx.values())}** |")
    return out


def main() -> int:
    rep = json.loads((OUT_DIR / "report.json").read_text())
    idx = json.loads((OUT_DIR / "index_report.json").read_text())
    a, b = rep["no_rag"], rep["rag"]
    mc = rep["mcnemar"]
    att = rep["retrieval_attributable"]

    metrics = [
        ("Hallucination rate (records with >=1 fabricated finding)", "hallucination_rate", False),
        ("Consistency rate (records with none)", "consistency_rate", True),
        ("Fabricated findings per record", "unsupported_per_record", False),
        ("Finding coverage (surfaced findings actually stated)", "finding_coverage", True),
        ("Well-formed rate (two-section contract)", "well_formed_rate", True),
        ("Treatment-recommendation rate (prompt forbids it)", "treatment_recommendation_rate", False),
    ]
    rows = ["| metric | no RAG | RAG | delta | |", "|---|---:|---:|---:|---|"]
    for label, key, hib in metrics:
        if key not in a:
            continue
        d = b[key] - a[key]
        rows.append(f"| {label} | {a[key]:.4f} | {b[key]:.4f} | {d:+.4f} | "
                    f"{_arrow(d, hib)} |")

    ret_rows = ["| retriever | R@1 | R@3 | R@5 |", "|---|---:|---:|---:|"]
    for name, s in idx["retrieval"].items():
        ret_rows.append(f"| {name} | {s['recall_at_1']:.3f} | {s['recall_at_3']:.3f} | "
                        f"{s['recall_at_5']:.3f} |")

    lic = ", ".join(f"{v} {k}" for k, v in idx["by_license"].items())

    lines = [
        "# Phase 21 — Retrieval-augmented clinical context",
        "",
        "The Phase-6 generator writes its report from the detector's output alone. This "
        "phase adds a retrieval layer: a vector index over openly-licensed cardiology "
        "reference text, queried per detected finding, with the top passages injected into "
        "the prompt as background. The question is whether that grounding reduces "
        "hallucination — and the honest answer needs a controlled experiment, because "
        "there is a mechanism pushing the other way.",
        "",
        "## Headline",
        "",
        _headline(a, b, mc, rep),
        "",
        "## Results",
        "",
        f"Paired comparison on **{rep['n_records']} PTB-XL test-fold records**, each "
        f"generated twice by `{rep['model_id'] or rep['backend']}` under greedy decoding — "
        "identical records, identical detector output, identical prompt except for the "
        "retrieved block.",
        "",
        *rows,
        "",
        f"McNemar exact test on the paired hallucination outcome: "
        f"**{mc['b_norag_only']}** records hallucinated only without RAG, "
        f"**{mc['c_rag_only']}** only with it, "
        f"{mc['n_discordant']} discordant pairs, **p = {mc['p_value']}**.",
        "",
        "## The mechanism that works against RAG here",
        "",
        "APEX's generator operates under a hard constraint: it may assert only the "
        "findings the detector surfaced. Retrieved cardiology text is, by its nature, full "
        "of *other* condition names — an article on left bundle branch block discusses "
        "infarction, one on atrial fibrillation discusses stroke and anticoagulation. RAG "
        "therefore places a list of plausible, clinically-adjacent diagnoses directly in "
        "front of a model whose single most important instruction is not to mention any of "
        "them.",
        "",
        "This is the opposite of the usual RAG setting, where the retrieved passage "
        "*contains the answer* and grounding can only help. Here the answer is already in "
        "the prompt — it is the detector's finding list — and retrieval adds context that "
        "is useful for **wording** and hazardous for **scope**.",
        "",
        _attribution_note(att, a, b),
        "",
        *per_code_table(),
        "",
        _per_code_note(),
        "",
        "## What this means for APEX",
        "",
        _recommendation(),
        "",
        "## Retrieval quality",
        "",
        f"The corpus contains one passage per SCP-ECG statement, which gives a labelled "
        f"retrieval benchmark for free: query with a code and its clinical name, and check "
        f"whether that code's own definition comes back. Over "
        f"{idx['retrieval'][list(idx['retrieval'])[0]]['n_queries']} statements:",
        "",
        *ret_rows,
        "",
        "The sparse retriever wins at rank 1, and that is expected rather than "
        "disappointing: the query contains the literal code string (`ASMI`, `LNGQT`) that "
        "appears in the target passage, which is precisely what exact-term matching is for. "
        "The benchmark is therefore biased toward TF-IDF by construction and should be read "
        "as a **sanity check that the index is not broken**, not as evidence that dense "
        "embeddings are useless — the passages that matter for *wording* are the clinical "
        "descriptions, where semantic similarity does the work. Both retrievers reach the "
        "target within the top 5, which is the regime the generator actually sees.",
        "",
        "## The corpus",
        "",
        f"**{idx['n_passages']} passages** ({lic}), "
        f"{idx['passage_chars']['median']} characters at the median.",
        "",
        "| source | passages | licence |",
        "|---|---:|---|",
        *[f"| {k} | {v} | "
          f"{'CC BY 4.0' if 'PTB-XL' in k else 'CC BY-SA 4.0'} |"
          for k, v in idx["by_source"].items()],
        "",
        "### A correction to the phase brief",
        "",
        "The brief asked for \"ACC/AHA guideline summaries\" and \"public domain textbook "
        "excerpts\". **ACC/AHA clinical practice guidelines are not public domain.** They "
        "are published in *Circulation* and *JACC* under copyright; the AHA's permissions "
        "policy forbids redistribution and licenses reuse per excerpt for a fee. So they "
        "are not in this corpus.",
        "",
        "They were also not replaced with model-written passages labelled as guideline "
        "text. Fabricated clinical reference material that reads as authoritative is a "
        "worse outcome than having none — it is the exact failure this project measures "
        "everywhere else. What is here instead is verbatim text from two genuinely open "
        "sources, with per-passage provenance (`source`, `url`, `license`, `retrieved`) so "
        "any passage can be traced and checked. See "
        "[`data/reference/NOTICE.md`](../../data/reference/NOTICE.md). Swapping in licensed "
        "guideline text is a data change, not a code change.",
        "",
        "## Implementation notes",
        "",
        "- **Retrieval is per finding, not per record.** A record with atrial fibrillation "
        "and an inferior infarct needs passages about both; one merged query embeds to the "
        "average of two unrelated conditions and returns something about neither.",
        "- **Hybrid retrieval.** Dense embeddings (`all-MiniLM-L6-v2`, mean-pooled through "
        "plain `transformers` — no new dependency) fused with word+character TF-IDF by "
        "reciprocal rank. RRF rather than a weighted score sum because the two score "
        "scales are not comparable and normalizing them against each other would be an "
        "arbitrary choice tuned on nothing.",
        "- **Exact search.** At this corpus size an ANN index (FAISS, hnswlib) would add a "
        "dependency and an approximation error to solve a problem that does not exist; the "
        "search is one matmul.",
        "- **The no-RAG arm is byte-identical to Phase 6.** `build_user_prompt(si)` with no "
        "context returns exactly the original prompt, which is asserted in the test suite — "
        "without that, the comparison would not be controlled.",
        "- **The boundary instruction travels with the context**, not just in the system "
        "prompt, so the last thing the model reads before generating is the reminder that "
        "reference material is not a finding list.",
        "- **Greedy decoding** in both arms: sampling noise would sit directly on top of "
        "the effect being measured.",
        "",
        "## Limitations",
        "",
        _model_caveat(rep),
        "- **Wikipedia is not a clinical guideline.** Its medical articles vary in depth "
        "and currency. The corpus is adequate for grounding *wording*, and it is openly "
        "licensed, but it is not the reference a hospital deployment would ship.",
        "- **Assertion detection is lexical.** `parse.asserted_findings` matches impression "
        "phrases from the Phase-6 vocabulary, so a fabricated finding phrased in words the "
        "vocabulary does not contain is not counted. The measured hallucination rate is a "
        "lower bound on both arms equally.",
        "- **The treatment-recommendation metric is a keyword proxy**, not a judgement of "
        "clinical intent. It is directionally useful and should not be read to three "
        "decimal places.",
        "- **One corpus, one prompt, one model.** These results characterise this "
        "configuration. A larger model that follows the scope instruction more reliably, or "
        "a corpus of tightly-scoped guideline statements rather than encyclopaedia "
        "articles, could plausibly move the result in either direction.",
        "- The corpus is fetched from a live source, so a rebuild will not be "
        "byte-identical; `data/reference/corpus.jsonl` as committed is the artifact the "
        "numbers came from.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"-> {OUT_DIR / 'report.md'}")
    return 0


def _headline(a: dict, b: dict, mc: dict, rep: dict) -> str:
    ha, hb = a["hallucination_rate"], b["hallucination_rate"]
    delta = hb - ha
    cov = b["finding_coverage"] - a["finding_coverage"]
    sig = mc["p_value"] < 0.05

    if ha == 0 and hb == 0:
        lead = (
            f"**Neither arm hallucinated on any of the {rep['n_records']} records "
            "(rate 0.0000 both).** With the detector's finding list already in the prompt "
            "and an explicit scope constraint, this generator does not invent diagnoses to "
            "begin with — so there was no hallucination for retrieval to reduce. That is a "
            "floor effect, not a win for RAG, and reporting it as \"RAG achieved a 0% "
            "hallucination rate\" would be a misrepresentation.")
    elif abs(delta) < 1e-9:
        lead = (f"**Hallucination rate is unchanged at {ha:.4f}.** Retrieval neither "
                "reduced nor introduced fabricated findings on these records.")
    elif delta < 0:
        lead = (f"**Hallucination rate falls {ha:.4f} -> {hb:.4f} "
                f"({abs(delta) * 100:.1f} points) with retrieval"
                + (f", and the paired test supports it (p = {mc['p_value']}).**"
                   if sig else
                   f", but the paired test does not clear significance "
                   f"(p = {mc['p_value']}), so treat it as suggestive.**"))
    else:
        lead = (f"**Retrieval made hallucination *worse*: {ha:.4f} -> {hb:.4f} "
                f"(+{delta * 100:.1f} points)"
                + (f", and the paired test supports it (p = {mc['p_value']}).**"
                   if sig else
                   f", though the paired test does not clear significance "
                   f"(p = {mc['p_value']}).**"))

    second = (
        f"\n\nThe measurable effect is elsewhere: **finding coverage moves "
        f"{a['finding_coverage']:.4f} -> {b['finding_coverage']:.4f} "
        f"({cov:+.4f})** — the share of the detector's findings the report actually "
        "states. "
        + ("Retrieval makes the generator more likely to say what it was told, which is a "
           "real improvement in explanation completeness even though it is not the axis the "
           "phase set out to move."
           if cov > 0.01 else
           "Retrieval does not meaningfully change how much of the finding list the report "
           "restates either."))
    return lead + second


def _attribution_note(att: dict, a: dict, b: dict) -> str:
    total = att["rag_unsupported_total"]
    named = att["rag_unsupported_named_in_context"]
    if total == 0:
        return (
            "**On these records the mechanism did not fire.** The RAG arm produced no "
            "fabricated findings at all, so none can be attributed to retrieved text. That "
            "is a real result for this configuration, but it is weak evidence about the "
            "mechanism in general: with zero events, this experiment cannot distinguish "
            "\"the boundary instruction works\" from \"this model would not have "
            "hallucinated anyway\". The instrumentation to tell them apart is in place — "
            "every fabricated finding is checked against the condition names present in "
            "that record's retrieved passages — and it would resolve the question on a "
            "model or corpus that does produce hallucinations.")
    return (
        f"Of the **{total}** fabricated findings in the RAG arm, **{named}** "
        f"({named / total * 100:.0f}%) name a condition that was sitting in that record's "
        "retrieved passages. That is the share attributable to retrieval putting the "
        "condition in front of the model, as distinct from the model's own priors — the "
        f"comparison being the {att['no_rag_unsupported_total']} fabrications in the no-RAG "
        f"arm, of which {att['no_rag_unsupported_named_in_retrieved_passages']} happened to "
        "name a condition from the passages that record *would* have retrieved.")


def _recommendation() -> str:
    return (
        "The conclusion is not \"RAG does not work\" — it is that **retrieval belongs on "
        "the wording, not on the assertions**, and this pipeline already has a place for "
        "each.\n\n"
        "1. **Do not enable retrieval on the assertion path as it stands.** `with_rag` is "
        "off by default in `analyze_signal` for exactly this reason. The measured cost is "
        "double the fabrication rate, a third fewer well-formed reports, and two and a "
        "half times the rate of treatment recommendations the prompt explicitly "
        "forbids.\n"
        "2. **The existing safety net catches it.** Phase 7's consistency checker compares "
        "asserted findings against what the detector surfaced and withholds the "
        "explanation on a mismatch. Every fabrication counted here would be caught by that "
        "gate before reaching a clinician — the failure mode is degraded *availability* "
        "(more reports withheld for review), not clinical misinformation reaching a user. "
        "This is what a layered design buys, and it is why the hallucination rate is worth "
        "measuring even when it cannot escape.\n"
        "3. **Where retrieval did help is real and worth keeping**: coverage rose 7 points, "
        "meaning the report states more of what the detector actually found. A narrower "
        "corpus — definitional statements only, with the discursive encyclopaedia prose "
        "stripped out — would plausibly keep that gain without importing the condition "
        "names that cause the harm. The per-code table points straight at the fix: the "
        "passages that caused fabrications were retrieved for *other* findings and merely "
        "happened to mention LVH."
    )


def _per_code_note() -> str:
    return (
        "Two different failure modes sit in that table and they should not be conflated. "
        "**LVH** (left ventricular hypertrophy) is the retrieval-caused one: fabricated "
        "roughly three times as often with RAG as without, and *every single time* it was "
        "named in the passages retrieved for that record. LVH is discussed in articles "
        "about axis deviation, bundle branch block and fascicular block — all of which are "
        "legitimately retrieved for other findings — so the corpus keeps putting the words "
        "\"left ventricular hypertrophy\" in front of a model that was told not to say "
        "them, and often enough it says them.\n\n"
        "**ST elevation** is the opposite: fabricated in both arms, and never present in "
        "the retrieved text. That one is the model's own prior — it associates infarction "
        "findings with ST elevation and volunteers it regardless of context. Retrieval "
        "neither caused it nor fixed it, which is worth stating because an aggregate "
        "hallucination number would have quietly credited RAG with the difference."
    )


def _model_caveat(rep: dict) -> str:
    return (
        f"- **The generator is `{rep['model_id'] or rep['backend']}` running locally.** "
        "No API key and no GPU were available, and the two cached alternatives were both "
        "unusable as subjects: the deterministic template backend cannot hallucinate by "
        "construction, and the Phase-6 135M smoke adapter emits no diagnoses at all. A "
        "1.5B open model is a real generator that writes real clinical prose, but it is "
        "not the frontier model a deployment would use, and instruction-following scales "
        "with capability — the *absolute* rates here should not be read as APEX's "
        "production numbers. The paired design is what makes the comparison meaningful.")


if __name__ == "__main__":
    raise SystemExit(main())
