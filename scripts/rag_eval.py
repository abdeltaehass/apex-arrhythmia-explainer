#!/usr/bin/env python3
"""Phase 21 — does retrieved clinical context reduce hallucination?

    python scripts/rag_eval.py --n 120
    python scripts/rag_eval.py --n 20 --backend template   # harness check (rate is 0 by construction)

A **paired** experiment. The same test-fold records, the same detector output, the same
greedy decoding, the same model — the single difference is whether the retrieved reference
block is present in the prompt. Pairing matters: hallucination varies enormously by record
(a record with six findings has far more opportunity than one with one), so comparing two
independent samples would drown the effect in case-mix. Every record is generated twice and
the outcome compared record by record, which also licenses McNemar's test on the
discordant pairs rather than a two-proportion test that assumes independence.

Metrics, all computed by the Phase-6/7 machinery rather than anything invented here:

- **hallucination rate** — fraction of records asserting at least one finding the detector
  never surfaced (`eval.hallucination.hallucination_rate`);
- **consistency rate** — fraction with no unsupported assertion at all
  (`eval.consistency`);
- **finding coverage** — of the findings the detector *did* surface, how many the report
  actually states. This is the other half of "explanation accuracy": a report that
  mentions nothing cannot hallucinate, and coverage is what stops that from looking like
  success;
- **well-formed rate** — did the output follow the two-section contract at all;
- **retrieval-attributable hallucinations** — of the fabricated findings, how many were
  named in the passages RAG put in the prompt. This is the measurement that decides
  whether retrieval *caused* a hallucination rather than merely failing to prevent one.

Writes docs/rag/report.{md,json} and per-record outputs to docs/rag/generations.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import CFG, ROOT  # noqa: E402
from src.data.labels import build_label_space, load_database, load_scp_statements  # noqa: E402
from src.detection.data_cache import build_split_cache  # noqa: E402
from src.eval.consistency import check, consistency_rate  # noqa: E402
from src.eval.hallucination import hallucination_rate  # noqa: E402
from src.generation.inference import generate_explanation  # noqa: E402
from src.generation.parse import asserted_findings, parse_report  # noqa: E402
from src.generation.templater import build_structured_input  # noqa: E402
from src.generation.vocab import impression_terms, leads_for  # noqa: E402
from src.grounding import load_detector  # noqa: E402
from src.rag import format_context, load_index, retrieve_for_findings  # noqa: E402

OUT_DIR = ROOT / "docs" / "rag"


def structured_inputs(n: int, seed: int = 42) -> list:
    """Build `StructuredInput`s from *real detector output* on the test fold.

    Deliberately not from ground-truth labels: the generator in deployment sees what the
    detector surfaced, including its mistakes, and the hallucination question is about
    what the generator does with that input.
    """
    X, _ = build_split_cache("test", 100)
    df = load_database()
    test_rows = df[df["strat_fold"] == 10]
    scp = load_scp_statements()
    label_space = build_label_space()

    import torch

    model, _, _ = load_detector(device="cpu")
    rng = np.random.default_rng(seed)
    # Sample records that have at least one finding: an empty findings list gives the
    # generator nothing to be wrong about and would dilute both arms equally but pointlessly.
    order = rng.permutation(len(X))
    out, ecg_ids = [], []
    for i in order:
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(X[i:i + 1])))[0].numpy()
        surfaced = [label_space[j] for j in range(len(label_space))
                    if probs[j] >= CFG.review_threshold]
        if not surfaced:
            continue
        row = test_rows.iloc[int(i)]
        si = build_structured_input(
            surfaced,
            confidences={c: float(probs[label_space.index(c)]) for c in surfaced},
            descriptions={c: (scp.loc[c, "description"] if c in scp.index else "")
                          for c in surfaced},
            leads_by_code={c: leads_for(c) for c in surfaced if leads_for(c)},
            review_threshold=CFG.review_threshold,
            age=None if np.isnan(row.get("age", np.nan)) else int(row["age"]),
            sex={0: "male", 1: "female"}.get(row.get("sex")),
        )
        out.append(si)
        ecg_ids.append(int(test_rows.index[int(i)]))
        if len(out) >= n:
            break
    return list(zip(out, ecg_ids, strict=True))


# The prompt forbids treatment recommendations outright ("Do not recommend treatment").
# Keyword matching is a crude proxy for whether that held, but it is a real safety-relevant
# instruction and a cheap way to see whether adding a page of clinical background — much of
# which discusses management — erodes compliance with it.
TREATMENT_CUES = (
    "anticoagul", "warfarin", "apixaban", "rivaroxaban", "heparin", "aspirin",
    "beta-blocker", "beta blocker", "statin", "amiodarone", "digoxin therapy",
    "cardioversion", "ablation", "pacemaker implant", "angioplasty", "stent",
    "bypass", "recommend treatment", "should be treated", "initiate therapy",
    "refer to", "further evaluation", "echocardiogra", "angiograph", "consider treatment",
)


def recommends_treatment(text: str) -> bool:
    low = text.lower()
    return any(cue in low for cue in TREATMENT_CUES)


def score(text: str, si, ctx_terms: set[str]) -> dict:
    """Score one generated report against what the detector surfaced."""
    surfaced = set(si.codes())
    asserted = asserted_findings(text)
    cons = check(asserted, surfaced)
    parsed = parse_report(text)
    covered = asserted & surfaced
    return {
        "asserted": sorted(asserted),
        "unsupported": sorted(cons.unsupported),
        "n_unsupported": len(cons.unsupported),
        "consistent": cons.consistent,
        "coverage": len(covered) / len(surfaced) if surfaced else float("nan"),
        "well_formed": parsed.well_formed,
        "recommends_treatment": recommends_treatment(text),
        # of the fabricated findings, which were sitting in the retrieved passages?
        "unsupported_in_context": sorted(cons.unsupported & ctx_terms),
        "consistency_result": cons,
    }


def mcnemar(pairs: list[tuple[bool, bool]]) -> dict:
    """Exact McNemar test on paired binary outcomes (hallucinated: no-RAG vs RAG).

    Only the discordant pairs carry information: records where both arms hallucinated, or
    neither did, say nothing about whether retrieval changed anything. The exact binomial
    form is used because the discordant count here is small enough that the chi-square
    approximation would be unreliable.
    """
    from math import comb

    b = sum(1 for a, c in pairs if a and not c)   # hallucinated without RAG only
    c_ = sum(1 for a, c in pairs if c and not a)  # hallucinated with RAG only
    n = b + c_
    if n == 0:
        return {"b_norag_only": 0, "c_rag_only": 0, "n_discordant": 0, "p_value": 1.0}
    k = min(b, c_)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"b_norag_only": b, "c_rag_only": c_, "n_discordant": n, "p_value": round(p, 5)}


def summarize(rows: list[dict], arm: str) -> dict:
    r = [x[arm] for x in rows]
    flags = [x[arm]["_flag"] for x in rows]
    cov = [x["coverage"] for x in r if not np.isnan(x["coverage"])]
    return {
        "n": len(r),
        "hallucination_rate": round(hallucination_rate(flags), 4),
        "consistency_rate": round(consistency_rate([x["consistency_result"] for x in r]), 4),
        "unsupported_per_record": round(float(np.mean([x["n_unsupported"] for x in r])), 3),
        "finding_coverage": round(float(np.mean(cov)), 4) if cov else float("nan"),
        "well_formed_rate": round(float(np.mean([x["well_formed"] for x in r])), 4),
        "treatment_recommendation_rate": round(
            float(np.mean([x["recommends_treatment"] for x in r])), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=120, help="records (each generated twice)")
    ap.add_argument("--backend", default="hf", choices=("hf", "local", "claude", "template"))
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--k-per-finding", type=int, default=2)
    ap.add_argument("--max-passages", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"building {args.n} structured inputs from detector output...")
    cases = structured_inputs(args.n, args.seed)
    print(f"  {len(cases)} records, "
          f"mean {np.mean([len(si.findings) for si, _ in cases]):.1f} findings/record")

    index = load_index()
    terms = impression_terms()                      # phrase -> code
    code_to_term = {c: t for t, c in terms.items()}  # code -> phrase

    gen_kwargs = {}
    if args.backend == "hf":
        gen_kwargs = {"model_id": args.model_id, "max_new_tokens": args.max_new_tokens}

    rows, t0 = [], time.time()
    for i, (si, ecg_id) in enumerate(cases, 1):
        ctx = retrieve_for_findings(si, index, k_per_finding=args.k_per_finding,
                                    max_passages=args.max_passages)
        context_block = format_context(ctx)
        blob = " ".join(h.passage.text.lower() for h in ctx.hits)
        ctx_terms = {c for c, t in code_to_term.items() if t and t.lower() in blob}

        txt_no = generate_explanation(si, backend=args.backend, context="", **gen_kwargs)
        txt_rag = generate_explanation(si, backend=args.backend, context=context_block,
                                       **gen_kwargs)

        s_no = score(txt_no, si, ctx_terms)
        s_rag = score(txt_rag, si, ctx_terms)
        for s in (s_no, s_rag):
            s["_flag"] = type("F", (), {"unsupported_findings": set(s["unsupported"])})()

        rows.append({
            "ecg_id": ecg_id,
            "surfaced": sorted(si.codes()),
            "n_findings": len(si.findings),
            "retrieved": ctx.sources(),
            "context_condition_codes": sorted(ctx_terms),
            "no_rag": s_no, "rag": s_rag,
            "text_no_rag": txt_no, "text_rag": txt_rag,
        })
        if i % 10 == 0 or i == len(cases):
            el = time.time() - t0
            print(f"  {i}/{len(cases)}  ({el:.0f}s, {el / i:.1f}s/record)")

    no_rag = summarize(rows, "no_rag")
    rag = summarize(rows, "rag")
    mc = mcnemar([(bool(r["no_rag"]["unsupported"]), bool(r["rag"]["unsupported"]))
                  for r in rows])

    attributable = sum(len(r["rag"]["unsupported_in_context"]) for r in rows)
    total_rag_unsupported = sum(r["rag"]["n_unsupported"] for r in rows)
    attributable_no = sum(len(r["no_rag"]["unsupported_in_context"]) for r in rows)
    total_no_unsupported = sum(r["no_rag"]["n_unsupported"] for r in rows)

    payload = {
        "backend": args.backend,
        "model_id": args.model_id if args.backend == "hf" else None,
        "n_records": len(rows),
        "k_per_finding": args.k_per_finding, "max_passages": args.max_passages,
        "threshold": CFG.review_threshold,
        "no_rag": no_rag, "rag": rag, "mcnemar": mc,
        "retrieval_attributable": {
            "rag_unsupported_total": total_rag_unsupported,
            "rag_unsupported_named_in_context": attributable,
            "no_rag_unsupported_total": total_no_unsupported,
            "no_rag_unsupported_named_in_retrieved_passages": attributable_no,
        },
        "mean_passages_retrieved": round(float(np.mean([len(r["retrieved"]) for r in rows])), 2),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(payload, indent=2))
    with (OUT_DIR / "generations.jsonl").open("w") as f:
        for r in rows:
            slim = {k: v for k, v in r.items() if k not in ("no_rag", "rag")}
            for arm in ("no_rag", "rag"):
                slim[arm] = {k: v for k, v in r[arm].items()
                             if k not in ("consistency_result", "_flag")}
            f.write(json.dumps(slim) + "\n")

    print(f"\n{'metric':<28}{'no RAG':>10}{'RAG':>10}{'delta':>10}")
    for key in ("hallucination_rate", "consistency_rate", "unsupported_per_record",
                "finding_coverage", "well_formed_rate", "treatment_recommendation_rate"):
        a, b = no_rag[key], rag[key]
        print(f"{key:<28}{a:>10.4f}{b:>10.4f}{b - a:>+10.4f}")
    print(f"\nMcNemar: b(no-RAG only)={mc['b_norag_only']} c(RAG only)={mc['c_rag_only']} "
          f"p={mc['p_value']}")
    print(f"-> {OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
