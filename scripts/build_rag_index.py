#!/usr/bin/env python3
"""Phase 21 — fetch the reference corpus, build the vector index, and score retrieval.

    python scripts/build_rag_index.py                 # fetch + index + evaluate
    python scripts/build_rag_index.py --no-fetch      # reindex the corpus already on disk
    python scripts/build_rag_index.py --sparse-only   # skip the embedding model

Fetching hits Wikipedia's API once per article with a polite delay, so a full build takes
a couple of minutes. `--no-fetch` reuses `data/reference/corpus.jsonl`.

**Retrieval is scored, not assumed.** The corpus contains one passage per SCP-ECG
statement, which gives a free labelled retrieval benchmark: query with a code and its
clinical name, and check whether that code's own definition comes back. It is an easy
task by construction — the point is not to celebrate a high number but to catch a broken
index, since a retriever that silently returns the wrong passages would still produce a
plausible-looking RAG report. Recall@k is written into the report so the generation
results can be read knowing what the retriever was actually doing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import ROOT  # noqa: E402
from src.data.labels import load_scp_statements  # noqa: E402
from src.rag.corpus import build_corpus, load_corpus  # noqa: E402
from src.rag.index import DenseIndex, HybridIndex, build_indexes  # noqa: E402
from src.rag.retrieve import build_query  # noqa: E402

OUT_DIR = ROOT / "docs" / "rag"


def score_retrieval(index, ks=(1, 3, 5)) -> dict:
    """Recall@k for "find the definition of this SCP code"."""
    scp = load_scp_statements()
    hits_at = {k: 0 for k in ks}
    n = 0
    misses = []
    for code, row in scp.iterrows():
        desc = str(row.get("description") or "").strip()
        if not desc:
            continue
        n += 1
        got = [h.passage.id for h in index.search(build_query(str(code), desc), k=max(ks))]
        target = f"scp::{code}"
        for k in ks:
            if target in got[:k]:
                hits_at[k] += 1
        if target not in got:
            misses.append(str(code))
    return {
        "n_queries": n,
        **{f"recall_at_{k}": round(hits_at[k] / n, 4) for k in ks},
        "missed_entirely": misses,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-fetch", action="store_true", help="reuse the corpus on disk")
    ap.add_argument("--sparse-only", action="store_true", help="skip dense embeddings")
    args = ap.parse_args()

    passages = load_corpus() if args.no_fetch else build_corpus()
    dense, sparse = build_indexes(passages, dense=not args.sparse_only)

    variants = {"tfidf (sparse)": sparse}
    if dense is not None:
        variants["minilm (dense)"] = dense
        variants["hybrid (RRF)"] = HybridIndex(dense, sparse)

    print("\nretrieval quality — 'find this SCP code's definition':")
    scores = {}
    for name, idx in variants.items():
        s = score_retrieval(idx)
        scores[name] = s
        print(f"  {name:<18} R@1 {s['recall_at_1']:.3f}  R@3 {s['recall_at_3']:.3f}  "
              f"R@5 {s['recall_at_5']:.3f}  (missed {len(s['missed_entirely'])})")

    by_license: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for p in passages:
        by_license[p.license] = by_license.get(p.license, 0) + 1
        by_source[p.source] = by_source.get(p.source, 0) + 1
    lens = [len(p.text) for p in passages]

    payload = {
        "n_passages": len(passages),
        "by_license": by_license,
        "by_source": by_source,
        "passage_chars": {"min": int(min(lens)), "median": int(np.median(lens)),
                          "max": int(max(lens))},
        "dense_model": DenseIndex.__init__.__defaults__[0] if dense is not None else None,
        "retrieval": scores,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "index_report.json").write_text(json.dumps(payload, indent=2))
    print(f"\n-> {OUT_DIR / 'index_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
