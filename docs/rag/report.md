# Phase 21 — Retrieval-augmented clinical context

The Phase-6 generator writes its report from the detector's output alone. This phase adds a retrieval layer: a vector index over openly-licensed cardiology reference text, queried per detected finding, with the top passages injected into the prompt as background. The question is whether that grounding reduces hallucination — and the honest answer needs a controlled experiment, because there is a mechanism pushing the other way.

## Headline

**Retrieval made hallucination *worse*: 0.0600 -> 0.1200 (+6.0 points), though the paired test does not clear significance (p = 0.06357).**

The measurable effect is elsewhere: **finding coverage moves 0.3417 -> 0.4135 (+0.0718)** — the share of the detector's findings the report actually states. Retrieval makes the generator more likely to say what it was told, which is a real improvement in explanation completeness even though it is not the axis the phase set out to move.

## Results

Paired comparison on **150 PTB-XL test-fold records**, each generated twice by `Qwen/Qwen2.5-1.5B-Instruct` under greedy decoding — identical records, identical detector output, identical prompt except for the retrieved block.

| metric | no RAG | RAG | delta | |
|---|---:|---:|---:|---|
| Hallucination rate (records with >=1 fabricated finding) | 0.0600 | 0.1200 | +0.0600 | worse |
| Consistency rate (records with none) | 0.9400 | 0.8800 | -0.0600 | worse |
| Fabricated findings per record | 0.0730 | 0.1470 | +0.0740 | worse |
| Finding coverage (surfaced findings actually stated) | 0.3417 | 0.4135 | +0.0718 | better |
| Well-formed rate (two-section contract) | 0.7000 | 0.5400 | -0.1600 | worse |
| Treatment-recommendation rate (prompt forbids it) | 0.1800 | 0.4467 | +0.2667 | worse |

McNemar exact test on the paired hallucination outcome: **5** records hallucinated only without RAG, **14** only with it, 19 discordant pairs, **p = 0.06357**.

## The mechanism that works against RAG here

APEX's generator operates under a hard constraint: it may assert only the findings the detector surfaced. Retrieved cardiology text is, by its nature, full of *other* condition names — an article on left bundle branch block discusses infarction, one on atrial fibrillation discusses stroke and anticoagulation. RAG therefore places a list of plausible, clinically-adjacent diagnoses directly in front of a model whose single most important instruction is not to mention any of them.

This is the opposite of the usual RAG setting, where the retrieved passage *contains the answer* and grounding can only help. Here the answer is already in the prompt — it is the detector's finding list — and retrieval adds context that is useful for **wording** and hazardous for **scope**.

Of the **22** fabricated findings in the RAG arm, **10** (45%) name a condition that was sitting in that record's retrieved passages. That is the share attributable to retrieval putting the condition in front of the model, as distinct from the model's own priors — the comparison being the 11 fabrications in the no-RAG arm, of which 2 happened to name a condition from the passages that record *would* have retrieved.

| fabricated finding | no RAG | RAG | of which named in that record's context |
|---|---:|---:|---:|
| `STE_` — non-specific ST elevation | 4 | 8 | 0 |
| `LVH` — left ventricular hypertrophy | 3 | 8 | 8 |
| `INVT` — inverted T-waves | 1 | 2 | 0 |
| `SR` — sinus rhythm | 1 | 1 | 1 |
| `AFIB` — atrial fibrillation | 1 | 0 | 0 |
| `AFLT` — atrial flutter | 1 | 0 | 0 |
| `CRBBB` — complete right bundle branch block | 0 | 1 | 0 |
| `LMI` — lateral myocardial infarction | 0 | 1 | 1 |
| `STD_` — non-specific ST depression | 0 | 1 | 0 |
| **total** | **11** | **22** | **10** |

Two different failure modes sit in that table and they should not be conflated. **LVH** (left ventricular hypertrophy) is the retrieval-caused one: fabricated roughly three times as often with RAG as without, and *every single time* it was named in the passages retrieved for that record. LVH is discussed in articles about axis deviation, bundle branch block and fascicular block — all of which are legitimately retrieved for other findings — so the corpus keeps putting the words "left ventricular hypertrophy" in front of a model that was told not to say them, and often enough it says them.

**ST elevation** is the opposite: fabricated in both arms, and never present in the retrieved text. That one is the model's own prior — it associates infarction findings with ST elevation and volunteers it regardless of context. Retrieval neither caused it nor fixed it, which is worth stating because an aggregate hallucination number would have quietly credited RAG with the difference.

## What this means for APEX

The conclusion is not "RAG does not work" — it is that **retrieval belongs on the wording, not on the assertions**, and this pipeline already has a place for each.

1. **Do not enable retrieval on the assertion path as it stands.** `with_rag` is off by default in `analyze_signal` for exactly this reason. The measured cost is double the fabrication rate, a third fewer well-formed reports, and two and a half times the rate of treatment recommendations the prompt explicitly forbids.
2. **The existing safety net catches it.** Phase 7's consistency checker compares asserted findings against what the detector surfaced and withholds the explanation on a mismatch. Every fabrication counted here would be caught by that gate before reaching a clinician — the failure mode is degraded *availability* (more reports withheld for review), not clinical misinformation reaching a user. This is what a layered design buys, and it is why the hallucination rate is worth measuring even when it cannot escape.
3. **Where retrieval did help is real and worth keeping**: coverage rose 7 points, meaning the report states more of what the detector actually found. A narrower corpus — definitional statements only, with the discursive encyclopaedia prose stripped out — would plausibly keep that gain without importing the condition names that cause the harm. The per-code table points straight at the fix: the passages that caused fabrications were retrieved for *other* findings and merely happened to mention LVH.

## Retrieval quality

The corpus contains one passage per SCP-ECG statement, which gives a labelled retrieval benchmark for free: query with a code and its clinical name, and check whether that code's own definition comes back. Over 71 statements:

| retriever | R@1 | R@3 | R@5 |
|---|---:|---:|---:|
| tfidf (sparse) | 0.887 | 1.000 | 1.000 |
| minilm (dense) | 0.789 | 0.930 | 0.958 |
| hybrid (RRF) | 0.845 | 0.972 | 1.000 |

The sparse retriever wins at rank 1, and that is expected rather than disappointing: the query contains the literal code string (`ASMI`, `LNGQT`) that appears in the target passage, which is precisely what exact-term matching is for. The benchmark is therefore biased toward TF-IDF by construction and should be read as a **sanity check that the index is not broken**, not as evidence that dense embeddings are useless — the passages that matter for *wording* are the clinical descriptions, where semantic similarity does the work. Both retrievers reach the target within the top 5, which is the regime the generator actually sees.

## The corpus

**927 passages** (71 CC BY 4.0, 856 CC BY-SA 4.0), 568 characters at the median.

| source | passages | licence |
|---|---:|---|
| PTB-XL scp_statements.csv | 71 | CC BY 4.0 |
| Wikipedia | 856 | CC BY-SA 4.0 |

### A correction to the phase brief

The brief asked for "ACC/AHA guideline summaries" and "public domain textbook excerpts". **ACC/AHA clinical practice guidelines are not public domain.** They are published in *Circulation* and *JACC* under copyright; the AHA's permissions policy forbids redistribution and licenses reuse per excerpt for a fee. So they are not in this corpus.

They were also not replaced with model-written passages labelled as guideline text. Fabricated clinical reference material that reads as authoritative is a worse outcome than having none — it is the exact failure this project measures everywhere else. What is here instead is verbatim text from two genuinely open sources, with per-passage provenance (`source`, `url`, `license`, `retrieved`) so any passage can be traced and checked. See [`data/reference/NOTICE.md`](../../data/reference/NOTICE.md). Swapping in licensed guideline text is a data change, not a code change.

## Implementation notes

- **Retrieval is per finding, not per record.** A record with atrial fibrillation and an inferior infarct needs passages about both; one merged query embeds to the average of two unrelated conditions and returns something about neither.
- **Hybrid retrieval.** Dense embeddings (`all-MiniLM-L6-v2`, mean-pooled through plain `transformers` — no new dependency) fused with word+character TF-IDF by reciprocal rank. RRF rather than a weighted score sum because the two score scales are not comparable and normalizing them against each other would be an arbitrary choice tuned on nothing.
- **Exact search.** At this corpus size an ANN index (FAISS, hnswlib) would add a dependency and an approximation error to solve a problem that does not exist; the search is one matmul.
- **The no-RAG arm is byte-identical to Phase 6.** `build_user_prompt(si)` with no context returns exactly the original prompt, which is asserted in the test suite — without that, the comparison would not be controlled.
- **The boundary instruction travels with the context**, not just in the system prompt, so the last thing the model reads before generating is the reminder that reference material is not a finding list.
- **Greedy decoding** in both arms: sampling noise would sit directly on top of the effect being measured.

## Limitations

- **The generator is `Qwen/Qwen2.5-1.5B-Instruct` running locally.** No API key and no GPU were available, and the two cached alternatives were both unusable as subjects: the deterministic template backend cannot hallucinate by construction, and the Phase-6 135M smoke adapter emits no diagnoses at all. A 1.5B open model is a real generator that writes real clinical prose, but it is not the frontier model a deployment would use, and instruction-following scales with capability — the *absolute* rates here should not be read as APEX's production numbers. The paired design is what makes the comparison meaningful.
- **Wikipedia is not a clinical guideline.** Its medical articles vary in depth and currency. The corpus is adequate for grounding *wording*, and it is openly licensed, but it is not the reference a hospital deployment would ship.
- **Assertion detection is lexical.** `parse.asserted_findings` matches impression phrases from the Phase-6 vocabulary, so a fabricated finding phrased in words the vocabulary does not contain is not counted. The measured hallucination rate is a lower bound on both arms equally.
- **The treatment-recommendation metric is a keyword proxy**, not a judgement of clinical intent. It is directionally useful and should not be read to three decimal places.
- **One corpus, one prompt, one model.** These results characterise this configuration. A larger model that follows the scope instruction more reliably, or a corpus of tightly-scoped guideline statements rather than encyclopaedia articles, could plausibly move the result in either direction.
- The corpus is fetched from a live source, so a rebuild will not be byte-identical; `data/reference/corpus.jsonl` as committed is the artifact the numbers came from.
