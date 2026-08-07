# Reference corpus — sources, licences and attribution

`corpus.jsonl` is the retrieval corpus for APEX's Phase-21 RAG layer. Every passage in it
was **copied verbatim** from one of the sources below and carries its own `source`,
`url`, `license` and `retrieved` fields, so any individual passage can be traced back and
checked. Nothing in this corpus was written by a language model.

Rebuild it with `make rag-index` (`python scripts/build_rag_index.py`).

## What is *not* in here, and why

The Phase-21 brief asked for "ACC/AHA guideline summaries" and "public domain textbook
excerpts". **ACC/AHA clinical practice guidelines are not public domain.** They are
published in *Circulation* and the *Journal of the American College of Cardiology* under
copyright; the American Heart Association's permissions policy prohibits redistribution
and licenses reuse per figure or excerpt for a fee. They are therefore excluded.

They were not replaced with model-written text dressed up as guideline material. A
fabricated passage that reads like an authoritative cardiology guideline is a worse
outcome than having no guideline text at all, and it is precisely the failure mode this
project exists to measure. If you have a licence for guideline text, dropping it into this
corpus is a data change, not a code change — the retriever does not care where passages
came from.

## Sources

### 1. PTB-XL `scp_statements.csv` — 71 passages

The definition of each SCP-ECG statement in the model's label space.

- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Source:** PTB-XL, a large publicly available electrocardiography dataset (v1.0.3),
  PhysioNet — <https://physionet.org/content/ptb-xl/1.0.3/>
- **Citation:** Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T.
  (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific
  Data*, 7, 154.

### 2. English Wikipedia — 856 passages

Clinical and morphological detail on the conditions behind those statements, fetched
verbatim through the MediaWiki `extracts` API and split into passages.

- **Licence:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **Attribution:** each passage records the article title and canonical URL in
  `corpus.jsonl`; article histories list the contributing authors.
- **ShareAlike:** the passages in `corpus.jsonl` remain under CC BY-SA 4.0. Redistributing
  them, or an adapted version of them, requires the same licence and attribution. This
  obligation attaches to **the corpus text**, not to the source code in this repository,
  which is a separate work that merely reads the file.
- **Articles used:** see `WIKI_ARTICLES` in `src/rag/corpus.py`.

Wikipedia is a general-reference encyclopaedia, not a clinical guideline, and its medical
articles vary in depth and currency. That is a genuine limitation of this corpus and is
stated as such in `docs/rag/report.md` rather than glossed over.

## Reproducibility

The corpus is fetched from a live source, so a rebuild months from now will not be
byte-identical — Wikipedia articles change. `corpus.jsonl` as committed is therefore the
artifact of record: it is what the reported Phase-21 results were produced from.
