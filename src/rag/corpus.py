"""Phase 21 — build the clinical reference corpus the RAG layer retrieves from.

**A correction to the phase spec, stated up front.** The brief asked for "ACC/AHA
guideline summaries" and "public domain textbook excerpts". ACC/AHA clinical practice
guidelines are *not* public domain — they are published in *Circulation* and *JACC* under
copyright, the AHA's own permissions policy forbids redistribution, and reuse is licensed
per figure/table for a fee. Reproducing them here would be a licence violation, and
*writing* passages and labelling them "ACC/AHA" would be worse: fabricated clinical
reference text that looks authoritative is exactly the failure mode this whole project is
built to avoid. So the corpus is assembled from sources that really are openly licensed,
and every passage carries its provenance so a reader can check it:

- **PTB-XL `scp_statements.csv`** (CC BY 4.0, already bundled in this repo) — the
  authoritative definition of each of the 71 SCP-ECG statements, its diagnostic class and
  its statement category. Short, but it is the ground truth for this exact label space.
- **Wikipedia** (CC BY-SA 4.0), fetched verbatim through the MediaWiki API — the
  clinical/morphological detail that PTB-XL's one-line descriptions lack: what a left
  bundle branch block does to the QRS, which leads an inferior infarct localizes to, and
  so on.

Passages are stored with ``source``, ``url``, ``license`` and ``retrieved`` so the report
can state exactly what the model was grounded on. Swapping in a licensed guideline corpus
later is a matter of replacing this file's output, not of changing the retriever.

Nothing here is authored by the model. Text is either copied from the bundled CSV or
fetched verbatim; the only transformation is splitting into passages.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from src.config import ROOT

CORPUS_PATH = ROOT / "data" / "reference" / "corpus.jsonl"
WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "APEX-arrhythmia-explainer/0.1 (research; contact via GitHub abdeltaehass)"

# Sections that carry no clinical content worth retrieving.
SKIP_SECTIONS = {
    "references", "external links", "see also", "further reading", "notes",
    "bibliography", "sources", "citations",
}

# Wikipedia articles covering the concepts behind PTB-XL's 71 SCP statements: rhythm,
# conduction, chamber enlargement, repolarization, infarction, and ECG basics.
WIKI_ARTICLES = [
    "Electrocardiography",
    "QRS complex",
    "P wave (electrocardiography)",
    "T wave",
    "QT interval",
    "ST segment",
    "ST elevation",
    "ST depression",
    "Sinus rhythm",
    "Sinus tachycardia",
    "Sinus bradycardia",
    "Sinus arrhythmia",
    "Atrial fibrillation",
    "Atrial flutter",
    "Supraventricular tachycardia",
    "Premature atrial contraction",
    "Premature ventricular contraction",
    "Wolff–Parkinson–White syndrome",
    "Bundle branch block",
    "Left bundle branch block",
    "Right bundle branch block",
    "Left anterior fascicular block",
    "Left posterior fascicular block",
    "Atrioventricular block",
    "First-degree atrioventricular block",
    "Second-degree atrioventricular block",
    "Third-degree atrioventricular block",
    "Left ventricular hypertrophy",
    "Right ventricular hypertrophy",
    "Left atrial enlargement",
    "Myocardial infarction",
    "Electrocardiography in myocardial infarction",
    "Coronary artery disease",
    "Cardiac muscle",
    "Artificial cardiac pacemaker",
    "Long QT syndrome",
    "Digoxin toxicity",
    "Right axis deviation",
    "Left axis deviation",
    "Electrical alternans",
    "Hypertrophic cardiomyopathy",
]


@dataclass
class Passage:
    """One retrievable chunk, with the provenance needed to audit it."""

    id: str
    text: str
    title: str
    source: str
    license: str
    url: str
    retrieved: str
    codes: list[str]  # SCP codes this passage was tagged with, when known

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# --- PTB-XL statement definitions -------------------------------------------
def scp_passages() -> list[Passage]:
    """One passage per SCP-ECG statement, from the bundled PTB-XL dictionary.

    These are the definitions of the exact label space the detector predicts, so they are
    the passages most likely to be *correctly* retrievable for a given finding. They are
    short by nature — a statement description plus its diagnostic class — which is why the
    Wikipedia articles are needed alongside them.
    """
    from src.data.labels import load_scp_statements

    df = load_scp_statements()
    today = date.today().isoformat()
    out = []
    for code, row in df.iterrows():
        desc = str(row.get("SCP-ECG Statement Description") or row.get("description") or "").strip()
        if not desc:
            continue
        category = str(row.get("Statement Category") or "").strip()
        dclass = str(row.get("diagnostic_class") or "").strip()
        bits = [f"{code} denotes {desc}."]
        if category and category.lower() != "nan":
            bits.append(f"It belongs to the SCP-ECG statement category: {category}.")
        if dclass and dclass.lower() != "nan":
            bits.append(f"Its PTB-XL diagnostic superclass is {dclass}.")
        out.append(Passage(
            id=f"scp::{code}",
            text=" ".join(bits),
            title=f"SCP-ECG statement {code}",
            source="PTB-XL scp_statements.csv",
            license="CC BY 4.0",
            url="https://physionet.org/content/ptb-xl/1.0.3/",
            retrieved=today,
            codes=[str(code)],
        ))
    return out


# --- Wikipedia --------------------------------------------------------------
def fetch_wikipedia_extract(title: str, timeout: int = 30, retries: int = 4) -> tuple[str, str] | None:
    """Verbatim plain-text extract of one article -> ``(text, canonical_url)``.

    Uses the MediaWiki ``extracts`` API with ``explaintext``, which returns the article
    body as plain text rather than HTML — no scraping, no model in the loop, so what lands
    in the corpus is what the article actually says.

    Retries on HTTP 429 with exponential backoff. Wikipedia rate-limits unauthenticated
    clients, and a build that silently drops two thirds of its articles to 429s produces a
    corpus with holes exactly where the retriever will look for them — the first attempt
    at this returned 253 passages and was missing every conduction-block article.
    """
    params = {
        "action": "query", "prop": "extracts", "explaintext": "1",
        "format": "json", "redirects": "1", "titles": title,
    }
    url = f"{WIKI_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (fixed host)
                payload = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            time.sleep(2.0 * (2 ** attempt))
    else:  # pragma: no cover - loop always breaks or raises
        return None
    pages = payload.get("query", {}).get("pages", {})
    for pid, page in pages.items():
        if pid == "-1" or "extract" not in page:
            continue
        canonical = ("https://en.wikipedia.org/wiki/"
                     + urllib.parse.quote(page["title"].replace(" ", "_")))
        return page["extract"], canonical
    return None


def split_sections(extract: str) -> list[tuple[str, str]]:
    """Split a plain-text extract into ``(section_heading, body)`` pairs.

    The API marks headings as ``== Heading ==``. Boilerplate sections (references,
    external links) are dropped — they are pure noise in a retrieval index.
    """
    parts: list[tuple[str, str]] = []
    current = "Summary"
    buf: list[str] = []
    for line in extract.splitlines():
        m = re.match(r"^\s*(={2,})\s*(.+?)\s*\1\s*$", line)
        if m:
            if buf:
                parts.append((current, "\n".join(buf).strip()))
            current, buf = m.group(2), []
        else:
            buf.append(line)
    if buf:
        parts.append((current, "\n".join(buf).strip()))
    return [(h, b) for h, b in parts if b and h.strip().lower() not in SKIP_SECTIONS]


def chunk_text(text: str, max_chars: int = 900, min_chars: int = 120) -> list[str]:
    """Split a section into passage-sized chunks on paragraph then sentence boundaries.

    Retrieval quality depends on chunks being one coherent idea: too long and the
    embedding averages several topics into mush, too short and it loses the context that
    makes it useful. Splitting on real boundaries rather than a fixed token stride keeps
    sentences intact, which matters when the passage is going into a clinical prompt.
    """
    text = re.sub(r"\n{2,}", "\n\n", text).strip()
    chunks: list[str] = []
    for para in text.split("\n\n"):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        buf = ""
        for s in sentences:
            if len(buf) + len(s) + 1 > max_chars and buf:
                chunks.append(buf.strip())
                buf = s
            else:
                buf = f"{buf} {s}".strip()
        if buf:
            chunks.append(buf.strip())
    # Merge runt chunks backward so no passage is a stray fragment — but never past
    # ``max_chars``, which exists to bound what the embedding model has to summarize into
    # one vector. A runt that cannot be merged without breaking that bound is kept as its
    # own passage; a slightly short passage is a smaller problem than an oversized one.
    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < min_chars and len(merged[-1]) + len(c) + 1 <= max_chars:
            merged[-1] = f"{merged[-1]} {c}"
        else:
            merged.append(c)
    return merged


def wikipedia_passages(titles: list[str] | None = None, verbose: bool = True,
                       delay: float = 2.5) -> list[Passage]:
    """Fetch and chunk each article, pausing ``delay`` seconds between requests to stay
    within Wikipedia's rate limit for unauthenticated clients."""
    titles = WIKI_ARTICLES if titles is None else titles
    today = date.today().isoformat()
    out: list[Passage] = []
    for n_done, title in enumerate(titles):
        if n_done:
            time.sleep(delay)
        try:
            got = fetch_wikipedia_extract(title)
        except Exception as e:  # network hiccup on one article shouldn't kill the build
            if verbose:
                print(f"  !! {title}: {type(e).__name__}: {e}")
            continue
        if not got:
            if verbose:
                print(f"  !! {title}: no extract returned")
            continue
        extract, url = got
        n = 0
        for heading, body in split_sections(extract):
            for i, chunk in enumerate(chunk_text(body)):
                slug = re.sub(r"[^a-z0-9]+", "_", f"{title}_{heading}".lower()).strip("_")
                out.append(Passage(
                    id=f"wiki::{slug}::{i}",
                    text=chunk,
                    title=f"{title} — {heading}",
                    source="Wikipedia",
                    license="CC BY-SA 4.0",
                    url=url,
                    retrieved=today,
                    codes=[],
                ))
                n += 1
        if verbose:
            print(f"  {title}: {n} passages")
    return out


# --- build / load ------------------------------------------------------------
def build_corpus(out_path: Path = CORPUS_PATH, titles: list[str] | None = None,
                 verbose: bool = True) -> list[Passage]:
    if verbose:
        print("PTB-XL SCP statement definitions...")
    passages = scp_passages()
    if verbose:
        print(f"  {len(passages)} passages")
        print("Wikipedia articles...")
    passages += wikipedia_passages(titles, verbose=verbose)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p in passages:
            f.write(p.to_json() + "\n")
    if verbose:
        print(f"\n{len(passages)} passages -> {out_path}")
    return passages


def load_corpus(path: Path = CORPUS_PATH) -> list[Passage]:
    if not path.exists():
        raise FileNotFoundError(
            f"no corpus at {path}; build it with `python scripts/build_rag_index.py`")
    return [Passage(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
