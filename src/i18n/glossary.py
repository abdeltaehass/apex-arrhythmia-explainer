"""Phase 27 — checking the Spanish terminology against real Spanish cardiology prose.

A hand-authored clinical vocabulary is only as good as its terminology, and "it looks right"
is not a check. This module fetches Spanish-language cardiology reference text and asks, for
each of the 71 terms, whether it actually occurs in writing by Spanish-speaking clinicians.

**What this can and cannot establish.** A term appearing in the reference corpus is evidence
it is the conventional Spanish usage. A term *not* appearing is not proof of error — the
corpus is finite, and several SCP statements (``non-diagnostic T-wave abnormality``) are
report-writing conventions rather than encyclopedia topics. So the output is a review list,
not a verdict, and the report says which terms went unconfirmed rather than quietly counting
them as correct.

The corpus is Spanish Wikipedia, fetched verbatim through the MediaWiki extracts API and
reusing the Phase-21 fetcher (retries and backoff included, since a corpus with holes in the
conduction-block articles is worse than no corpus). CC BY-SA 4.0, same licensing footing as
Phase 21's reference corpus: it is quoted for validation, not redistributed.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from src.config import ROOT
from src.i18n.languages import get_language

CORPUS_PATH = ROOT / "data" / "reference" / "cardiology_es.json"

# Spanish Wikipedia cardiology and electrocardiography articles.
ES_ARTICLES = [
    "Electrocardiograma", "Arritmia cardíaca", "Fibrilación auricular",
    "Aleteo auricular", "Taquicardia supraventricular", "Taquicardia ventricular",
    "Bradicardia", "Taquicardia", "Bloqueo auriculoventricular",
    "Bloqueo de rama", "Síndrome de Wolff-Parkinson-White", "Extrasístole",
    "Infarto agudo de miocardio", "Isquemia miocárdica", "Angina de pecho",
    "Hipertrofia ventricular izquierda", "Cardiopatía isquémica",
    "Síndrome de QT largo", "Marcapasos artificial", "Sístole", "Diástole",
    "Nodo sinusal", "Sistema de conducción eléctrica del corazón",
    "Complejo QRS", "Onda T", "Segmento ST", "Intervalo QT",
]


@dataclass
class TermCheck:
    code: str
    term: str
    found: bool
    occurrences: int
    example: str = ""


def fetch_corpus(titles: list[str] | None = None, verbose: bool = True,
                 delay_s: float = 1.5) -> list[dict]:
    """Fetch the Spanish reference articles as ``[{title, url, text}]``.

    ``delay_s`` spaces the requests. Wikipedia rate-limits unauthenticated clients hard
    enough that a tight loop exhausts the fetcher's retries about halfway through this list
    — and the failure is silent unless you count what came back, leaving the corpus missing
    exactly the conduction-block articles the rarer terms need.
    """
    import time

    import src.rag.corpus as rag_corpus

    titles = titles or ES_ARTICLES
    original = rag_corpus.WIKI_API
    rag_corpus.WIKI_API = "https://es.wikipedia.org/w/api.php"
    out: list[dict] = []
    try:
        for i, title in enumerate(titles):
            if i:
                time.sleep(delay_s)
            try:
                got = rag_corpus.fetch_wikipedia_extract(title)
            except Exception as e:                       # noqa: BLE001 — one article is not the run
                if verbose:
                    print(f"    error: {title}: {e}")
                continue
            if got is None:
                if verbose:
                    print(f"    miss: {title}")
                continue
            text, url = got
            out.append({"title": title, "url": url, "text": text})
            if verbose:
                print(f"    {title}: {len(text):,} chars")
    finally:
        rag_corpus.WIKI_API = original
    return out


def save_corpus(passages: list[dict], path: Path = CORPUS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "source": "Spanish Wikipedia (es.wikipedia.org), MediaWiki extracts API",
        "license": "CC BY-SA 4.0",
        "note": "Quoted for terminology validation only; not redistributed as a corpus.",
        "articles": passages,
    }, indent=2, ensure_ascii=False))
    return path


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("articles", [])


def _fold(text: str) -> str:
    text = "".join(c for c in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower())


def _term_pattern(phrase: str) -> re.Pattern:
    """A regex matching ``phrase`` with each word optionally pluralized.

    Spanish pluralizes by ``+s`` after a vowel and ``+es`` after a consonant, and a single
    phrase mixes both: *extrasístoles auriculares* is *extrasístole* + s and *auricular* +
    es. Stripping one fixed suffix from every word — the first thing tried here — produced
    "extrasistol auriculare" and matched nothing, so a standard term was reported
    unconfirmed over a letter. Matching each word against its stem with an optional suffix
    handles the mixed case and, more importantly, works in the direction that matters:
    report vocabulary is plural, reference prose defines the singular.
    """
    words = []
    for word in phrase.split():
        stem = word[:-1] if (len(word) > 3 and word.endswith("s")) else word
        if len(stem) > 3 and stem.endswith("e"):
            # Both Spanish plural forms collapse here: "extrasistoles" wants the stem
            # "extrasistole" (+s) while "auriculares" wants "auricular" (+es). Stripping a
            # fixed suffix gets one of them wrong whichever you pick, so the final vowel is
            # made optional and one pattern covers both.
            words.append(re.escape(stem[:-1]) + "e?s?")
        else:
            words.append(re.escape(stem) + "s?")
    return re.compile(r"\b" + r"\s+".join(words), re.IGNORECASE)


def _patterns(entry) -> list[tuple[str, re.Pattern]]:
    """Surface forms that count as the same term: the full phrase, then the head noun."""
    forms = [(entry.impression, _term_pattern(_fold(entry.impression)))]
    if entry.territory:
        # Reference prose names the entity and the territory separately far more often than
        # it writes "isquemia anterolateral" verbatim.
        head = _fold(entry.impression.replace(entry.territory, "")).strip(" ,")
        if head:
            forms.append((head, _term_pattern(head)))
    return forms


def check_terms(passages: list[dict], lang: str = "es") -> list[TermCheck]:
    """For each vocabulary term, count occurrences in the reference corpus.

    Matching folds accents and whitespace, tolerates singular/plural, and accepts the head
    noun phrase for territory-qualified entries. Anything still unmatched is a genuine
    review item rather than a matching artefact.
    """
    bank = get_language(lang)
    blob = _fold(" ".join(p["text"] for p in passages))
    checks: list[TermCheck] = []
    for code, entry in sorted(bank.vocab.items()):
        if not entry.impression:
            continue
        n, example = 0, ""
        for _label, pattern in _patterns(entry):
            hits = list(pattern.finditer(blob))
            if hits:
                n = len(hits)
                at = hits[0].start()
                example = blob[max(0, at - 60):hits[0].end() + 60].strip()
                break
        checks.append(TermCheck(code=code, term=entry.impression, found=n > 0,
                                occurrences=n, example=example))
    return checks
