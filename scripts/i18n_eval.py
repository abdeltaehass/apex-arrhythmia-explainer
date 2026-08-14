#!/usr/bin/env python3
"""Phase 27 — does the Spanish path carry the same safety guarantees as the English one?

Two measurements, in order of importance.

**1. Consistency-gate parity.** The Phase-7 gate is what stops APEX asserting findings the
detector never surfaced. It matches English impression phrases, so before this phase a
Spanish report parsed as asserting *nothing* and passed unconditionally. This block
quantifies that: over real PTB-XL records it renders each report in both languages, checks
that the gate recovers exactly the findings that went in, and then injects a fabricated
finding to confirm the gate catches it — in both languages, and under the pre-Phase-27
English-only parser for contrast.

An equity claim that stops at "we added Spanish" is decoration. The claim worth making is
that the Spanish output is held to the same measured standard, and this is the measurement.

**2. Terminology validation.** Each Spanish term is checked against Spanish-language
cardiology prose (`src/i18n/glossary.py`). Confirmation is evidence the term is
conventional; non-confirmation is a review item, not an error — the corpus is 27
encyclopedia articles and many SCP statements are report-writing conventions that
encyclopedic prose has no reason to use.

    python scripts/i18n_eval.py                 # uses the cached corpus
    python scripts/i18n_eval.py --fetch         # re-fetch the Spanish reference corpus
    python scripts/i18n_eval.py --n 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT  # noqa: E402
from src.data.labels import load_database, present_codes  # noqa: E402
from src.generation.parse import asserted_findings as asserted_en  # noqa: E402
from src.generation.templater import NORM_COMPATIBLE, build_structured_input  # noqa: E402
from src.i18n.glossary import check_terms, fetch_corpus, load_corpus, save_corpus  # noqa: E402
from src.i18n.languages import supported_languages  # noqa: E402
from src.i18n.parse import asserted_findings, detect_language, parse_report  # noqa: E402
from src.i18n.render import render_text  # noqa: E402

OUT = ROOT / "docs" / "i18n"
# A finding no rendered report contains, injected to test that the gate notices.
FABRICATION = {"en": "Complete left bundle branch block.",
               "es": "Bloqueo completo de rama izquierda."}


def gate_parity(n: int, seed: int = 0) -> dict:
    """Render real records in each language and audit the consistency gate."""
    import numpy as np

    db = load_database()
    rng = np.random.default_rng(seed)
    ids = rng.choice(db.index.to_numpy(), size=min(n, len(db)), replace=False)

    langs = supported_languages()
    stats = {lang: {"n": 0, "well_formed": 0, "roundtrip_exact": 0, "caught": 0,
                    "language_detected": 0, "n_informative": 0,
                    "roundtrip_exact_informative": 0} for lang in langs}
    n_collapsed = 0
    # The pre-Phase-27 behaviour: English parser pointed at whatever came out.
    legacy = {"n": 0, "caught": 0, "asserted_any": 0}
    misses: list[dict] = []

    for ecg_id in ids:
        codes = [c for c in present_codes(db.loc[int(ecg_id), "scp_codes"])]
        si = build_structured_input(codes, confidences=dict.fromkeys(codes, 0.9))
        surfaced = set(si.codes())
        if not surfaced:
            continue
        # A normal study deliberately collapses its impression to "Normal ECG", which
        # suppresses the rhythm term (Phase 6, NORM_COMPATIBLE). Those records can never
        # round-trip every surfaced code, in either language, so they are counted
        # separately rather than being charged to the translation.
        collapsed = "NORM" in surfaced and not (surfaced - NORM_COMPATIBLE)
        n_collapsed += int(collapsed)

        for lang in langs:
            text = render_text(si, lang)
            s = stats[lang]
            s["n"] += 1
            s["well_formed"] += int(parse_report(text, lang).well_formed)
            s["language_detected"] += int(detect_language(text) == lang)

            got = asserted_findings(text, lang)
            # A rendered report should assert exactly the findings that carry an impression
            # term; codes without one (HVOLT, QWAVE...) legitimately never appear.
            expected = {c for c in surfaced
                        if __import__("src.i18n.languages", fromlist=["get_language"])
                        .get_language(lang).vocab[c].impression}
            if not collapsed:
                s["n_informative"] += 1
            if got == expected:
                s["roundtrip_exact"] += 1
                if not collapsed:
                    s["roundtrip_exact_informative"] += 1
            elif not collapsed and len(misses) < 12:
                misses.append({"ecg_id": int(ecg_id), "lang": lang,
                               "missing": sorted(expected - got), "extra": sorted(got - expected)})

            fabricated = text + " " + FABRICATION[lang]
            s["caught"] += int("CLBBB" in asserted_findings(fabricated, lang))

            if lang != "en":
                legacy["n"] += 1
                legacy["caught"] += int("CLBBB" in asserted_en(fabricated))
                legacy["asserted_any"] += int(bool(asserted_en(text)))

    for s in stats.values():
        for key in ("well_formed", "roundtrip_exact", "caught", "language_detected"):
            s[f"{key}_rate"] = round(s[key] / s["n"], 4) if s["n"] else None
        s["roundtrip_exact_informative_rate"] = (
            round(s["roundtrip_exact_informative"] / s["n_informative"], 4)
            if s["n_informative"] else None)
    legacy["caught_rate"] = round(legacy["caught"] / legacy["n"], 4) if legacy["n"] else None
    legacy["asserted_any_rate"] = (round(legacy["asserted_any"] / legacy["n"], 4)
                                   if legacy["n"] else None)
    return {"by_language": stats, "legacy_english_only_parser": legacy,
            "n_norm_collapsed": n_collapsed, "roundtrip_misses": misses}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400, help="records for the gate audit")
    ap.add_argument("--fetch", action="store_true", help="re-fetch the Spanish corpus")
    args = ap.parse_args()

    print(f"[1/2] Consistency-gate parity over {args.n} PTB-XL records")
    parity = gate_parity(args.n)
    for lang, s in parity["by_language"].items():
        print(f"  {lang}: n={s['n']}  well-formed {s['well_formed_rate']:.1%}  "
              f"language detected {s['language_detected_rate']:.1%}  "
              f"round-trip {s['roundtrip_exact_rate']:.1%} overall / "
              f"{s['roundtrip_exact_informative_rate']:.1%} excl. normal-collapsed  "
              f"fabrication caught {s['caught_rate']:.1%}")
    print(f"  ({parity['n_norm_collapsed']} of {args.n} records are normal studies whose "
          "impression collapses to \"Normal ECG\", suppressing the rhythm term in both "
          "languages by design)")
    legacy = parity["legacy_english_only_parser"]
    print(f"  pre-Phase-27 (English parser on Spanish text): fabrication caught "
          f"{legacy['caught_rate']:.1%}, any finding recognised {legacy['asserted_any_rate']:.1%}")
    if parity["roundtrip_misses"]:
        print(f"  round-trip misses (first {len(parity['roundtrip_misses'])}):")
        for m in parity["roundtrip_misses"][:5]:
            print(f"    {m['ecg_id']} [{m['lang']}] missing={m['missing']} extra={m['extra']}")

    print("\n[2/2] Spanish terminology against Spanish-language cardiology prose")
    if args.fetch or not load_corpus():
        print("  fetching es.wikipedia reference articles...")
        save_corpus(fetch_corpus(verbose=True))
    passages = load_corpus()
    checks = check_terms(passages)
    confirmed = [c for c in checks if c.found]
    print(f"  corpus: {len(passages)} articles, "
          f"{sum(len(p['text']) for p in passages):,} characters")
    print(f"  confirmed: {len(confirmed)}/{len(checks)} terms appear in the reference prose")
    print("  unconfirmed (review list, not an error list):")
    for c in checks:
        if not c.found:
            print(f"    {c.code:8s} {c.term}")

    report = {
        "gate_parity": parity,
        "terminology": {
            "n_articles": len(passages),
            "n_chars": sum(len(p["text"]) for p in passages),
            "confirmed": len(confirmed), "total": len(checks),
            "checks": [vars(c) for c in checks],
            "source": "Spanish Wikipedia (CC BY-SA 4.0), quoted for validation only",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {(OUT / 'eval.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
