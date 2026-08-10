#!/usr/bin/env python3
"""Phase 23 — verify every ICD-10-CM and LOINC code against the live NLM service.

A billing mapping authored from clinical knowledge is a liability unless it can be
re-checked mechanically. ICD-10-CM is revised every October and the revisions are not
cosmetic: FY2024 subdivided I47.1 into I47.10/I47.11/I47.19, which silently turned a
billable code into a non-billable category header. A table that was right when written
becomes wrong without anyone touching it.

So this script re-derives the whole table from the U.S. National Library of Medicine's
Clinical Table Search Service and checks three things per code:

1. it exists in the current release;
2. the ``display`` string in `src/ehr/codes.py` matches the official description exactly;
3. **it is billable** — the code is a leaf, not a parent with children. This is the check
   that catches the I47.1 class of failure, and it is the one that would cost a hospital a
   rejected claim.

Needs network access. `--offline` runs only the structural checks (code format, tier
consistency, full SCP coverage), which is what CI without egress can still enforce.

    python scripts/verify_terminology.py
    python scripts/verify_terminology.py --offline
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ehr.codes import (  # noqa: E402
    ICD10_MAP,
    ICD10CM_PATTERN,
    LOINC_ECG_IMPRESSION,
    LOINC_ECG_STUDY,
    LOINC_MEASUREMENTS,
    LOINC_PATTERN,
    TIER_DEFINITIONAL,
    TIER_NOT_CODABLE,
    TIER_SUGGESTIVE,
)
from src.generation.vocab import VOCAB  # noqa: E402

ICD10_API = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
LOINC_API = "https://clinicaltables.nlm.nih.gov/api/loinc_items/v3/search"
TIMEOUT = 30


def _get(url: str, params: dict) -> list:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(full, timeout=TIMEOUT) as fh:   # noqa: S310 (fixed https host)
        return json.loads(fh.read().decode())


def lookup_icd10(code: str) -> list[tuple[str, str]]:
    """All ICD-10-CM codes whose code field starts with ``code``, as (code, display)."""
    blob = _get(ICD10_API, {"sf": "code", "terms": code, "maxList": 200})
    return [(c, n) for c, n in blob[3]]


def lookup_loinc(code: str) -> list[tuple[str, str]]:
    blob = _get(LOINC_API, {"terms": code, "sf": "LOINC_NUM",
                            "df": "LOINC_NUM,LONG_COMMON_NAME", "maxList": 20})
    return [(r[0], r[1]) for r in (blob[3] or [])]


def structural_checks() -> list[str]:
    """Everything checkable without network: coverage, format, tier consistency."""
    problems: list[str] = []

    missing = set(VOCAB) - set(ICD10_MAP)
    if missing:
        problems.append(f"SCP codes with no ICD-10 mapping: {sorted(missing)}")
    unknown = set(ICD10_MAP) - set(VOCAB)
    if unknown:
        problems.append(f"mappings for codes not in the vocabulary: {sorted(unknown)}")

    for scp, m in sorted(ICD10_MAP.items()):
        if m.tier not in (TIER_DEFINITIONAL, TIER_SUGGESTIVE, TIER_NOT_CODABLE):
            problems.append(f"{scp}: unknown tier {m.tier!r}")
        if m.tier == TIER_NOT_CODABLE:
            if m.icd10 is not None:
                problems.append(f"{scp}: not-codable but carries a code {m.icd10}")
            continue
        if not m.icd10 or not re.match(ICD10CM_PATTERN, m.icd10):
            problems.append(f"{scp}: {m.icd10!r} is not a well-formed ICD-10-CM code")
        if not m.display:
            problems.append(f"{scp}: no display text")
        if m.tier == TIER_SUGGESTIVE and m.icd10 != "R94.31":
            problems.append(f"{scp}: suggestive findings must code to R94.31, got {m.icd10}")
        if m.tier == TIER_DEFINITIONAL and m.candidate:
            problems.append(f"{scp}: definitional mapping should not need a candidate")
        if m.candidate:
            if not re.match(ICD10CM_PATTERN, m.candidate):
                problems.append(f"{scp}: candidate {m.candidate!r} is malformed")
            if not m.requires:
                problems.append(f"{scp}: candidate {m.candidate} has no `requires` evidence "
                                "— a more specific code must state what would justify it")

    for key, lc in {**LOINC_MEASUREMENTS, "study": LOINC_ECG_STUDY,
                    "impression": LOINC_ECG_IMPRESSION}.items():
        if not re.match(LOINC_PATTERN, lc.code):
            problems.append(f"LOINC {key}: {lc.code!r} is malformed")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="structural checks only")
    args = ap.parse_args()

    print("Structural checks")
    problems = structural_checks()
    for p in problems:
        print(f"  FAIL {p}")
    if not problems:
        print(f"  ok — all {len(ICD10_MAP)} SCP codes mapped, formats and tiers consistent")

    if args.offline:
        print("\n(--offline: skipping the live terminology check)")
        return 1 if problems else 0

    # --- ICD-10-CM ------------------------------------------------------------
    print(f"\nICD-10-CM against {ICD10_API}")
    wanted: dict[str, str] = {}
    for m in ICD10_MAP.values():
        if m.icd10:
            wanted[m.icd10] = m.display
        if m.candidate and m.candidate_display:
            wanted[m.candidate] = m.candidate_display

    ok = 0
    for code, display in sorted(wanted.items()):
        try:
            hits = lookup_icd10(code)
        except Exception as e:                                  # noqa: BLE001
            problems.append(f"{code}: lookup failed ({e})")
            print(f"  ERR  {code}: {e}")
            continue
        exact = [(c, n) for c, n in hits if c == code]
        children = [c for c, _ in hits if c.startswith(code) and c != code]
        if not exact:
            problems.append(f"{code}: not found in the current ICD-10-CM release")
            print(f"  FAIL {code}: not found"
                  + (f" — did it become {children}?" if children else ""))
            continue
        official = exact[0][1]
        if official.strip().lower() != display.strip().lower():
            problems.append(f"{code}: display drift — ours {display!r}, official {official!r}")
            print(f"  FAIL {code}: display mismatch\n         ours     {display!r}"
                  f"\n         official {official!r}")
            continue
        if children:
            problems.append(f"{code}: NOT BILLABLE — subdivided into {children}")
            print(f"  FAIL {code}: not billable, has children {children}")
            continue
        ok += 1
    print(f"  {ok}/{len(wanted)} ICD-10-CM codes verified (exist, exact display, billable)")

    # --- LOINC ----------------------------------------------------------------
    print(f"\nLOINC against {LOINC_API}")
    loincs = {**{f"measurement:{k}": v for k, v in LOINC_MEASUREMENTS.items()},
              "study": LOINC_ECG_STUDY, "impression": LOINC_ECG_IMPRESSION}
    lok = 0
    for key, lc in loincs.items():
        try:
            hits = lookup_loinc(lc.code)
        except Exception as e:                                  # noqa: BLE001
            problems.append(f"LOINC {lc.code}: lookup failed ({e})")
            print(f"  ERR  {lc.code}: {e}")
            continue
        exact = [n for c, n in hits if c == lc.code]
        if not exact:
            problems.append(f"LOINC {lc.code} ({key}): not found")
            print(f"  FAIL {lc.code} ({key}): not found")
            continue
        if exact[0].strip().lower() != lc.display.strip().lower():
            problems.append(f"LOINC {lc.code}: display drift — ours {lc.display!r}, "
                            f"official {exact[0]!r}")
            print(f"  FAIL {lc.code}: ours {lc.display!r} vs official {exact[0]!r}")
            continue
        lok += 1
    print(f"  {lok}/{len(loincs)} LOINC codes verified")

    print(f"\n{'FAILED — ' + str(len(problems)) + ' problem(s)' if problems else 'All checks passed.'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
