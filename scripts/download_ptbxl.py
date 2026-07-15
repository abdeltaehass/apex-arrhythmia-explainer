#!/usr/bin/env python3
"""Download and verify the PTB-XL dataset from PhysioNet.

PTB-XL is open access (no credentialing / DUA required). This script downloads
the versioned release archive, verifies it, and unpacks it into
``data/raw/ptbxl/``.

Usage:
    python scripts/download_ptbxl.py                 # full dataset (~3 GB, 100+500 Hz)
    python scripts/download_ptbxl.py --metadata-only # just ptbxl_database.csv + scp_statements.csv
    python scripts/download_ptbxl.py --force         # re-download even if present

The full archive is large. It is downloaded to a temp file and only moved into
place after extraction succeeds, so partial downloads never look complete.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Pinned version so runs are reproducible. Update deliberately, not accidentally.
PTBXL_VERSION = "1.0.3"
BASE = f"https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-{PTBXL_VERSION}.zip"
# Lightweight metadata files (served individually by PhysioNet).
META_BASE = f"https://physionet.org/files/ptb-xl/{PTBXL_VERSION}"
META_FILES = ("ptbxl_database.csv", "scp_statements.csv")
# Per-record waveform sources. PhysioNet throttles bandwidth per connection; the AWS
# Open Data S3 mirror does not, so it is the default for the bulk per-file download.
RECORD_SOURCES = {
    "s3": f"https://physionet-open.s3.amazonaws.com/ptb-xl/{PTBXL_VERSION}",
    "physionet": META_BASE,
}

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "ptbxl"


_last_pct = -1.0


def _report(block_num: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    global _last_pct
    done = min(block_num * block_size, total)
    pct = 100 * done / total
    # Throttle: only redraw every whole percent so logs/pipes stay readable.
    if pct - _last_pct < 1.0 and done < total:
        return
    _last_pct = pct
    mb = done / 1e6
    sys.stdout.write(f"\r  {pct:5.1f}%  ({mb:,.0f} MB)")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")
        _last_pct = -1.0


def download_metadata(force: bool = False) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in META_FILES:
        out = DEST / name
        if out.exists() and not force:
            print(f"  [skip] {name} already present")
            continue
        url = f"{META_BASE}/{name}"
        print(f"  [get ] {name}")
        urllib.request.urlretrieve(url, out, _report)  # noqa: S310 (trusted host)
        print()


def download_full(force: bool = False) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    marker = DEST / f".complete-{PTBXL_VERSION}"
    if marker.exists() and not force:
        print(f"PTB-XL {PTBXL_VERSION} already downloaded at {DEST}")
        return

    tmp_zip = DEST / f"_ptbxl-{PTBXL_VERSION}.zip.part"
    print(f"Downloading PTB-XL {PTBXL_VERSION} (this is several GB)...")
    urllib.request.urlretrieve(BASE, tmp_zip, _report)  # noqa: S310
    print("\nExtracting...")
    with zipfile.ZipFile(tmp_zip) as zf:
        zf.extractall(DEST)
    tmp_zip.unlink(missing_ok=True)

    # The archive extracts into a long-named subfolder; flatten it.
    inner = next((p for p in DEST.iterdir() if p.is_dir() and p.name.startswith("ptb-xl")), None)
    if inner is not None:
        for item in inner.iterdir():
            shutil.move(str(item), str(DEST / item.name))
        inner.rmdir()

    marker.touch()
    print(f"Done. PTB-XL {PTBXL_VERSION} is at {DEST}")


_SESSION = None  # pooled, keepalive HTTP session (set up in download_records)
_REC_BASE = RECORD_SOURCES["s3"]  # active waveform source


def _get_file(rel: str, force: bool) -> str | None:
    """Download one record file (<rel>) unless present. Returns rel on failure."""
    out = DEST / rel
    if out.exists() and not force:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = _SESSION.get(f"{_REC_BASE}/{rel}", timeout=30)
        r.raise_for_status()
        # Guard against truncated bodies (a 200 with a short/dropped payload): the
        # written file must match the advertised Content-Length, or we discard + retry.
        expected = int(r.headers.get("Content-Length", len(r.content)))
        if len(r.content) != expected:
            raise OSError(f"short read {len(r.content)}/{expected}")
        out.write_bytes(r.content)
        return None
    except Exception:  # noqa: BLE001 - transient errors reported for a resume/retry
        out.unlink(missing_ok=True)  # never leave a partial file behind
        return rel


def download_records(rate: int = 100, workers: int = 32, force: bool = False,
                     source: str = "s3") -> None:
    """Download only the <rate> Hz waveforms (parallel), skipping the other rate.

    Phase 3 trains at 100 Hz, so this pulls only the ~0.5 GB of 100 Hz records instead of
    the full archive. Defaults to the AWS S3 mirror (unthrottled); pass source="physionet"
    to use physionet.org. Idempotent: existing files are skipped, so it resumes cleanly.
    """
    global _SESSION, _REC_BASE
    import pandas as pd
    import requests
    from requests.adapters import HTTPAdapter

    download_metadata(force=False)
    marker = DEST / f".records{rate}-complete-{PTBXL_VERSION}"
    if marker.exists() and not force:
        print(f"records{rate} for {PTBXL_VERSION} already complete at {DEST}")
        return

    _REC_BASE = RECORD_SOURCES[source]
    _SESSION = requests.Session()
    _SESSION.mount("https://", HTTPAdapter(pool_connections=workers, pool_maxsize=workers, max_retries=3))
    print(f"source: {_REC_BASE}")

    col = "filename_lr" if rate == 100 else "filename_hr"
    rels = pd.read_csv(DEST / "ptbxl_database.csv")[col].tolist()
    files = [rel + ext for rel in rels for ext in (".hea", ".dat")]
    todo = [f for f in files if force or not (DEST / f).exists()]
    print(f"records{rate}: {len(files)} files total, {len(todo)} to fetch, {workers} workers")

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_get_file, f, force): f for f in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            if (bad := fut.result()) is not None:
                failed.append(bad)
            if i % 2000 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done ({len(failed)} failed)")

    if failed:
        print(f"WARNING: {len(failed)} files failed (e.g. {failed[:3]}). Re-run to retry.")
    else:
        marker.touch()
        print(f"Done. records{rate} ({len(files)} files) at {DEST}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata-only", action="store_true", help="download only the CSV metadata")
    ap.add_argument("--records", type=int, choices=(100, 500),
                    help="download only this sampling rate's waveforms (parallel, subset)")
    ap.add_argument("--workers", type=int, default=32, help="parallel download workers for --records")
    ap.add_argument("--source", choices=("s3", "physionet"), default="s3",
                    help="waveform mirror for --records (s3 is unthrottled; default)")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    if args.metadata_only:
        download_metadata(force=args.force)
    elif args.records:
        download_records(rate=args.records, workers=args.workers, force=args.force, source=args.source)
    else:
        download_full(force=args.force)
        download_metadata(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
