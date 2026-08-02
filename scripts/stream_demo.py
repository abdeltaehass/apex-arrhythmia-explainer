#!/usr/bin/env python3
"""Phase 16 — live ECG monitor demo in the terminal.

    python scripts/stream_demo.py                      # stream a normal record
    python scripts/stream_demo.py --ecg-id 598         # stream atrial fibrillation
    python scripts/stream_demo.py --playlist 9,598     # normal, then AF: does it notice?
    python scripts/stream_demo.py --speed 4            # 4x fast
    python scripts/stream_demo.py --duration 30        # stop after 30 stream-seconds

Replays a PTB-XL recording at wall-clock pace into :class:`~src.streaming.StreamMonitor`
and redraws a findings panel in place, the way a bedside monitor would. The rolling 10 s
window is re-analyzed every hop; findings are only promoted to CONFIRMED once they persist
(see `docs/streaming/report.md` for why, and what it costs).

The signal is replayed from a recording, not captured live — see the report's limitations
section. This is a Holter/telemetry-style **12-lead** stream; a smartwatch delivers one
lead and cannot feed this model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data.labels import load_database, load_scp_statements, present_codes  # noqa: E402
from src.streaming import (  # noqa: E402
    ReplaySource,
    StreamMonitor,
    build_playlist,
    load_record_signal,
)

DIM, RESET, BOLD = "\033[2m", "\033[0m", "\033[1m"
RED, YELLOW, GREEN, CYAN = "\033[31m", "\033[33m", "\033[32m", "\033[36m"
CLEAR, HOME, HIDE, SHOW = "\033[2J", "\033[H", "\033[?25l", "\033[?25h"

SPARK = " ▁▂▃▄▅▆▇█"


def sparkline(x: np.ndarray, width: int = 60) -> str:
    """A tiny ASCII trace of the most recent signal, for the "is it alive" feel."""
    if x.size == 0:
        return ""
    step = max(1, x.size // width)
    xs = x[::step][:width]
    lo, hi = float(xs.min()), float(xs.max())
    if hi - lo < 1e-9:
        return SPARK[0] * len(xs)
    idx = ((xs - lo) / (hi - lo) * (len(SPARK) - 1)).astype(int)
    return "".join(SPARK[i] for i in idx)


def render(update, lead2: np.ndarray, truth: set[str], desc: dict[str, str],
           elapsed: float, log: list[str]) -> str:
    """Build the whole panel as one string, so the redraw doesn't flicker."""
    alarm = update.alarm
    banner_color = RED if alarm else (YELLOW if update.confirmed else GREEN)
    banner = ("URGENT PATTERN — ESCALATE" if alarm
              else ("FINDINGS PRESENT — clinician review" if update.confirmed
                    else "NO CONFIRMED FINDINGS"))

    out = [
        f"{BOLD}APEX — live monitor{RESET}   "
        f"{DIM}12-lead · {update.window_s:.0f}s rolling window · replayed stream{RESET}",
        f"{banner_color}{BOLD}  {banner}  {RESET}",
        "",
        f"  {CYAN}{sparkline(lead2)}{RESET}  {DIM}lead II{RESET}",
        "",
        f"  {DIM}stream {update.t_s:6.1f}s   wall {elapsed:6.1f}s   "
        f"inference {update.latency_ms:5.1f} ms{RESET}",
        "",
        f"  {BOLD}CONFIRMED{RESET} {DIM}(persisted across windows){RESET}",
    ]
    if update.confirmed:
        for f in update.confirmed:
            mark = f"{RED}!{RESET}" if f.urgent else " "
            tick = "✓" if f.code in truth else f"{DIM}?{RESET}"
            held = f"{f.hits}/{f.window_of}"
            fading = f.releasing(update.confirm_k)
            name = f"{DIM}{f.code:<8}{RESET}" if fading else f"{BOLD}{f.code:<8}{RESET}"
            state = f"{DIM}fading{RESET}" if fading else f"{DIM}since {f.confirmed_at_s:.0f}s{RESET}"
            out.append(f"   {mark} {name} {f.confidence:.2f}  "
                       f"{DIM}held {held}{RESET}  {state}  {tick} "
                       f"{DIM}{desc.get(f.code, '')[:38]}{RESET}")
    else:
        out.append(f"   {DIM}— none —{RESET}")

    out += ["", f"  {BOLD}PENDING{RESET} {DIM}(seen, not yet persistent){RESET}"]
    if update.pending:
        for f in update.pending[:6]:
            out.append(f"     {f.code:<8} {f.confidence:.2f}  "
                       f"{DIM}{f.hits}/{f.window_of} windows{RESET}")
    else:
        out.append(f"   {DIM}— none —{RESET}")

    out += ["", f"  {BOLD}EVENTS{RESET}"]
    out += [f"   {line}" for line in (log[-6:] or [f"{DIM}— none —{RESET}"])]
    out += ["", f"  {DIM}✓ = in this record's ground truth · ? = not in ground truth "
                f"(likely over-flag){RESET}",
            f"  {DIM}Ctrl-C to stop. Decision support only — not a diagnosis.{RESET}"]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ecg-id", type=int, default=9, help="PTB-XL record to stream")
    ap.add_argument("--playlist", type=str, default="",
                    help="comma-separated ecg_ids spliced into one stream (rhythm change)")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time, 0 = as fast as possible")
    ap.add_argument("--duration", type=float, default=60.0, help="stream seconds to play")
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--confirm-k", type=int, default=3)
    ap.add_argument("--confirm-m", type=int, default=5)
    ap.add_argument("--no-clear", action="store_true", help="append instead of redrawing")
    args = ap.parse_args()

    df = load_database()
    scp = load_scp_statements()
    desc = {c: (scp.loc[c, "description"] if c in scp.index else "") for c in scp.index}

    ids = [int(x) for x in args.playlist.split(",") if x.strip()] or [args.ecg_id]
    signals, truth = [], set()
    for ecg_id in ids:
        sig, fs = load_record_signal(ecg_id)
        signals.append(sig)
        truth |= set(present_codes(df.loc[ecg_id, "scp_codes"]))
    signal = build_playlist(signals, fs=fs) if len(signals) > 1 else signals[0]
    label = "+".join(str(i) for i in ids)

    print(f"streaming ecg {label} ({signal.shape[1] / fs:.0f}s of signal, looped) "
          f"at {args.speed}x — warming model...")
    mon = StreamMonitor(fs=fs, hop_s=args.hop, confirm_k=args.confirm_k,
                        confirm_m=args.confirm_m)
    mon._ensure_model()          # pay the cold load before the clock starts

    src = ReplaySource(signal, fs=fs, chunk_s=0.2, speed=args.speed, loop=True,
                       max_duration_s=args.duration, source=label)
    log: list[str] = []
    import time
    t0 = time.monotonic()
    if not args.no_clear:
        sys.stdout.write(HIDE + CLEAR)
    try:
        for chunk in src:
            u = mon.push(chunk.samples)
            if u is None:
                continue
            for e in u.events:
                colour = RED if e.urgent else (GREEN if e.kind == "onset" else DIM)
                verb = "ONSET " if e.kind == "onset" else "resolved"
                log.append(f"{colour}[{e.t_s:6.1f}s] {verb} {e.code} "
                           f"({e.confidence:.2f}){RESET}")
            frame = render(u, mon._buf.window(min(400, mon.window_samples))[1],
                           truth, desc, time.monotonic() - t0, log)
            if args.no_clear:
                print(frame + "\n")
            else:
                sys.stdout.write(HOME + CLEAR + frame)
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_clear:
            sys.stdout.write(SHOW + "\n")
        sys.stdout.flush()

    confirmed = sorted(f.code for f in (mon._state.values()) if f.confirmed)
    print(f"\nstream ended. confirmed: {confirmed or '—'}")
    print(f"ground truth:  {sorted(truth)}")
    print(f"missed:        {sorted(truth - set(confirmed)) or '—'}")
    print(f"over-flagged:  {sorted(set(confirmed) - truth) or '—'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
