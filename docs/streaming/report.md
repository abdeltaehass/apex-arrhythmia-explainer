# Phase 16 — Real-time streaming behaviour

40 PTB-XL test records replayed as live streams (40 s each, 1 s hop, 10 s rolling window). Regenerate with `python scripts/stream_eval.py`.

## The problem streaming creates

Batch APEX makes one decision per recording. A monitor makes a new one every hop. Phase 13 measured 5.09 spurious labels per record at the shipped 0.5 threshold — in a stream that is a fresh handful of spurious labels *every second*, and the findings panel becomes unreadable. The batch error profile does not transfer.

## Naive panel vs temporal persistence

Persistence rule: a finding is confirmed once it holds in **3 of the last 5** windows, and released only after it is absent from all 5 (hysteresis, so findings don't flap at the boundary).

| | naive (per-window) | persistent (N-of-M) |
|---|---:|---:|
| Panel changes per record | 24.4 | **5.7** |
| Distinct spurious codes shown | 9.68 | **7.55** |
| True-finding recall | 0.891 | **0.849** |

**Panel churn falls 76%** — the display stops flickering and becomes readable, which is the whole point.

**Two costs, both real, neither hidden:**

1. **Recall drops 0.891 → 0.849** (4.2 points). Requiring persistence discards true findings that only appear intermittently. This is not a free win, and anyone reporting only the churn number is telling half the story.
2. **Detection latency: 3.3 s** on average between a finding first clearing threshold and being announced. For continuous monitoring that is a fine trade; for a time-critical alarm it may not be — which is why `confirm_k` is a parameter, and why the curve below matters more than any single setting.

### The trade-off curve (`confirm_m` = 5, n = 20 records)

| `confirm_k` | panel changes | spurious/rec | true recall | confirm lag |
|---|---:|---:|---:|---:|
| 1 of 5 | 10.3 | 8.55 | 0.857 | 0.0 s |
| 2 of 5 | 7.4 | 7.40 | 0.839 | 2.3 s |
| 3 of 5 | 5.7 | 6.60 | 0.821 | 3.5 s |
| 4 of 5 | 4.1 | 5.50 | 0.786 | 4.7 s |

`confirm_k = 1` is the naive panel with hysteresis on release only. Raising `k` buys stability and pays in recall and latency, monotonically. There is no free setting here — the right one depends on whether the deployment is a ward monitor (favour stability) or an arrhythmia alarm (favour latency).

## What persistence does *not* fix

**Systematically over-confident labels survive it.** Debouncing removes findings that *flicker*; a spurious code that the miscalibrated model asserts in every single window is confirmed just like a real one. Phase 13 traced the over-flagging to calibration (ECE ≈ 0.90), and this phase does not fix that — it only removes the noise on top. Calibration remains the outstanding work, and it is what would move the remaining 7.55 spurious codes per record.

## Real-time feasibility

| | |
|---|---:|
| Median inference per window | **6.6 ms** |
| p95 | 7.3 ms |
| First window (cold model load) | 775 ms |
| Hop budget | 1000 ms |
| Headroom | **~150.9x** |

A 10 s window is re-analyzed every 1 s in 6.6 ms on CPU — roughly **150.9x faster than real time**, so a single core could carry many concurrent streams. The cold first window pays a one-off model load; the monitor warms the cache before the buffer fills in any real deployment.

## Honest limitations of this simulation

- **The stream is replayed, not live.** PTB-XL records are 10 s — exactly one window — so a continuous stream requires looping (or splicing a playlist). Joins are raised-cosine crossfaded to avoid injecting a step discontinuity, but a window spanning a join is still part-one-record, part-another. That is a property of the simulation, not of a real monitor.
- **Looping repeats the same 10 s.** Windows are rotations of one recording, so consecutive windows are more correlated than genuinely new data would be. This *flatters* the persistence layer: real streams vary more, so real churn would be higher and persistence more valuable, but the exact numbers above would move.
- **12-lead only.** An Apple Watch delivers a single lead and a typical Holter three; the detector's input convolution is fixed at 12 and the service hard-rejects anything else. This simulation is therefore *Holter/telemetry-style 12-lead*, not a smartwatch. Supporting single-lead input needs a retrained model, not an adapter.
- **No generation or grounding in the loop.** Each hop runs detection only; the explanation and grounding layers run on demand, not per window.
- **Ground truth is per-record, not per-window.** A looped record carries the same labels throughout, so "recall" here means "did the stream ever confirm the record's true codes", not a time-resolved measurement.
