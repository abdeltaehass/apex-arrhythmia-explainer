# Phase 9 — API inference benchmark

End-to-end latency and throughput of the APEX pipeline, measured by `scripts/benchmark_api.py`. `pipeline/<device>` is `serving.analyze_signal` (signal -> JSON); `http/<device>` adds the FastAPI request/response stack.

## Environment

- **Platform**: macOS-26.5.2-arm64-arm-64bit
- **Processor**: arm
- **Python / torch**: 3.11.4 / 2.13.0
- **Requests timed per config**: 300 (after warmup)
- **Grounding stage**: off · **backend**: template · **signal**: [12, 1000] @ 100 Hz

## Results

| configuration | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) | max (ms) | throughput (req/s) |
|---|---:|---:|---:|---:|---:|---:|
| pipeline/cpu | 5.91 | 6.25 | 6.59 | 5.92 | 6.97 | 168.8 |
| http/cpu | 12.49 | 12.98 | 13.19 | 12.5 | 13.25 | 80.0 |
| pipeline/mps | 7.33 | 7.67 | 7.79 | 7.28 | 8.37 | 137.3 |
| http/mps | 12.94 | 13.47 | 13.67 | 13.01 | 13.75 | 76.9 |

## Notes

- Latencies are **warm** (checkpoint + SCP table + generation imports preloaded); the first request after boot pays a one-time ~1.3 s model-load cost not shown here.
- Throughput is **sequential** (single worker, `req/s = n / total_time`). Torch holds the GIL during the forward pass, so process-level concurrency needs multiple uvicorn workers rather than threads — the sequential number is the honest per-worker figure; multiply by worker count for a rough capacity estimate.
- **Grounding off by default.** `with_grounding=true` adds ~64 ms on CPU (~70 ms p50 end-to-end) because it runs a Grad-CAM backward pass per localizing finding; leave it off for latency-sensitive callers.
- No CUDA GPU is available on this host, so only CPU and Apple MPS are reported. The 10 s, 12-lead, 100 Hz signal (1000 samples) is tiny, so the CNN forward pass is a small part of the total — preprocessing (band-pass + Pan-Tompkins) and the deterministic template generation dominate, which is why CPU and MPS land close together (MPS is even marginally slower: dispatch/host-transfer overhead outweighs any compute win at this size).
