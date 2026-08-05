#!/usr/bin/env bash
# Phase 19 distillation sweep.
#
# Three student capacities, each trained twice: once with the teacher's soft labels
# (`--alpha 0.7`) and once with ground truth alone (`--alpha 0`). The second run is the
# control that makes the result mean something — without it, "the small model works fine"
# is indistinguishable from "distillation helped", because a 254k-parameter CNN may simply
# be enough for this task. Everything else (architecture, data, schedule, seed) is held
# fixed, so the only variable is the teacher signal.
#
# Each run writes a per-label AUROC table under docs/distillation/<run>/ and appends a
# record to docs/distillation/runs.jsonl. Then build the trade-off report.
#
# Run with the project venv active (or: PATH=.venv/bin:$PATH bash scripts/run_distillation.sh).
set -euo pipefail

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_SILENT=true
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"
PY="python -m src.detection.distill"

for W in 8 16 32; do
  $PY --width "$W" --blocks 1 --alpha 0.7 --temperature 2.0   # distilled
  $PY --width "$W" --blocks 1 --alpha 0.0                     # from-scratch control
done

python scripts/distill_report.py
echo "Phase 19 sweep complete. See docs/distillation/report.md"
