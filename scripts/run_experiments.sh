#!/usr/bin/env bash
# Phase 4 model-improvement sweep. Each run trains 20 epochs, evaluates on the val and
# test folds, writes a per-label AUROC table under docs/model_comparison/<run>/, and
# appends a record to docs/model_comparison/runs.jsonl. Then build the comparison table.
#
# Run with the project venv active (or: PATH=.venv/bin:$PATH bash scripts/run_experiments.sh).
set -euo pipefail

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_SILENT=true
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"
PY="python -m src.detection.train"

# 1) CNN + class-weighted BCE  (the Phase 3 baseline, re-run for uniform logging)
$PY --model cnn --loss bce --eval-test

# 2) CNN + focal loss  (more aggressive imbalance handling)
$PY --model cnn --loss focal --eval-test

# 3) 1D transformer (PatchTST-style) + class-weighted BCE
$PY --model transformer --loss bce --d-model 192 --depth 4 --heads 6 --lr 5e-4 --eval-test

# 4) 1D transformer + focal loss
$PY --model transformer --loss focal --d-model 192 --depth 4 --heads 6 --lr 5e-4 --eval-test

python scripts/build_comparison.py
echo "Phase 4 sweep complete. See docs/model_comparison/comparison.md"
