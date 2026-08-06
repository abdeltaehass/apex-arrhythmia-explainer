#!/usr/bin/env bash
# Phase 20 federated-learning sweep.
#
# Every run costs the same 30 epoch-equivalents of gradient work (rounds x local_epochs),
# so the configurations are comparable at fixed compute and the only thing varying is how
# that work is divided between local computation and communication.
#
# The two baselines at the end are what make the federated numbers interpretable:
# "centralized" is Phase 4's model (already trained, docs/model_comparison/runs.jsonl), and
# "local-only" is a hospital that refuses to federate and trains on its own data alone.
# FedAvg is only worth anything if it lands between them.
#
# Run with the project venv active (or: PATH=.venv/bin:$PATH bash scripts/run_federated.sh).
set -euo pipefail

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"
PY="python -m src.federated.train"

# --- FedAvg on the natural (non-IID) device split, at fixed compute -----------
$PY --by device --rounds 30 --local-epochs 1     # most communication, least drift
$PY --by device --rounds 15 --local-epochs 2
$PY --by device --rounds 6  --local-epochs 5     # least communication, most drift

# --- Control: same client count and sizes, label skew removed ----------------
$PY --by iid    --rounds 30 --local-epochs 1

# --- Baselines: single hospitals that never federate -------------------------
$PY --local-only "CS100    3" --rounds 30 --local-epochs 1   # largest client (35% of data)
$PY --local-only "CS-12   E" --rounds 30 --local-epochs 1    # most skewed client (82% NORM)

# --- Run to convergence ------------------------------------------------------
# The fixed-compute runs above all peak at their final round, i.e. they are still
# improving when the budget ends. Quoting a gap from an unconverged curve would overstate
# what federation costs, so the best of those settings (E=5) is also run far past the
# budget until validation AUROC plateaus. These are the runs the headline is drawn from.
$PY --by device --rounds 20 --local-epochs 5
$PY --by iid    --rounds 20 --local-epochs 5

# --- Diagnostic: is the gap BatchNorm? ---------------------------------------
# GroupNorm keeps no cross-batch running statistics, so there is nothing distribution-
# dependent for the server to average. If BN-stat averaging were driving the gap, this
# closes it.
$PY --by device --rounds 20 --local-epochs 5 --norm gn

python scripts/federated_report.py
echo "Phase 20 sweep complete. See docs/federated/report.md"
