"""Central configuration for APEX.

Single source of truth for paths, dataset constants, target metrics, and the
Weights & Biases project. Import `CFG` everywhere rather than hard-coding paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_DIR = DATA_DIR / "manifests"
PTBXL_DIR = RAW_DIR / "ptbxl"  # populated by scripts/download_ptbxl.py


# --- Dataset constants -----------------------------------------------------
# PTB-XL: 21,837 records, 10 s each, at two sampling rates.
NUM_LEADS = 12
RECORD_SECONDS = 10
SAMPLING_RATES = (100, 500)  # Hz; low-res for fast iteration, high-res for final
DEFAULT_SAMPLING_RATE = 100

# 71 SCP-ECG statement categories (the multi-label detection target).
NUM_LABELS = 71

# Official PTB-XL stratified folds.
TRAIN_FOLDS = (1, 2, 3, 4, 5, 6, 7, 8)
VAL_FOLD = 9
TEST_FOLD = 10


# --- Weights & Biases ------------------------------------------------------
@dataclass
class WandbConfig:
    project: str = "apex-arrhythmia-explainer"
    entity: str | None = os.environ.get("WANDB_ENTITY")  # your W&B username/team
    # "online" logs to the cloud; set WANDB_MODE=offline for no-network runs.
    mode: str = os.environ.get("WANDB_MODE", "online")
    tags: tuple[str, ...] = ("ptb-xl", "multi-label", "ecg")


# --- Target metrics (mirrors docs/target_metrics.md) -----------------------
@dataclass
class Targets:
    macro_auroc: float = 0.90
    macro_f1: float = 0.75
    micro_f1: float = 0.80
    ece: float = 0.05           # max acceptable calibration error
    consistency_rate: float = 0.98
    hallucination_rate: float = 0.02  # max acceptable
    grounding_coverage: float = 0.95
    p50_latency_s: float = 1.5
    p95_latency_s: float = 4.0


@dataclass
class Config:
    seed: int = 42
    sampling_rate: int = DEFAULT_SAMPLING_RATE
    # confidence below this routes a prediction to manual review
    review_threshold: float = 0.5
    # a SEPARATE, higher bar (Phase 7): a label can clear `review_threshold` and still
    # be surfaced/asserted, but if its confidence sits below this it is additionally
    # tagged "low confidence — manual review recommended" by src/eval/reliability.py.
    low_confidence_threshold: float = 0.7
    wandb: WandbConfig = field(default_factory=WandbConfig)
    targets: Targets = field(default_factory=Targets)


CFG = Config()
