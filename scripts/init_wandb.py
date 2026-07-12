#!/usr/bin/env python3
"""Initialize the Weights & Biases project for APEX.

Run once after `wandb login` to create the project and log the target-metric
baselines so the dashboard has the goal lines from day one.

    export WANDB_ENTITY=<your-wandb-username-or-team>
    wandb login
    python scripts/init_wandb.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG  # noqa: E402


def main() -> int:
    try:
        import wandb
    except ImportError:
        print("wandb is not installed. Run: pip install -r requirements.txt")
        return 1

    run = wandb.init(
        project=CFG.wandb.project,
        entity=CFG.wandb.entity,
        mode=CFG.wandb.mode,
        tags=list(CFG.wandb.tags) + ["phase0-init"],
        job_type="setup",
        name="phase0-init",
        config={
            "seed": CFG.seed,
            "num_labels": CFG.__dict__.get("num_labels", 71),
            "sampling_rate": CFG.sampling_rate,
            "review_threshold": CFG.review_threshold,
            "targets": vars(CFG.targets),
        },
    )
    # Log target lines so charts show the goals immediately.
    wandb.summary["target/macro_auroc"] = CFG.targets.macro_auroc
    wandb.summary["target/macro_f1"] = CFG.targets.macro_f1
    wandb.summary["target/hallucination_rate"] = CFG.targets.hallucination_rate
    wandb.summary["target/p95_latency_s"] = CFG.targets.p95_latency_s
    print(f"Initialized W&B project '{CFG.wandb.project}'.")
    print(f"View it at: {run.get_url()}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
