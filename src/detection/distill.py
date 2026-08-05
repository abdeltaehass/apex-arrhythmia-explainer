#!/usr/bin/env python3
"""Phase 19 — knowledge distillation of the detector into a lightweight student.

    python -m src.detection.distill --width 16 --blocks 1              # KD student
    python -m src.detection.distill --width 16 --blocks 1 --alpha 0    # scratch control
    python -m src.detection.distill --smoke                           # tiny end-to-end check

The Phase-4 winner (`cnn_bce`, 8.8M parameters, 33.6 MB) is far larger than this task
needs, and every millisecond of its forward pass is paid on every request. Distillation
trains a small student to imitate the **teacher's probabilities** rather than only the
0/1 ground truth.

**Why soft targets carry more than hard ones.** A ground-truth vector says "inferior
MI: yes, everything else: no". The teacher's output says "inferior MI 0.91, lateral MI
0.34, non-diagnostic T abnormalities 0.22, atrial fibrillation 0.002" — it encodes which
*other* findings this ECG resembles, and by how much. Hinton et al. (2015) call that the
dark knowledge: the ratios among the wrong answers describe the geometry the teacher
learned, and for a 71-label problem where labels genuinely co-occur (an anterior MI does
raise the odds of ST-T change) that structure is exactly what a small model cannot
easily discover on its own from sparse binary labels.

**The multi-label form is not the softmax one.** The usual KD loss is a KL between two
softmax distributions over mutually exclusive classes. Here the 71 outputs are
independent sigmoids — conditions coexist — so the correct object is a *per-label
Bernoulli KL*, summed over labels (:class:`BinaryKDLoss`). Using a softmax KD loss on
sigmoid outputs would force the labels to compete for a fixed probability budget and
quietly destroy the multi-label semantics.

**Temperature.** ``T`` softens both sides: ``sigmoid(z / T)`` pulls probabilities toward
0.5, which magnifies the small teacher probabilities that hold the inter-label
structure. Gradients through a ``1/T``-scaled logit shrink as ``1/T``, so the KD term is
multiplied by ``T²`` to keep its gradient magnitude comparable to the hard-label term as
``T`` varies — the standard Hinton correction.

The teacher is frozen and the cached training tensors are not augmented, so teacher
logits are computed **once** and cached to ``outputs/`` rather than recomputed every
epoch; distillation then costs no more per epoch than ordinary training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.config import CFG, ROOT
from src.data.labels import build_label_space
from src.detection.data_cache import build_split_cache
from src.detection.losses import build_loss
from src.detection.model import build_model, count_parameters
from src.detection.train import (
    SMOKE_TRAIN_IDS,
    SMOKE_VAL_IDS,
    eval_split,
    pick_device,
    pos_weight_from,
    predict,
    set_seed,
    write_auroc_table,
)
from src.eval.metrics import expected_calibration_error, f1_scores, macro_auroc

OUT_DIR = ROOT / "outputs"
DISTILL_DIR = ROOT / "docs" / "distillation"
TEACHER_CKPT = OUT_DIR / "final_best.pt"


# --- losses ------------------------------------------------------------------
class BinaryKDLoss(nn.Module):
    """Temperature-scaled per-label Bernoulli KL, ``KL(teacher ‖ student)``.

    For each of the L independent sigmoid outputs, with ``q = σ(z_T / T)`` and
    ``p = σ(z_S / T)``::

        KL = q·log(q/p) + (1−q)·log((1−q)/(1−p))

    averaged over labels and batch, and multiplied by ``T²``.

    *Averaged*, not summed, over labels — softmax KD sums over classes because they form
    one distribution, but here there are L separate distributions and the hard-label term
    (``BCEWithLogitsLoss``) already averages over them. Summing would silently scale the
    soft term by 71x and make ``alpha`` mean nothing; matching the hard term's reduction
    keeps ``alpha`` an honest interpolation weight.

    Minimizing this KL is equivalent (in gradient) to minimizing the cross-entropy
    ``BCEWithLogits(z_S / T, target=q)``, since the two differ only by the teacher's
    entropy — a constant with respect to the student. The KL form is used anyway because
    its *value* is interpretable: it is exactly zero when the student reproduces the
    teacher, which makes divergence readable during training and testable in unit tests.
    """

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        t = self.temperature
        zs, zt = student_logits / t, teacher_logits / t
        # log-sigmoid form: exact and stable for large |z| (no log of a clamped sigmoid).
        log_p, log_1mp = F.logsigmoid(zs), F.logsigmoid(-zs)
        log_q, log_1mq = F.logsigmoid(zt), F.logsigmoid(-zt)
        q = torch.sigmoid(zt)
        kl = q * (log_q - log_p) + (1.0 - q) * (log_1mq - log_1mp)
        return (t * t) * kl.mean()


class DistillationLoss(nn.Module):
    """``α · T² · KL(teacher‖student) + (1 − α) · hard_loss(student, y)``.

    ``alpha=0`` reduces to the ordinary supervised objective, which is how the
    train-from-scratch control is run: identical architecture, data, schedule and seed,
    with only the teacher signal removed. ``alpha=1`` ignores the ground truth entirely.

    The hard term keeps the teacher's class-weighted BCE (per-label ``pos_weight``). The
    soft term is *not* class-weighted: the teacher's probabilities already reflect the
    weighting it was trained under, so re-applying ``pos_weight`` there would double-count
    the imbalance correction.
    """

    def __init__(self, hard_loss: nn.Module, temperature: float = 2.0, alpha: float = 0.7):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.hard_loss = hard_loss
        self.kd = BinaryKDLoss(temperature)
        self.alpha = float(alpha)

    def forward(
        self,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Returns ``(total_loss, {"hard": ..., "soft": ...})`` for logging."""
        hard = self.hard_loss(student_logits, targets)
        soft = (
            self.kd(student_logits, teacher_logits)
            if self.alpha > 0
            else torch.zeros((), device=student_logits.device)
        )
        total = self.alpha * soft + (1.0 - self.alpha) * hard
        return total, {"hard": float(hard.detach()), "soft": float(soft.detach())}


# --- teacher logits ----------------------------------------------------------
def _ckpt_tag(checkpoint: Path) -> str:
    """Short content hash of a checkpoint, so a cache can't outlive its teacher."""
    h = hashlib.sha256()
    with open(checkpoint, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


@torch.no_grad()
def compute_logits(model: nn.Module, X: np.ndarray, device, batch: int = 256) -> np.ndarray:
    """Raw pre-sigmoid logits for a cached ``(N, 12, T)`` array."""
    model.eval()
    out = []
    for s in range(0, len(X), batch):
        xb = torch.from_numpy(X[s : s + batch]).to(device)
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


def teacher_logits_for(
    split: str,
    X: np.ndarray,
    checkpoint: Path = TEACHER_CKPT,
    device: str = "cpu",
    use_cache: bool = True,
) -> np.ndarray:
    """Teacher logits for a split, memoized on disk.

    The teacher is frozen and the cached tensors are not augmented, so its outputs are a
    pure function of ``(checkpoint, split)``. Recomputing them every epoch would make
    distillation strictly slower than normal training for no reason; caching makes it
    free. The cache filename carries a hash of the checkpoint bytes, so pointing at a
    different (or retrained) teacher can never silently reuse stale logits.
    """
    from src.grounding import load_detector

    cache = OUT_DIR / f"teacher_logits_{split}_{_ckpt_tag(checkpoint)}.npy"
    if use_cache and cache.exists():
        z = np.load(cache)
        if len(z) == len(X):
            return z

    model, _, _ = load_detector(checkpoint, device=device)
    z = compute_logits(model, X, device).astype(np.float32)
    if use_cache:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, z)
    return z


def agreement(student_prob: np.ndarray, teacher_prob: np.ndarray, threshold: float) -> dict:
    """How closely the student reproduces the teacher, independent of ground truth.

    A student can match the teacher's AUROC while making different *decisions*; for a
    drop-in replacement that matters as much as the aggregate score. Reports mean absolute
    probability difference and the rate at which both models land on the same side of the
    surfacing threshold.
    """
    same = (student_prob >= threshold) == (teacher_prob >= threshold)
    return {
        "mean_abs_prob_diff": float(np.abs(student_prob - teacher_prob).mean()),
        "decision_agreement": float(same.mean()),
        "decision_agreement_positives": (
            float(same[teacher_prob >= threshold].mean())
            if (teacher_prob >= threshold).any()
            else float("nan")
        ),
    }


# --- training ----------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="cnn", choices=("cnn", "transformer"))
    ap.add_argument("--teacher", default=str(TEACHER_CKPT), help="teacher checkpoint")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="weight on the KD term; 0 = train-from-scratch control")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--loss", default="bce", choices=("bce", "focal"),
                    help="the hard-label term")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--run-name", default=None, help="defaults to student_w<width>_<kd|scratch>")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    # student capacity
    ap.add_argument("--width", type=int, default=16, help="stem channels (teacher uses 64)")
    ap.add_argument("--blocks", type=int, default=1, help="residual blocks/stage (teacher uses 2)")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--pos-weight-cap", type=float, default=50.0)
    ap.add_argument("--sampling-rate", type=int, default=100, choices=(100, 500))
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-eval-test", action="store_true", help="skip the test-fold report")
    ap.add_argument("--wandb-mode", default=CFG.wandb.mode, choices=("online", "offline", "disabled"))
    ap.add_argument("--smoke", action="store_true", help="tiny run on the sample records")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.batch_size, args.num_workers = min(args.epochs, 2), 2, 0
        args.wandb_mode = "offline"
        args.no_eval_test = True
    kind = "scratch" if args.alpha == 0 else "kd"
    run_name = args.run_name or f"student_w{args.width}b{args.blocks}_{kind}"

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"run={run_name}  width={args.width} blocks={args.blocks}  alpha={args.alpha} "
          f"T={args.temperature}  device={device}  epochs={args.epochs}")

    tr_ids = SMOKE_TRAIN_IDS if args.smoke else None
    va_ids = SMOKE_VAL_IDS if args.smoke else None
    print("loading/caching data...")
    Xtr, Ytr = build_split_cache("train", args.sampling_rate, num_workers=args.num_workers,
                                 ecg_ids=tr_ids)
    Xva, Yva = build_split_cache("val", args.sampling_rate, num_workers=args.num_workers,
                                 ecg_ids=va_ids)
    print(f"train {Xtr.shape}  val {Xva.shape}")

    # Teacher logits: computed once (frozen teacher, un-augmented cache), then reused.
    teacher_path = Path(args.teacher)
    if args.alpha > 0:
        t0 = time.time()
        # On the training device: a full pass over ~17k records is minutes on CPU and
        # seconds on MPS/CUDA. Paid once, then read from the cache by every later run.
        Ztr = teacher_logits_for("train", Xtr, teacher_path, device=str(device),
                                 use_cache=not args.smoke)
        print(f"teacher logits {Ztr.shape} ready in {time.time() - t0:.1f}s")
    else:
        Ztr = np.zeros_like(Ytr)  # unused when alpha == 0; keeps the loader signature fixed

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr), torch.from_numpy(Ztr)),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)),
                            batch_size=256)

    label_space = build_label_space()
    student = build_model(args.model, width=args.width, blocks=args.blocks,
                          dropout=args.dropout).to(device)
    n_params = count_parameters(student)
    teacher_params = count_parameters(
        build_model("cnn", width=64, blocks=2, dropout=0.2)
    )
    print(f"student params: {n_params:,}  ({teacher_params / n_params:.1f}x smaller than teacher)")

    pos_weight = pos_weight_from(Ytr, args.pos_weight_cap).to(device)
    criterion = DistillationLoss(
        build_loss(args.loss, pos_weight=pos_weight, focal_gamma=args.focal_gamma),
        temperature=args.temperature, alpha=args.alpha,
    ).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    import wandb

    wandb.init(project=CFG.wandb.project, name=run_name, mode=args.wandb_mode,
               config={**vars(args), "params": n_params,
                       "compression": teacher_params / n_params},
               tags=list(CFG.wandb.tags) + ["distillation", kind])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = OUT_DIR / f"{run_name}_best.pt"
    best_auroc, best_epoch, train_time = -1.0, 0, 0.0
    for epoch in range(1, args.epochs + 1):
        student.train()
        t0, running, parts_sum = time.time(), 0.0, {"hard": 0.0, "soft": 0.0}
        for xb, yb, zb in train_loader:
            xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
            opt.zero_grad()
            loss, parts = criterion(student(xb), yb, zb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
            for k, v in parts.items():
                parts_sum[k] += v * len(xb)
        sched.step()
        train_time += time.time() - t0
        n_seen = max(1, len(train_loader) * args.batch_size)
        train_loss = running / n_seen

        y_true, y_prob = predict(student, val_loader, device)
        m = {
            "macro_auroc": macro_auroc(y_true, y_prob),
            "ece": expected_calibration_error(y_true, y_prob),
            **f1_scores(y_true, (y_prob >= 0.5).astype(int)),
        }
        wandb.log({"epoch": epoch, "train_loss": train_loss, "lr": sched.get_last_lr()[0],
                   "hard_loss": parts_sum["hard"] / n_seen,
                   "kd_loss": parts_sum["soft"] / n_seen, **m})
        print(f"epoch {epoch:2d}/{args.epochs}  loss={train_loss:.4f} "
              f"(hard {parts_sum['hard'] / n_seen:.4f} / kd {parts_sum['soft'] / n_seen:.4f})  "
              f"macroAUROC={m['macro_auroc']:.4f}  ({time.time() - t0:.1f}s)")

        if m["macro_auroc"] > best_auroc:
            best_auroc, best_epoch = m["macro_auroc"], epoch
            # `args` carries width/blocks/dropout, so src.grounding.load_detector can
            # rebuild this student with no extra bookkeeping — it is a drop-in checkpoint.
            torch.save({"model": student.state_dict(), "args": vars(args),
                        "epoch": epoch, "params": n_params}, best_path)

    student.load_state_dict(torch.load(best_path)["model"])
    subtitle = (f"student width={args.width} blocks={args.blocks}, {n_params:,} params, "
                f"alpha={args.alpha}, T={args.temperature}, best epoch {best_epoch}")
    y_true, y_prob = predict(student, val_loader, device)
    val_headline = write_auroc_table(y_true, y_prob, label_space, Ytr.sum(axis=0),
                                     DISTILL_DIR / run_name,
                                     title=f"Distilled student ({kind})", subtitle=subtitle)

    record = {
        "run_name": run_name, "kind": kind, "width": args.width, "blocks": args.blocks,
        "alpha": args.alpha, "temperature": args.temperature, "loss": args.loss,
        "params": n_params, "teacher_params": teacher_params,
        "compression": round(teacher_params / n_params, 2),
        "epochs": args.epochs, "best_epoch": best_epoch, "lr": args.lr, "seed": args.seed,
        "train_time_s": round(train_time, 1),
        "checkpoint": best_path.name,
        "val_macro_auroc": val_headline["macro_auroc"], "val_macro_f1": val_headline["macro_f1"],
        "val_micro_f1": val_headline["micro_f1"], "val_ece": val_headline["ece"],
    }
    print(f"\nBEST val macro-AUROC {best_auroc:.4f} (epoch {best_epoch}, {train_time:.0f}s train)")

    if not args.no_eval_test:
        Xte, Yte = build_split_cache("test", args.sampling_rate, num_workers=args.num_workers)
        test_m = eval_split(student, Xte, Yte, device)
        record.update({"test_macro_auroc": test_m["macro_auroc"],
                       "test_macro_f1": test_m["macro_f1"],
                       "test_micro_f1": test_m["micro_f1"]})
        print(f"TEST macro-AUROC {test_m['macro_auroc']:.4f}  macro-F1 {test_m['macro_f1']:.4f}")

    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    with (DISTILL_DIR / "runs.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"per-label table -> {DISTILL_DIR / run_name}  |  record -> distillation/runs.jsonl")
    if wandb.run is not None:
        wandb.summary.update({k: v for k, v in record.items() if isinstance(v, (int, float))})
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
