"""Phase 25 — the ablation: does synthetic augmentation earn its complexity?

One arm per way of manufacturing extra rare-class examples, all trained identically and
scored on the untouched test fold.

    baseline             the masked training set, nothing added
    oversample           the kept real examples, repeated
    classical            the kept real examples, repeated *and* perturbed
    synthetic            diffusion samples conditioned on the rare label
    synthetic+classical  both

**Every arm adds the same number of rare-positive rows.** Otherwise the comparison is
between augmentation methods *and* effective class weights at once, and a difference could
not be attributed to either. Oversampling is in the table because it is free, requires no
model, and adds no information — which makes it the yardstick that says how much
information the synthetic samples actually carry.

Scoring is per-label AUROC on the test fold with bootstrap confidence intervals. With
40-112 test positives per target the intervals are wide, and quoting point estimates
without them would manufacture findings out of resampling noise — the same trap the
underpowered genuinely-rare labels fall into (see :mod:`src.synthesis.rarity`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.detection.model import build_model
from src.synthesis.augment import AugmentConfig, augment_batch
from src.synthesis.rarity import RarityScenario

ARMS = ("baseline", "oversample", "classical", "synthetic", "synthetic+classical")


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    # Extra rare-positive rows each augmenting arm adds per target label. Chosen to bring
    # the 50 surviving annotations back up to roughly the label's original prevalence.
    n_added: int = 350


@dataclass
class ArmResult:
    arm: str
    seed: int
    auroc: dict[str, float] = field(default_factory=dict)          # target label -> AUROC
    auroc_common: float = float("nan")                             # macro over non-targets
    n_train_rows: int = 0
    notes: list[str] = field(default_factory=list)


def build_arm_data(arm: str, X: np.ndarray, scenario: RarityScenario,
                   label_space: list[str],
                   synthetic: dict[str, tuple[np.ndarray, np.ndarray]] | None,
                   cfg: TrainConfig, rng: np.random.Generator
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Assemble ``(X, Y)`` for one arm: the masked set plus that arm's added rows."""
    y = scenario.y_masked
    if arm == "baseline":
        return X, y

    extra_x: list[np.ndarray] = []
    extra_y: list[np.ndarray] = []
    for label in scenario.targets:
        rows = scenario.positive_rows(label)
        if not len(rows):
            continue

        if arm in ("oversample", "classical"):
            pick = rng.choice(rows, size=cfg.n_added, replace=True)
            batch = X[pick]
            if arm == "classical":
                batch = augment_batch(batch, cfg=AugmentConfig(), rng=rng)
            extra_x.append(batch)
            # Copies carry the source record's full label row, not a synthetic one-hot:
            # a recording with left bundle branch block also has a rhythm, and dropping
            # that would teach the model that this class appears without one.
            extra_y.append(y[pick])

        elif arm.startswith("synthetic"):
            entry = (synthetic or {}).get(label)
            if entry is None or not len(entry[0]):
                continue
            pool, pool_labels = entry
            pick = rng.choice(len(pool), size=cfg.n_added, replace=len(pool) < cfg.n_added)
            batch = pool[pick]
            if arm == "synthetic+classical":
                batch = augment_batch(batch, cfg=AugmentConfig(), rng=rng)
            extra_x.append(batch)
            # Each synthetic row keeps the *exact* label vector it was generated from. An
            # earlier version drew a fresh label row here, which meant a sample conditioned
            # on {CLBBB, SR} could be trained as {CLBBB, STACH} — the target label stayed
            # right, but every co-occurring label became noise, and only the synthetic arms
            # paid for it. Conditioning and supervision must be the same vector.
            extra_y.append(pool_labels[pick])

    if not extra_x:
        return X, y
    return (np.concatenate([X, *extra_x]).astype(np.float32),
            np.concatenate([y, *extra_y]).astype(np.float32))


def train_classifier(X: np.ndarray, Y: np.ndarray, cfg: TrainConfig, device: str,
                     seed: int) -> torch.nn.Module:
    """Train the Phase-4 CNN. Identical settings across arms — only the data differs."""
    torch.manual_seed(seed)
    model = build_model("cnn", dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    crit = torch.nn.BCEWithLogitsLoss()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict(model: torch.nn.Module, X: np.ndarray, device: str,
            batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(X), batch):
        xb = torch.from_numpy(X[start:start + batch]).to(device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def evaluate(probs: np.ndarray, y_true: np.ndarray, label_space: list[str],
             targets) -> tuple[dict[str, float], float]:
    """Per-target AUROC, and macro AUROC over every *other* label with test positives.

    The second number is the one that decides whether an augmentation method is usable at
    all: buying rare-class performance by degrading the other 63 labels is not a win, and
    it is the cost a rare-class paper most often forgets to report.
    """
    per: dict[str, float] = {}
    for label in targets:
        j = label_space.index(label)
        if 0 < y_true[:, j].sum() < len(y_true):
            per[label] = float(roc_auc_score(y_true[:, j], probs[:, j]))

    target_idx = {label_space.index(t) for t in targets}
    others = [j for j in range(len(label_space))
              if j not in target_idx and 0 < y_true[:, j].sum() < len(y_true)]
    common = float(np.mean([roc_auc_score(y_true[:, j], probs[:, j]) for j in others]))
    return per, common


def bootstrap_auroc(y_true: np.ndarray, score: np.ndarray, n: int = 1000,
                    seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for one label's AUROC, resampling test records."""
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        take = rng.choice(idx, size=len(idx), replace=True)
        yt = y_true[take]
        if 0 < yt.sum() < len(yt):
            vals.append(roc_auc_score(yt, score[take]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_delta(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                           n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """CI for AUROC(b) - AUROC(a) on the *same* resampled records.

    Paired, because both arms are scored on one test set: resampling them independently
    would add variance that is not in the comparison and would hide real differences.
    """
    rng = np.random.default_rng(seed)
    deltas = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        take = rng.choice(idx, size=len(idx), replace=True)
        yt = y_true[take]
        if 0 < yt.sum() < len(yt):
            deltas.append(roc_auc_score(yt, score_b[take]) - roc_auc_score(yt, score_a[take]))
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    return (float(np.mean(deltas)), float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)))


def sample_for_targets(model, scenario: RarityScenario, label_space: list[str],
                       n_per_label: int, diff_cfg, device: str, seed: int
                       ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic recordings for each target label.

    Returns ``{label: (signals, conditioning_labels)}`` — the label vectors are returned
    alongside so the training rows can carry exactly what each sample was conditioned on.

    Conditioning vectors are copied from the *real kept examples* rather than built as a
    one-hot. A left bundle branch block recording also carries a rhythm label and often a
    repolarization one, and conditioning on a bare one-hot would ask the generator for a
    recording with a conduction defect and no rhythm — a combination it has never seen and
    which does not exist.
    """
    from src.synthesis.diffusion import sample

    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for label in scenario.targets:
        rows = scenario.positive_rows(label)
        if not len(rows):
            continue
        pick = rng.choice(rows, size=n_per_label, replace=True)
        cond = scenario.y_masked[pick]
        out[label] = (sample(model, cond, diff_cfg, device=device,
                             seed=seed + label_space.index(label)), cond)
    return out
