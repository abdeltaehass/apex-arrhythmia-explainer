"""Losses for multi-label ECG classification.

Phase 3 used class-weighted BCE. Phase 4 adds focal loss to push harder on the class
imbalance: focal down-weights easy (well-classified) examples so training focuses on the
hard, rare positives. Both operate on raw logits (shape (B, num_labels)).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-label sigmoid focal loss (Lin et al., 2017), optionally class-weighted.

    ``gamma`` controls the focusing strength (0 == plain BCE). ``pos_weight`` (per label)
    can still be supplied to further up-weight positives, i.e. focal and the Phase-3
    imbalance weighting compose.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)  # prob assigned to the true class
        focal = (1 - p_t).clamp(min=1e-6) ** self.gamma * bce
        return focal.mean()


def build_loss(name: str, pos_weight: torch.Tensor | None = None, focal_gamma: float = 2.0) -> nn.Module:
    if name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "focal":
        return FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
    raise ValueError(f"unknown loss: {name!r}")
