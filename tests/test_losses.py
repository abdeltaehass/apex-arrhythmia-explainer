"""Unit tests for Phase 4 losses (focal) and the model factory."""

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from src.detection.losses import FocalLoss, build_loss  # noqa: E402
from src.detection.model import build_model  # noqa: E402


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(8, 71, generator=g)
    targets = (torch.rand(8, 71, generator=g) > 0.85).float()
    return logits, targets


def test_focal_gamma0_equals_bce():
    logits, targets = _batch()
    focal = FocalLoss(gamma=0.0)(logits, targets)
    bce = nn.BCEWithLogitsLoss()(logits, targets)
    assert torch.allclose(focal, bce, atol=1e-6)


def test_focal_downweights_vs_bce():
    logits, targets = _batch()
    focal = FocalLoss(gamma=2.0)(logits, targets)
    bce = nn.BCEWithLogitsLoss()(logits, targets)
    assert focal.item() < bce.item()  # focusing shrinks easy-example loss
    assert focal.item() > 0


def test_focal_accepts_pos_weight():
    logits, targets = _batch()
    pw = torch.linspace(1, 10, 71)
    loss = FocalLoss(gamma=2.0, pos_weight=pw)(logits, targets)
    assert torch.isfinite(loss) and loss.item() > 0


def test_build_loss_types():
    assert isinstance(build_loss("bce"), nn.BCEWithLogitsLoss)
    assert isinstance(build_loss("focal"), FocalLoss)
    with pytest.raises(ValueError):
        build_loss("hinge")


def test_build_model_factory():
    x = torch.randn(2, 12, 1000)
    for name in ("cnn", "transformer"):
        out = build_model(name, width=16, blocks=1, d_model=64, depth=2, heads=4)(x)
        assert out.shape == (2, 71)
    with pytest.raises(ValueError):
        build_model("mlp")


def test_transformer_handles_batchnorm_free_single_sample():
    # transformer has no BatchNorm, so a batch of 1 must work (unlike the CNN in train mode)
    m = build_model("transformer", d_model=64, depth=2, heads=4).train()
    out = m(torch.randn(1, 12, 1000))
    assert out.shape == (1, 71)
