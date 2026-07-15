"""Unit tests for the baseline detector model + training helpers (no dataset needed)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip this module entirely if torch is absent

from src.config import NUM_LABELS, NUM_LEADS  # noqa: E402
from src.detection.model import ECGResNet1d, count_parameters  # noqa: E402
from src.detection.train import pos_weight_from  # noqa: E402


def test_forward_shape_and_logits():
    model = ECGResNet1d(width=16, blocks=1).eval()
    x = torch.randn(4, NUM_LEADS, 1000)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, NUM_LABELS)
    assert torch.isfinite(out).all()


def test_variable_batch_and_length():
    model = ECGResNet1d(width=8, blocks=1).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 12, 1000)).shape == (1, 71)
        assert model(torch.randn(3, 12, 2500)).shape == (3, 71)  # GAP handles any length


def test_param_count_positive():
    assert count_parameters(ECGResNet1d(width=16, blocks=1)) > 0


def test_pos_weight_capped_and_shaped():
    # label 0: 1 pos / 9 neg -> weight 9 ; label 1: all positive -> weight 1
    Y = np.zeros((10, NUM_LABELS), dtype=np.float32)
    Y[0, 0] = 1
    Y[:, 1] = 1
    w = pos_weight_from(Y, cap=5.0)
    assert w.shape == (NUM_LABELS,)
    assert w[0].item() == pytest.approx(5.0)  # 9 capped to 5
    assert w[1].item() == pytest.approx(1.0)  # no positives-vs-neg -> 1.0
    assert (w <= 5.0).all()
