"""Unit tests for Phase 19 knowledge distillation (no dataset or checkpoint needed)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from src.config import NUM_LABELS, NUM_LEADS  # noqa: E402
from src.detection.distill import (  # noqa: E402
    BinaryKDLoss,
    DistillationLoss,
    agreement,
    compute_logits,
)
from src.detection.model import ECGResNet1d, build_model, count_parameters  # noqa: E402


@pytest.fixture
def logits():
    g = torch.Generator().manual_seed(0)
    return torch.randn(16, NUM_LABELS, generator=g)


# --- the KD loss -------------------------------------------------------------
def test_kd_is_zero_when_student_matches_teacher(logits):
    """KL(q‖p) == 0 iff the distributions agree — the property that makes the value readable."""
    for t in (1.0, 2.0, 5.0):
        assert BinaryKDLoss(t)(logits, logits).item() == pytest.approx(0.0, abs=1e-6)


def test_kd_is_positive_and_finite_for_a_mismatch(logits):
    g = torch.Generator().manual_seed(1)
    student = torch.randn(16, NUM_LABELS, generator=g)
    loss = BinaryKDLoss(2.0)(student, logits)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_kd_stable_at_saturating_logits():
    """log-sigmoid form must not produce inf/nan where sigmoid would saturate to 0 or 1."""
    big = torch.full((4, NUM_LABELS), 80.0)
    loss = BinaryKDLoss(1.0)(-big, big)  # maximally opposed, saturated both sides
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_kd_grows_with_disagreement(logits):
    """A student closer to the teacher must score lower than one further away."""
    kd = BinaryKDLoss(2.0)
    near = kd(logits + 0.1, logits).item()
    far = kd(logits + 2.0, logits).item()
    assert near < far


def test_temperature_squared_keeps_gradient_scale_stable(logits):
    """The Hinton T² factor: gradient magnitude must not collapse as T grows."""
    norms = []
    for t in (1.0, 2.0, 4.0, 8.0):
        student = torch.randn(16, NUM_LABELS, generator=torch.Generator().manual_seed(2))
        student.requires_grad_(True)
        BinaryKDLoss(t)(student, logits).backward()
        norms.append(student.grad.norm().item())
    assert min(norms) > 0.5 * max(norms), f"T² correction not holding gradients flat: {norms}"


def test_kd_rejects_bad_temperature():
    with pytest.raises(ValueError):
        BinaryKDLoss(0.0)
    with pytest.raises(ValueError):
        BinaryKDLoss(-1.0)


def test_kd_matches_bce_against_soft_targets_in_gradient(logits):
    """KL and cross-entropy differ by the teacher's entropy — a student-independent constant,
    so the two must produce identical gradients."""
    t = 2.0
    q = torch.sigmoid(logits / t)

    a = torch.zeros(16, NUM_LABELS, requires_grad=True)
    BinaryKDLoss(t)(a, logits).backward()

    b = torch.zeros(16, NUM_LABELS, requires_grad=True)
    (t * t * nn.BCEWithLogitsLoss()(b / t, q)).backward()

    assert torch.allclose(a.grad, b.grad, atol=1e-6)


def test_kd_averages_over_labels_not_sums(logits):
    """Scale must not depend on label count, or `alpha` stops meaning anything."""
    g = torch.Generator().manual_seed(3)
    student = torch.randn(16, NUM_LABELS, generator=g)
    kd = BinaryKDLoss(2.0)
    full = kd(student, logits).item()
    half = kd(student[:, :40], logits[:, :40]).item()
    # a mean is within noise of the subset's mean; a sum would be ~1.8x apart
    assert 0.5 < full / half < 2.0


# --- the combined objective --------------------------------------------------
def test_alpha_zero_is_pure_supervision(logits):
    """The from-scratch control must be exactly the ordinary loss, not an approximation."""
    y = (torch.rand(16, NUM_LABELS) > 0.9).float()
    student = torch.randn(16, NUM_LABELS)
    hard = nn.BCEWithLogitsLoss()
    total, parts = DistillationLoss(hard, 2.0, alpha=0.0)(student, y, logits)
    assert total.item() == pytest.approx(hard(student, y).item())
    assert parts["soft"] == 0.0


def test_alpha_one_ignores_ground_truth(logits):
    y_a = (torch.rand(16, NUM_LABELS) > 0.9).float()
    y_b = torch.zeros(16, NUM_LABELS)
    student = torch.randn(16, NUM_LABELS)
    loss = DistillationLoss(nn.BCEWithLogitsLoss(), 2.0, alpha=1.0)
    assert loss(student, y_a, logits)[0].item() == pytest.approx(
        loss(student, y_b, logits)[0].item()
    )


def test_alpha_interpolates(logits):
    y = (torch.rand(16, NUM_LABELS) > 0.9).float()
    student = torch.randn(16, NUM_LABELS)
    hard = nn.BCEWithLogitsLoss()
    _, parts = DistillationLoss(hard, 2.0, 0.5)(student, y, logits)
    mid = DistillationLoss(hard, 2.0, 0.5)(student, y, logits)[0].item()
    assert mid == pytest.approx(0.5 * parts["soft"] + 0.5 * parts["hard"])


def test_distillation_loss_rejects_bad_alpha():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            DistillationLoss(nn.BCEWithLogitsLoss(), 2.0, bad)


def test_gradients_flow_to_the_student(logits):
    student = ECGResNet1d(width=8, blocks=1)
    x = torch.randn(2, NUM_LEADS, 1000)
    y = torch.zeros(2, NUM_LABELS)
    total, _ = DistillationLoss(nn.BCEWithLogitsLoss(), 2.0, 0.7)(
        student(x), y, logits[:2]
    )
    total.backward()
    grads = [p.grad for p in student.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


# --- student architecture ----------------------------------------------------
def test_student_is_smaller_but_shape_compatible():
    """The student must be a drop-in: same input/output contract, fewer parameters."""
    teacher = build_model("cnn", width=64, blocks=2)
    student = build_model("cnn", width=16, blocks=1)
    x = torch.randn(2, NUM_LEADS, 1000)
    with torch.no_grad():
        assert student(x).shape == teacher(x).shape == (2, NUM_LABELS)
    assert count_parameters(student) < count_parameters(teacher) / 10


def test_student_keeps_the_gradcam_target():
    """Phase 5 grounding hooks `model.stages`; a student without it would break the UI."""
    from src.grounding.loader import default_cam_target

    assert default_cam_target(build_model("cnn", width=16, blocks=1)) is not None


def test_gradcam_runs_end_to_end_on_a_student():
    """The drop-in claim is only worth making if grounding actually produces valid maps
    on the smaller model, not merely if the attribute it hooks still exists."""
    from src.grounding import ground

    student = build_model("cnn", width=16, blocks=1).eval()
    sig = np.random.default_rng(0).standard_normal((NUM_LEADS, 1000)).astype(np.float32)
    sal = ground(student, sig, [0, 5], fs=100)

    for s in sal.values():
        assert s.per_lead.shape == (NUM_LEADS, 1000)
        assert s.temporal.shape == (1000,)
        assert np.isfinite(s.per_lead).all()
        assert 0.0 <= s.per_lead.min() and s.per_lead.max() <= 1.0
        assert s.lead_importance.shape == (NUM_LEADS,)


# --- helpers -----------------------------------------------------------------
def test_compute_logits_shape_and_batching():
    model = ECGResNet1d(width=8, blocks=1).eval()
    X = np.random.default_rng(0).standard_normal((5, NUM_LEADS, 1000)).astype(np.float32)
    z = compute_logits(model, X, "cpu", batch=2)  # deliberately not a divisor of 5
    assert z.shape == (5, NUM_LABELS)
    assert np.isfinite(z).all()


def test_agreement_perfect_and_partial():
    p = np.array([[0.9, 0.1], [0.2, 0.8]])
    perfect = agreement(p, p, 0.5)
    assert perfect["decision_agreement"] == 1.0
    assert perfect["mean_abs_prob_diff"] == 0.0

    flipped = agreement(1 - p, p, 0.5)
    assert flipped["decision_agreement"] == 0.0
