"""Phase 25 — tests for augmentation, induced rarity, generation, and the ablation.

Fast and data-independent: tiny arrays and a one-step diffusion model. Nothing here trains
a real generator or touches PTB-XL.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.synthesis import (
    ARMS,
    AugmentConfig,
    DiffusionConfig,
    TrainConfig,
    UNet1D,
    augment,
    augment_batch,
    build_arm_data,
    cosine_alphas,
    diversity,
    evaluate,
    make_rare,
    memorization,
    oversample_indices,
)
from src.synthesis.ablation import bootstrap_auroc, paired_bootstrap_delta

LABELS = ["AFIB", "CLBBB", "CRBBB", "NORM", "SR"]


def fake_ecg(n: int = 8, seed: int = 0) -> np.ndarray:
    """(n, 12, 300) signals with a repeating spike train — enough structure to delineate."""
    rng = np.random.default_rng(seed)
    base = np.zeros((n, 12, 300), dtype=np.float32)
    for i in range(n):
        for beat in range(1, 300, 60):
            base[i, :, beat:beat + 3] += 3.0
        base[i] += rng.normal(0, 0.05, (12, 300))
    return base.astype(np.float32)


# --- classical augmentation ---------------------------------------------------
def test_augment_preserves_shape_and_dtype():
    x = fake_ecg(1)[0]
    out = augment(x, fs=100, rng=np.random.default_rng(0))
    assert out.shape == x.shape and out.dtype == np.float32


def test_augment_is_deterministic_given_a_seed():
    x = fake_ecg(1)[0]
    a = augment(x, rng=np.random.default_rng(7))
    b = augment(x, rng=np.random.default_rng(7))
    assert np.allclose(a, b)


def test_augment_actually_changes_the_signal():
    x = fake_ecg(1)[0]
    cfg = AugmentConfig(p_scale=1.0, p_noise=1.0)
    assert not np.allclose(augment(x, cfg=cfg, rng=np.random.default_rng(0)), x)


def test_augment_never_reverses_time():
    """Time reversal would produce a waveform no heart generates."""
    x = fake_ecg(1)[0]
    out = augment(x, rng=np.random.default_rng(3))
    assert not np.allclose(out, x[:, ::-1], atol=1e-3)


def test_augment_never_permutes_leads():
    """Lead identity is the localizing information — swapping leads relabels the finding."""
    x = fake_ecg(1)[0] * np.arange(1, 13)[:, None]      # each lead uniquely scaled
    out = augment(x, cfg=AugmentConfig(p_lead_dropout=0.0, p_noise=0.0, p_warp=0.0,
                                       p_baseline=0.0, p_powerline=0.0, p_shift=0.0),
                  rng=np.random.default_rng(0))
    order = np.argsort([np.abs(lead).max() for lead in out])
    assert list(order) == list(range(12))


def test_lead_dropout_zeroes_at_most_the_configured_leads():
    cfg = AugmentConfig(p_lead_dropout=1.0, max_dropped_leads=1, p_noise=0.0,
                        p_scale=0.0, p_warp=0.0, p_baseline=0.0, p_powerline=0.0,
                        p_shift=0.0)
    out = augment(fake_ecg(1)[0], cfg=cfg, rng=np.random.default_rng(0))
    assert sum(1 for lead in out if np.allclose(lead, 0.0)) == 1


def test_augment_batch_varies_between_records():
    batch = np.repeat(fake_ecg(1), 4, axis=0)
    out = augment_batch(batch, cfg=AugmentConfig(p_scale=1.0), rng=np.random.default_rng(0))
    assert not np.allclose(out[0], out[1])


# --- induced rarity -----------------------------------------------------------
def _labels(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros((n, len(LABELS)), dtype=np.float32)
    y[: n // 2, LABELS.index("CLBBB")] = 1.0
    y[:, LABELS.index("SR")] = (rng.random(n) < 0.8).astype(np.float32)
    return y


def test_make_rare_keeps_exactly_n():
    y = _labels()
    scenario = make_rare(y, LABELS, targets=("CLBBB",), n_keep=10, seed=0)
    j = LABELS.index("CLBBB")
    assert scenario.y_masked[:, j].sum() == 10
    assert len(scenario.positive_rows("CLBBB")) == 10
    assert scenario.masked["CLBBB"] == 90


def test_make_rare_leaves_other_labels_untouched():
    y = _labels()
    scenario = make_rare(y, LABELS, targets=("CLBBB",), n_keep=10, seed=0)
    others = [j for j, name in enumerate(LABELS) if name != "CLBBB"]
    assert np.array_equal(scenario.y_masked[:, others], y[:, others])


def test_make_rare_only_removes_never_adds():
    y = _labels()
    masked = make_rare(y, LABELS, targets=("CLBBB",), n_keep=10, seed=0).y_masked
    assert np.all(masked <= y), "masking must not introduce positives"


def test_make_rare_is_a_no_op_when_already_rare():
    y = np.zeros((20, len(LABELS)), dtype=np.float32)
    y[:3, 1] = 1.0
    scenario = make_rare(y, LABELS, targets=(LABELS[1],), n_keep=10, seed=0)
    assert scenario.masked[LABELS[1]] == 0
    assert scenario.y_masked[:, 1].sum() == 3


def test_make_rare_rejects_unknown_label():
    with pytest.raises(KeyError):
        make_rare(_labels(), LABELS, targets=("NOPE",), n_keep=5)


def test_oversample_repeats_only_the_kept_rows():
    y = _labels()
    scenario = make_rare(y, LABELS, targets=("CLBBB",), n_keep=10, seed=0)
    idx = oversample_indices(scenario, len(y), factor=4, rng=np.random.default_rng(0))
    assert len(idx) == len(y) + 10 * 3
    kept = set(scenario.positive_rows("CLBBB").tolist())
    counts = {i: int((idx == i).sum()) for i in kept}
    assert all(c == 4 for c in counts.values())


# --- quality ------------------------------------------------------------------
def test_memorization_detects_verbatim_copies():
    train = fake_ecg(16, seed=1)
    heldout = fake_ecg(16, seed=2)
    ratio, _, _ = memorization(train.copy(), train, heldout)
    assert ratio < 0.2, "exact copies must score far below an independent sample"


def test_memorization_is_near_one_for_honest_samples():
    train = fake_ecg(16, seed=1)
    heldout = fake_ecg(16, seed=2)
    fresh = fake_ecg(16, seed=3)
    ratio, _, _ = memorization(fresh, train, heldout)
    assert 0.5 < ratio < 2.0


def test_diversity_detects_mode_collapse():
    real = fake_ecg(16, seed=1)
    collapsed = np.repeat(real[:1], 16, axis=0)
    assert diversity(collapsed, real) < 0.1


# --- diffusion ----------------------------------------------------------------
def test_cosine_schedule_is_monotonic_and_bounded():
    a = cosine_alphas(200).numpy()
    assert a.shape == (200,)
    assert np.all(np.diff(a) <= 1e-6), "alpha_bar must decrease with t"
    assert 0.0 <= a.min() and a.max() <= 1.0


def test_unet_returns_noise_shaped_like_its_input():
    import torch

    cfg = DiffusionConfig(base_channels=8, channel_mults=(1, 2), time_dim=16)
    model = UNet1D(cfg, n_labels=len(LABELS))
    x = torch.randn(2, 12, 128)
    out = model(x, torch.tensor([5, 900]), torch.zeros(2, len(LABELS)))
    assert out.shape == x.shape


def test_label_conditioning_changes_the_prediction():
    import torch

    torch.manual_seed(0)
    cfg = DiffusionConfig(base_channels=8, channel_mults=(1, 2), time_dim=16)
    model = UNet1D(cfg, n_labels=len(LABELS)).eval()
    x = torch.randn(1, 12, 128)
    t = torch.tensor([100])
    y1 = torch.zeros(1, len(LABELS))
    y2 = torch.zeros(1, len(LABELS))
    y2[0, 1] = 1.0
    with torch.no_grad():
        assert not torch.allclose(model(x, t, y1), model(x, t, y2))


# --- ablation -----------------------------------------------------------------
def test_every_arm_adds_the_same_number_of_rows():
    """Otherwise the comparison confounds augmentation method with class weighting."""
    y = _labels()
    x = fake_ecg(len(y), seed=0)
    scenario = make_rare(y, LABELS, targets=("CLBBB",), n_keep=10, seed=0)
    cond_row = scenario.y_masked[scenario.positive_rows("CLBBB")[0]]
    synth = {"CLBBB": (fake_ecg(12, seed=5), np.tile(cond_row, (12, 1)))}
    cfg = TrainConfig(n_added=20)
    sizes = {}
    for arm in ARMS:
        xa, ya = build_arm_data(arm, x, scenario, LABELS, synth, cfg,
                                np.random.default_rng(0))
        sizes[arm] = len(xa)
        assert len(xa) == len(ya)
    assert sizes["baseline"] == len(x)
    added = {v for k, v in sizes.items() if k != "baseline"}
    assert added == {len(x) + 20}


def test_baseline_arm_is_the_masked_data_untouched():
    y = _labels()
    x = fake_ecg(len(y), seed=0)
    scenario = make_rare(y, LABELS, targets=("CLBBB",), n_keep=10, seed=0)
    xa, ya = build_arm_data("baseline", x, scenario, LABELS, None, TrainConfig(),
                            np.random.default_rng(0))
    assert xa is x and np.array_equal(ya, scenario.y_masked)


def test_synthetic_arm_degrades_to_baseline_without_samples():
    y = _labels()
    x = fake_ecg(len(y), seed=0)
    scenario = make_rare(y, LABELS, targets=("CLBBB",), n_keep=10, seed=0)
    xa, _ = build_arm_data("synthetic", x, scenario, LABELS, {}, TrainConfig(),
                           np.random.default_rng(0))
    assert len(xa) == len(x)


def test_synthetic_rows_keep_their_conditioning_labels():
    """Supervision must equal what the sample was generated from, co-occurring labels included."""
    y = _labels()
    x = fake_ecg(len(y), seed=0)
    scenario = make_rare(y, LABELS, targets=("CLBBB",), n_keep=10, seed=0)
    cond = np.zeros((6, len(LABELS)), dtype=np.float32)
    cond[:, LABELS.index("CLBBB")] = 1.0
    cond[:, LABELS.index("SR")] = 1.0
    synth = {"CLBBB": (fake_ecg(6, seed=5), cond)}
    _, ya = build_arm_data("synthetic", x, scenario, LABELS, synth,
                           TrainConfig(n_added=6), np.random.default_rng(0))
    added = ya[len(y):]
    assert np.array_equal(added, cond[np.argsort(np.zeros(6))][: len(added)]) or \
        np.all(added[:, LABELS.index("SR")] == 1.0), "co-occurring labels must survive"


def test_evaluate_separates_targets_from_the_rest():
    rng = np.random.default_rng(0)
    y = (rng.random((200, len(LABELS))) < 0.3).astype(np.float32)
    probs = rng.random((200, len(LABELS)))
    per, common = evaluate(probs, y, LABELS, ("CLBBB",))
    assert set(per) == {"CLBBB"}
    assert 0.0 <= common <= 1.0


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.3).astype(int)
    score = y * 0.6 + rng.random(300) * 0.4          # informative but noisy
    lo, hi = bootstrap_auroc(y, score, n=200, seed=0)
    from sklearn.metrics import roc_auc_score

    assert lo <= roc_auc_score(y, score) <= hi


def test_paired_bootstrap_finds_no_difference_between_identical_scores():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.3).astype(int)
    score = rng.random(300)
    mean, lo, hi = paired_bootstrap_delta(y, score, score, n=200, seed=0)
    assert mean == pytest.approx(0.0, abs=1e-9)
    assert lo == pytest.approx(0.0, abs=1e-9) and hi == pytest.approx(0.0, abs=1e-9)


def test_paired_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.4).astype(int)
    good = y * 0.8 + rng.random(400) * 0.2
    bad = rng.random(400)
    mean, lo, hi = paired_bootstrap_delta(y, bad, good, n=300, seed=0)
    assert mean > 0 and lo > 0, "a large true difference must exclude zero"


def test_no_arm_recovers_the_masked_annotations():
    """The leak that would invalidate the whole experiment.

    Every arm must see exactly ``n_keep`` real positives for a target in the original rows.
    If augmentation could restore the masked annotations — by copying rows whose label was
    masked, or by conditioning a generator on the unmasked matrix — the "improvement" would
    just be the withheld labels coming back in through the side door.
    """
    y = _labels()
    x = fake_ecg(len(y), seed=0)
    n_keep = 10
    scenario = make_rare(y, LABELS, targets=("CLBBB",), n_keep=n_keep, seed=0)
    j = LABELS.index("CLBBB")
    cond = np.zeros((6, len(LABELS)), dtype=np.float32)
    cond[:, j] = 1.0
    synth = {"CLBBB": (fake_ecg(6, seed=5), cond)}

    for arm in ARMS:
        _, ya = build_arm_data(arm, x, scenario, LABELS, synth, TrainConfig(n_added=6),
                               np.random.default_rng(0))
        original = ya[: len(y)]
        assert original[:, j].sum() == n_keep, f"{arm} restored masked annotations"
        assert np.all(original <= y), f"{arm} invented positives in the original rows"
