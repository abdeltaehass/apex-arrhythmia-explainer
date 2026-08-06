"""Unit tests for Phase 20 federated learning (no dataset or checkpoint needed)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from src.config import NUM_LABELS, NUM_LEADS  # noqa: E402
from src.detection.model import ECGResNet1d, build_model, make_norm  # noqa: E402
from src.federated.fedavg import (  # noqa: E402
    average_state_dicts,
    clone_to,
    local_train,
    select_clients,
)
from src.federated.partition import (  # noqa: E402
    Client,
    _gini,
    build_clients,
    label_skew,
    partition_summary,
)


# --- aggregation -------------------------------------------------------------
def _state(v: float, n_batches: int = 3) -> dict:
    return {
        "w": torch.full((2, 2), v),
        "bn.running_mean": torch.full((2,), v),
        "bn.num_batches_tracked": torch.tensor(n_batches),
    }


def test_average_is_weighted_by_sample_count():
    """FedAvg weights by n_k; a uniform mean would be a different (wrong) algorithm."""
    out = average_state_dicts([_state(0.0), _state(10.0)], [9.0, 1.0])
    assert out["w"].flatten()[0].item() == pytest.approx(1.0)  # not 5.0


def test_average_of_identical_states_is_identity():
    out = average_state_dicts([_state(2.5), _state(2.5), _state(2.5)], [1, 5, 100])
    assert torch.allclose(out["w"], torch.full((2, 2), 2.5))


def test_weights_are_normalized_not_summed():
    a = average_state_dicts([_state(1.0), _state(3.0)], [1.0, 1.0])
    b = average_state_dicts([_state(1.0), _state(3.0)], [500.0, 500.0])
    assert torch.allclose(a["w"], b["w"])


def test_integer_buffers_stay_integer():
    """num_batches_tracked must not silently become a float and break load_state_dict."""
    out = average_state_dicts([_state(0.0, 3), _state(0.0, 8)], [1.0, 1.0])
    assert not torch.is_floating_point(out["bn.num_batches_tracked"])
    assert out["bn.num_batches_tracked"].item() == 6  # round(5.5) under banker's rounding


def test_buffer_mode_keep_global_preserves_server_stats():
    g = _state(99.0)
    out = average_state_dicts([_state(0.0), _state(10.0)], [1, 1],
                              buffer_mode="keep_global", global_state=g)
    assert out["bn.running_mean"].flatten()[0].item() == pytest.approx(99.0)
    assert out["w"].flatten()[0].item() == pytest.approx(5.0)  # params still averaged


def test_buffer_mode_largest_client_takes_the_heaviest_shard():
    out = average_state_dicts([_state(1.0), _state(7.0)], [1.0, 9.0],
                              buffer_mode="largest_client")
    assert out["bn.running_mean"].flatten()[0].item() == pytest.approx(7.0)


def test_average_rejects_bad_input():
    with pytest.raises(ValueError):
        average_state_dicts([], [])
    with pytest.raises(ValueError):
        average_state_dicts([_state(1.0)], [1.0, 2.0])
    with pytest.raises(ValueError):
        average_state_dicts([_state(1.0)], [0.0])
    with pytest.raises(ValueError):
        average_state_dicts([_state(1.0)], [1.0], buffer_mode="nonsense")


def test_averaging_a_real_model_state_reloads():
    """The aggregated dict must be a valid state_dict, dtypes and all."""
    models = [ECGResNet1d(width=8, blocks=1) for _ in range(3)]
    out = average_state_dicts([m.state_dict() for m in models], [3.0, 1.0, 1.0])
    ECGResNet1d(width=8, blocks=1).load_state_dict(out)  # raises if anything is off


# --- local training ----------------------------------------------------------
def test_local_train_changes_weights_and_leaves_global_alone():
    g = ECGResNet1d(width=8, blocks=1)
    before = {k: v.clone() for k, v in g.state_dict().items()}
    local = clone_to(g, g.state_dict(), "cpu")
    X = np.random.default_rng(0).standard_normal((8, NUM_LEADS, 1000)).astype(np.float32)
    Y = np.zeros((8, NUM_LABELS), dtype=np.float32)
    state, loss = local_train(local, X, Y, nn.BCEWithLogitsLoss(), epochs=1, lr=1e-3,
                              weight_decay=0.0, batch_size=4, device="cpu")
    assert np.isfinite(loss)
    assert not torch.equal(state["head.weight"].cpu(), before["head.weight"])
    # the server's copy must be untouched — clients train on a clone, not the global model
    assert torch.equal(g.state_dict()["head.weight"], before["head.weight"])


def test_local_train_skips_singleton_batches():
    """BatchNorm cannot take a variance over one sample; a 5-record client with
    batch_size 4 would otherwise crash on its trailing batch of 1."""
    local = ECGResNet1d(width=8, blocks=1)
    X = np.random.default_rng(1).standard_normal((5, NUM_LEADS, 1000)).astype(np.float32)
    Y = np.zeros((5, NUM_LABELS), dtype=np.float32)
    _, loss = local_train(local, X, Y, nn.BCEWithLogitsLoss(), epochs=1, lr=1e-3,
                          weight_decay=0.0, batch_size=4, device="cpu")
    assert np.isfinite(loss)


def test_select_clients_full_and_partial():
    clients = [Client(f"c{i}", np.arange(10)) for i in range(8)]
    rng = np.random.default_rng(0)
    assert len(select_clients(clients, 1.0, rng)) == 8
    assert len(select_clients(clients, 0.5, rng)) == 4
    assert len(select_clients(clients, 0.01, rng)) == 1  # never selects zero clients


# --- partitioning ------------------------------------------------------------
@pytest.fixture
def fake_df():
    pd = pytest.importorskip("pandas")
    n = 600
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "strat_fold": [1] * n + [10] * 50,          # 600 train rows + a test fold
        "device": (["A"] * 400 + ["B"] * 180 + ["C"] * 20) + ["A"] * 50,
        "scp_codes": [{"NORM": 100.0}] * (n + 50),
        "patient_id": rng.integers(0, 500, n + 50),
    })


def test_build_clients_covers_every_row_exactly_once(fake_df):
    Y = np.zeros((600, 3), dtype=np.float32)
    clients = build_clients(fake_df, Y, by="device", min_records=10)
    allidx = np.concatenate([c.indices for c in clients])
    assert len(allidx) == 600
    assert len(np.unique(allidx)) == 600  # no row assigned twice, none dropped


def test_small_clients_are_pooled_not_dropped(fake_df):
    Y = np.zeros((600, 3), dtype=np.float32)
    clients = build_clients(fake_df, Y, by="device", min_records=100)
    assert sum(c.n for c in clients) == 600           # C (20 rows) kept, not discarded
    assert any("other" in c.name for c in clients)


def test_iid_control_matches_device_sizes_exactly(fake_df):
    Y = np.zeros((600, 3), dtype=np.float32)
    dev = build_clients(fake_df, Y, by="device", min_records=100)
    iid = build_clients(fake_df, Y, by="iid", min_records=100)
    assert sorted(c.n for c in dev) == sorted(c.n for c in iid)


def test_partition_rejects_a_mismatched_cache(fake_df):
    with pytest.raises(ValueError):
        build_clients(fake_df, np.zeros((599, 3), dtype=np.float32), by="device")


def test_label_skew_is_zero_for_identical_mixes():
    p = np.array([0.5, 0.3, 0.2])
    assert label_skew(p, p) == pytest.approx(0.0)
    assert label_skew(p * 7, p) == pytest.approx(0.0)  # normalized: shape, not magnitude


def test_label_skew_is_one_for_disjoint_mixes():
    assert label_skew(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_gini_extremes():
    assert _gini(np.array([5.0, 5.0, 5.0])) == pytest.approx(0.0)
    assert _gini(np.array([0.0, 0.0, 9.0])) > 0.6


def test_partition_summary_counts_coverage_holes():
    Y = np.zeros((10, 3), dtype=np.float32)
    Y[:5, 0] = 1          # label 0 only in client A
    Y[5:, 1] = 1          # label 1 only in client B
    clients = [Client("A", np.arange(5), Y[:5].mean(0)),
               Client("B", np.arange(5, 10), Y[5:].mean(0))]
    s = partition_summary(clients, Y, ["L0", "L1", "L2"])
    assert s["n_clients"] == 2
    assert s["labels_absent_from_some_client"] == 2  # L2 has no positives anywhere, excluded


# --- GroupNorm variant -------------------------------------------------------
def test_make_norm_groups_divide_channels():
    for c in (8, 16, 64, 128, 512):
        gn = make_norm("gn", c)
        assert c % gn.num_groups == 0


def test_groupnorm_model_has_no_averaging_prone_buffers():
    """The point of the GroupNorm variant: no running statistics to average."""
    bn = build_model("cnn", width=16, blocks=1, norm="bn")
    gn = build_model("cnn", width=16, blocks=1, norm="gn")
    assert any("running_mean" in k for k in bn.state_dict())
    assert not any("running_mean" in k for k in gn.state_dict())
    with torch.no_grad():
        assert gn(torch.randn(2, NUM_LEADS, 1000)).shape == (2, NUM_LABELS)


def test_unknown_norm_rejected():
    with pytest.raises(ValueError):
        make_norm("layernorm", 16)
