"""Phase 20 — carve PTB-XL into simulated hospital clients.

Federated learning exists because clinical data cannot leave the institution that
collected it. To simulate that honestly the partition has to be **natural**, not random:
if clients are drawn by shuffling, every client sees the same distribution and FedAvg has
nothing hard to do. PTB-XL ships a ``device`` column (the ECG cart that recorded each
study) and a ``site`` column, and the device a hospital owns is a good proxy for the
hospital itself — different carts live in different units, and different units see
different patients.

That proxy produces real heterogeneity. On the training folds, NORM prevalence runs from
25% on ``CS100 3`` to 82% on ``CS-12 E``, and client sizes span 6018 records down to 49.
Both matter, and they are different problems: label skew is what makes averaged updates
disagree, and size skew is what makes the average dominated by one client.

Three partition modes, because the comparison is the point:

- ``device`` / ``site`` — the natural, non-IID split. The realistic case.
- ``iid`` — random assignment that **reproduces the same client sizes**. This is the
  control: it holds the federated algorithm, the client count and the size skew fixed and
  removes only the label skew, so any gap between ``iid`` and ``device`` is attributable
  to heterogeneity rather than to federation itself.

Only the *training* folds are partitioned. Validation (fold 9) and test (fold 10) stay
global and untouched, so every model in Phase 20 is scored on exactly the data the
centralized Phase-4 model was scored on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.config import TRAIN_FOLDS

# Clients below this many records are folded into an "other" shard rather than simulated
# as standalone hospitals: three PTB-XL devices hold ~50 records each, which is too few to
# form a meaningful local training set and would mostly measure divide-by-small-n noise.
MIN_CLIENT_RECORDS = 100


@dataclass
class Client:
    """One simulated hospital: a name and the row indices it owns in the train cache."""

    name: str
    indices: np.ndarray
    label_prevalence: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(len(self.indices))

    def __repr__(self) -> str:  # keep round logs readable
        return f"Client({self.name!r}, n={self.n})"


def train_rows(df):
    """The training-fold rows of ``ptbxl_database.csv``, in cache order.

    ``build_split_cache`` iterates ``PTBXLDataset`` with ``shuffle=False``, and that
    dataset keeps the database's own row order after filtering by fold. So row *i* of the
    cached ``X``/``Y`` is row *i* of this frame — the assumption every partition here
    rests on, and the one :func:`assert_aligned` checks rather than trusts.
    """
    return df[df["strat_fold"].isin(TRAIN_FOLDS)]


def assert_aligned(df, Y: np.ndarray, label_space: list[str]) -> None:
    """Verify the cached label matrix matches the metadata rows, or refuse to proceed.

    If the cache and the frame ever fell out of order, every client would be assigned the
    wrong records and the experiment would still *run* — producing plausible, wrong
    numbers. Cheap to check, so check it.
    """
    from src.data.labels import encode

    rows = train_rows(df)
    if len(rows) != len(Y):
        raise ValueError(f"train cache has {len(Y)} rows, metadata has {len(rows)}")
    ref = np.stack([encode(c, label_space) for c in rows["scp_codes"]])
    if not np.array_equal(ref, Y):
        raise ValueError(
            "train cache is not row-aligned with the fold 1-8 metadata order; "
            "client assignment would be silently wrong"
        )


def _prevalence(Y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return Y[idx].mean(axis=0) if len(idx) else np.zeros(Y.shape[1], dtype=np.float64)


def label_skew(client_prev: np.ndarray, global_prev: np.ndarray) -> float:
    """Total-variation distance between a client's label mix and the global one.

    Both vectors are normalized to sum to 1 first, so this measures the *shape* of the
    label mix — which conditions this hospital sees relative to everyone else — and not
    simply how many labels per record it carries. 0 = identical mix, 1 = disjoint.
    """
    a, b = client_prev.sum(), global_prev.sum()
    if a <= 0 or b <= 0:
        return float("nan")
    return float(0.5 * np.abs(client_prev / a - global_prev / b).sum())


def build_clients(
    df,
    Y: np.ndarray,
    by: str = "device",
    min_records: int = MIN_CLIENT_RECORDS,
    seed: int = 42,
) -> list[Client]:
    """Partition the training rows into simulated hospitals.

    ``by="device"`` / ``"site"`` use the real metadata column; ``by="iid"`` shuffles rows
    at random into shards whose *sizes match the device partition exactly*, which is the
    control that isolates label skew from size skew.
    """
    rows = train_rows(df)
    if len(rows) != len(Y):
        raise ValueError(f"train cache has {len(Y)} rows, metadata has {len(rows)}")

    if by == "iid":
        sizes = [c.n for c in build_clients(df, Y, by="device", min_records=min_records)]
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(Y))
        clients, start = [], 0
        for k, size in enumerate(sizes):
            idx = np.sort(order[start:start + size])
            start += size
            clients.append(Client(f"iid_{k}", idx, _prevalence(Y, idx),
                                  {"source": "iid control"}))
        return clients

    if by not in rows.columns:
        raise ValueError(f"unknown partition column {by!r}")

    # Position within the cache, not the ecg_id: the cache is 0..N-1 in frame order.
    pos = np.arange(len(rows))
    key = rows[by].astype("object").where(rows[by].notna(), "unknown").astype(str).to_numpy()

    groups: dict[str, np.ndarray] = {}
    for name in np.unique(key):
        groups[name] = pos[key == name]

    # Fold the too-small groups together. They are kept (dropping them would change the
    # training set relative to centralized, which would confound the comparison) but
    # pooled into one shard rather than simulated as standalone hospitals.
    small = [n for n, idx in groups.items() if len(idx) < min_records]
    clients = [
        Client(name, idx, _prevalence(Y, idx), {"source": by})
        for name, idx in groups.items()
        if len(idx) >= min_records
    ]
    if small:
        pooled = np.sort(np.concatenate([groups[n] for n in small]))
        clients.append(Client(
            f"other({len(small)} small {by}s)", pooled, _prevalence(Y, pooled),
            {"source": by, "pooled_from": small},
        ))

    clients.sort(key=lambda c: -c.n)
    total = sum(c.n for c in clients)
    if total != len(Y):
        raise ValueError(f"partition lost rows: {total} assigned of {len(Y)}")
    return clients


def partition_summary(clients: list[Client], Y: np.ndarray, label_space: list[str],
                      min_positives: int = 10) -> dict:
    """Per-client size, label skew and coverage, plus whole-partition summary stats."""
    global_prev = Y.mean(axis=0)
    sizes = np.array([c.n for c in clients], dtype=float)
    per_client = []
    for c in clients:
        y = Y[c.indices]
        counts = y.sum(axis=0)
        per_client.append({
            "name": c.name,
            "n": c.n,
            "share": round(float(c.n / sizes.sum()), 4),
            "label_skew_tvd": round(label_skew(c.label_prevalence, global_prev), 4),
            "labels_with_positives": int((counts > 0).sum()),
            "labels_trainable": int((counts >= min_positives).sum()),
            "norm_rate": round(float(y[:, label_space.index("NORM")].mean()), 4)
            if "NORM" in label_space else None,
            "mean_labels_per_record": round(float(y.sum(axis=1).mean()), 3),
        })
    skews = np.array([p["label_skew_tvd"] for p in per_client], dtype=float)
    return {
        "n_clients": len(clients),
        "n_records": int(sizes.sum()),
        "largest_client_share": round(float(sizes.max() / sizes.sum()), 4),
        "size_ratio_max_min": round(float(sizes.max() / sizes.min()), 1),
        "size_gini": round(_gini(sizes), 4),
        "mean_label_skew_tvd": round(float(np.nanmean(skews)), 4),
        "max_label_skew_tvd": round(float(np.nanmax(skews)), 4),
        "labels_absent_from_some_client": int(
            sum(1 for j in range(Y.shape[1])
                if any(Y[c.indices][:, j].sum() == 0 for c in clients)
                and Y[:, j].sum() > 0)
        ),
        "clients": per_client,
    }


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of client sizes: 0 = all equal, ->1 = one client holds everything."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
