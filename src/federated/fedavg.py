"""Phase 20 — FedAvg: local training on each client, weighted averaging on the server.

The algorithm (McMahan et al., 2017) is short enough to state completely: the server holds
one global model; each round it ships those weights to the selected clients, each client
trains on its own data for ``E`` local epochs and ships the weights back, and the server
replaces the global model with the average of what came back, weighted by how many records
each client trained on. No gradients, no raw records and no labels ever leave a client —
only weights.

Two details in here are where a naive implementation quietly goes wrong.

**1. Weighting.** The average must be weighted by client sample count ``n_k``, not uniform.
With PTB-XL's device split the largest client holds 35% of the records and the smallest
0.9%; a uniform average would hand a 151-record shard the same authority as a 6018-record
one, which is neither what FedAvg specifies nor what anyone would want.

**2. BatchNorm buffers.** ``ECGResNet1d`` is full of ``BatchNorm1d``, whose ``running_mean``
and ``running_var`` are *not* parameters — they are statistics estimated from whatever data
the client happened to hold. Under a non-IID split those statistics genuinely differ between
clients, and averaging them produces a normalization that matches no client's actual input
distribution. This is a well-documented FedAvg failure mode rather than an implementation
bug, so :func:`average_state_dicts` averages them by default (that *is* FedAvg) but the
behaviour is switchable, and ``ECGResNet1d(norm="gn")`` offers GroupNorm as the structural
fix — GroupNorm normalizes per sample and keeps no cross-batch statistics at all, so there
is nothing distribution-dependent left to average.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BUFFER_MODES = ("average", "keep_global", "largest_client")


def average_state_dicts(
    states: list[dict],
    weights: list[float],
    buffer_mode: str = "average",
    global_state: dict | None = None,
) -> dict:
    """Weighted average of client ``state_dict``s — the FedAvg aggregation step.

    ``weights`` are normalized internally, so passing raw client sample counts is the
    intended use. ``buffer_mode`` controls the non-parameter tensors (BatchNorm running
    statistics):

    - ``"average"`` — average them like everything else. Plain FedAvg.
    - ``"keep_global"`` — leave the server's existing buffers untouched.
    - ``"largest_client"`` — adopt the buffers of the highest-weighted client, so the
      normalization at least matches *one* real data distribution instead of an average
      of several that matches none.

    Integer buffers (``num_batches_tracked``) are averaged and rounded back to their
    original dtype; they only influence BatchNorm when ``momentum=None``, which is not
    the case here, so the choice is immaterial but must not silently produce a float.
    """
    if not states:
        raise ValueError("no client states to aggregate")
    if len(states) != len(weights):
        raise ValueError(f"{len(states)} states but {len(weights)} weights")
    if buffer_mode not in BUFFER_MODES:
        raise ValueError(f"buffer_mode must be one of {BUFFER_MODES}, got {buffer_mode!r}")

    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        raise ValueError("client weights sum to zero")
    w = w / w.sum()

    # A tensor is a "buffer" here if it is not a float parameter shape we average, i.e.
    # anything the model registered as a buffer. Detect by name against the first state.
    ref = states[0]
    out: dict = {}
    biggest = int(np.argmax(w))

    for key, ref_val in ref.items():
        is_float = torch.is_floating_point(ref_val)
        if not is_float:
            # integer buffer (num_batches_tracked): round the weighted mean back to dtype
            acc = sum(float(s[key]) * wi for s, wi in zip(states, w, strict=True))
            out[key] = torch.tensor(round(acc), dtype=ref_val.dtype, device=ref_val.device)
            continue
        acc = torch.zeros_like(ref_val, dtype=torch.float64)
        for s, wi in zip(states, w, strict=True):
            acc += s[key].to(torch.float64) * wi
        out[key] = acc.to(ref_val.dtype)

    if buffer_mode != "average":
        source = global_state if buffer_mode == "keep_global" else states[biggest]
        if source is not None:
            for key, val in source.items():
                if _is_bn_buffer(key):
                    out[key] = val.clone()
    return out


def _is_bn_buffer(key: str) -> bool:
    return key.endswith(("running_mean", "running_var", "num_batches_tracked"))


@dataclass
class ClientUpdate:
    """What one client sends back: weights, how much data backed them, and local loss."""

    name: str
    n: int
    state: dict
    train_loss: float
    epochs: int


@dataclass
class RoundRecord:
    """One communication round, for the convergence curve."""

    round: int
    lr: float
    clients: list[str]
    n_total: int
    mean_train_loss: float
    val: dict = field(default_factory=dict)
    seconds: float = 0.0


def local_train(
    model: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    criterion: nn.Module,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device,
    seed: int = 0,
) -> tuple[dict, float]:
    """Train a *copy* of the global model on one client's data. Returns (state, loss).

    The optimizer is constructed fresh each round, which is plain FedAvg: only weights
    cross the network, so Adam's moment estimates cannot persist on the server. That
    restart is a real (and known) source of FedAvg's slower convergence versus centralized
    training, not something this implementation papers over.

    ``drop_last`` is off here, unlike centralized training: the smallest client holds 151
    records, and dropping a partial batch would discard a meaningful slice of it. Batches
    of size 1 are dropped instead, since BatchNorm cannot compute a variance over one
    sample.
    """
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)),
        batch_size=batch_size, shuffle=True, generator=g,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    running, seen = 0.0, 0
    for _ in range(epochs):
        for xb, yb in loader:
            if len(xb) < 2:  # BatchNorm needs >1 sample to estimate a variance
                continue
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * len(xb)
            seen += len(xb)
    return model.state_dict(), (running / seen if seen else float("nan"))


def clone_to(model: nn.Module, state: dict, device) -> nn.Module:
    """A fresh copy of ``model`` loaded with ``state`` — the 'ship the model down' step."""
    local = copy.deepcopy(model)
    local.load_state_dict(state)
    return local.to(device)


def select_clients(clients: list, fraction: float, rng: np.random.Generator) -> list:
    """Sample the participating clients for one round.

    Cross-silo federation (hospitals, not phones) normally runs with every client every
    round — they are few, known and reliably online — so ``fraction=1.0`` is the default
    elsewhere in this package. Partial participation is supported because it is the other
    half of the FedAvg literature and changes convergence noticeably.
    """
    if fraction >= 1.0:
        return list(clients)
    k = max(1, int(round(fraction * len(clients))))
    idx = rng.choice(len(clients), size=k, replace=False)
    return [clients[i] for i in sorted(idx)]
