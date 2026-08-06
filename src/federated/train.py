#!/usr/bin/env python3
"""Phase 20 — train the detector with FedAvg across simulated hospital clients.

    python -m src.federated.train --rounds 30 --local-epochs 1        # FedAvg, device split
    python -m src.federated.train --by iid --rounds 30                # IID control
    python -m src.federated.train --local-only "CS-12   E"            # one hospital alone
    python -m src.federated.train --norm gn                           # GroupNorm variant
    python -m src.federated.train --smoke                             # tiny end-to-end check

The comparison against Phase 4 is only meaningful if *nothing else changes*, so this
deliberately reuses the centralized recipe wholesale: same architecture, same
class-weighted BCE, same AdamW, same cosine schedule, same cached tensors, and the same
global validation and test folds. Only the training loop differs — folds 1-8 are
partitioned across clients and never pooled.

**Compute budget.** ``rounds x local_epochs`` is the number of passes each client makes
over its own data, so it is also the number of epoch-equivalents of gradient work over the
whole training set. Setting ``rounds=30, local_epochs=1`` costs the same as 30 centralized
epochs. Holding that product fixed while varying ``local_epochs`` is the honest way to ask
what more local work buys per unit of communication.

**What is measured each round**, beyond validation AUROC:

- *client drift* — the mean relative L2 distance between each client's post-training
  weights and the server average. This is the quantity that non-IID data inflates and that
  FedAvg's averaging step has to absorb, so it is the mechanism behind any gap, not just
  evidence that one exists.
- *per-client local loss*, which exposes clients whose data is too small or too skewed to
  train on at all.

**Privacy accounting, stated plainly.** Only weights cross the network. The one global
quantity used here is the per-label positive count for ``pos_weight``, taken over the union
of clients (``--pos-weight global``, the default): an aggregate count, the kind of statistic
federated deployments share through secure aggregation, and shared so the comparison
isolates FedAvg's cost rather than the cost of not knowing the global class balance.
``--pos-weight local`` removes even that and is reported as an ablation.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import ROOT
from src.data.labels import build_label_space, load_database
from src.detection.data_cache import build_split_cache
from src.detection.losses import build_loss
from src.detection.model import build_model, count_parameters
from src.detection.train import eval_split, pick_device, pos_weight_from, predict, set_seed
from src.eval.metrics import expected_calibration_error, f1_scores, macro_auroc
from src.federated.fedavg import (
    ClientUpdate,
    RoundRecord,
    average_state_dicts,
    clone_to,
    local_train,
    select_clients,
)
from src.federated.partition import assert_aligned, build_clients, partition_summary

OUT_DIR = ROOT / "outputs"
FED_DIR = ROOT / "docs" / "federated"


def relative_drift(states: list[dict], averaged: dict) -> float:
    """Mean ``||w_k - w_avg|| / ||w_avg||`` over clients — how far apart the clients pulled.

    Computed over floating-point tensors only (integer BatchNorm counters are not weights).
    A single number per round, so the convergence plot can show drift alongside AUROC.
    """
    keys = [k for k, v in averaged.items() if torch.is_floating_point(v)]
    denom = torch.sqrt(sum((averaged[k].to(torch.float64) ** 2).sum() for k in keys))
    if float(denom) == 0:
        return float("nan")
    dists = []
    for s in states:
        d = torch.sqrt(sum(((s[k].to(torch.float64) - averaged[k].to(torch.float64)) ** 2).sum()
                           for k in keys))
        dists.append(float(d / denom))
    return float(np.mean(dists))


def evaluate(model, loader, device) -> dict:
    y_true, y_prob = predict(model, loader, device)
    return {
        "macro_auroc": macro_auroc(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        **f1_scores(y_true, (y_prob >= 0.5).astype(int)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--by", default="device", choices=("device", "site", "iid"),
                    help="how to partition folds 1-8 into clients")
    ap.add_argument("--rounds", type=int, default=30, help="communication rounds")
    ap.add_argument("--local-epochs", type=int, default=1, help="E: local passes per round")
    ap.add_argument("--client-fraction", type=float, default=1.0,
                    help="fraction of clients per round (cross-silo default: all)")
    ap.add_argument("--buffer-mode", default="average",
                    choices=("average", "keep_global", "largest_client"),
                    help="how to aggregate BatchNorm running statistics")
    ap.add_argument("--norm", default="bn", choices=("bn", "gn"))
    ap.add_argument("--pos-weight", default="global", choices=("global", "local"))
    ap.add_argument("--local-only", default=None,
                    help="skip federation: train on this client alone (baseline)")
    ap.add_argument("--min-client-records", type=int, default=100)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--loss", default="bce", choices=("bce", "focal"))
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--pos-weight-cap", type=float, default=50.0)
    ap.add_argument("--sampling-rate", type=int, default=100, choices=(100, 500))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-eval-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.rounds, args.batch_size = 2, 8
        args.width, args.blocks = 8, 1
        args.no_eval_test = True

    model_tag = "" if args.norm == "bn" else f"_{args.norm}"
    if args.local_only:
        default_name = f"local_{_slug(args.local_only)}{model_tag}"
    else:
        default_name = f"fedavg_{args.by}_e{args.local_epochs}r{args.rounds}{model_tag}"
        if args.buffer_mode != "average":
            default_name += f"_{args.buffer_mode}"
        if args.pos_weight != "global":
            default_name += "_localpw"
    run_name = args.run_name or default_name

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"run={run_name}  by={args.by}  rounds={args.rounds}  E={args.local_epochs}  "
          f"norm={args.norm}  device={device}")

    print("loading cached splits...")
    Xtr, Ytr = build_split_cache("train", args.sampling_rate)
    Xva, Yva = build_split_cache("val", args.sampling_rate)
    label_space = build_label_space()
    df = load_database()
    assert_aligned(df, Ytr, label_space)  # refuse to run on a misaligned cache

    clients = build_clients(df, Ytr, by=args.by, min_records=args.min_client_records,
                            seed=args.seed)
    summary = partition_summary(clients, Ytr, label_space)
    if args.smoke:  # keep the smoke run to a couple of tiny clients
        clients = [type(c)(c.name, c.indices[:64], c.label_prevalence, c.meta)
                   for c in clients[:2]]

    if args.local_only:
        matches = [c for c in clients if c.name == args.local_only]
        if not matches:
            print(f"no client named {args.local_only!r}; available: "
                  f"{[c.name for c in clients]}")
            return 1
        clients = matches
        print(f"LOCAL-ONLY baseline: {clients[0].name} ({clients[0].n} records, "
              f"{clients[0].n / len(Ytr) * 100:.1f}% of the training set)")
    else:
        print(f"{len(clients)} clients | sizes {[c.n for c in clients]} | "
              f"mean label skew (TVD) {summary['mean_label_skew_tvd']}")

    val_loader = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)),
                            batch_size=256)

    global_model = build_model("cnn", width=args.width, blocks=args.blocks,
                               dropout=args.dropout, norm=args.norm).to(device)
    n_params = count_parameters(global_model)
    print(f"model params: {n_params:,}")

    global_pw = pos_weight_from(Ytr, args.pos_weight_cap)
    rng = np.random.default_rng(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = OUT_DIR / f"{run_name}_best.pt"
    best_auroc, best_round, history = -1.0, 0, []
    t_start = time.time()

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        # Cosine over rounds — the server-side analogue of the centralized schedule.
        lr = 0.5 * args.lr * (1 + np.cos(np.pi * (rnd - 1) / max(1, args.rounds)))
        participants = select_clients(clients, args.client_fraction, rng)
        global_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}

        updates: list[ClientUpdate] = []
        for c in participants:
            local = clone_to(global_model, global_state, device)
            pw = (pos_weight_from(Ytr[c.indices], args.pos_weight_cap)
                  if args.pos_weight == "local" else global_pw).to(device)
            criterion = build_loss(args.loss, pos_weight=pw, focal_gamma=args.focal_gamma)
            state, loss = local_train(
                local, Xtr[c.indices], Ytr[c.indices], criterion,
                epochs=args.local_epochs, lr=lr, weight_decay=args.weight_decay,
                batch_size=args.batch_size, device=device,
                seed=args.seed + rnd * 1000 + len(updates),
            )
            updates.append(ClientUpdate(c.name, c.n, {k: v.detach().cpu().clone()
                                                      for k, v in state.items()},
                                        loss, args.local_epochs))

        states = [u.state for u in updates]
        weights = [float(u.n) for u in updates]
        new_state = average_state_dicts(
            states, weights, buffer_mode=args.buffer_mode,
            global_state={k: v.cpu() for k, v in global_state.items()},
        )
        drift = relative_drift(states, new_state) if len(states) > 1 else 0.0
        global_model.load_state_dict(new_state)
        global_model.to(device)

        m = evaluate(global_model, val_loader, device)
        n_total = sum(weights)
        rec = RoundRecord(
            round=rnd, lr=float(lr), clients=[u.name for u in updates], n_total=int(n_total),
            mean_train_loss=float(np.average([u.train_loss for u in updates], weights=weights)),
            val=m, seconds=round(time.time() - t0, 1),
        )
        history.append({**rec.__dict__, "client_drift": round(drift, 5),
                        "client_losses": {u.name: round(u.train_loss, 4) for u in updates}})
        print(f"round {rnd:3d}/{args.rounds}  loss={rec.mean_train_loss:.4f}  "
              f"valAUROC={m['macro_auroc']:.4f}  drift={drift:.4f}  ({rec.seconds:.0f}s)")

        if m["macro_auroc"] > best_auroc:
            best_auroc, best_round = m["macro_auroc"], rnd
            torch.save({"model": global_model.state_dict(),
                        "args": {**vars(args), "model": "cnn"},
                        "epoch": rnd, "params": n_params}, best_path)

    train_time = time.time() - t_start
    global_model.load_state_dict(torch.load(best_path)["model"])
    global_model.to(device)
    val_m = evaluate(global_model, val_loader, device)

    record = {
        "run_name": run_name,
        "mode": "local_only" if args.local_only else "fedavg",
        "partition": args.by, "n_clients": len(clients),
        "rounds": args.rounds, "local_epochs": args.local_epochs,
        "epoch_equivalents": args.rounds * args.local_epochs,
        "client_fraction": args.client_fraction, "buffer_mode": args.buffer_mode,
        "norm": args.norm, "pos_weight": args.pos_weight,
        "local_only_client": args.local_only,
        "params": n_params, "lr": args.lr, "seed": args.seed,
        "best_round": best_round, "train_time_s": round(train_time, 1),
        "checkpoint": best_path.name,
        "val_macro_auroc": val_m["macro_auroc"], "val_macro_f1": val_m["macro_f1"],
        "val_micro_f1": val_m["micro_f1"], "val_ece": val_m["ece"],
        "final_client_drift": history[-1]["client_drift"] if history else None,
        "partition_summary": summary,
    }
    print(f"\nBEST val macro-AUROC {best_auroc:.4f} (round {best_round}, {train_time:.0f}s)")

    if not args.no_eval_test:
        Xte, Yte = build_split_cache("test", args.sampling_rate)
        test_m = eval_split(global_model, Xte, Yte, device)
        record.update({"test_macro_auroc": test_m["macro_auroc"],
                       "test_macro_f1": test_m["macro_f1"],
                       "test_micro_f1": test_m["micro_f1"]})
        print(f"TEST macro-AUROC {test_m['macro_auroc']:.4f}  "
              f"macro-F1 {test_m['macro_f1']:.4f}")

    FED_DIR.mkdir(parents=True, exist_ok=True)
    (FED_DIR / f"history_{run_name}.json").write_text(
        json.dumps({"run": record, "history": history}, indent=2))
    with (FED_DIR / "runs.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"history -> {FED_DIR / f'history_{run_name}.json'}  |  record -> runs.jsonl")
    return 0


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_").replace("__", "_")


if __name__ == "__main__":
    raise SystemExit(main())
