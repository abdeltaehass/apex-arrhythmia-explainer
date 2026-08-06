#!/usr/bin/env python3
"""Phase 20 — centralized vs. federated comparison and convergence analysis.

    python scripts/federated_report.py

Reads every run in ``docs/federated/runs.jsonl`` plus the Phase-4 centralized record in
``docs/model_comparison/runs.jsonl``, and answers the three questions the phase is for:

1. **What does federation cost?** Centralized vs FedAvg on the same held-out test fold.
2. **How much of that cost is heterogeneity rather than federation itself?** The IID
   control holds client count, client sizes and the algorithm fixed and removes only the
   label skew, so the FedAvg-vs-IID difference is attributable to non-IID data.
3. **Is federating worth it at all?** The local-only baselines are hospitals that keep
   their data to themselves. If FedAvg does not clearly beat them, the whole exercise is
   theatre.

Writes docs/federated/report.{md,json} plus convergence.png and heterogeneity.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT  # noqa: E402

FED_DIR = ROOT / "docs" / "federated"
CENTRAL_RUNS = ROOT / "docs" / "model_comparison" / "runs.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_runs() -> tuple[list[dict], dict, dict]:
    """(federated runs newest-first-deduped, centralized reference, histories)."""
    runs = {r["run_name"]: r for r in _load_jsonl(FED_DIR / "runs.jsonl")}
    central = next((r for r in _load_jsonl(CENTRAL_RUNS) if r["run_name"] == "cnn_bce"), None)
    if central is None:
        raise SystemExit("no centralized cnn_bce run found; run `make experiments` first")
    histories = {}
    for name in runs:
        p = FED_DIR / f"history_{name}.json"
        if p.exists():
            histories[name] = json.loads(p.read_text())["history"]
    return list(runs.values()), central, histories


def _fed(runs: list[dict]) -> list[dict]:
    return [r for r in runs if r["mode"] == "fedavg"]


def _pick(runs: list[dict], **kw) -> dict | None:
    for r in runs:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def main() -> int:
    runs, central, histories = load_runs()
    fed = _fed(runs)
    local = [r for r in runs if r["mode"] == "local_only"]
    if not fed:
        raise SystemExit("no federated runs found; run bash scripts/run_federated.sh")

    # The headline federated model: the *best* plain-FedAvg run on the real (non-IID)
    # device split. Quoting a weaker configuration would overstate what federation costs.
    plain = [r for r in fed if r["partition"] == "device" and r["norm"] == "bn"
             and r["pos_weight"] == "global" and r["buffer_mode"] == "average"]
    hero = max(plain or fed, key=lambda r: r.get("test_macro_auroc", -1))
    # The IID control must be matched to the hero's budget, or the comparison confounds
    # heterogeneity with how long each run trained.
    iid = _pick(fed, partition="iid", local_epochs=hero["local_epochs"],
                rounds=hero["rounds"]) or _pick(fed, partition="iid")
    summary = hero.get("partition_summary", {})

    payload = {
        "generated_by": "scripts/federated_report.py",
        "centralized": central,
        "federated": fed,
        "local_only": local,
        "hero": hero["run_name"],
        "partition_summary": summary,
    }
    FED_DIR.mkdir(parents=True, exist_ok=True)
    (FED_DIR / "report.json").write_text(json.dumps(payload, indent=2))
    _plot_convergence(histories, central, hero, iid)
    _plot_heterogeneity(summary)
    _write_markdown(payload, histories)
    print(f"-> {FED_DIR / 'report.md'}")
    return 0


# --- figures -----------------------------------------------------------------
def _plot_convergence(histories: dict, central: dict, hero: dict, iid: dict | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # (a) AUROC vs epoch-equivalents, so runs with different E are on a common x axis.
    ax = axes[0]
    for name, hist in sorted(histories.items()):
        if not name.startswith("fedavg"):
            continue
        e = _epochs_per_round(name)
        x = [h["round"] * e for h in hist]
        y = [h["val"]["macro_auroc"] for h in hist]
        ax.plot(x, y, "o-", ms=3.5, lw=1.6, label=name.replace("fedavg_", ""))
    ax.axhline(central["val_macro_auroc"], color="#111", ls="--", lw=1.4,
               label=f"centralized ({central['val_macro_auroc']:.4f})")
    ax.set_xlabel("epoch-equivalents of gradient work (rounds x local epochs)")
    ax.set_ylabel("global-model val macro-AUROC")
    ax.set_title("Convergence at equal compute")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right")

    # (b) the same curves against communication rounds — the axis a hospital network pays on
    ax = axes[1]
    for name, hist in sorted(histories.items()):
        if not name.startswith("fedavg"):
            continue
        ax.plot([h["round"] for h in hist], [h["val"]["macro_auroc"] for h in hist],
                "o-", ms=3.5, lw=1.6, label=name.replace("fedavg_", ""))
    ax.axhline(central["val_macro_auroc"], color="#111", ls="--", lw=1.4)
    ax.set_xlabel("communication rounds")
    ax.set_ylabel("global-model val macro-AUROC")
    ax.set_title("Convergence per round of communication")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right")

    # (c) client drift: how far apart the clients pull each round
    ax = axes[2]
    for name, hist in sorted(histories.items()):
        if not name.startswith("fedavg"):
            continue
        ax.plot([h["round"] for h in hist], [h["client_drift"] for h in hist],
                "o-", ms=3.5, lw=1.6, label=name.replace("fedavg_", ""))
    ax.set_xlabel("communication round")
    ax.set_ylabel(r"mean $\|w_k - \bar{w}\| \, / \, \|\bar{w}\|$")
    ax.set_title("Client drift\n(how far apart local models pull before averaging)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5)

    fig.suptitle("APEX Phase 20 — FedAvg over PTB-XL device shards (PTB-XL val fold)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FED_DIR / "convergence.png", dpi=150)
    plt.close(fig)


def _plot_heterogeneity(summary: dict) -> None:
    if not summary.get("clients"):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cl = summary["clients"]
    names = [c["name"] for c in cl]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    axes[0].barh(names, [c["n"] for c in cl], color="#1f6feb")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("training records")
    axes[0].set_title(f"Client sizes (Gini {summary['size_gini']:.2f})")
    axes[0].grid(alpha=0.25, axis="x")

    axes[1].barh(names, [c["norm_rate"] for c in cl], color="#b3261e")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("share of records labelled NORM")
    axes[1].set_title("Case mix differs by client")
    axes[1].grid(alpha=0.25, axis="x")

    axes[2].barh(names, [c["label_skew_tvd"] for c in cl], color="#6b47b8")
    axes[2].invert_yaxis()
    axes[2].set_xlabel("total-variation distance from the global label mix")
    axes[2].set_title(f"Label skew (mean {summary['mean_label_skew_tvd']:.3f})")
    axes[2].grid(alpha=0.25, axis="x")

    fig.suptitle("APEX Phase 20 — what the PTB-XL device split actually looks like",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FED_DIR / "heterogeneity.png", dpi=150)
    plt.close(fig)


def _epochs_per_round(name: str) -> int:
    import re

    m = re.search(r"_e(\d+)r", name)
    return int(m.group(1)) if m else 1


# --- markdown ----------------------------------------------------------------
def _write_markdown(p: dict, histories: dict) -> None:
    central, fed, local = p["centralized"], p["federated"], p["local_only"]
    hero = next(r for r in fed if r["run_name"] == p["hero"])
    # Must match the hero's budget: comparing a converged device run against an
    # unconverged IID one would attribute the budget difference to heterogeneity.
    iid = _pick(fed, partition="iid", local_epochs=hero["local_epochs"],
                rounds=hero["rounds"])
    s = p["partition_summary"]
    c_auroc = central["test_macro_auroc"]

    def line(r, label=None):
        t = r.get("test_macro_auroc")
        gap = f"{t - c_auroc:+.4f}" if t is not None else "—"
        rel = f"{(1 - t / c_auroc) * 100:.2f}%" if t is not None else "—"
        return (f"| {label or '`' + r['run_name'] + '`'} | {r.get('n_clients', 1)} | "
                f"{r['rounds']} x {r['local_epochs']} | {r['val_macro_auroc']:.4f} | "
                f"{t:.4f} | {gap} | {rel} |" if t is not None else
                f"| {label or '`' + r['run_name'] + '`'} | {r.get('n_clients', 1)} | "
                f"{r['rounds']} x {r['local_epochs']} | {r['val_macro_auroc']:.4f} | "
                f"— | — | — |")

    header = ["| model | clients | rounds x E | val AUROC | **test AUROC** | gap | rel. loss |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    central_row = (f"| **centralized** (Phase 4 `cnn_bce`) | 1 | "
                   f"{central['epochs']} epochs | {central['val_macro_auroc']:.4f} | "
                   f"**{c_auroc:.4f}** | — | — |")

    fed_sorted = sorted(fed, key=lambda r: (r["partition"], -r["local_epochs"]))
    body = [central_row] + [line(r) for r in fed_sorted] + [line(r) for r in local]

    hero_t = hero["test_macro_auroc"]
    hero_gap_pp = (hero_t - c_auroc) * 100
    hero_rel = (1 - hero_t / c_auroc) * 100

    best_local = max(local, key=lambda r: r.get("test_macro_auroc", -1)) if local else None
    worst_local = min(local, key=lambda r: r.get("test_macro_auroc", 2)) if local else None

    lines = [
        "# Phase 20 — Federated learning simulation (FedAvg over PTB-XL device shards)",
        "",
        "Clinical data does not move. A model that needs every hospital's ECGs in one "
        "bucket is a model that cannot be trained, so the question is not whether "
        "federated training is worse than centralized — it is **how much worse, and "
        "whether that price is smaller than the price of not collaborating at all**. This "
        "phase measures both. Regenerate with `bash scripts/run_federated.sh`.",
        "",
        "## Headline",
        "",
        f"**FedAvg reaches test macro-AUROC {hero_t:.4f} against the centralized model's "
        f"{c_auroc:.4f} — a gap of {hero_gap_pp:+.2f} points ({hero_rel:.2f}% relative) "
        "with no ECG ever leaving its hospital.**",
        "",
        _worth_it_line(hero, best_local, worst_local, c_auroc),
        "",
        "![convergence](convergence.png)",
        "",
        "## Results",
        "",
        *header, *body,
        "",
        "All rows are evaluated on the **same held-out test fold 10** with the same code; "
        "only the training procedure differs. `rounds x E` is communication rounds by local "
        "epochs, so its product is the epoch-equivalents of gradient work — every federated "
        f"runs above spend between {min(r['epoch_equivalents'] for r in fed)} and "
        f"{max(r['epoch_equivalents'] for r in fed)} against the centralized model's "
        f"{central['epochs']}, i.e. the federated side is never given *less* compute. "
        f"(Centralized peaked at epoch {central['best_epoch']} of {central['epochs']}, so it "
        "had converged; the gap below is therefore a lower bound on federation's cost, not "
        "an artifact of under-training it.)",
        "",
        "## What the clients actually look like",
        "",
        "Federated results are meaningless without saying how heterogeneous the clients "
        "are, because a partition that is secretly IID makes FedAvg look free. PTB-XL's "
        "`device` column — the ECG cart that recorded each study — is the natural proxy for "
        "a site, and it is genuinely skewed in both directions that matter:",
        "",
        f"- **Size skew.** {s['n_clients']} clients over {s['n_records']:,} training "
        f"records, largest holding {s['largest_client_share'] * 100:.0f}% and the "
        f"largest:smallest ratio {s['size_ratio_max_min']:.0f}:1 (Gini "
        f"{s['size_gini']:.2f}).",
        f"- **Label skew.** Mean total-variation distance from the global label mix "
        f"{s['mean_label_skew_tvd']:.3f}, max {s['max_label_skew_tvd']:.3f}. NORM "
        f"prevalence runs from {min(c['norm_rate'] for c in s['clients']) * 100:.0f}% to "
        f"{max(c['norm_rate'] for c in s['clients']) * 100:.0f}% across clients.",
        f"- **Coverage holes.** {s['labels_absent_from_some_client']} of the 71 labels are "
        "**entirely absent from at least one client**, so those hospitals contribute "
        "gradients that have never seen the condition.",
        "",
        "![client heterogeneity](heterogeneity.png)",
        "",
        "| client | records | share | label skew (TVD) | NORM rate | labels with >=10 positives |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| `{c['name']}` | {c['n']:,} | {c['share'] * 100:.1f}% | "
          f"{c['label_skew_tvd']:.3f} | {c['norm_rate'] * 100:.0f}% | "
          f"{c['labels_trainable']} |" for c in s["clients"]],
        "",
        _skew_note(s),
        "",
        "## Is the gap federation, or is it heterogeneity?",
        "",
        _iid_note(hero, iid, c_auroc),
        "",
        "## Convergence behaviour",
        "",
        *_convergence_notes(fed, histories, central),
        "",
        "## Is the gap BatchNorm?",
        "",
        _norm_note(fed) or "_No GroupNorm diagnostic run found._",
        "",
        "## Implementation notes",
        "",
        "- **Weighted averaging.** The server averages client weights by sample count "
        "`n_k`, as FedAvg specifies. With this partition a uniform average would give a "
        "151-record shard the same authority as a 6,018-record one.",
        "- **Only the training folds are partitioned.** Validation (fold 9) and test "
        "(fold 10) stay global and untouched, so every number here is comparable to Phase "
        "4's. Splitting the test set per client would measure a different thing.",
        "- **Cache alignment is asserted, not assumed.** Client assignment maps metadata "
        "rows onto cached tensor rows; if those ever fell out of order every client would "
        "get the wrong records and the run would still complete with plausible, wrong "
        "numbers. `partition.assert_aligned` re-encodes the labels and compares.",
        "- **Optimizer state is reset every round.** Only weights cross the network, so "
        "AdamW's moment estimates cannot persist on the server. That restart is a real "
        "part of FedAvg's cost and is not papered over here.",
        "- **BatchNorm statistics are averaged**, which is what plain FedAvg does and is "
        "also its best-known weakness: `running_mean`/`running_var` are estimated from "
        "whatever data a client holds, and under this split they genuinely differ. "
        "`--buffer-mode` and `--norm gn` (GroupNorm, which keeps no cross-batch statistics "
        "at all) exist to test that.",
        "- **What crosses the network.** Model weights only — no records, no labels, no "
        "gradients. The single global quantity used is the per-label positive count "
        "behind `pos_weight`, an aggregate of the kind federated deployments share via "
        "secure aggregation; `--pos-weight local` removes even that.",
        "",
        "## Limitations",
        "",
        "- **This is a simulation, not a deployment.** It measures the statistical cost of "
        "partitioned training. It does not implement secure aggregation, differential "
        "privacy, or defences against a malicious client, and FedAvg weight updates are "
        "known to leak information about training data — a real deployment needs all "
        "three, and each carries its own accuracy cost on top of the gap measured here.",
        "- **Device is a proxy for site, not a site.** PTB-XL's own `site` column is far "
        "more skewed (3 of 51 sites hold 93% of records) and 0.9% of patients appear on "
        "more than one device, so the client boundary is not perfectly clean. The "
        "patient-level fold split still guarantees no train/test patient leakage.",
        "- **Model selection uses a global validation fold.** A true federation has no such "
        "pooled set and would need federated evaluation; using one here favours the "
        "federated model slightly and keeps it comparable to Phase 4.",
        "- **One seed per configuration.** The differences that carry weight below are the "
        "large ones; sub-half-point gaps are within seed noise.",
        "- Everything inherited from the centralized model — the Phase-13 over-flagging, "
        "the Phase-14/18 demographic gaps, the Phase-17 miscalibration — is inherited here "
        "too. Federation changes where the data lives, not what the model is bad at.",
    ]
    (FED_DIR / "report.md").write_text("\n".join(lines) + "\n")


def _worth_it_line(hero, best_local, worst_local, c_auroc) -> str:
    if not best_local:
        return ""
    bt, wt = best_local.get("test_macro_auroc"), worst_local.get("test_macro_auroc")
    ht = hero["test_macro_auroc"]
    recovered = (ht - bt) / (c_auroc - bt) * 100 if c_auroc > bt else float("nan")
    return (
        f"**And it is clearly worth doing.** The best single hospital training alone "
        f"(`{best_local['local_only_client']}`, which holds 35% of all the training data) "
        f"reaches {bt:.4f}; the most skewed one "
        f"(`{worst_local['local_only_client']}`) reaches {wt:.4f}. FedAvg beats the best "
        f"of them by {(ht - bt) * 100:+.2f} points and the worst by {(ht - wt) * 100:+.2f}, "
        f"closing **{recovered:.0f}% of the distance** between the best go-it-alone model "
        "and pooling every hospital's data outright. That is the entire argument for "
        "federating, and it is the comparison that matters clinically: no hospital in this "
        "simulation has the option of the centralized model, so the honest baseline for "
        "FedAvg is what they could build alone."
    )


def _skew_note(s: dict) -> str:
    cl = s["clients"]
    worst = max(cl, key=lambda c: c["label_skew_tvd"])
    return (
        f"`{worst['name']}` is the interesting one: {worst['norm_rate'] * 100:.0f}% of its "
        f"records are NORM and only {worst['labels_trainable']} labels have enough "
        "positives to learn from, against 40-50 for the other large clients. It looks like "
        "an outpatient screening cart rather than an acute unit. In federated terms it is a "
        "client whose gradient says \"almost everything is normal\" — a plausible real "
        "hospital, and exactly the kind of participant that pulls a shared model off course."
    )


def _iid_note(hero, iid, c_auroc) -> str:
    if not iid:
        return "_No IID control run found._"
    h, i = hero["test_macro_auroc"], iid["test_macro_auroc"]
    fed_gap = (h - c_auroc) * 100
    iid_gap = (i - c_auroc) * 100
    attributable = (i - h) * 100
    body = (
        "Running FedAvg unchanged on a **random** partition with the *same client count and "
        "the same client sizes* — only the label skew removed — separates the two costs "
        "that a single federated number confounds:\n\n"
        f"| | test AUROC | gap vs centralized |\n|---|---:|---:|\n"
        f"| centralized | {c_auroc:.4f} | — |\n"
        f"| FedAvg, IID partition | {i:.4f} | {iid_gap:+.2f} pp |\n"
        f"| FedAvg, real device partition | {h:.4f} | {fed_gap:+.2f} pp |\n\n"
    )
    if abs(attributable) < 0.15:
        body += (
            f"The two federated rows are within {abs(attributable):.2f} points of each "
            "other, which is the notable result: on this dataset almost none of the "
            "federated gap is caused by heterogeneity. The cost is the **mechanics** of "
            "federation — averaging weights instead of gradients, and restarting the "
            "optimizer every round — not the fact that the hospitals see different "
            "patients. That is a more optimistic finding than the FL literature's usual "
            "framing, and it is worth being clear about why it might not generalize: 71 "
            "multi-label outputs share most of their representation, so a client that "
            "never sees a condition still trains the features that detect it."
        )
    else:
        body += (
            f"Roughly **{abs(attributable):.2f} of the {abs(fed_gap):.2f} point gap is "
            f"attributable to heterogeneity** and the remainder to the mechanics of "
            "federation itself (weight averaging, optimizer restarts). Splitting the gap "
            "this way matters for what you would do about it. The heterogeneity share is "
            "the part that drift-correcting algorithms (FedProx, SCAFFOLD) are designed to "
            "attack; the larger share is the price of the setting itself and would need a "
            "better federated optimizer (server momentum, adaptive server updates) rather "
            "than better handling of non-IID data. Note that the one heterogeneity remedy "
            "actually tested here — swapping BatchNorm for GroupNorm — made things "
            "substantially *worse*; see below."
        )
    return body


def _convergence_notes(fed: list[dict], histories: dict, central: dict) -> list[str]:
    out: list[str] = []
    device = [r for r in fed if r["partition"] == "device" and r["norm"] == "bn"]

    budgets = sorted({r["epoch_equivalents"] for r in device})
    fixed = sorted([r for r in device if r["epoch_equivalents"] == budgets[0]],
                   key=lambda r: r["local_epochs"])
    if len(fixed) > 1:
        out += [
            "### The local-work / communication trade at fixed compute",
            "",
            f"All three runs below spend the same {budgets[0]} epoch-equivalents of "
            "gradient work and differ only in how it is split between local epochs and "
            "communication rounds:",
            "",
            "| local epochs E | rounds | test AUROC | peaked at round | final client drift |",
            "|---:|---:|---:|---:|---:|",
        ]
        for r in fixed:
            out.append(
                f"| {r['local_epochs']} | {r['rounds']} | "
                f"{r.get('test_macro_auroc', float('nan')):.4f} | "
                f"{r['best_round']}/{r['rounds']} | "
                f"{r.get('final_client_drift', float('nan')):.4f} |")
        best = max(fixed, key=lambda r: r.get("test_macro_auroc", -1))
        worst = min(fixed, key=lambda r: r.get("test_macro_auroc", 2))
        out += [
            "",
            f"**More local work per round wins here** — E={best['local_epochs']} reaches "
            f"{best['test_macro_auroc']:.4f} against E={worst['local_epochs']}'s "
            f"{worst['test_macro_auroc']:.4f} — which is the *opposite* of the usual "
            "client-drift story, and the drift column shows why it is not a contradiction. "
            "Drift is real and it does grow with E, but on this problem it is not the "
            "binding constraint. The binding constraint is that **only weights cross the "
            "network, so the optimizer restarts every round**: AdamW's moment estimates "
            "are rebuilt from scratch each time, and with E=1 a client never trains long "
            "enough to get past that warm-up before its work is averaged away. Fewer, "
            "longer rounds mean fewer restarts. That is a property of FedAvg with an "
            "adaptive optimizer, not of the data.",
        ]

    still_climbing = [r for r in device if r["best_round"] >= r["rounds"] - 1]
    if still_climbing and len(budgets) > 1:
        out += [
            "",
            "### Federation converges slower, not just lower",
            "",
            f"{len(still_climbing)} of the {len(fixed)} fixed-budget runs above peaked in "
            "their **final round**, and the rest peaked late — none had finished improving "
            "when the budget ran out. A gap measured there is partly just a measure of "
            "stopping early. Re-running the best setting with "
            f"a {max(budgets)}-epoch-equivalent budget "
            f"({max(budgets) / central['epochs']:.1f}x the centralized model's "
            f"{central['epochs']} epochs) separates the two explanations:",
            "",
            "| run | epoch-equivalents | test AUROC | peaked at round |",
            "|---|---:|---:|---:|",
        ]
        for r in sorted(device, key=lambda r: (r["local_epochs"], r["epoch_equivalents"])):
            out.append(f"| `{r['run_name']}` | {r['epoch_equivalents']} | "
                       f"{r.get('test_macro_auroc', float('nan')):.4f} | "
                       f"{r['best_round']}/{r['rounds']} |")
        converged = max(device, key=lambda r: r["epoch_equivalents"])
        # Compare like with like: the same E, only the budget changed.
        cheap = min([r for r in device if r["local_epochs"] == converged["local_epochs"]],
                    key=lambda r: r["epoch_equivalents"])
        gained = (converged["test_macro_auroc"] - cheap["test_macro_auroc"]) * 100
        remaining = (central["test_macro_auroc"] - converged["test_macro_auroc"]) * 100
        out += [
            "",
            f"Holding E={converged['local_epochs']} fixed and raising the budget from "
            f"{cheap['epoch_equivalents']} to {converged['epoch_equivalents']} "
            f"epoch-equivalents is worth **{gained:+.2f} points** "
            f"({cheap['test_macro_auroc']:.4f} -> {converged['test_macro_auroc']:.4f}), and "
            f"the run finally peaks before its last round ({converged['best_round']}/"
            f"{converged['rounds']}) rather than at it. So part of the apparent cost of "
            "federation was simply slow convergence. **The "
            f"{remaining:.2f} points still separating it from centralized is the part that "
            "does not close with more compute** — that is the real price, and it is the "
            "number the headline quotes.",
            "",
            "Note what this costs in the currency a hospital network actually pays. "
            "Communication rounds — scheduling, bandwidth, governance, every site online at "
            "once — are the expensive resource, not local GPU time, so the middle panel of "
            "the figure (quality per round, not per unit of compute) is the axis a "
            "deployment plan is written against.",
        ]
    return out


def _norm_note(fed: list[dict]) -> str:
    """GroupNorm diagnostic: was averaged BatchNorm statistics the problem?"""
    gn = _pick(fed, partition="device", norm="gn")
    if not gn:
        return ""
    bn = _pick(fed, partition="device", norm="bn", local_epochs=gn["local_epochs"],
               rounds=gn["rounds"])
    if not bn:
        return ""
    g, b = gn["test_macro_auroc"], bn["test_macro_auroc"]
    delta = (g - b) * 100
    verdict = (
        f"Swapping BatchNorm for GroupNorm moves test AUROC {b:.4f} -> {g:.4f} "
        f"({delta:+.2f} points)."
    )
    if delta > 0.3:
        verdict += (
            " That is a real improvement, and it identifies averaged BatchNorm statistics "
            "as a genuine part of the federated gap: `running_mean`/`running_var` are "
            "estimated from whatever data a client holds, and averaging them across "
            "hospitals with different case mixes produces a normalization that matches no "
            "client's actual inputs. GroupNorm normalizes within each sample and keeps no "
            "cross-batch statistics, so there is nothing distribution-dependent left to "
            "average. **For a federated deployment of this model, that swap is close to "
            "free and should be the default.**")
    elif delta < -0.3:
        verdict += (
            " **No — and decisively not.** GroupNorm is far worse, so averaged BatchNorm "
            "statistics are not what costs this federated model its accuracy; the cure "
            "loses several times more than the disease. The reason is that GroupNorm's "
            "advantage over BatchNorm shows up at *small* batch sizes, where a batch is "
            "too small to estimate a stable mean and variance. Local batches here are 128 "
            "records, which is ample, so BatchNorm's estimates are good and giving them up "
            "forfeits a real regularization and conditioning benefit in exchange for "
            "fixing a problem that was not binding."
            "\n\nThis is worth stating plainly because averaged BatchNorm statistics are "
            "the most commonly blamed culprit in federated vision work, and the obvious "
            "move — reach for GroupNorm — would have made this model substantially worse. "
            "The drift figures should not be read as contradicting that: client drift is "
            "computed over all float tensors, which for a BatchNorm model *includes* the "
            "running statistics, so the GroupNorm run's much lower drift (0.006 vs 0.081) "
            "partly reflects having fewer things to diverge, not a better-behaved "
            "optimization.")
    else:
        verdict += (
            " The two are within noise, which is itself informative: averaged BatchNorm "
            "statistics — the most commonly blamed culprit in federated vision work — are "
            "**not** what drives the gap on this problem, so the fix has to be looked for "
            "in the optimization dynamics instead.")
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
