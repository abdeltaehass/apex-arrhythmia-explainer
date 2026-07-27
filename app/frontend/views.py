"""Rendering for the APEX dashboard — the ECG+grounding figure and the report HTML.

Kept separate from the Gradio wiring (`app.py`) and free of any gradio import so the
figure/HTML builders can be unit-tested directly. Colours are assigned one per finding
so the same label reads the same in the plot overlay, its legend, and the report card.
"""

from __future__ import annotations

import html as _html

import numpy as np

from src.grounding.saliency import LEAD_NAMES
from src.serving.schema import APEXReport, FlagType
from src.serving.severity import banner_meta

# A qualitative palette; findings are assigned colours in a stable order.
_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
            "#e377c2", "#8c564b", "#bcbd22", "#7f7f7f", "#393b79", "#5254a3"]


def finding_colors(codes) -> dict[str, str]:
    """Stable code -> hex colour map (same order the report lists them)."""
    return {code: _PALETTE[i % len(_PALETTE)] for i, code in enumerate(codes)}


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def ecg_figure(clean_signal, saliency_by_code: dict, fs: int = 100,
               highlight: str | None = None, colors: dict[str, str] | None = None):
    """12-lead ECG with per-finding grounding overlays.

    ``clean_signal`` is the ``(12, T)`` preprocessed signal; ``saliency_by_code`` maps a
    finding code to its `grounding.LeadSaliency`. Each grounded finding shades the region
    of its most-important lead in the finding's colour. ``highlight`` emphasises one
    finding (others dim); ``None`` shows them all at equal weight.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    clean = np.asarray(clean_signal)
    n_leads, T = clean.shape
    t = np.arange(T) / fs
    colors = colors or finding_colors(list(saliency_by_code))
    spacing = 4.0
    offsets = [(n_leads - 1 - i) * spacing for i in range(n_leads)]  # lead I at top

    fig, ax = plt.subplots(figsize=(11, 8.5))
    for i in range(n_leads):
        ax.plot(t, clean[i] + offsets[i], color="#222", lw=0.6, zorder=3)
        ax.text(-0.35, offsets[i], LEAD_NAMES[i], va="center", ha="right",
                fontsize=9, fontweight="bold", color="#333")

    for code, sal in saliency_by_code.items():
        color = colors.get(code, "#d62728")
        lead = sal.top_lead
        s = _smooth(np.asarray(sal.per_lead[lead]), max(1, fs // 12))  # tidy the stripes
        base = offsets[lead]
        if highlight is None:
            alpha = 0.45
        elif highlight == code:
            alpha = 0.9
        else:
            alpha = 0.08
        ax.fill_between(t, base - spacing / 2, base + spacing / 2, where=s > 0.2,
                        color=color, alpha=alpha, lw=0, zorder=2, interpolate=False)

    handles = [Line2D([0], [0], color=colors.get(c, "#d62728"), lw=6)
               for c in saliency_by_code]
    labels = [f"{c} (lead {LEAD_NAMES[s.top_lead]})" for c, s in saliency_by_code.items()]
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.9,
                  title="grounded findings")
    ax.set_xlabel("time (s)")
    ax.set_yticks([])
    ax.set_xlim(-0.4, t[-1])
    ax.set_title("12-lead ECG with grounding overlays"
                 + (f" — highlighting {highlight}" if highlight else ""), fontsize=11)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


# --- HTML panels -------------------------------------------------------------
_FLAG_STYLE = {
    FlagType.LOW_CONFIDENCE: ("#8a6d00", "#fef7e0", "low confidence"),
    FlagType.GROUNDING_CONFLICT: ("#7a4f01", "#fbe9d8", "grounding conflict"),
    FlagType.MUTUAL_EXCLUSIVITY: ("#8a1c1c", "#fce8e6", "mutually exclusive"),
    FlagType.UNRELIABLE_INPUT: ("#5f2c91", "#efe6fa", "unreliable input"),
}


def _confidence_bar(conf: float, color: str) -> str:
    pct = round(conf * 100)
    return (
        f'<div style="background:#eee;border-radius:4px;height:9px;width:120px;'
        f'display:inline-block;vertical-align:middle;overflow:hidden">'
        f'<div style="background:{color};height:100%;width:{pct}%"></div></div>'
        f'<span style="font-size:12px;color:#555;margin-left:6px">{pct}%</span>'
    )


def _flag_chip(flag) -> str:
    fg, bg, text = _FLAG_STYLE.get(flag.type, ("#555", "#eee", str(flag.type)))
    return (f'<span title="{_html.escape(flag.message)}" style="background:{bg};color:{fg};'
            f'font-size:11px;padding:1px 7px;border-radius:10px;margin-right:4px;'
            f'white-space:nowrap">{text}</span>')


def disclaimer_html() -> str:
    return (
        '<div style="background:#eef3fb;border:1px solid #c9d8ef;border-radius:6px;'
        'padding:9px 12px;font-size:13px;color:#2c3e50">'
        '⚕️ <b>APEX is a clinical decision-support tool.</b> All outputs require clinical '
        'review and should not be used as a standalone diagnosis.</div>'
    )


def severity_banner_html(level: str, urgent: list[str] | None = None) -> str:
    meta = banner_meta(level)
    icon = {"red": "🔴", "yellow": "🟡", "green": "🟢"}[level]
    extra = ""
    if level == "red" and urgent:
        extra = f' <span style="font-weight:600">({", ".join(urgent)})</span>'
    return (
        f'<div style="background:{meta["bg"]};border-left:6px solid {meta["color"]};'
        f'border-radius:6px;padding:12px 14px">'
        f'<div style="font-size:16px;font-weight:700;color:{meta["color"]}">'
        f'{icon} {meta["label"]}{extra}</div>'
        f'<div style="font-size:13px;color:#444;margin-top:2px">{meta["detail"]}</div></div>'
    )


def report_html(report: APEXReport, colors: dict[str, str] | None = None) -> str:
    """Right-panel report: findings with confidence bars + flags, impression, explanation."""
    colors = colors or finding_colors([f.label for f in report.findings])
    rows = []
    for f in report.findings:
        color = colors.get(f.label, "#1f77b4")
        review = ('<span style="color:#b3261e;font-size:11px;font-weight:700;'
                  'margin-left:6px">● REVIEW</span>' if f.needs_review else "")
        chips = "".join(_flag_chip(fl) for fl in f.flags)
        desc = _html.escape(f.description or f.label)
        leads = (f'<span style="color:#888;font-size:11px">leads '
                 f'{_html.escape(", ".join(f.leads))}</span>' if f.leads else "")
        rows.append(
            f'<div style="padding:8px 0;border-bottom:1px solid #eee">'
            f'<div><span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
            f'background:{color};margin-right:7px"></span>'
            f'<b>{_html.escape(f.label)}</b> — {desc}{review}</div>'
            f'<div style="margin:4px 0 4px 17px">{_confidence_bar(f.confidence, color)} '
            f'&nbsp; {leads}</div>'
            f'<div style="margin-left:17px">{chips}</div></div>'
        )
    findings_html = "".join(rows) or '<div style="color:#888">No findings surfaced.</div>'

    cons = report.consistency
    cons_html = (
        '<span style="color:#0b6b3a">✓ explanation consistent with detector</span>'
        if cons.consistent else
        f'<span style="color:#b3261e">✗ inconsistent: {", ".join(cons.unsupported)}</span>'
    )
    return (
        '<div style="font-family:system-ui,-apple-system,sans-serif">'
        f'<h3 style="margin:6px 0">Findings ({len(report.findings)})</h3>{findings_html}'
        f'<h3 style="margin:14px 0 4px">Impression</h3>'
        f'<div style="font-size:14px;line-height:1.5">{_html.escape(report.impression)}</div>'
        f'<h3 style="margin:14px 0 4px">Full report</h3>'
        f'<pre style="white-space:pre-wrap;font-size:12.5px;background:#f7f7f9;padding:10px;'
        f'border-radius:6px;line-height:1.45">{_html.escape(report.explanation)}</pre>'
        f'<div style="font-size:12px;color:#555;margin-top:8px">{cons_html}</div>'
        f'<div style="font-size:11px;color:#999;margin-top:6px">{_html.escape(report.disclaimer)}</div>'
        '</div>'
    )
