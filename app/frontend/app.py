"""APEX clinical dashboard (Phase 11) — a Gradio front end over the full pipeline.

Layout:
  - a fixed decision-support disclaimer + a dynamic severity banner (green/yellow/red)
  - upload panel: a signal file (.npy/.csv/.json) OR a photo of a paper ECG
  - left: the 12-lead ECG with per-finding grounding overlays; pick a finding to
    highlight just its saliency
  - right: the structured report — findings with confidence bars, flags, impression,
    and the full explanation

Runs the pipeline in-process (`serving.analyze_detailed`), so the Space needs the
detector checkpoint (`outputs/final_best.pt`); see `docs/frontend/deploy.md`. This file
is the Hugging Face Space entry point (`app_file: app.py`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gradio as gr  # noqa: E402

from app.frontend.views import (  # noqa: E402
    disclaimer_html,
    ecg_figure,
    finding_colors,
    report_html,
    severity_banner_html,
)
from src.serving.schema import InputValidationError  # noqa: E402
from src.serving.serializer import analyze_detailed  # noqa: E402
from src.serving.severity import severity, urgent_findings  # noqa: E402

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
BACKEND = os.environ.get("APEX_BACKEND", "template")


def _analyze(file_path):
    """Upload -> (severity banner, ECG figure, report HTML, finding choices, state)."""
    if not file_path:
        return (severity_banner_html("green"), None,
                '<div style="color:#888">Upload an ECG to begin.</div>',
                gr.update(choices=[], value=None), None)
    from src.serving.loaders import parse_signal_upload

    content = Path(file_path).read_bytes()
    try:
        signal, sr = parse_signal_upload(Path(file_path).name, content)
        res = analyze_detailed(signal, sampling_rate=sr or 100, backend=BACKEND)
    except InputValidationError as e:
        msg = "; ".join(e.validation.errors) or "invalid input"
        return (severity_banner_html("yellow"), None,
                f'<div style="color:#b3261e">Input rejected: {msg}</div>',
                gr.update(choices=[], value=None), None)
    except Exception as e:  # noqa: BLE001 - surface any pipeline error to the user
        return (severity_banner_html("yellow"), None,
                f'<div style="color:#b3261e">Could not analyze this file: {e}</div>',
                gr.update(choices=[], value=None), None)

    colors = finding_colors([f.label for f in res.report.findings])
    level = severity(res.report)
    banner = severity_banner_html(level, urgent_findings(res.report))
    fig = ecg_figure(res.clean_signal, res.saliency_by_code, res.fs, colors=colors)
    html = report_html(res.report, colors=colors)
    choices = list(res.saliency_by_code)
    state = {"clean": res.clean_signal, "saliency": res.saliency_by_code,
             "fs": res.fs, "colors": colors}
    return banner, fig, html, gr.update(choices=choices, value=None), state


def _highlight(code, state):
    """Re-draw the ECG emphasising one finding's saliency (others dimmed)."""
    if not state:
        return None
    return ecg_figure(state["clean"], state["saliency"], state["fs"],
                      highlight=code or None, colors=state["colors"])


def _examples() -> list[list[str]]:
    if not EXAMPLES_DIR.exists():
        return []
    return [[str(p)] for p in sorted(EXAMPLES_DIR.iterdir())
            if p.suffix.lower() in (".npy", ".csv", ".json", ".png", ".jpg", ".jpeg")]


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="APEX — Arrhythmia Pattern Explainer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# APEX — Arrhythmia Pattern Explainer\n"
                    "Upload a 12-lead ECG (signal file **or a photo of a paper ECG**) "
                    "for a grounded, explained reading.")
        gr.HTML(disclaimer_html())
        banner = gr.HTML(severity_banner_html("green"))

        with gr.Row():
            file_in = gr.File(label="ECG signal (.npy/.csv/.json) or paper-ECG image",
                              type="filepath", file_types=[".npy", ".csv", ".json",
                                                           ".png", ".jpg", ".jpeg"])
            analyze_btn = gr.Button("Analyze", variant="primary", scale=0)
        if _examples():
            gr.Examples(_examples(), inputs=file_in, label="Examples")

        with gr.Row():
            with gr.Column(scale=3):
                ecg_plot = gr.Plot(label="ECG + grounding")
                highlight = gr.Radio(choices=[], label="Highlight a finding's grounding",
                                     interactive=True)
            with gr.Column(scale=2):
                report = gr.HTML('<div style="color:#888">Upload an ECG to begin.</div>')

        state = gr.State()
        analyze_btn.click(_analyze, inputs=file_in,
                          outputs=[banner, ecg_plot, report, highlight, state])
        file_in.change(_analyze, inputs=file_in,
                       outputs=[banner, ecg_plot, report, highlight, state])
        highlight.change(_highlight, inputs=[highlight, state], outputs=ecg_plot)
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0")
