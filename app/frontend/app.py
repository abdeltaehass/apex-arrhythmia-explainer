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
    stream_events_html,
    stream_panel_html,
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


STREAM_EXAMPLES = {
    "9 — normal sinus rhythm": "9",
    "598 — atrial fibrillation": "598",
    "175 — inferior MI": "175",
    "9,598 — normal, then AF (rhythm change)": "9,598",
}


def _stream(record_spec: str, speed: float, duration: float, confirm_k: int):
    """Generator: replay a record and yield a live panel per hop.

    Yields ``(banner, panel, events, plot)``. Gradio streams each yield to the browser,
    so the findings panel updates as the stream advances.
    """
    import matplotlib.pyplot as plt

    from src.data.labels import load_database, load_scp_statements, present_codes
    from src.streaming import ReplaySource, StreamMonitor, build_playlist, load_record_signal

    ids = [int(x) for x in str(record_spec).split(",") if x.strip()]
    if not ids:
        yield severity_banner_html("green"), "<div>Pick a record.</div>", "", None
        return

    df, scp = load_database(), load_scp_statements()
    desc = {c: str(scp.loc[c, "description"]) for c in scp.index}
    signals, truth = [], set()
    for ecg_id in ids:
        sig, fs = load_record_signal(ecg_id)
        signals.append(sig)
        truth |= set(present_codes(df.loc[ecg_id, "scp_codes"]))
    signal = build_playlist(signals, fs=fs) if len(signals) > 1 else signals[0]

    mon = StreamMonitor(fs=fs, hop_s=1.0, confirm_k=int(confirm_k), confirm_m=5)
    mon._ensure_model()                      # pay the cold load before the clock starts
    src = ReplaySource(signal, fs=fs, chunk_s=0.25, speed=float(speed), loop=True,
                       max_duration_s=float(duration))
    log = []
    for chunk in src:
        u = mon.push(chunk.samples)
        if u is None:
            continue
        log.extend(u.events)
        level = "red" if u.alarm else ("yellow" if u.confirmed else "green")

        window = mon._buf.window(mon.window_samples)
        fig, ax = plt.subplots(figsize=(9, 2.2))
        t = [i / fs for i in range(window.shape[1])]
        ax.plot(t, window[1], lw=0.8, color="#b3261e" if u.alarm else "#1f3b5c")
        ax.set_xlabel("seconds (rolling window)")
        ax.set_ylabel("lead II")
        ax.set_title(f"stream t = {u.t_s:.0f}s")
        ax.margins(x=0)
        fig.tight_layout()

        yield (severity_banner_html(level, u.urgent_active),
               stream_panel_html(u, truth, desc),
               stream_events_html(log),
               fig)
        plt.close(fig)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="APEX — Arrhythmia Pattern Explainer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# APEX — Arrhythmia Pattern Explainer\n"
                    "Upload a 12-lead ECG (signal file **or a photo of a paper ECG**) "
                    "for a grounded, explained reading.")
        gr.HTML(disclaimer_html())

        with gr.Tabs():
            with gr.Tab("Single recording"):
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
                        highlight = gr.Radio(choices=[],
                                             label="Highlight a finding's grounding",
                                             interactive=True)
                    with gr.Column(scale=2):
                        report = gr.HTML('<div style="color:#888">Upload an ECG to begin.</div>')

                state = gr.State()
                analyze_btn.click(_analyze, inputs=file_in,
                                  outputs=[banner, ecg_plot, report, highlight, state])
                file_in.change(_analyze, inputs=file_in,
                               outputs=[banner, ecg_plot, report, highlight, state])
                highlight.change(_highlight, inputs=[highlight, state], outputs=ecg_plot)

            with gr.Tab("Live monitor"):
                gr.Markdown(
                    "Replays a PTB-XL recording as a **live 12-lead stream** (Holter / "
                    "telemetry style) into a rolling 10 s window, re-analyzed every second. "
                    "Findings are promoted to **confirmed** only once they persist across "
                    "windows — see `docs/streaming/report.md` for what that buys and what "
                    "it costs.\n\n"
                    "_The signal is replayed from a recording, not captured live. A "
                    "smartwatch delivers a single lead and cannot feed this 12-lead model._"
                )
                s_banner = gr.HTML(severity_banner_html("green"))
                with gr.Row():
                    rec = gr.Dropdown(choices=list(STREAM_EXAMPLES),
                                      value=list(STREAM_EXAMPLES)[0], label="Recording")
                    speed = gr.Slider(1, 20, value=8, step=1, label="Speed (x real time)")
                    dur = gr.Slider(20, 180, value=60, step=10, label="Stream seconds")
                    k = gr.Slider(1, 5, value=3, step=1,
                                  label="Confirm after k of 5 windows")
                    start = gr.Button("Start stream", variant="primary", scale=0)
                with gr.Row():
                    with gr.Column(scale=3):
                        s_plot = gr.Plot(label="Rolling window (lead II)")
                        s_events = gr.HTML("<div style='color:#999'>No events yet.</div>",
                                           label="Events")
                    with gr.Column(scale=2):
                        s_panel = gr.HTML(
                            '<div style="color:#888">Press <b>Start stream</b>.</div>')

                def _run_stream(name, sp, d, kk):
                    yield from _stream(STREAM_EXAMPLES.get(name, name), sp, d, kk)

                start.click(_run_stream, inputs=[rec, speed, dur, k],
                            outputs=[s_banner, s_panel, s_events, s_plot])
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0")
