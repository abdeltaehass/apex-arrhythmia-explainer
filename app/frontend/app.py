"""APEX Gradio frontend (Phase-0 skeleton).

Lets a clinician upload/select an ECG, calls the backend, and renders findings +
explanation with the review flag prominent. Wire BACKEND_URL to the FastAPI service.
"""

from __future__ import annotations

import os

BACKEND_URL = os.environ.get("APEX_BACKEND_URL", "http://localhost:8000")


def build_demo():
    import gradio as gr

    def analyze(_file):
        # TODO(phase2+): read the uploaded record, POST to {BACKEND_URL}/analyze.
        return "Backend not wired yet (Phase 0 skeleton).", "⚠️ Needs manual review"

    with gr.Blocks(title="APEX — Arrhythmia Pattern Explainer") as demo:
        gr.Markdown("# APEX\nDecision support for 12-lead ECG. **Verify before acting.**")
        inp = gr.File(label="12-lead ECG record")
        btn = gr.Button("Analyze")
        out_text = gr.Textbox(label="Explanation")
        out_flag = gr.Textbox(label="Review status")
        btn.click(analyze, inputs=inp, outputs=[out_text, out_flag])
    return demo


if __name__ == "__main__":
    build_demo().launch()
