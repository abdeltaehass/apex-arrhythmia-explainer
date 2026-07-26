"""Phase 10 — ECG image digitization.

Turns a scanned/photographed paper ECG back into a ``(12, T)`` numeric signal that
feeds the Phase-2 preprocessing pipeline, and renders signals to paper-ECG images (the
source of paired data, and the demo's "upload a photo" input).

    from src.digitization import render_ecg, digitize_image, digitize_bytes

    img = render_ecg(signal_12xT, fs=100)          # signal -> paper ECG image
    recon = digitize_image(img)                    # paper ECG image -> (12, T) signal
"""

from src.digitization.digitize import digitize_bytes, digitize_image
from src.digitization.grid import GridInfo, detect_grid
from src.digitization.render import LEAD_NAMES, render_ecg

__all__ = [
    "render_ecg",
    "digitize_image",
    "digitize_bytes",
    "detect_grid",
    "GridInfo",
    "LEAD_NAMES",
]
