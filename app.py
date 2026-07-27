"""Hugging Face Spaces entry point.

HF Spaces (Gradio SDK) runs the file named in the Space README's ``app_file`` — by
default ``app.py`` at the repo root, which is this shim. It just launches the real
dashboard defined in `app/frontend/app.py`. Locally, ``python app.py`` (or ``make ui``)
does the same. See `docs/frontend/deploy.md` for deployment.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.frontend.app import build_demo  # noqa: E402

if __name__ == "__main__":
    build_demo().launch()
