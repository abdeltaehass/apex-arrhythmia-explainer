"""Make a clean rendered ECG look photographed — for robustness eval and realistic demos.

Real uploads are phone photos of paper: rotated a little, unevenly lit, blurred, and
JPEG-compressed. These transforms approximate that so the digitizer can be evaluated
against something harder than a pristine render (see `scripts/eval_digitization.py`).
They are *not* a substitute for a real-world photo dataset — perspective/skew and heavy
shadows are out of scope for the classical pipeline.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter


def add_noise(image: Image.Image, sigma: float = 8.0, rng=None) -> Image.Image:
    rng = rng or np.random.default_rng()
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    arr += rng.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def adjust_brightness(image: Image.Image, factor: float = 0.9) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.float32) * factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def blur(image: Image.Image, radius: float = 0.8) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius))


def rotate(image: Image.Image, degrees: float = 1.5) -> Image.Image:
    return image.rotate(degrees, resample=Image.BICUBIC, fillcolor=(255, 255, 255), expand=False)


def jpeg(image: Image.Image, quality: int = 70) -> Image.Image:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)


def photograph(image: Image.Image, level: float = 1.0, rng=None) -> Image.Image:
    """Apply the full stack (blur -> brightness -> noise -> JPEG), scaled by ``level``.

    ``level=0`` returns the image unchanged; ``level=1`` is a moderate phone-photo look.
    ``rotate`` is left out of the default stack because it shifts the plotting rectangle
    the digitizer assumes; apply it separately to test skew tolerance.
    """
    rng = rng or np.random.default_rng()
    if level <= 0:
        return image
    out = blur(image, radius=0.6 * level)
    out = adjust_brightness(out, factor=1.0 - 0.12 * level)
    out = add_noise(out, sigma=8.0 * level, rng=rng)
    return jpeg(out, quality=int(85 - 20 * level))
