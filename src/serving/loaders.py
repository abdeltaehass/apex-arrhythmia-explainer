"""Parse an uploaded recording into a ``(12, T)`` signal array.

``POST /analyze`` accepts a signal *file* (or a JSON body). This module turns the raw
upload bytes into the ``(leads, samples)`` array the pipeline expects, inferring lead
orientation. Supported: ``.npy``, ``.csv``/``.txt`` (delimited), ``.json``
(``{"signal": [[...]], "sampling_rate": N}`` or a bare 2-D array).

ECG *image* uploads are recognized and politely refused — turning a scanned/photographed
ECG back into a signal (image digitization) is a separate problem this project does not
implement, and silently faking it would be worse than a clear error.
"""

from __future__ import annotations

import io
import json

import numpy as np

from src.config import NUM_LEADS

SIGNAL_EXTS = (".npy", ".csv", ".txt", ".json")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp")


class UnsupportedUploadError(ValueError):
    """The uploaded file type can't be turned into a signal (e.g. an ECG image)."""


def _ext(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _orient(arr: np.ndarray) -> np.ndarray:
    """Return the array as ``(leads, samples)``.

    Uses the 12-lead convention to disambiguate: if exactly one axis has length 12 it is
    the lead axis. If neither does, the array is returned first-axis-as-leads and left
    for `validate_signal` to reject with a clear message.
    """
    if arr.ndim != 2:
        raise UnsupportedUploadError(f"expected a 2-D signal, got {arr.ndim}-D array")
    rows, cols = arr.shape
    if cols == NUM_LEADS and rows != NUM_LEADS:
        return arr.T
    return arr


def _from_json(content: bytes) -> tuple[np.ndarray, int | None]:
    obj = json.loads(content)
    if isinstance(obj, dict):
        signal = obj.get("signal")
        sr = obj.get("sampling_rate")
    else:
        signal, sr = obj, None
    if signal is None:
        raise UnsupportedUploadError("JSON upload has no 'signal' field")
    return _orient(np.asarray(signal, dtype=np.float32)), sr


def _from_delimited(content: bytes) -> np.ndarray:
    text = content.decode("utf-8", errors="replace").strip()
    delimiter = "," if "," in text.splitlines()[0] else None  # comma else whitespace
    arr = np.loadtxt(io.StringIO(text), delimiter=delimiter, dtype=np.float32)
    return _orient(np.atleast_2d(arr))


def parse_signal_upload(filename: str, content: bytes) -> tuple[np.ndarray, int | None]:
    """``(filename, bytes)`` -> ``((12, T) float32 array, sampling_rate_or_None)``.

    ``sampling_rate`` is only recoverable from a JSON upload; for ``.npy``/``.csv`` the
    caller must supply it (via the form field / default). Raises
    :class:`UnsupportedUploadError` for image uploads and unknown extensions.
    """
    ext = _ext(filename)
    if ext in IMAGE_EXTS:
        raise UnsupportedUploadError(
            f"ECG image uploads ({ext}) are not supported — image digitization "
            "(image -> signal) is out of scope. Upload the sampled signal (.npy/.csv/.json)."
        )
    if ext == ".npy":
        return _orient(np.load(io.BytesIO(content))).astype(np.float32), None
    if ext == ".json":
        return _from_json(content)
    if ext in (".csv", ".txt"):
        return _from_delimited(content), None
    raise UnsupportedUploadError(
        f"unsupported file type {ext or '(none)'}; expected one of {SIGNAL_EXTS} or a JSON body"
    )
