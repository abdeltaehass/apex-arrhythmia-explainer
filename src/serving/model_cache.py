"""Process-level caches for the heavy, reusable pieces of the pipeline.

`analyze_signal` (and the API) must not reload the checkpoint from disk or re-read
`scp_statements.csv` on every request. These helpers memoize the detector (keyed by
checkpoint path + device) and the SCP description table for the process lifetime, so the
first call pays the load cost and the rest are warm.
"""

from __future__ import annotations

from src.grounding import load_detector

_DETECTORS: dict[tuple[str, str], tuple] = {}
_SCP = None


def get_detector(checkpoint=None, device: str = "cpu"):
    """Cached ``(model, label_space, args)`` for ``(checkpoint, device)``.

    ``checkpoint=None`` resolves to the default (`grounding.loader.DEFAULT_CHECKPOINT`).
    The model is returned in eval mode; grounding needs gradients, so callers must not
    wrap forward passes in ``torch.no_grad()`` when they intend to run Grad-CAM.
    """
    key = (str(checkpoint) if checkpoint else "__default__", device)
    if key not in _DETECTORS:
        _DETECTORS[key] = (
            load_detector(checkpoint, device=device) if checkpoint
            else load_detector(device=device)
        )
    return _DETECTORS[key]


def get_scp_statements():
    """Cached ``scp_statements.csv`` table (code -> description/superclass)."""
    global _SCP
    if _SCP is None:
        from src.data.labels import load_scp_statements

        _SCP = load_scp_statements()
    return _SCP


def warmup(checkpoint=None, device: str = "cpu") -> None:
    """Eagerly populate both caches (e.g. at API startup, so the first request is warm)."""
    get_detector(checkpoint, device)
    get_scp_statements()


def clear_caches() -> None:
    """Drop the caches (tests / explicit reloads)."""
    global _SCP
    _DETECTORS.clear()
    _SCP = None
