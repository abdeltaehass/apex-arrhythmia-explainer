"""Grounding layer: map each predicted label back to the signal evidence.

This is the Phase-5 "Grad-CAM equivalent for 1D signals". Given a recording and a
predicted label it returns a **per-lead saliency trace** — a ``(12, T)`` array, aligned
sample-for-sample with the raw ECG, whose magnitude says how much each lead at each
instant drove that one prediction. The explanation layer cites *where* a finding is
supported; `src/eval/hallucination.py` flags findings whose saliency lands nowhere.

Why not plain Grad-CAM alone? The CNN's stem (`Conv1d(12 -> width)`) mixes all 12
leads at the very first layer, so every downstream activation is lead-agnostic. Grad-CAM
on the last conv stage therefore yields a **time-only** importance map ``(T',)`` — it
localizes *when* the model looked but not *which lead*. To recover the lead axis we
combine that class-discriminative temporal envelope with the magnitude of the gradient
at the *input* (which is naturally ``(12, T)``). This "guided Grad-CAM" is the default:

    per_lead[l, t] = |d logit_c / d x[l, t]|  ·  cam_c(t)

- ``cam_c(t)``  — Grad-CAM over the last conv stage, upsampled to raw time, in [0, 1].
  Robustly class-discriminative and smooth; answers *when*.
- input gradient — answers *which lead*, but is noisy on its own; the CAM envelope
  gates it to the class-relevant window.

Methods (``method=``):
  - ``"guided_gradcam"`` (default): the product above — per-lead *and* class-specific.
  - ``"input_grad"``: input-gradient magnitude only (per-lead, less class-localized).
  - ``"gradcam"``: the temporal CAM broadcast across all leads (no lead resolution).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # keep importable before torch is installed
    torch = None  # type: ignore
    F = None  # type: ignore

METHODS = ("guided_gradcam", "input_grad", "gradcam")


@dataclass
class LeadSaliency:
    """Per-lead grounding for a single (recording, label) pair.

    Attributes:
        label_index: column into the 71-label space.
        label: SCP code string (may be empty if not resolved by the caller).
        method: which saliency method produced this (see ``METHODS``).
        logit / prob: the model's raw logit and sigmoid probability for the label.
        per_lead: ``(12, T)`` saliency, globally normalized to [0, 1] so lead
            magnitudes are comparable — a quiet lead reads low, the driving lead high.
        temporal: ``(T,)`` class-discriminative Grad-CAM envelope in [0, 1] (the
            "when", lead-agnostic). Equals ``per_lead`` collapsed over leads for the
            ``gradcam`` method.
        lead_importance: ``(12,)`` fraction of total saliency mass per lead (sums to 1);
            ``argmax`` is the single most influential lead.
        fs: sampling rate of the trace (Hz), so time windows map back to seconds.
    """

    label_index: int
    label: str
    method: str
    logit: float
    prob: float
    per_lead: np.ndarray
    temporal: np.ndarray
    lead_importance: np.ndarray
    fs: int

    @property
    def top_lead(self) -> int:
        return int(np.argmax(self.lead_importance))

    def lead_ranking(self, lead_names: list[str] | None = None) -> list[tuple]:
        """Leads sorted by importance, as ``(name_or_index, fraction)`` pairs."""
        order = np.argsort(self.lead_importance)[::-1]
        names = lead_names or list(range(len(self.lead_importance)))
        return [(names[i], float(self.lead_importance[i])) for i in order]


# PTB-XL lead order (I, II, III, aVR, aVL, aVF, V1..V6).
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _as_input_tensor(signal, device):
    """Coerce a ``(12, T)`` array / ``(1, 12, T)`` tensor to a leaf input tensor."""
    if isinstance(signal, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(signal, dtype=np.float32))
    else:
        x = signal.detach().clone().float()
    if x.dim() == 2:
        x = x.unsqueeze(0)  # (1, 12, T)
    if x.dim() != 3 or x.size(0) != 1:
        raise ValueError(f"expected (12, T) or (1, 12, T), got {tuple(x.shape)}")
    return x.to(device).requires_grad_(True)


def _smooth(x: np.ndarray, fs: int, smooth_s: float) -> np.ndarray:
    """Light moving-average along the last axis; a no-op when ``smooth_s <= 0``."""
    w = int(round(smooth_s * fs))
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    pad = w // 2
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="edge")
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid")[: x.shape[-1]], -1, padded)


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def lead_saliency(
    model,
    signal,
    target_label: int,
    cam_layer=None,
    method: str = "guided_gradcam",
    smooth_s: float = 0.02,
    fs: int = 100,
    label: str = "",
) -> LeadSaliency:
    """Per-lead saliency trace for one (recording, label). Core Phase-5 entrypoint.

    Args:
        model: a trained detector in eval mode (grad enabled — do **not** use no_grad).
        signal: ``(12, T)`` preprocessed input (the tensor the model was trained on),
            as a numpy array or torch tensor. A batch dim of 1 is added if absent.
        target_label: column index into the 71-label space to explain.
        cam_layer: the conv module to hook for Grad-CAM (defaults to the CNN's last
            stage via ``loader.default_cam_target``). Unused for ``method="input_grad"``.
        method: one of ``METHODS``.
        smooth_s: moving-average width (seconds) applied to the input-gradient magnitude
            to tame single-sample noise; the CAM envelope is already smooth. 0 disables.
        fs: sampling rate of ``signal`` (Hz), stored on the result for time alignment.
        label: optional SCP code string, carried through onto the result.

    Returns:
        A :class:`LeadSaliency` with a ``(12, T)`` per-lead trace and a ``(T,)`` temporal
        envelope, both aligned to the input samples.
    """
    if torch is None:
        raise RuntimeError("torch is required for saliency")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    device = next(model.parameters()).device
    x = _as_input_tensor(signal, device)
    T = x.shape[-1]

    acts: dict = {}
    grads: dict = {}
    handles = []
    need_cam = method in ("guided_gradcam", "gradcam")
    if need_cam:
        if cam_layer is None:
            from src.grounding.loader import default_cam_target

            cam_layer = default_cam_target(model)
        handles.append(cam_layer.register_forward_hook(lambda _m, _i, o: acts.__setitem__("v", o)))
        handles.append(
            cam_layer.register_full_backward_hook(lambda _m, _gi, go: grads.__setitem__("v", go[0].detach()))
        )

    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logit = logits[0, target_label]
        logit.backward()

        input_grad = x.grad[0].detach().cpu().numpy()  # (12, T)

        if need_cam:
            a = acts["v"][0].detach()          # (C, T')
            g = grads["v"][0]                  # (C, T')
            weights = g.mean(dim=-1, keepdim=True)             # GAP over time -> (C, 1)
            cam = torch.relu((weights * a).sum(dim=0))         # (T',)
            cam = cam.unsqueeze(0).unsqueeze(0)                # (1, 1, T')
            cam = F.interpolate(cam, size=T, mode="linear", align_corners=False)
            cam = cam[0, 0].cpu().numpy()                      # (T,)
            cam = _norm01(cam)
        else:
            cam = None
    finally:
        for h in handles:
            h.remove()

    ig = _smooth(np.abs(input_grad), fs, smooth_s)  # (12, T)

    if method == "gradcam":
        per_lead = np.repeat(cam[None, :], ig.shape[0], axis=0)  # broadcast, no lead info
        temporal = cam
    elif method == "input_grad":
        per_lead = _norm01(ig)
        temporal = _norm01(ig.sum(axis=0))
    else:  # guided_gradcam
        per_lead = _norm01(ig * cam[None, :])
        temporal = cam

    mass = per_lead.sum(axis=1)
    lead_importance = mass / mass.sum() if mass.sum() > 0 else np.full_like(mass, 1.0 / len(mass))

    with torch.no_grad():
        prob = float(torch.sigmoid(logits[0, target_label]).cpu())
    return LeadSaliency(
        label_index=int(target_label),
        label=label,
        method=method,
        logit=float(logit.detach().cpu()),
        prob=prob,
        per_lead=per_lead.astype(np.float32),
        temporal=temporal.astype(np.float32),
        lead_importance=lead_importance.astype(np.float32),
        fs=fs,
    )


def ground(
    model,
    signal,
    labels,
    label_space: list[str] | None = None,
    fs: int = 100,
    **kwargs,
) -> dict[str, LeadSaliency]:
    """Ground one or more predicted labels for a recording.

    ``labels`` may be label indices or SCP code strings (resolved via ``label_space``).
    Returns a dict keyed by SCP code (or ``str(index)`` when no label space is given).
    This is the module's public "recording + predicted label -> per-lead trace" API.
    """
    if isinstance(labels, (int, str)):
        labels = [labels]
    idx_of = {c: i for i, c in enumerate(label_space)} if label_space else {}
    out: dict[str, LeadSaliency] = {}
    for lab in labels:
        if isinstance(lab, str):
            if lab not in idx_of:
                raise KeyError(f"label {lab!r} not in label space")
            i, code = idx_of[lab], lab
        else:
            i = int(lab)
            code = label_space[i] if label_space else str(i)
        out[code] = lead_saliency(model, signal, i, fs=fs, label=code, **kwargs)
    return out


# --- Backward-compatible helpers -------------------------------------------

def grad_cam_1d(model, signal, target_label: int, conv_layer) -> np.ndarray:
    """Temporal Grad-CAM over time for one label -> ``(T',)`` in [0, 1].

    The original Phase-5 stub API, kept for callers that only want the lead-agnostic
    "when" map at conv resolution. New code should prefer :func:`lead_saliency`.
    """
    if torch is None:
        raise RuntimeError("torch is required for saliency")
    activations, gradients = {}, {}
    h1 = conv_layer.register_forward_hook(lambda _m, _i, o: activations.__setitem__("v", o.detach()))
    h2 = conv_layer.register_full_backward_hook(
        lambda _m, _gi, go: gradients.__setitem__("v", go[0].detach())
    )
    x = signal if hasattr(signal, "dim") else torch.from_numpy(np.asarray(signal, dtype=np.float32))
    if x.dim() == 2:
        x = x.unsqueeze(0)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, target_label].backward()
        acts = activations["v"][0]
        grads = gradients["v"][0]
        weights = grads.mean(dim=-1, keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=0)).cpu().numpy()
    finally:
        h1.remove()
        h2.remove()
    return _norm01(cam)


def is_grounded(saliency: np.ndarray, rel_threshold: float = 0.5, min_frac: float = 0.02) -> bool:
    """True if a meaningful fraction of the strip exceeds the relative threshold.

    Accepts either a ``(T,)`` temporal trace or a ``(12, T)`` per-lead trace; for the
    latter it collapses to the per-time max across leads before thresholding.
    """
    s = np.asarray(saliency)
    if s.ndim == 2:
        s = s.max(axis=0)
    return float((s >= rel_threshold).mean()) >= min_frac
