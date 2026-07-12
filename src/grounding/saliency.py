"""Grounding layer: map each predicted label back to the signal evidence.

Produces, per predicted label, a saliency map over (lead, time) so the explanation
layer can cite *where* in the ECG a finding is supported. Grad-CAM on the last
conv stage of the 1D-CNN is the v1 method; attention rollout will be added for the
transformer variant.

A finding is considered "grounded" if its saliency mass concentrates in at least
one contiguous lead+time region above a relative threshold — this feeds
`src/eval/hallucination.py`.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore


def grad_cam_1d(model, signal, target_label: int, conv_layer) -> np.ndarray:
    """Grad-CAM saliency over time for one label.

    Args:
        model: the trained ECGResNet1d.
        signal: (1, 12, T) input tensor.
        target_label: index into the 71-label space.
        conv_layer: the nn.Module whose activations/gradients to use (last conv).

    Returns:
        (T,) saliency vector normalized to [0, 1].
    """
    activations, gradients = {}, {}

    def fwd_hook(_m, _inp, out):
        activations["value"] = out.detach()

    def bwd_hook(_m, _gin, gout):
        gradients["value"] = gout[0].detach()

    h1 = conv_layer.register_forward_hook(fwd_hook)
    h2 = conv_layer.register_full_backward_hook(bwd_hook)
    try:
        logits = model(signal)
        model.zero_grad()
        logits[0, target_label].backward()
        acts = activations["value"][0]           # (C, T')
        grads = gradients["value"][0]            # (C, T')
        weights = grads.mean(dim=-1, keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=0))  # (T',)
        cam = cam.cpu().numpy()
    finally:
        h1.remove()
        h2.remove()

    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam


def is_grounded(saliency: np.ndarray, rel_threshold: float = 0.5, min_frac: float = 0.02) -> bool:
    """True if a meaningful fraction of the strip exceeds the relative threshold."""
    return float((saliency >= rel_threshold).mean()) >= min_frac
