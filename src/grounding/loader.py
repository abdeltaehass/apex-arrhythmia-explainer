"""Reconstruct a trained detector from a checkpoint for grounding.

Training (`src/detection/train.py`) saves ``{"model": state_dict, "args": vars(args),
"epoch": ...}``. The ``args`` dict carries everything needed to rebuild the exact
architecture (``model`` name + width/blocks/dropout for the CNN, patch/d_model/depth/
heads for the transformer), so grounding can load a checkpoint without the caller
having to remember how it was trained.
"""

from __future__ import annotations

from pathlib import Path

from src.config import ROOT
from src.data.labels import build_label_space
from src.detection.model import build_model

try:
    import torch
    import torch.nn as nn
except ImportError:  # keep importable before torch is installed
    torch = None  # type: ignore
    nn = object  # type: ignore

DEFAULT_CHECKPOINT = ROOT / "outputs" / "final_best.pt"


def load_detector(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    device: str = "cpu",
) -> tuple[nn.Module, list[str], dict]:
    """Load a detector checkpoint -> ``(model, label_space, args)``.

    The model is returned in ``eval`` mode on ``device``. ``label_space`` is the
    canonical 71-code SCP order the head's columns correspond to. Grad-CAM needs
    gradients, so callers should *not* wrap forward passes in ``torch.no_grad()``.
    """
    if torch is None:
        raise RuntimeError("torch is required to load the detector")
    ckpt = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    args = ckpt.get("args", {}) or {}
    # Phase-3 baseline checkpoints predate the --model flag; they are the CNN.
    name = args.get("model") or "cnn"
    model = build_model(
        name,
        width=args.get("width", 64),
        blocks=args.get("blocks", 2),
        dropout=args.get("dropout", 0.2),
        d_model=args.get("d_model", 128),
        depth=args.get("depth", 3),
        heads=args.get("heads", 4),
        patch=args.get("patch", 25),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, build_label_space(), args


def default_cam_target(model: nn.Module) -> nn.Module:
    """The activation map Grad-CAM should hook for ``ECGResNet1d``.

    The output of the final residual stage (``model.stages``) is the last, most
    class-specific ``(B, C, T')`` feature map before global pooling — the standard
    Grad-CAM target. Raises for architectures without a conv feature stack (e.g. the
    transformer), which need attention-based grounding instead.
    """
    stages = getattr(model, "stages", None)
    if stages is None:
        raise ValueError(
            f"{type(model).__name__} has no conv `stages`; Grad-CAM needs a "
            "convolutional feature map. Use input-gradient or attention grounding."
        )
    return stages
