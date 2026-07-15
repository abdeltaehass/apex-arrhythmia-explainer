"""1D CNN baseline for multi-label ECG classification.

A compact residual 1D-CNN over the 12-lead signal (ResNet-style, but 1D over time):
a wide stem, four residual stages that halve time / double channels, global average
pooling over the time axis, dropout, and a linear head that returns raw logits over the
71 SCP-ECG labels. Multiple conditions can coexist, so the head is multi-label — use
``BCEWithLogitsLoss`` (with per-label ``pos_weight``) on the logits, not softmax.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:  # keep importable before torch is installed
    torch = None  # type: ignore
    nn = object  # type: ignore

from src.config import NUM_LABELS, NUM_LEADS

if torch is not None:

    class ResBlock1d(nn.Module):
        """Two 7-wide conv layers + a projected skip connection."""

        def __init__(self, c_in: int, c_out: int, stride: int = 1, dropout: float = 0.0):
            super().__init__()
            self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=7, stride=stride, padding=3, bias=False)
            self.bn1 = nn.BatchNorm1d(c_out)
            self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=7, padding=3, bias=False)
            self.bn2 = nn.BatchNorm1d(c_out)
            self.act = nn.ReLU(inplace=True)
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.down = (
                nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out))
                if (stride != 1 or c_in != c_out)
                else nn.Identity()
            )

        def forward(self, x):
            residual = self.down(x)
            x = self.act(self.bn1(self.conv1(x)))
            x = self.drop(x)
            x = self.bn2(self.conv2(x))
            return self.act(x + residual)

    class ECGResNet1d(nn.Module):
        """Input:  (batch, 12 leads, time). Output: (batch, 71) logits.

        `width` sets the stem/stage-1 channel count (stages double it); `blocks` is the
        number of residual blocks per stage; `dropout` applies inside blocks and before
        the head.
        """

        def __init__(
            self,
            num_leads: int = NUM_LEADS,
            num_labels: int = NUM_LABELS,
            width: int = 64,
            blocks: int = 2,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(num_leads, width, kernel_size=15, stride=2, padding=7, bias=False),
                nn.BatchNorm1d(width),
                nn.ReLU(inplace=True),
            )
            channels = [width, width * 2, width * 4, width * 8]
            stages = []
            c_in = width
            for c_out in channels:
                # first block of each stage downsamples (stride 2); the rest keep shape.
                stages.append(ResBlock1d(c_in, c_out, stride=2, dropout=dropout))
                for _ in range(blocks - 1):
                    stages.append(ResBlock1d(c_out, c_out, stride=1, dropout=dropout))
                c_in = c_out
            self.stages = nn.Sequential(*stages)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(channels[-1], num_labels)

        def forward(self, x):
            x = self.stem(x)
            x = self.stages(x)
            x = self.pool(x).squeeze(-1)
            x = self.drop(x)
            return self.head(x)  # logits

    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
