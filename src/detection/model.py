"""1D CNN baseline for multi-label ECG classification.

A compact residual 1D-CNN over the 12-lead signal. This is the Phase-1 baseline;
a transformer variant will live alongside it and share the dataset/train loop. The
head returns raw logits over the 71 SCP-ECG labels — use BCEWithLogitsLoss.
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
        def __init__(self, c_in: int, c_out: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=7, stride=stride, padding=3)
            self.bn1 = nn.BatchNorm1d(c_out)
            self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=7, padding=3)
            self.bn2 = nn.BatchNorm1d(c_out)
            self.act = nn.ReLU(inplace=True)
            self.down = (
                nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride), nn.BatchNorm1d(c_out))
                if (stride != 1 or c_in != c_out)
                else nn.Identity()
            )

        def forward(self, x):
            residual = self.down(x)
            x = self.act(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return self.act(x + residual)

    class ECGResNet1d(nn.Module):
        """Input:  (batch, 12 leads, time). Output: (batch, 71) logits."""

        def __init__(self, num_leads: int = NUM_LEADS, num_labels: int = NUM_LABELS, width: int = 64):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(num_leads, width, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(width),
                nn.ReLU(inplace=True),
            )
            self.stage1 = ResBlock1d(width, width, stride=2)
            self.stage2 = ResBlock1d(width, width * 2, stride=2)
            self.stage3 = ResBlock1d(width * 2, width * 4, stride=2)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(width * 4, num_labels)

        def forward(self, x):
            x = self.stem(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.pool(x).squeeze(-1)
            return self.head(x)  # logits
