"""Phase 25 — a conditional 1D denoising diffusion model for 12-lead ECG.

Generates ``(12, 1000)`` recordings conditioned on a multi-hot SCP label vector, so a rare
class can be sampled on demand. Diffusion rather than a GAN, for a reason specific to this
setting: GAN training on a few dozen examples of a class is dominated by mode collapse and
discriminator memorization, and the failure is silent — the samples look fine and carry no
information the training set did not already have. A denoising objective is a plain
regression loss with no adversarial dynamics to destabilize, which is the difference
between an experiment that measures augmentation and one that measures whether the GAN
happened to converge.

**One conditional model, not one model per class.** Training a separate generator on the 50
examples of a rare label is hopeless; a single model trained across the whole training set
learns what an ECG *is* from 17,000 recordings and uses the label vector only to steer.
That is what makes generation from a handful of conditioning examples plausible at all.

**The leakage trap this design has to avoid.** The generator must be trained on the *same
masked labels* the classifier sees. A generator trained on the full label matrix and then
used to augment a classifier that was deliberately starved of that class would be smuggling
the withheld annotations back in through the samples, and the resulting improvement would
measure nothing but the leak. :func:`train_diffusion` therefore takes the masked ``Y`` from
:mod:`src.synthesis.rarity`, and ``scripts/synth_ablation.py`` re-trains the generator per
seed rather than reusing one across scenarios.

Sampling uses DDIM, which lets 50 steps stand in for the 1000-step training chain — the
difference between minutes and hours per arm, at negligible sample-quality cost in this
regime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import NUM_LABELS, NUM_LEADS


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    base_channels: int = 64
    channel_mults: tuple[int, ...] = (1, 2, 4)
    time_dim: int = 128
    epochs: int = 30
    batch_size: int = 64
    lr: float = 2e-4
    # 200, not the usual 50. Measured: at 50 steps the samples come out under-denoised with
    # standard deviation 1.50 against real data's 1.00; 200 steps lands at 1.05. Cheap to
    # get wrong and invisible unless you check the marginal statistics.
    ddim_steps: int = 200
    # Classifier-free guidance: the label is dropped this often during training so the
    # model learns both conditional and unconditional scores, letting the conditioning be
    # amplified at sample time. Without it, conditioning on a label seen 50 times barely
    # moves the sample.
    cond_dropout: float = 0.15
    guidance: float = 3.0


def cosine_alphas(T: int) -> torch.Tensor:
    """Cosine noise schedule — gentler at the ends than linear, better for smooth signals."""
    s = 0.008
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = f / f[0]
    betas = torch.clip(1 - alphas_bar[1:] / alphas_bar[:-1], 0.0001, 0.999)
    return torch.cumprod(1.0 - betas, dim=0)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class Block(nn.Module):
    """Conv block with FiLM-style conditioning from the time+label embedding."""

    def __init__(self, c_in: int, c_out: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 5, padding=2)
        self.conv2 = nn.Conv1d(c_out, c_out, 5, padding=2)
        self.norm1 = nn.GroupNorm(min(8, c_out), c_out)
        self.norm2 = nn.GroupNorm(min(8, c_out), c_out)
        self.emb = nn.Linear(emb_dim, c_out * 2)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        scale, shift = self.emb(emb)[:, :, None].chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class UNet1D(nn.Module):
    """Small 1D UNet predicting the noise added to a 12-lead recording."""

    def __init__(self, cfg: DiffusionConfig, n_labels: int = NUM_LABELS,
                 n_leads: int = NUM_LEADS):
        super().__init__()
        c = cfg.base_channels
        emb = cfg.time_dim
        self.time = TimeEmbedding(emb)
        self.label = nn.Sequential(nn.Linear(n_labels, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.stem = nn.Conv1d(n_leads, c, 7, padding=3)

        chans = [c * m for m in cfg.channel_mults]
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = c
        for ch in chans:
            self.downs.append(Block(prev, ch, emb))
            self.pools.append(nn.Conv1d(ch, ch, 4, stride=2, padding=1))
            prev = ch
        self.mid = Block(prev, prev, emb)
        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for ch in reversed(chans):
            self.upsamples.append(nn.ConvTranspose1d(prev, ch, 4, stride=2, padding=1))
            self.ups.append(Block(ch * 2, ch, emb))
            prev = ch
        self.out = nn.Sequential(nn.GroupNorm(min(8, prev), prev), nn.SiLU(),
                                 nn.Conv1d(prev, n_leads, 7, padding=3))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emb = self.time(t) + self.label(y)
        h = self.stem(x)
        skips = []
        for block, pool in zip(self.downs, self.pools, strict=True):
            h = block(h, emb)
            skips.append(h)
            h = pool(h)
        h = self.mid(h, emb)
        for up, block, skip in zip(self.upsamples, self.ups, reversed(skips), strict=True):
            h = up(h)
            h = block(torch.cat([h, skip], dim=1), emb)
        return self.out(h)


def train_diffusion(X: np.ndarray, Y: np.ndarray, cfg: DiffusionConfig | None = None,
                    device: str = "cpu", seed: int = 0, verbose: bool = True) -> UNet1D:
    """Fit the model on ``(N, 12, T)`` signals with ``(N, L)`` **masked** labels.

    ``Y`` must be the masked matrix from :mod:`src.synthesis.rarity`. Passing the full
    labels here is the leak that would invalidate the whole experiment.
    """
    cfg = cfg or DiffusionConfig()
    torch.manual_seed(seed)
    model = UNet1D(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    alphas_bar = cosine_alphas(cfg.timesteps).to(device)

    xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(Y, dtype=np.float32))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xt, yt), batch_size=cfg.batch_size, shuffle=True,
        drop_last=True)

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            # Classifier-free guidance needs an unconditional branch to interpolate from.
            drop = (torch.rand(len(yb), 1, device=device) < cfg.cond_dropout).float()
            yb = yb * (1.0 - drop)
            t = torch.randint(0, cfg.timesteps, (len(xb),), device=device)
            noise = torch.randn_like(xb)
            a = alphas_bar[t][:, None, None]
            noisy = a.sqrt() * xb + (1 - a).sqrt() * noise
            loss = F.mse_loss(model(noisy, t, yb), noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(xb)
        if verbose:
            print(f"    diffusion epoch {epoch:2d}/{cfg.epochs}  loss={total / len(xt):.4f}",
                  flush=True)
    return model


@torch.no_grad()
def sample(model: UNet1D, labels: np.ndarray, cfg: DiffusionConfig | None = None,
           device: str = "cpu", seed: int = 0, length: int = 1000) -> np.ndarray:
    """DDIM-sample one recording per row of ``labels`` ``(N, L)``.

    Classifier-free guidance is applied at ``cfg.guidance``; with a label seen only a few
    dozen times the unguided conditional signal is too weak to steer the sample anywhere in
    particular.
    """
    cfg = cfg or DiffusionConfig()
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    y = torch.from_numpy(np.asarray(labels, dtype=np.float32)).to(device)
    n = len(y)
    x = torch.randn(n, NUM_LEADS, length, generator=g).to(device)

    alphas_bar = cosine_alphas(cfg.timesteps).to(device)
    steps = torch.linspace(cfg.timesteps - 1, 0, cfg.ddim_steps).long().to(device)
    zeros = torch.zeros_like(y)

    for i, t in enumerate(steps):
        tb = t.repeat(n)
        eps_cond = model(x, tb, y)
        if cfg.guidance and cfg.guidance != 1.0:
            eps_uncond = model(x, tb, zeros)
            eps = eps_uncond + cfg.guidance * (eps_cond - eps_uncond)
        else:
            eps = eps_cond
        a_t = alphas_bar[t]
        x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        x0 = x0.clamp(-6, 6)                       # z-scored input; beyond this is divergence
        if i + 1 < len(steps):
            a_prev = alphas_bar[steps[i + 1]]
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        else:
            x = x0
    return x.cpu().numpy().astype(np.float32)
