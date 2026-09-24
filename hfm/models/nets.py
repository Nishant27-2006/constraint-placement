"""Backbones shared by every neural method in the benchmark.

Every learned baseline (DDPM, flow matching, projected flow matching, cVAE,
cGAN) uses the *same* ``TemporalTransformer`` with the same width/depth, so
differences in the results tables are attributable to the objective and the
constraint handling rather than to architecture search.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TemporalTransformer", "ResMLP", "timestep_embedding", "count_params", "flops_per_eval"]


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 1e4) -> torch.Tensor:
    """Sinusoidal embedding of a continuous time in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    args = t.reshape(-1, 1) * 1000.0 * freqs.reshape(1, -1)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, hidden: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * hidden)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s, b = self.to_scale_shift(c).chunk(2, dim=-1)
        return h * (1 + s.unsqueeze(1)) + b.unsqueeze(1)


class Block(nn.Module):
    def __init__(self, hidden: int, heads: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.f1 = FiLM(cond_dim, hidden)
        self.n2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden))
        self.f2 = FiLM(cond_dim, hidden)

    def forward(self, h, c):
        z = self.f1(self.n1(h), c)
        a, _ = self.attn(z, z, z, need_weights=False)
        h = h + a
        z = self.f2(self.n2(h), c)
        return h + self.mlp(z)


class TemporalTransformer(nn.Module):
    """Maps ``(B, T, D)`` states plus a scalar time and optional context to
    ``(B, T, D)`` outputs (a velocity, a noise prediction, or a mean)."""

    def __init__(self, d_in: int, T: int, hidden: int = 256, depth: int = 4,
                 heads: int = 4, cond_dim: int = 0, d_out: Optional[int] = None,
                 dropout: float = 0.0, time_dim: int = 128):
        super().__init__()
        d_out = d_in if d_out is None else d_out
        self.T, self.d_in, self.d_out = T, d_in, d_out
        self.inp = nn.Linear(d_in, hidden)
        self.pos = nn.Parameter(torch.randn(1, T, hidden) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden)) if cond_dim > 0 else None
        self.blocks = nn.ModuleList([Block(hidden, heads, hidden, dropout) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, d_out)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.hidden, self.depth, self.heads = hidden, depth, heads

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None):
        B = x.shape[0]
        if t.dim() == 0:
            t = t.expand(B)
        c = self.time_mlp(timestep_embedding(t.to(x.dtype), self.time_dim))
        if self.cond_mlp is not None and cond is not None:
            c = c + self.cond_mlp(cond.to(x.dtype))
        h = self.inp(x) + self.pos
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(self.out_norm(h))


class ResMLP(nn.Module):
    """Flat-vector alternative used for the low-dimensional reduced-coordinate
    baseline, where the temporal axis has been absorbed into the basis."""

    def __init__(self, d_in: int, hidden: int = 512, depth: int = 4, cond_dim: int = 0,
                 d_out: Optional[int] = None, time_dim: int = 128):
        super().__init__()
        d_out = d_in if d_out is None else d_out
        self.d_in, self.d_out, self.time_dim, self.cond_dim = d_in, d_out, time_dim, cond_dim
        self.inp = nn.Linear(d_in, hidden)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden)) if cond_dim > 0 else None
        self.layers = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                                   nn.SiLU(), nn.Linear(hidden, hidden))
                                     for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, d_out))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, cond=None):
        shp = x.shape
        xf = x.reshape(shp[0], -1)
        if t.dim() == 0:
            t = t.expand(shp[0])
        c = self.time_mlp(timestep_embedding(t.to(x.dtype), self.time_dim))
        if self.cond_mlp is not None and cond is not None:
            c = c + self.cond_mlp(cond.to(x.dtype))
        h = self.inp(xf) + c
        for lyr in self.layers:
            h = h + lyr(h)
        y = self.out(h)
        return y.reshape(shp[0], -1) if self.d_out != self.d_in else y.reshape(shp)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def flops_per_eval(model: nn.Module, x: torch.Tensor, cond=None) -> int:
    """Multiply-accumulate count for one forward pass, counted analytically.

    We count 2 FLOPs per MAC for linear layers and attention matmuls.  This is
    hardware-independent and exactly reproducible, unlike wall-clock or
    device-specific joules, and it is the primary efficiency axis in the paper.
    """
    B, T = x.shape[0], x.shape[1]
    total = 0
    if isinstance(model, TemporalTransformer):
        H, L = model.hidden, model.depth
        total += 2 * B * T * model.d_in * H                      # input proj
        for _ in range(L):
            total += 2 * B * T * H * (3 * H)                     # qkv
            total += 2 * B * T * T * H * 2                       # attn scores + apply
            total += 2 * B * T * H * H                           # attn out proj
            total += 2 * B * T * H * (4 * H) * 2                 # mlp
            total += 2 * B * H * (2 * H) * 2                     # two FiLMs
        total += 2 * B * T * H * model.d_out
        total += 2 * B * (model.time_dim * H + H * H)
    else:
        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                total += 2 * B * mod.in_features * mod.out_features
    return int(total)
