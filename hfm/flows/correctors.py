"""Inference-time constraint correctors, used as baselines.

``PCFMCorrector`` is our reimplementation of the scheme described by Utkarsh
et al., *Physics-Constrained Flow Matching: Sampling Generative Models with Hard
Constraints* (NeurIPS 2025).  It is a **zero-shot** method: it takes an already
trained, unconstrained flow model and steers sampling so that the terminal state
satisfies the constraints exactly.  At each integrator step it

  1. forms the one-step endpoint prediction  ``x1_hat = x_t + (1 - t) v(x_t, t)``,
  2. projects ``x1_hat`` onto the constraint set (a single linear solve for an
     affine set; damped Gauss-Newton for a nonlinear manifold),
  3. re-anchors the state by interpolating the corrected endpoint back to the
     next time, ``x_{t+h} = x_t + h (x1_hat_proj - x_t) / (1 - t)``.

We report it because it is the closest published competitor, and because the
comparison we care about is not "who satisfies constraints" -- both do, exactly
-- but what each costs and what each does to distributional fidelity.

Any deviation from the original is ours, not the authors'; we label the row as a
reimplementation everywhere it appears.
"""

from __future__ import annotations

import torch

__all__ = ["PCFMCorrector", "sample_with_corrector"]


class PCFMCorrector:
    def __init__(self, affine=None, manifold=None, gn_iters: int = 2):
        self.affine = affine
        self.manifold = manifold
        self.gn_iters = gn_iters
        self.extra_solves = 0

    def project_endpoint(self, x1: torch.Tensor) -> torch.Tensor:
        self.extra_solves += 1
        if self.affine is not None:
            return self.affine.project_point(x1)
        if self.manifold is not None:
            return self.manifold.retract(x1, iters=self.gn_iters)
        return x1


@torch.no_grad()
def sample_with_corrector(model, x0, cond=None, steps: int = 50,
                          corrector: PCFMCorrector | None = None,
                          final_project: bool = True):
    """Explicit-Euler sampling with per-step endpoint correction."""
    model.counter.reset()
    if corrector is not None:
        corrector.extra_solves = 0
    x = x0
    dev, dt = x0.device, x0.dtype
    ts = torch.linspace(0, 1, steps + 1, device=dev, dtype=dt)
    for i in range(steps):
        t, h = ts[i], ts[i + 1] - ts[i]
        tb = torch.full((x.shape[0],), float(t), device=dev, dtype=dt)
        v = model(x, tb, cond)
        rem = float(1.0 - t)
        if corrector is not None and rem > 1e-6:
            x1_hat = x + rem * v
            x1_hat = corrector.project_endpoint(x1_hat)
            x = x + float(h) * (x1_hat - x) / rem
        else:
            x = x + float(h) * v
    if corrector is not None and final_project:
        x = corrector.project_endpoint(x)
    return x, model.counter.n, (corrector.extra_solves if corrector else 0)
