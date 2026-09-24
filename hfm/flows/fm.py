"""Flow matching with exact constraint projection, and ODE samplers that count
every network evaluation.

Notation follows Lipman et al. (2023) / Tong et al. (2024) conditional flow
matching: with ``x_t = (1-t) x_0 + t x_1`` the conditional target is the constant
``u = x_1 - x_0``, and the regression optimum is ``v*(x,t) = E[x_1 - x_0 | x_t = x]``.

The one change that carries the whole method: if the data *and* the base
distribution are supported on an affine set ``A = {x : A x = b}`` with direction
space ``V = null(A)``, then

  (i)   ``x_t in A`` for all t (convexity),
  (ii)  ``u = x_1 - x_0 in V``,
  (iii) ``v*(x,t) = E[u | x_t] in V`` because ``V`` is a linear subspace and
        conditional expectation is linear,

so restricting the hypothesis class to ``V``-valued fields is *bias-free*, and by
Pythagoras it can only lower the flow-matching loss.  Integrating a ``V``-valued
field from ``x_0 in A`` with any Runge-Kutta scheme keeps every iterate in ``A``
up to floating-point round-off, because every increment is a linear combination
of vectors in ``V``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn

__all__ = ["NFECounter", "ProjectedVelocity", "cfm_loss", "sample_ode", "SOLVERS"]


class NFECounter:
    """Exact count of velocity-network evaluations during a solve."""

    def __init__(self):
        self.n = 0

    def reset(self):
        self.n = 0
        return self

    def tick(self, k: int = 1):
        self.n += k


class ProjectedVelocity(nn.Module):
    """Wraps a raw velocity network with the constraint machinery.

    ``mode``:
      ``"none"``       -- plain flow matching (baseline).
      ``"affine"``     -- Level 1: tangential projection onto ``null(A)``.
      ``"manifold"``   -- Level 2: state-dependent tangential projection onto
                          ``T_x M`` for ``M = {g(x) = 0}``.
    ``corrector`` adds the Level-3 dissipative term ``-gamma(t) P grad E(x)``
    driving operational-limit violations down along the generative path.
    """

    def __init__(self, net: nn.Module, affine=None, manifold=None,
                 mode: str = "affine", corrector=None, gamma_max: float = 0.0,
                 counter: Optional[NFECounter] = None):
        super().__init__()
        self.net = net
        self.affine = affine
        self.manifold = manifold
        self.mode = mode
        self.corrector = corrector
        self.gamma_max = gamma_max
        self.counter = counter or NFECounter()

    def gamma(self, t: float) -> float:
        """Corrector strength: zero early, ramping in as t -> 1 so that the
        dissipative term shapes only the terminal distribution."""
        if self.gamma_max <= 0:
            return 0.0
        return self.gamma_max * float(t) ** 2

    def raw(self, x: torch.Tensor, t: torch.Tensor, cond=None) -> torch.Tensor:
        self.counter.tick()
        return self.net(x, t, cond)

    def project(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "affine" and self.affine is not None:
            return self.affine.project_tangent(v)
        if self.mode == "manifold" and self.manifold is not None:
            return self.manifold.project_tangent(v, x)
        return v

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond=None) -> torch.Tensor:
        v = self.project(self.raw(x, t, cond), x)
        if self.corrector is not None and self.gamma_max > 0:
            tt = float(t.reshape(-1)[0])
            g = self.gamma(tt)
            if g > 0:
                with torch.enable_grad():
                    xg = x.detach().requires_grad_(True)
                    E = self.corrector.energy(xg).sum()
                    (grad,) = torch.autograd.grad(E, xg)
                v = v - g * self.project(grad, x)
        return v


def cfm_loss(model: ProjectedVelocity, x1: torch.Tensor, x0: torch.Tensor,
             cond=None, t: Optional[torch.Tensor] = None,
             project_target: bool = True) -> torch.Tensor:
    """Conditional flow-matching loss on the straight-line interpolant."""
    B = x1.shape[0]
    if t is None:
        t = torch.rand(B, device=x1.device, dtype=x1.dtype)
    tb = t.view(B, *([1] * (x1.dim() - 1)))
    xt = (1 - tb) * x0 + tb * x1
    u = x1 - x0
    if project_target:
        u = model.project(u, xt)
    v = model(xt, t, cond)
    return (v - u).pow(2).mean()


# --------------------------------------------------------------------- solvers
def _euler(f, x, ts, on_step=None, **kw):
    for i in range(len(ts) - 1):
        t, h = ts[i], ts[i + 1] - ts[i]
        x = x + h * f(x, t)
        if on_step is not None:
            x = on_step(x, i + 1)
    return x, len(ts) - 1


def _heun(f, x, ts, on_step=None, **kw):
    for i in range(len(ts) - 1):
        t, h = ts[i], ts[i + 1] - ts[i]
        k1 = f(x, t)
        k2 = f(x + h * k1, ts[i + 1])
        x = x + 0.5 * h * (k1 + k2)
        if on_step is not None:
            x = on_step(x, i + 1)
    return x, 2 * (len(ts) - 1)


def _rk4(f, x, ts, on_step=None, **kw):
    for i in range(len(ts) - 1):
        t, h = ts[i], ts[i + 1] - ts[i]
        k1 = f(x, t)
        k2 = f(x + 0.5 * h * k1, t + 0.5 * h)
        k3 = f(x + 0.5 * h * k2, t + 0.5 * h)
        k4 = f(x + h * k3, t + h)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if on_step is not None:
            x = on_step(x, i + 1)
    return x, 4 * (len(ts) - 1)


# Dormand-Prince 5(4) tableau
_DP_C = [0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0]
_DP_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_DP_B5 = [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0]
_DP_B4 = [5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]


def _dopri5(f, x, ts, rtol=1e-3, atol=1e-4, h_init=0.1, max_steps=2000,
             on_step=None, **kw):
    """Adaptive Dormand-Prince 5(4) with PI step control.

    This is the *principled* version of the draft's "curvature scheduler": the
    embedded 4th-order estimate gives a local error proxy, so the step count
    adapts to how curved the learned field actually is, with a user-set accuracy
    budget rather than a hand-tuned threshold.
    """
    t0, t1 = float(ts[0]), float(ts[-1])
    t, h, nfe, steps = t0, min(h_init, t1 - t0), 0, 0
    k7, accepted = None, 0
    while t < t1 - 1e-12 and steps < max_steps:
        h = min(h, t1 - t)
        ks = []
        for i in range(7):
            if i == 0 and k7 is not None:
                ks.append(k7)                      # FSAL
                continue
            xi = x
            for j, a in enumerate(_DP_A[i]):
                if a != 0.0:
                    xi = xi + h * a * ks[j]
            ks.append(f(xi, t + _DP_C[i] * h))
            nfe += 1
        x5 = x
        for b, k in zip(_DP_B5, ks):
            if b != 0.0:
                x5 = x5 + h * b * k
        err = torch.zeros((), device=x.device, dtype=x.dtype)
        diff = torch.zeros_like(x)
        for b5, b4, k in zip(_DP_B5, _DP_B4, ks):
            if b5 - b4 != 0.0:
                diff = diff + h * (b5 - b4) * k
        scale = atol + rtol * torch.maximum(x.abs(), x5.abs())
        err = torch.sqrt((diff / scale).pow(2).mean())
        if float(err) <= 1.0 or h <= 1e-8:
            t = t + h
            x = x5
            k7 = ks[-1]
            accepted += 1
            if on_step is not None:
                xr = on_step(x, accepted)
                if xr is not x:
                    x, k7 = xr, None     # FSAL is invalid once the state moves
        else:
            k7 = None
        fac = 0.9 * float(err + 1e-12) ** (-0.2)
        h = h * min(5.0, max(0.2, fac))
        steps += 1
    return x, nfe


SOLVERS = {"euler": _euler, "heun": _heun, "rk4": _rk4, "dopri5": _dopri5}


@torch.no_grad()
def sample_ode(model: ProjectedVelocity, x0: torch.Tensor, cond=None,
               solver: str = "euler", steps: int = 50,
               retract_every: int = 0, **solver_kw):
    """Integrate ``dx/dt = v(x,t)`` from ``t=0`` to ``t=1``.

    ``retract_every`` > 0 applies the Level-2 Gauss-Newton pull-back every k
    steps, which is what turns a tangential method into a *projection method*
    with bounded invariant drift (Hairer et al., Ch. IV.4).
    """
    model.counter.reset()
    dev, dt = x0.device, x0.dtype
    step_idx = {"n": 0}

    def f(x, t):
        tt = torch.full((x.shape[0],), float(t), device=dev, dtype=dt)
        v = model(x, tt, cond)
        step_idx["n"] += 1
        return v

    ts = torch.linspace(0, 1, steps + 1, device=dev, dtype=dt).tolist()
    fn = SOLVERS[solver]
    # Retract every ``retract_every`` steps DURING integration -- that is what
    # bounds the invariant drift (Hairer et al., Ch. IV.4).  Previously this
    # pull-back ran once after the solve returned, so ``retract_every`` acted as
    # a boolean and the method was tangential-with-cleanup, not a projection
    # method.  The final retraction below still runs so the returned sample is
    # on the manifold regardless of where the last in-loop retraction landed.
    retractor = None
    if retract_every > 0 and model.manifold is not None:
        def retractor(xk, k):
            return model.manifold.retract(xk) if k % retract_every == 0 else xk
    x1, nfe = fn(f, x0, ts, on_step=retractor, **solver_kw)

    if retract_every > 0 and model.manifold is not None:
        x1 = model.manifold.retract(x1)
    if model.affine is not None and model.mode == "affine":
        pass   # exact by construction; no clean-up applied (this is the claim)
    return x1, model.counter.n
