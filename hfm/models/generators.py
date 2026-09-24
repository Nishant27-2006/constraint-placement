"""Scenario generators: one interface, ten methods, one shared backbone.

``ScenarioGenerator.sample`` returns an ensemble of shape ``(N, M, T, D)`` for
``N`` conditioning contexts and ``M`` members, together with an exact count of
network evaluations (NFE) and analytic FLOPs, so that accuracy and compute are
reported on the same footing.

Constraint handling is an explicit axis of the comparison:

  ``none``        unconstrained (DDPM, FM, cVAE, cGAN, copula, kNN)
  ``posthoc``     unconstrained sampling then a single orthogonal projection
  ``penalty``     soft quadratic penalty on the residual during training
  ``reduced``     generate in a nullspace basis and map back (exact, linear only)
  ``projected``   *ours*: tangential projection inside the flow (exact, and the
                  only one that transfers to a constraint set unseen at training)
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hfm.flows.fm import ProjectedVelocity, cfm_loss, sample_ode, NFECounter
from hfm.models.nets import TemporalTransformer, ResMLP, count_params, flops_per_eval

__all__ = ["GaussianCopulaGen", "KNNResampleGen", "CVAEGen", "CGANGen",
           "DDPMGen", "FlowMatchingGen", "GENERATORS"]


class ScenarioGenerator:
    name = "base"
    is_neural = False

    def fit(self, X, C, **kw):
        raise NotImplementedError

    def sample(self, C, M, seed=0):
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return 0


# ----------------------------------------------------------------- statistical
class GaussianCopulaGen(ScenarioGenerator):
    """Gaussian copula with empirical marginals, conditioned by covariate-space
    nearest neighbours.  The standard non-deep benchmark in the power-systems
    scenario-generation literature."""

    name = "GaussianCopula"

    def __init__(self, k_neighbors: int = 200, ridge: float = 1e-3):
        self.k = k_neighbors
        self.ridge = ridge

    def fit(self, X, C, **kw):
        self.X = np.asarray(X, dtype=np.float64)           # (N,T,D)
        self.C = np.asarray(C, dtype=np.float64)
        self.Cm, self.Cs = self.C.mean(0), self.C.std(0) + 1e-8
        N, T, D = self.X.shape
        Z = self.X.reshape(N, -1)
        # rank -> normal scores
        r = np.argsort(np.argsort(Z, axis=0), axis=0) + 1.0
        from scipy.stats import norm
        G = norm.ppf(r / (N + 1.0))
        self.Sigma = np.corrcoef(G, rowvar=False)
        self.Sigma = self.Sigma + self.ridge * np.eye(self.Sigma.shape[0])
        self.L = np.linalg.cholesky(self.Sigma)
        self.sorted_Z = np.sort(Z, axis=0)
        return self

    def sample(self, C, M, seed=0):
        from scipy.stats import norm
        rng = np.random.default_rng(seed)
        C = np.asarray(C, dtype=np.float64)
        N, T, D = C.shape[0], self.X.shape[1], self.X.shape[2]
        d = ((C[:, None] - self.C[None]) / self.Cs) ** 2
        nn_idx = np.argsort(d.sum(-1), axis=1)[:, : self.k]
        out = np.empty((N, M, T * D))
        for i in range(N):
            loc = np.sort(self.X[nn_idx[i]].reshape(self.k, -1), axis=0)
            g = rng.standard_normal((M, self.L.shape[0])) @ self.L.T
            u = norm.cdf(g).clip(1e-6, 1 - 1e-6)
            q = (u * (self.k - 1)).astype(int)
            out[i] = np.take_along_axis(loc, q, axis=0)
        return out.reshape(N, M, T, D), 0, 0


class KNNResampleGen(ScenarioGenerator):
    """Resample historical analogue days from covariate-space neighbours.  A
    deceptively strong baseline that generative papers often omit."""

    name = "kNN-Historical"

    def __init__(self, k_neighbors: int = 50, jitter: float = 0.0):
        self.k, self.jitter = k_neighbors, jitter

    def fit(self, X, C, **kw):
        self.X = np.asarray(X, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.Cs = self.C.std(0) + 1e-8
        return self

    def sample(self, C, M, seed=0):
        rng = np.random.default_rng(seed)
        C = np.asarray(C, dtype=np.float64)
        d = (((C[:, None] - self.C[None]) / self.Cs) ** 2).sum(-1)
        idx = np.argsort(d, axis=1)[:, : self.k]
        pick = rng.integers(0, self.k, size=(C.shape[0], M))
        sel = np.take_along_axis(idx, pick, axis=1)
        out = self.X[sel]
        if self.jitter > 0:
            out = out + self.jitter * rng.standard_normal(out.shape) * self.X.std(0)
        return out, 0, 0


# ---------------------------------------------------------------- neural base
class _NeuralGen(ScenarioGenerator):
    is_neural = True

    def __init__(self, T, D, cond_dim, affine=None, manifold=None, ineq=None,
                 hidden=256, depth=4, heads=4, device="cpu", dtype=torch.float32,
                 lr=2e-4, epochs=300, batch=128, ema=0.999, seed=0, constraint="none",
                 penalty_weight=0.0, gamma_max=0.0, verbose=False):
        self.T, self.D, self.cond_dim = T, D, cond_dim
        self.affine, self.manifold, self.ineq = affine, manifold, ineq
        self.hidden, self.depth, self.heads = hidden, depth, heads
        self.device, self.dtype = device, dtype
        self.lr, self.epochs, self.batch, self.ema_decay = lr, epochs, batch, ema
        self.seed, self.constraint = seed, constraint
        self.penalty_weight, self.gamma_max = penalty_weight, gamma_max
        self.verbose = verbose
        self.train_seconds = 0.0

    def _standardise_fit(self, X):
        self.mu = X.mean(0, keepdims=True)
        self.sd = X.std(0, keepdims=True) + 1e-6
        return (X - self.mu) / self.sd

    def _make_backbone(self, d_in, d_out=None):
        torch.manual_seed(self.seed)
        return TemporalTransformer(d_in=d_in, T=self.T, hidden=self.hidden,
                                   depth=self.depth, heads=self.heads,
                                   cond_dim=self.cond_dim, d_out=d_out).to(self.device, self.dtype)

    @property
    def n_params(self):
        return count_params(self.net) if hasattr(self, "net") else 0


class DDPMGen(_NeuralGen):
    """Conditional DDPM (Ho et al. 2020) with a cosine schedule, sharing the
    benchmark backbone.  Ancestral sampling; NFE equals the number of steps."""

    name = "DDPM"

    def __init__(self, *a, n_steps=200, sample_steps=None, x0_clip=5.0, **kw):
        super().__init__(*a, **kw)
        self.n_steps = n_steps
        self.x0_clip = x0_clip
        self.sample_steps = sample_steps or n_steps
        s = 0.008
        ts = torch.linspace(0, 1, n_steps + 1)
        ac = torch.cos((ts + s) / (1 + s) * math.pi / 2) ** 2
        ac = ac / ac[0]
        self.alpha_bar = ac[1:].to(self.device, self.dtype)
        self.betas = (1 - self.alpha_bar / torch.cat([ac[:1].to(self.device, self.dtype),
                                                      self.alpha_bar[:-1]])).clamp(1e-8, 0.999)

    def fit(self, X, C, **kw):
        t0 = time.time()
        Xs = self._standardise_fit(np.asarray(X, np.float32))
        self.net = self._make_backbone(self.D)
        xb = torch.tensor(Xs, dtype=self.dtype, device=self.device)
        cb = torch.tensor(np.asarray(C, np.float32), dtype=self.dtype, device=self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        N = xb.shape[0]
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        for ep in range(self.epochs):
            perm = torch.randperm(N, generator=g).to(self.device)
            for i in range(0, N, self.batch):
                idx = perm[i:i + self.batch]
                x0, c = xb[idx], cb[idx]
                n = torch.randint(0, self.n_steps, (x0.shape[0],), device=self.device)
                ab = self.alpha_bar[n].view(-1, 1, 1)
                eps = torch.randn_like(x0)
                xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
                pred = self.net(xt, n.to(self.dtype) / self.n_steps, c)
                loss = F.mse_loss(pred, eps)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            sched.step()
        self.train_seconds = time.time() - t0
        return self

    @torch.no_grad()
    def sample(self, C, M, seed=0):
        torch.manual_seed(seed)
        C = np.asarray(C, np.float32)
        N = C.shape[0]
        cb = torch.tensor(C, dtype=self.dtype, device=self.device).repeat_interleave(M, 0)
        x = torch.randn(N * M, self.T, self.D, device=self.device, dtype=self.dtype)
        nfe = 0
        fl = 0
        for n in reversed(range(self.n_steps)):
            nn_ = torch.full((x.shape[0],), n, device=self.device, dtype=self.dtype) / self.n_steps
            eps = self.net(x, nn_, cb); nfe += 1
            fl += flops_per_eval(self.net, x)
            ab = self.alpha_bar[n]
            ab_prev = self.alpha_bar[n - 1] if n > 0 else torch.ones_like(ab)
            # Clip the x0 prediction.  With a cosine schedule alpha_bar -> 0 at
            # the start of sampling, so dividing by sqrt(alpha_bar) amplifies any
            # error without bound and the chain diverges.  Data is standardised,
            # so the clip is in units of standard deviations and is the usual
            # static-thresholding fix.
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-4)
            x0 = x0.clamp(-self.x0_clip, self.x0_clip)
            beta = (1 - ab / ab_prev).clamp(1e-8, 0.999)
            mean = ab_prev.sqrt() * x0 * beta / (1 - ab) + ((1 - beta).sqrt() * (1 - ab_prev) / (1 - ab)) * x
            if n > 0:
                x = mean + (beta * (1 - ab_prev) / (1 - ab)).sqrt() * torch.randn_like(x)
            else:
                x = mean
        out = x.cpu().numpy().reshape(N, M, self.T, self.D) * self.sd + self.mu
        return out, nfe, fl // max(1, N * M)


class CVAEGen(_NeuralGen):
    """Conditional VAE with the benchmark backbone as both encoder and decoder."""

    name = "cVAE"

    def __init__(self, *a, latent=64, beta=1.0, **kw):
        super().__init__(*a, **kw)
        self.latent, self.beta = latent, beta

    def fit(self, X, C, **kw):
        t0 = time.time()
        Xs = self._standardise_fit(np.asarray(X, np.float32))
        torch.manual_seed(self.seed)
        self.enc = self._make_backbone(self.D, d_out=2 * self.latent)
        self.dec = self._make_backbone(self.latent, d_out=self.D)
        self.net = self.dec
        xb = torch.tensor(Xs, dtype=self.dtype, device=self.device)
        cb = torch.tensor(np.asarray(C, np.float32), dtype=self.dtype, device=self.device)
        params = list(self.enc.parameters()) + list(self.dec.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        N = xb.shape[0]
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        zero = torch.zeros(1, device=self.device, dtype=self.dtype)
        for ep in range(self.epochs):
            perm = torch.randperm(N, generator=g).to(self.device)
            kl_w = self.beta * min(1.0, ep / max(1, 0.3 * self.epochs))
            for i in range(0, N, self.batch):
                idx = perm[i:i + self.batch]
                x0, c = xb[idx], cb[idx]
                h = self.enc(x0, zero.expand(x0.shape[0]), c)
                mu, lv = h[..., : self.latent], h[..., self.latent:].clamp(-8, 8)
                z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
                rec = self.dec(z, zero.expand(x0.shape[0]), c)
                loss = F.mse_loss(rec, x0) + kl_w * (-0.5 * (1 + lv - mu ** 2 - lv.exp()).mean())
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            sched.step()
        self.train_seconds = time.time() - t0
        return self

    @torch.no_grad()
    def sample(self, C, M, seed=0):
        torch.manual_seed(seed)
        C = np.asarray(C, np.float32); N = C.shape[0]
        cb = torch.tensor(C, dtype=self.dtype, device=self.device).repeat_interleave(M, 0)
        z = torch.randn(N * M, self.T, self.latent, device=self.device, dtype=self.dtype)
        zero = torch.zeros(N * M, device=self.device, dtype=self.dtype)
        x = self.dec(z, zero, cb)
        fl = flops_per_eval(self.dec, z) // max(1, N * M)
        return x.cpu().numpy().reshape(N, M, self.T, self.D) * self.sd + self.mu, 1, fl


class CGANGen(_NeuralGen):
    """Conditional WGAN-GP, the architecture family introduced for renewable
    scenario generation by Chen et al. (IEEE TPWRS 2018)."""

    name = "cWGAN-GP"

    def __init__(self, *a, latent=64, n_critic=5, gp=10.0, **kw):
        super().__init__(*a, **kw)
        self.latent, self.n_critic, self.gp = latent, n_critic, gp

    def fit(self, X, C, **kw):
        t0 = time.time()
        Xs = self._standardise_fit(np.asarray(X, np.float32))
        torch.manual_seed(self.seed)
        self.gen = self._make_backbone(self.latent, d_out=self.D)
        self.crit = self._make_backbone(self.D, d_out=1)
        self.net = self.gen
        xb = torch.tensor(Xs, dtype=self.dtype, device=self.device)
        cb = torch.tensor(np.asarray(C, np.float32), dtype=self.dtype, device=self.device)
        og = torch.optim.AdamW(self.gen.parameters(), lr=self.lr, betas=(0.5, 0.9))
        oc = torch.optim.AdamW(self.crit.parameters(), lr=self.lr, betas=(0.5, 0.9))
        N = xb.shape[0]
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        for ep in range(self.epochs):
            perm = torch.randperm(N, generator=g).to(self.device)
            for i in range(0, N, self.batch):
                idx = perm[i:i + self.batch]
                xr, c = xb[idx], cb[idx]
                B = xr.shape[0]
                zero = torch.zeros(B, device=self.device, dtype=self.dtype)
                for _ in range(self.n_critic):
                    z = torch.randn(B, self.T, self.latent, device=self.device, dtype=self.dtype)
                    xf = self.gen(z, zero, c).detach()
                    eps = torch.rand(B, 1, 1, device=self.device, dtype=self.dtype)
                    xi = (eps * xr + (1 - eps) * xf).requires_grad_(True)
                    d_i = self.crit(xi, zero, c).mean(dim=(1, 2)).sum()
                    (grad,) = torch.autograd.grad(d_i, xi, create_graph=True)
                    gp = ((grad.reshape(B, -1).norm(dim=1) - 1) ** 2).mean()
                    loss_c = (self.crit(xf, zero, c).mean() - self.crit(xr, zero, c).mean()
                              + self.gp * gp)
                    oc.zero_grad(); loss_c.backward(); oc.step()
                z = torch.randn(B, self.T, self.latent, device=self.device, dtype=self.dtype)
                loss_g = -self.crit(self.gen(z, zero, c), zero, c).mean()
                og.zero_grad(); loss_g.backward(); og.step()
        self.train_seconds = time.time() - t0
        return self

    @torch.no_grad()
    def sample(self, C, M, seed=0):
        torch.manual_seed(seed)
        C = np.asarray(C, np.float32); N = C.shape[0]
        cb = torch.tensor(C, dtype=self.dtype, device=self.device).repeat_interleave(M, 0)
        z = torch.randn(N * M, self.T, self.latent, device=self.device, dtype=self.dtype)
        zero = torch.zeros(N * M, device=self.device, dtype=self.dtype)
        x = self.gen(z, zero, cb)
        fl = flops_per_eval(self.gen, z) // max(1, N * M)
        return x.cpu().numpy().reshape(N, M, self.T, self.D) * self.sd + self.mu, 1, fl


class FlowMatchingGen(_NeuralGen):
    """Conditional flow matching, with every constraint-handling mode.

    ``constraint``:
      ``"none"``      plain CFM.
      ``"posthoc"``   plain CFM, then one orthogonal projection of the samples.
      ``"penalty"``   plain CFM plus ``lambda ||A x_1_hat - b||^2`` on the
                      one-step-predicted endpoint (the usual soft-constraint fix).
      ``"reduced"``   CFM in a nullspace basis, decoded back to physical space.
      ``"projected"`` *ours*: the velocity is projected onto ``null(A)`` at every
                      evaluation, in training and in sampling, and the base
                      distribution is pushed onto the feasible set.
      ``"manifold"``  *ours, nonlinear*: state-dependent tangential projection
                      plus Gauss-Newton retraction.
    """

    name = "FlowMatching"

    def __init__(self, *a, solver="euler", steps=50, rtol=1e-3, atol=1e-4,
                 retract_every=0, **kw):
        super().__init__(*a, **kw)
        self.solver, self.steps, self.rtol, self.atol = solver, steps, rtol, atol
        self.retract_every = retract_every

    # ------------------------------------------------------------------ fit
    def fit(self, X, C, **kw):
        t0 = time.time()
        X = np.asarray(X, np.float32)
        Xs = self._standardise_fit(X)
        mu = torch.tensor(self.mu, dtype=torch.float64)
        sd = torch.tensor(self.sd, dtype=torch.float64)

        # express the constraint in standardised coordinates
        self.aff_s = self.affine.standardise(mu, sd) if self.affine is not None else None
        self.aff_dev = self.aff_s.to(self.device, self.dtype) if self.aff_s is not None else None
        # Row-normalised copy for the soft-constraint baseline, so that its
        # weight sweep is meaningful rather than saturated at every value.
        self.aff_pen = (self.aff_s.row_normalised().to(self.device, self.dtype)
                        if self.aff_s is not None else None)
        # the nonlinear manifold needs the same change of variables
        self.man_s = None
        if self.manifold is not None and self.constraint == "manifold":
            from hfm.grid.constraints import StandardisedManifold
            self.man_s = StandardisedManifold(self.manifold, mu, sd)

        if self.constraint == "reduced":
            Nb = self.aff_s.null_basis()
            self.basis = Nb.to(self.device, self.dtype)                  # (T,D,k)
            self.x_part = self.aff_s.x_part.to(self.device, self.dtype)  # (T,D)
            d_model = Nb.shape[-1]
        elif self.constraint == "completion":
            comp = self.aff_s.completion()
            ks = {len(f) for f in comp["free"]}
            assert len(ks) == 1, "DC3 completion needs a uniform free-variable count per hour"
            self.c_free = torch.tensor(np.stack(comp["free"]), device=self.device)   # (T,k)
            self.c_dep = torch.tensor(np.stack(comp["dep"]), device=self.device)     # (T,r)
            self.c_Ainv = torch.tensor(np.stack(comp["Ad_inv"]), device=self.device,
                                       dtype=self.dtype)
            self.c_Af = torch.tensor(np.stack(comp["Af"]), device=self.device, dtype=self.dtype)
            self.c_b = torch.tensor(np.stack(comp["b_act"]), device=self.device, dtype=self.dtype)
            d_model = ks.pop()
        else:
            d_model = self.D

        if self.gamma_max > 0 and self.ineq is not None:
            # The inequality set is defined on physical quantities (MW), but the
            # network operates on standardised coordinates.  Enabling the
            # dissipative corrector without the same change of variables applied
            # to the affine and manifold constraints would evaluate h() on
            # numbers that are not physical quantities -- exactly the class of
            # silent error that produced a wrong AC result.  Refuse loudly.
            raise NotImplementedError(
                "the inequality corrector needs a standardised-coordinate "
                "pull-back of h(); see StandardisedManifold for the pattern. "
                "Inequality violations are currently measured and reported, "
                "not enforced.")

        mode = {"projected": "affine", "manifold": "manifold"}.get(self.constraint, "none")
        # "pcfm" trains an ordinary unconstrained model; the constraint is applied
        # only at sampling time by the inference-time corrector.
        self.net = self._make_backbone(d_model, d_out=d_model)
        self.model = ProjectedVelocity(
            self.net,
            affine=self.aff_dev if mode == "affine" else None,
            manifold=self.man_s if mode == "manifold" else None,
            mode=mode,
            corrector=self.ineq if self.gamma_max > 0 else None,
            gamma_max=self.gamma_max,
        )

        xb = torch.tensor(Xs, dtype=self.dtype, device=self.device)
        if self.constraint == "reduced":
            # exact coordinates: c = B^T (z - x_part)
            xb = torch.einsum("tdk,btd->btk", self.basis, xb - self.x_part)
        elif self.constraint == "completion":
            xb = torch.gather(xb, 2, self.c_free.unsqueeze(0).expand(xb.shape[0], -1, -1))
        cb = torch.tensor(np.asarray(C, np.float32), dtype=self.dtype, device=self.device)

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        self.ema = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        N = xb.shape[0]
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        torch.manual_seed(self.seed)

        for ep in range(self.epochs):
            perm = torch.randperm(N, generator=g).to(self.device)
            for i in range(0, N, self.batch):
                idx = perm[i:i + self.batch]
                x1, c = xb[idx], cb[idx]
                x0 = torch.randn_like(x1)
                if self.constraint == "projected":
                    x0 = self.aff_dev.project_point(x0)      # base law lives on the set
                elif self.constraint == "manifold":
                    x0 = self.man_s.retract(x0, iters=4)     # base law lives on the manifold
                loss = cfm_loss(self.model, x1, x0, cond=c,
                                project_target=self.constraint in ("projected", "manifold"))
                if self.constraint == "penalty" and self.penalty_weight > 0:
                    B = x1.shape[0]
                    t = torch.rand(B, device=self.device, dtype=self.dtype)
                    tb = t.view(B, 1, 1)
                    xt = (1 - tb) * x0 + tb * x1
                    v = self.model(xt, t, c)
                    x1_hat = xt + (1 - tb) * v               # one-step endpoint estimate
                    r = self.aff_pen.residual(x1_hat)
                    loss = loss + self.penalty_weight * r.pow(2).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    for k, v in self.net.state_dict().items():
                        if v.dtype.is_floating_point:
                            self.ema[k].mul_(self.ema_decay).add_(v, alpha=1 - self.ema_decay)
                        else:
                            self.ema[k].copy_(v)
            sched.step()
        self.net.load_state_dict(self.ema)
        self.train_seconds = time.time() - t0
        return self

    def _complete(self, x_free: torch.Tensor) -> torch.Tensor:
        """DC3 completion: restore dependent channels from the free ones."""
        B = x_free.shape[0]
        out = torch.zeros(B, self.T, self.D, device=x_free.device, dtype=x_free.dtype)
        out.scatter_(2, self.c_free.unsqueeze(0).expand(B, -1, -1), x_free)
        rhs = self.c_b.unsqueeze(0) - torch.einsum("tmk,btk->btm", self.c_Af, x_free)
        xd = torch.einsum("tdm,btm->btd", self.c_Ainv, rhs)
        out.scatter_(2, self.c_dep.unsqueeze(0).expand(B, -1, -1), xd)
        return out

    # --------------------------------------------------------------- sample
    @torch.no_grad()
    def sample(self, C, M, seed=0, solver=None, steps=None, rtol=None, atol=None):
        solver = solver or self.solver
        steps = self.steps if steps is None else steps
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        torch.manual_seed(seed)
        C = np.asarray(C, np.float32); N = C.shape[0]
        cb = torch.tensor(C, dtype=self.dtype, device=self.device).repeat_interleave(M, 0)
        d_model = self.net.d_in
        x0 = torch.randn(N * M, self.T, d_model, device=self.device, dtype=self.dtype)
        if self.constraint == "projected":
            x0 = self.aff_dev.project_point(x0)
        elif self.constraint == "manifold":
            x0 = self.man_s.retract(x0, iters=4)
        if self.constraint == "pcfm":
            from hfm.flows.correctors import PCFMCorrector, sample_with_corrector
            corr = PCFMCorrector(affine=self.aff_dev, manifold=self.manifold)
            x1, nfe, self.extra_solves = sample_with_corrector(
                self.model, x0, cond=cb, steps=steps, corrector=corr)
        else:
            kw = {"rtol": rtol, "atol": atol} if solver == "dopri5" else {}
            x1, nfe = sample_ode(self.model, x0, cond=cb, solver=solver, steps=steps,
                                 retract_every=self.retract_every, **kw)
        fl = flops_per_eval(self.net, x0) * nfe // max(1, N * M)
        if self.constraint == "reduced":
            x1 = self.x_part + torch.einsum("tdk,btk->btd", self.basis, x1)
        elif self.constraint == "completion":
            x1 = self._complete(x1)
        z = x1.detach().cpu().to(torch.float64)
        if self.constraint == "posthoc":
            z = self.aff_s.project_point(z)
        out = z.numpy().reshape(N, M, self.T, self.D) * self.sd + self.mu
        return out, nfe, fl


GENERATORS = {
    "GaussianCopula": GaussianCopulaGen,
    "kNN-Historical": KNNResampleGen,
    "cVAE": CVAEGen,
    "cWGAN-GP": CGANGen,
    "DDPM": DDPMGen,
    "FlowMatching": FlowMatchingGen,
}


class NormalizingFlowGen(_NeuralGen):
    """Conditional RealNVP over the flattened day.

    Included because Dumas et al. (*Applied Energy* 305:117871, 2022) established
    conditional normalizing flows as the strongest deep baseline for energy
    scenario generation under exactly this evaluation shape (scoring rules plus a
    downstream optimisation task), so a comparison that omits them is incomplete.
    """

    name = "NormFlow-RealNVP"

    def __init__(self, *a, n_coupling=8, hidden_nf=256, **kw):
        super().__init__(*a, **kw)
        self.n_coupling, self.hidden_nf = n_coupling, hidden_nf

    def _build(self, P):
        import torch.nn as nn
        torch.manual_seed(self.seed)
        self.masks = []
        nets = []
        for i in range(self.n_coupling):
            m = torch.zeros(P)
            m[i % 2::2] = 1.0                       # alternating checkerboard
            self.masks.append(m.to(self.device, self.dtype))
            nets.append(nn.Sequential(
                nn.Linear(P + self.cond_dim, self.hidden_nf), nn.SiLU(),
                nn.Linear(self.hidden_nf, self.hidden_nf), nn.SiLU(),
                nn.Linear(self.hidden_nf, 2 * P)))
            nn.init.zeros_(nets[-1][-1].weight); nn.init.zeros_(nets[-1][-1].bias)
        self.net = nn.ModuleList(nets).to(self.device, self.dtype)

    def _fwd(self, x, c):
        """Data -> latent, accumulating log|det J|."""
        ld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for m, net in zip(self.masks, self.net):
            xm = x * m
            h = net(torch.cat([xm, c], -1))
            s, t = h.chunk(2, -1)
            s = torch.tanh(s) * (1 - m)
            t = t * (1 - m)
            x = xm + (1 - m) * (x * torch.exp(s) + t)
            ld = ld + s.sum(-1)
        return x, ld

    def _inv(self, z, c):
        for m, net in zip(reversed(self.masks), reversed(self.net)):
            zm = z * m
            h = net(torch.cat([zm, c], -1))
            s, t = h.chunk(2, -1)
            s = torch.tanh(s) * (1 - m)
            t = t * (1 - m)
            z = zm + (1 - m) * ((z - t) * torch.exp(-s))
        return z

    def fit(self, X, C, **kw):
        t0 = time.time()
        Xs = self._standardise_fit(np.asarray(X, np.float32))
        P = self.T * self.D
        self._build(P)
        xb = torch.tensor(Xs.reshape(-1, P), dtype=self.dtype, device=self.device)
        cb = torch.tensor(np.asarray(C, np.float32), dtype=self.dtype, device=self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        N = xb.shape[0]
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        for ep in range(self.epochs):
            perm = torch.randperm(N, generator=g).to(self.device)
            for i in range(0, N, self.batch):
                idx = perm[i:i + self.batch]
                z, ld = self._fwd(xb[idx], cb[idx])
                nll = 0.5 * (z ** 2).sum(-1).mean() - ld.mean()
                opt.zero_grad(); nll.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); opt.step()
            sched.step()
        self.train_seconds = time.time() - t0
        return self

    @torch.no_grad()
    def sample(self, C, M, seed=0):
        torch.manual_seed(seed)
        C = np.asarray(C, np.float32); N = C.shape[0]
        cb = torch.tensor(C, dtype=self.dtype, device=self.device).repeat_interleave(M, 0)
        z = torch.randn(N * M, self.T * self.D, device=self.device, dtype=self.dtype)
        x = self._inv(z, cb)
        fl = sum(2 * m.in_features * m.out_features for m in self.net.modules()
                 if isinstance(m, nn.Linear))
        out = x.cpu().numpy().reshape(N, M, self.T, self.D) * self.sd + self.mu
        return out, 1, fl


GENERATORS["NormFlow-RealNVP"] = NormalizingFlowGen
