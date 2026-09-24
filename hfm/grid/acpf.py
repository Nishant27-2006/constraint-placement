"""AC power flow: the nonlinear feasible manifold (Suite C).

State per hour is $x = [V_m, V_a, P, Q] \\in \\mathbb{R}^{4 n_b}$ and the manifold is

    g(x) = [ P - P(V_m,V_a) ;  Q - Q(V_m,V_a) ] = 0,      g: R^{4n} -> R^{2n}

with the standard polar injections

    P_i = V_i sum_j V_j ( G_ij cos(θ_i-θ_j) + B_ij sin(θ_i-θ_j) )
    Q_i = V_i sum_j V_j ( G_ij sin(θ_i-θ_j) - B_ij cos(θ_i-θ_j) ).

Unlike the DC case this is a curved manifold: the straight-line flow-matching
interpolant leaves it, so Theorem 1 does not apply and we fall back to the
tangential-projection-plus-retraction scheme of Proposition 2.

Dataset construction solves a genuine Newton-Raphson power flow, batched over
hours in numpy with the analytic MATPOWER Jacobian, so every training point lies
on the manifold to solver tolerance (we assert < 1e-10 p.u.).
"""
from __future__ import annotations

import numpy as np
import torch

from hfm.grid.network import build_ybus, REF_BUS, PV_BUS, BUS_TYPE, VM, PD, QD, GEN_BUS, VG

__all__ = ["ACSystem", "ac_injections_torch", "ACManifoldFn"]


def ac_injections_torch(Vm: torch.Tensor, Va: torch.Tensor,
                        G: torch.Tensor, B: torch.Tensor):
    """Polar power injections, differentiable.  ``Vm, Va``: ``(..., nb)``."""
    d = Va.unsqueeze(-1) - Va.unsqueeze(-2)              # (..., nb, nb)
    c, s = torch.cos(d), torch.sin(d)
    VV = Vm.unsqueeze(-1) * Vm.unsqueeze(-2)
    P = (VV * (G * c + B * s)).sum(-1)
    Q = (VV * (G * s - B * c)).sum(-1)
    return P, Q


class ACManifoldFn:
    """Callable ``g`` for :class:`~hfm.grid.constraints.NonlinearManifold`.

    Channel layout is ``[Vm | Va | P | Q]``; a module-level class rather than a
    closure so built datasets stay picklable.
    """

    def __init__(self, G: np.ndarray, B: np.ndarray):
        self.G = np.asarray(G, np.float64)
        self.B = np.asarray(B, np.float64)
        self.nb = self.G.shape[0]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n = self.nb
        G = torch.as_tensor(self.G, dtype=x.dtype, device=x.device)
        B = torch.as_tensor(self.B, dtype=x.dtype, device=x.device)
        Vm, Va, P, Q = x[..., :n], x[..., n:2 * n], x[..., 2 * n:3 * n], x[..., 3 * n:]
        Pc, Qc = ac_injections_torch(Vm, Va, G, B)
        return torch.cat([P - Pc, Q - Qc], dim=-1)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Analytic Jacobian ``dg/dx``, shape ``(..., 2n, 4n)``.

        Autograd is not an option here: ``jacrev`` needs one backward pass per
        output, i.e. ``2n = 114`` per element and ~1.8e5 per training step, which
        makes the manifold model unusable.  These are the standard polar
        power-flow Jacobian blocks (H, N, M, L), written batched.

        With ``g = [P - P(Vm,Va); Q - Q(Vm,Va)]`` and column order
        ``[Vm | Va | P | Q]`` the Jacobian is ``[[-N, -H, I, 0], [-L, -M, 0, I]]``.
        """
        n = self.nb
        G = torch.as_tensor(self.G, dtype=x.dtype, device=x.device)
        B = torch.as_tensor(self.B, dtype=x.dtype, device=x.device)
        Vm, Va = x[..., :n], x[..., n:2 * n]
        d = Va.unsqueeze(-1) - Va.unsqueeze(-2)              # theta_ij
        c, s = torch.cos(d), torch.sin(d)
        Gc_Bs = G * c + B * s
        Gs_Bc = G * s - B * c
        VV = Vm.unsqueeze(-1) * Vm.unsqueeze(-2)
        P = (VV * Gc_Bs).sum(-1)
        Q = (VV * Gs_Bc).sum(-1)

        eye = torch.eye(n, dtype=x.dtype, device=x.device)
        off = 1.0 - eye
        H = VV * Gs_Bc * off                                  # dP/dVa, i != j
        N = Vm.unsqueeze(-1) * Gc_Bs * off                    # dP/dVm
        M = -VV * Gc_Bs * off                                 # dQ/dVa
        L = Vm.unsqueeze(-1) * Gs_Bc * off                    # dQ/dVm
        gdiag = torch.diagonal(G).expand_as(Vm)
        bdiag = torch.diagonal(B).expand_as(Vm)
        H = H + torch.diag_embed(-Q - bdiag * Vm ** 2)
        N = N + torch.diag_embed(P / Vm.clamp(min=1e-8) + gdiag * Vm)
        M = M + torch.diag_embed(P - gdiag * Vm ** 2)
        L = L + torch.diag_embed(Q / Vm.clamp(min=1e-8) - bdiag * Vm)

        Z = torch.zeros_like(H)
        I = eye.expand_as(H)
        top = torch.cat([-N, -H, I, Z], dim=-1)
        bot = torch.cat([-L, -M, Z, I], dim=-1)
        return torch.cat([top, bot], dim=-2)


class ACSystem:
    """Newton-Raphson AC power flow, batched over hours."""

    def __init__(self, case, grid):
        self.case, self.grid = case, grid
        self.Y = build_ybus(case, grid)
        self.G, self.B = self.Y.real.copy(), self.Y.imag.copy()
        self.nb = case.nb
        bt = case.bus[:, BUS_TYPE]
        self.ref = int(np.where(bt == REF_BUS)[0][0]) if (bt == REF_BUS).any() else 0
        self.pv = np.where(bt == PV_BUS)[0]
        self.pq = np.array([i for i in range(self.nb)
                            if i != self.ref and i not in set(self.pv.tolist())])
        vg = np.ones(self.nb)
        vg[grid.gen_bus] = np.nan_to_num(case.gen[:, VG], nan=1.0)
        self.vset = np.where(np.isnan(case.bus[:, VM]), 1.0, case.bus[:, VM])
        self.vset[self.pv] = vg[self.pv]
        self.vset[self.ref] = vg[self.ref]

    def _inj(self, Vm, Va):
        d = Va[..., :, None] - Va[..., None, :]
        c, s = np.cos(d), np.sin(d)
        VV = Vm[..., :, None] * Vm[..., None, :]
        return (VV * (self.G * c + self.B * s)).sum(-1), (VV * (self.G * s - self.B * c)).sum(-1)

    def _dS_dV(self, Vm, Va):
        """Analytic ``dSbus_dVm`` / ``dSbus_dVa`` (MATPOWER polar form), batched.

        Written with broadcasting rather than explicit diagonal matrices:
        ``diag(V) @ M == V[...,:,None] * M`` and ``M @ diag(d) == M * d[...,None,:]``.
        """
        V = Vm * np.exp(1j * Va)                                   # (N, nb)
        Ibus = V @ self.Y.T                                        # Ybus @ V
        Vnorm = V / np.abs(V)
        eye = np.eye(self.nb)
        Y_dVn = self.Y[None] * Vnorm[:, None, :]                   # Ybus @ diag(Vnorm)
        Y_dV = self.Y[None] * V[:, None, :]                        # Ybus @ diag(V)
        dS_dVm = (V[:, :, None] * np.conj(Y_dVn)
                  + eye * (np.conj(Ibus) * Vnorm)[:, None, :])
        dS_dVa = 1j * V[:, :, None] * np.conj(eye * Ibus[:, None, :] - Y_dV)
        return dS_dVm, dS_dVa

    def solve(self, Pspec: np.ndarray, Qspec: np.ndarray, tol=1e-12, max_iter=25,
              chunk: int = 4000, verbose: bool = False):
        """Batched NR.  ``Pspec, Qspec``: ``(N, nb)`` in per-unit.

        Solved in chunks: the dense Jacobian is ``(N, 2n, 2n)``, which for a full
        five-year record at hourly resolution would be tens of gigabytes.
        """
        N = Pspec.shape[0]
        if N > chunk:
            outs = []
            for a in range(0, N, chunk):
                outs.append(self._solve_chunk(Pspec[a:a + chunk], Qspec[a:a + chunk],
                                              tol, max_iter))
                if verbose:
                    print(f"    AC chunk {min(a+chunk,N)}/{N}", flush=True)
            Vm = np.concatenate([o[0] for o in outs]); Va = np.concatenate([o[1] for o in outs])
            Pc = np.concatenate([o[2] for o in outs]); Qc = np.concatenate([o[3] for o in outs])
            ps = np.concatenate([o[4] for o in outs])
            return Vm, Va, Pc, Qc, ps, max(o[5] for o in outs)
        return self._solve_chunk(Pspec, Qspec, tol, max_iter)

    def _solve_chunk(self, Pspec, Qspec, tol=1e-12, max_iter=25):
        N = Pspec.shape[0]
        Vm = np.tile(self.vset, (N, 1)).astype(np.float64)
        Va = np.zeros((N, self.nb))
        pvpq = np.concatenate([self.pv, self.pq]).astype(int)
        pq = self.pq.astype(int)
        for it in range(max_iter):
            Pc, Qc = self._inj(Vm, Va)
            dP = (Pspec - Pc)[:, pvpq]
            dQ = (Qspec - Qc)[:, pq]
            F = np.concatenate([dP, dQ], axis=1)
            if np.abs(F).max() < tol:
                break
            dS_dVm, dS_dVa = self._dS_dV(Vm, Va)
            J11 = dS_dVa.real[:, pvpq][:, :, pvpq]
            J12 = dS_dVm.real[:, pvpq][:, :, pq]
            J21 = dS_dVa.imag[:, pq][:, :, pvpq]
            J22 = dS_dVm.imag[:, pq][:, :, pq]
            J = np.block([[J11, J12], [J21, J22]])
            dx = np.linalg.solve(J, F[..., None])[..., 0]
            Va[:, pvpq] += dx[:, : pvpq.size]
            Vm[:, pq] += dx[:, pvpq.size:]
        Pc, Qc = self._inj(Vm, Va)
        dP = (Pspec - Pc)[:, pvpq]
        dQ = (Qspec - Qc)[:, pq]
        per_sample = np.abs(np.concatenate([dP, dQ], axis=1)).max(axis=1)
        return Vm, Va, Pc, Qc, per_sample, it
