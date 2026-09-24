"""Constraint algebra for physically-consistent scenario generation.

The central object is a *feasible set* on which grid scenarios provably live.  We
support three nested levels of structure, matching the three levels of the paper:

  Level 1  ``AffineConstraintSet``     -- exact linear invariants  {x : A x = b}
           (Kirchhoff/DC power flow, accounting identities, night-time zeros).
           Admits an exact orthogonal projector; the tangent space is ``null(A)``.

  Level 2  ``NonlinearManifold``       -- smooth invariants        {x : g(x) = 0}
           (AC power-flow equations).  Tangent projection + Gauss-Newton
           retraction give a projection integrator in the sense of
           Hairer-Lubich-Wanner, Geometric Numerical Integration, Ch. IV.4.

  Level 3  ``InequalitySet``           -- operational limits       {x : h(x) <= 0}
           (line ratings, ramp rates, generator bounds).  No cheap exact
           projection exists.  These are **measured and reported**, not
           enforced: no method in the benchmark actively satisfies them, and the
           paper makes no claim that any does.

Design notes
------------
*  All scenarios are tensors of shape ``(B, T, D)``: batch, hours, channels.
*  Grid invariants are *per-hour* (block diagonal in time), so we never form the
   dense ``(T*D, T*D)`` projector.  ``AffineConstraintSet`` stores one small
   ``(D, D)`` projector per hour, which also lets the night-time mask differ
   across hours.  Temporal constraints (daily energy budgets, ramps) are handled
   by the general dense path.
*  Projectors are built in float64 on CPU (MPS has no float64) and then cast to
   the working dtype.  Residual *reporting* always happens in float64 so that a
   claim of "exact" satisfaction is a statement about the arithmetic, not about
   the tolerance of float32.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Sequence

import numpy as np
import torch

__all__ = [
    "AffineConstraintSet",
    "NonlinearManifold",
    "InequalitySet",
    "build_blockdiag_affine",
]

# Singular values below this fraction of the largest are treated as zero when
# forming pseudo-inverses.  1e-10 is comfortably below float64 noise for the
# matrix norms we encounter (PTDF entries are O(1)).
_RCOND = 1e-10


def _pinv_and_projector(A: np.ndarray, rcond: float = _RCOND):
    """Return ``(A_pinv, P_tan, rank)`` for a single constraint block.

    ``P_tan = I - A^+ A`` is the orthogonal projector onto ``null(A)``; it is the
    tangent-space projector of the affine set ``{x : A x = b}``.
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    if m == 0:
        return np.zeros((n, 0)), np.eye(n), 0
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    tol = rcond * (s[0] if s.size else 0.0)
    keep = s > tol
    rank = int(keep.sum())
    s_inv = np.zeros_like(s)
    s_inv[keep] = 1.0 / s[keep]
    A_pinv = (Vt.T * s_inv) @ U.T                     # (n, m)
    V_r = Vt[:rank].T                                  # (n, rank) row-space basis
    P_tan = np.eye(n) - V_r @ V_r.T
    return A_pinv, P_tan, rank


@dataclasses.dataclass
class AffineConstraintSet:
    """Exact linear invariants, block-diagonal over the time axis.

    Represents ``{x in R^{T x D} : A_t x_t = b_t for every hour t}``.

    Parameters
    ----------
    A : ``(T, M, D)`` array -- per-hour constraint matrix (rows may be all-zero
        padding so that every hour has the same ``M``; zero rows are inert).
    b : ``(T, M)`` array -- per-hour right-hand side.
    names : optional per-row labels, used for diagnostic break-downs.
    """

    A: torch.Tensor          # (T, M, D) float64
    b: torch.Tensor          # (T, M)    float64
    names: Optional[Sequence[str]] = None

    # -- derived, filled in __post_init__ --
    A_pinv: torch.Tensor = dataclasses.field(init=False)   # (T, D, M)
    P_tan: torch.Tensor = dataclasses.field(init=False)    # (T, D, D)
    x_part: torch.Tensor = dataclasses.field(init=False)   # (T, D) particular sol.
    rank: torch.Tensor = dataclasses.field(init=False)     # (T,)

    def __post_init__(self):
        A = self.A.detach().to(torch.float64).cpu()
        b = self.b.detach().to(torch.float64).cpu()
        assert A.dim() == 3 and b.dim() == 2 and A.shape[:2] == b.shape, (A.shape, b.shape)
        T, M, D = A.shape
        pinvs, projs, parts, ranks = [], [], [], []
        for t in range(T):
            At = A[t].numpy()
            Ap, P, r = _pinv_and_projector(At)
            pinvs.append(torch.from_numpy(Ap))
            projs.append(torch.from_numpy(P))
            parts.append(torch.from_numpy(Ap @ b[t].numpy()))
            ranks.append(r)
        self.A = A
        self.b = b
        self.A_pinv = torch.stack(pinvs)
        self.P_tan = torch.stack(projs)
        self.x_part = torch.stack(parts)
        self.rank = torch.tensor(ranks)

    # ---------------------------------------------------------------- shapes
    @property
    def T(self) -> int:
        return self.A.shape[0]

    @property
    def D(self) -> int:
        return self.A.shape[2]

    @property
    def codim(self) -> int:
        """Total number of independent scalar constraints across the day."""
        return int(self.rank.sum())

    @property
    def dof(self) -> int:
        """Dimension of the feasible affine set."""
        return self.T * self.D - self.codim

    def to(self, device, dtype=torch.float32) -> "AffineConstraintSet":
        obj = object.__new__(AffineConstraintSet)
        obj.A = self.A.to(device=device, dtype=dtype)
        obj.b = self.b.to(device=device, dtype=dtype)
        obj.A_pinv = self.A_pinv.to(device=device, dtype=dtype)
        obj.P_tan = self.P_tan.to(device=device, dtype=dtype)
        obj.x_part = self.x_part.to(device=device, dtype=dtype)
        obj.rank = self.rank.to(device)
        obj.names = self.names
        return obj

    # ------------------------------------------------------------ operators
    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """``A_t x_t - b_t`` for every hour.  Shape ``(B, T, M)``."""
        A = self.A.to(x.dtype).to(x.device)
        b = self.b.to(x.dtype).to(x.device)
        return torch.einsum("tmd,btd->btm", A, x) - b

    def project_point(self, x: torch.Tensor) -> torch.Tensor:
        """Orthogonal projection of a *state* onto the affine set.

        ``Pi(x) = x - A^+ (A x - b)``.  Idempotent and distance-minimising.
        """
        Ap = self.A_pinv.to(x.dtype).to(x.device)
        r = self.residual(x)
        return x - torch.einsum("tdm,btm->btd", Ap, r)

    def project_tangent(self, v: torch.Tensor) -> torch.Tensor:
        """Orthogonal projection of a *velocity* onto ``null(A)``.

        This is the operator that makes the generative flow constraint-exact:
        if ``x_0`` satisfies the constraints and every increment lies in
        ``null(A)``, then so does ``x_1``, for *any* consistent ODE integrator.
        """
        P = self.P_tan.to(v.dtype).to(v.device)
        return torch.einsum("tde,bte->btd", P, v)

    # ------------------------------------------------- change of coordinates
    def standardise(self, mu: torch.Tensor, sd: torch.Tensor) -> "AffineConstraintSet":
        """Push the constraint through ``x = mu + sd * z``.

        Networks train on standardised coordinates, so the constraint must be
        expressed there too, or "exactness" would silently be lost in the
        de-normalisation.  With ``A x = b`` and ``x = mu + diag(sd) z`` we get
        ``(A diag(sd)) z = b - A mu``.  Degenerate channels (``sd = 0``, e.g. a
        solar site that is exactly zero at that hour across the whole record)
        are handled by the floor already applied to ``sd``.
        """
        mu = mu.detach().to(torch.float64).cpu().reshape(self.T, self.D)
        sd = sd.detach().to(torch.float64).cpu().reshape(self.T, self.D)
        A_new = self.A * sd.unsqueeze(1)                       # (T, M, D)
        b_new = self.b - torch.einsum("tmd,td->tm", self.A, mu)
        return AffineConstraintSet(A_new, b_new, names=self.names)

    def row_normalised(self) -> "AffineConstraintSet":
        """Scale each constraint row to unit norm.

        The feasible set and its projectors are unchanged -- scaling a row of
        ``Ax=b`` scales both sides -- but the *residual* becomes a geometric
        distance rather than a quantity in whatever units the row happened to
        carry.  This matters for any method that puts the residual in a loss: in
        standardised coordinates ``A diag(sd)`` has entries of order ``sd``
        (~1e4 MW here), so an unnormalised quadratic penalty is ~1e8 while the
        flow-matching term is ~1, and every penalty weight saturates.
        """
        A = self.A.clone()
        b = self.b.clone()
        nrm = A.norm(dim=-1, keepdim=True)                   # (T, M, 1)
        scale = torch.where(nrm > 0, 1.0 / nrm, torch.ones_like(nrm))
        return AffineConstraintSet(A * scale, b * scale.squeeze(-1), names=self.names)

    def null_basis(self) -> torch.Tensor:
        """Per-hour orthonormal basis of ``null(A_t)``, zero-padded to a common
        width.  Used by the reduced-coordinate baseline."""
        bases, ks = [], []
        for t in range(self.T):
            At = self.A[t].numpy()
            _, s, Vt = np.linalg.svd(At, full_matrices=True)
            tol = _RCOND * (s[0] if s.size else 0.0)
            r = int((s > tol).sum()) if s.size else 0
            bases.append(torch.from_numpy(Vt[r:].T.copy()))     # (D, D-r)
            ks.append(self.D - r)
        k = max(ks)
        out = torch.zeros(self.T, self.D, k, dtype=torch.float64)
        for t, Bt in enumerate(bases):
            out[t, :, : Bt.shape[1]] = Bt
        return out

    # ----------------------------------------------- variable completion (DC3)
    def completion(self):
        """Pick dependent/independent variable partitions for a DC3-style
        completion layer (Donti, Rolnick & Kolter, ICLR 2021).

        For each hour, rank-revealing QR of ``A_t`` selects ``r`` pivot columns
        ``d`` (dependent) whose submatrix is well conditioned; the remaining
        ``f`` columns are free.  A network then emits only ``x_f`` and the
        completion ``x_d = A_d^{-1} (b - A_f x_f)`` restores exact feasibility.
        Unlike the orthogonal nullspace basis this keeps the free variables
        *axis aligned*, i.e. they remain physically interpretable channels.
        """
        import scipy.linalg as sla
        dep, free, Ad_inv, Af = [], [], [], []
        for t in range(self.T):
            At = self.A[t].numpy()
            act = np.where(np.abs(At).sum(1) > 0)[0]
            At = At[act]
            r = int(self.rank[t])
            if r == 0:
                dep.append(np.array([], dtype=int)); free.append(np.arange(self.D))
                Ad_inv.append(np.zeros((0, 0))); Af.append(np.zeros((0, self.D)))
                continue
            _, _, piv = sla.qr(At, pivoting=True, mode="economic")
            d = np.sort(piv[:r]); f = np.array([i for i in range(self.D) if i not in set(d.tolist())])
            Ad = At[:, d]
            dep.append(d); free.append(f)
            Ad_inv.append(np.linalg.pinv(Ad))
            Af.append(At[:, f])
        return {"dep": dep, "free": free, "Ad_inv": Ad_inv, "Af": Af,
                "b_act": [self.b[t].numpy()[np.where(np.abs(self.A[t].numpy()).sum(1) > 0)[0]]
                          for t in range(self.T)]}

    # ------------------------------------------------------------ reporting
    @torch.no_grad()
    def violation(self, x: torch.Tensor) -> dict:
        """Constraint-violation statistics, computed in float64 on CPU."""
        xd = x.detach().to(torch.float64).cpu()
        A = self.A.to(torch.float64).cpu()
        b = self.b.to(torch.float64).cpu()
        r = torch.einsum("tmd,btd->btm", A, xd) - b
        # Only score rows that are actually active (non-zero constraint rows).
        active = (A.abs().sum(-1) > 0).unsqueeze(0)          # (1, T, M)
        r = r * active
        n_active = active.sum().clamp(min=1)
        return {
            "eq_rmse": float(torch.sqrt((r ** 2).sum() / (r.shape[0] * n_active))),
            "eq_mae": float(r.abs().sum() / (r.shape[0] * n_active)),
            "eq_max": float(r.abs().max()) if r.numel() else 0.0,
            "eq_l2_per_scenario": float(r.pow(2).sum(dim=(1, 2)).sqrt().mean()),
        }


def build_blockdiag_affine(rows_per_hour, D: int, T: int, names=None) -> AffineConstraintSet:
    """Assemble an :class:`AffineConstraintSet` from ragged per-hour rows.

    ``rows_per_hour[t]`` is a list of ``(coef_vector, rhs)`` pairs.  Hours with
    fewer rows are zero-padded, which leaves the projector unchanged.
    """
    M = max(1, max(len(r) for r in rows_per_hour))
    A = np.zeros((T, M, D), dtype=np.float64)
    b = np.zeros((T, M), dtype=np.float64)
    for t, rows in enumerate(rows_per_hour):
        for j, (coef, rhs) in enumerate(rows):
            A[t, j] = coef
            b[t, j] = rhs
    return AffineConstraintSet(torch.from_numpy(A), torch.from_numpy(b), names=names)


class NonlinearManifold:
    """Smooth equality manifold ``{x : g(x) = 0}`` with tangent projection and
    Gauss-Newton retraction.

    ``g`` maps ``(B, T, D) -> (B, T, M)`` and is applied hour-wise; the Jacobian
    is therefore block diagonal in time and is obtained with ``torch.func.jacrev``
    over the channel axis only.
    """

    def __init__(self, g: Callable[[torch.Tensor], torch.Tensor], D: int, n_out: int,
                 retract_iters: int = 3, damping: float = 1e-8):
        self.g = g
        self.D = D
        self.n_out = n_out
        self.retract_iters = retract_iters
        self.damping = damping

    def _jac(self, x: torch.Tensor) -> torch.Tensor:
        """Per-hour Jacobian ``dg/dx``; shape ``(B, T, M, D)``.

        Uses an analytic Jacobian when the constraint supplies one.  Falling back
        to ``jacrev`` costs one backward pass per output row, which for the AC
        equations is ``2 n_b`` passes per hour and is far too slow to train with.
        """
        jac_fn = getattr(self.g, "jacobian", None)
        if jac_fn is not None:
            return jac_fn(x)
        from torch.func import jacrev, vmap
        flat = x.reshape(-1, self.D)
        J = vmap(jacrev(lambda z: self.g(z.reshape(1, 1, -1)).reshape(-1)))(flat)
        return J.reshape(*x.shape[:2], self.n_out, self.D)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        return self.g(x)

    def project_tangent(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``(I - J^+ J) v`` at the current state -- state-dependent projector."""
        J = self._jac(x)
        return v - self._lstsq_apply(J, torch.einsum("btmd,btd->btm", J, v))

    def retract(self, x: torch.Tensor, iters: Optional[int] = None) -> torch.Tensor:
        """Gauss-Newton pull-back onto the manifold: ``x <- x - J^+ g(x)``."""
        iters = self.retract_iters if iters is None else iters
        for _ in range(iters):
            r = self.g(x)
            if float(r.abs().max()) < 1e-12:
                break
            J = self._jac(x)
            x = x - self._lstsq_apply(J, r)
        return x

    def _lstsq_apply(self, J: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Apply the damped pseudo-inverse: ``J^T (J J^T + eps I)^{-1} r``."""
        JJt = torch.einsum("btmd,btnd->btmn", J, J)
        eye = torch.eye(JJt.shape[-1], device=J.device, dtype=J.dtype)
        JJt = JJt + self.damping * eye
        sol = torch.linalg.solve(JJt, r.unsqueeze(-1)).squeeze(-1)
        return torch.einsum("btmd,btm->btd", J, sol)

    @torch.no_grad()
    def violation(self, x: torch.Tensor) -> dict:
        r = self.g(x.to(torch.float64)).detach().cpu()
        return {
            "nl_rmse": float(torch.sqrt(r.pow(2).mean())),
            "nl_max": float(r.abs().max()),
            "nl_l2_per_scenario": float(r.pow(2).sum(dim=(1, 2)).sqrt().mean()),
        }


class StandardisedManifold(NonlinearManifold):
    """Pull a nonlinear manifold back through the standardising map.

    Networks train on ``z`` with ``x = mu + sd * z``.  The affine path handles
    this by transforming ``A`` and ``b`` (:meth:`AffineConstraintSet.standardise`);
    the nonlinear path needs the analogous change of variables, or else ``g`` is
    being evaluated on numbers that are not physical quantities at all.

    Here the pulled-back constraint is ``g_std(z) = g(mu + sd * z)``, whose
    Jacobian is ``J diag(sd)``.  Tangential projection and Gauss-Newton
    retraction then act on the correct manifold in ``z``-coordinates, and
    ``mu + sd * z`` lies on the original manifold in physical units.
    """

    def __init__(self, base: "NonlinearManifold", mu, sd, **kw):
        self.base = base
        self._mu = torch.as_tensor(mu, dtype=torch.float64).reshape(1, -1, base.D)
        self._sd = torch.as_tensor(sd, dtype=torch.float64).reshape(1, -1, base.D)
        super().__init__(self._g, D=base.D, n_out=base.n_out,
                         retract_iters=kw.get("retract_iters", base.retract_iters),
                         damping=kw.get("damping", base.damping))

    def _cast(self, like):
        return (self._mu.to(like.dtype).to(like.device),
                self._sd.to(like.dtype).to(like.device))

    def _g(self, z: torch.Tensor) -> torch.Tensor:
        mu, sd = self._cast(z)
        return self.base.g(mu + sd * z)

    def _jac(self, z: torch.Tensor) -> torch.Tensor:
        """Chain rule through the standardising map: ``J_std = J(x) diag(sd)``.

        Without this override the wrapper hides the base constraint's analytic
        Jacobian behind a bound method and silently falls back to ``jacrev``,
        which is slow enough to look like a hang rather than an error.
        """
        mu, sd = self._cast(z)
        x = mu + sd * z
        base_jac = getattr(self.base.g, "jacobian", None)
        if base_jac is not None:
            J = base_jac(x)                                   # (B, T, M, D)
        else:
            J = NonlinearManifold._jac(self.base, x)
        return J * sd.unsqueeze(-2)


class InequalitySet:
    """Operational limits ``h(x) <= 0`` -- reported, and softly enforced."""

    def __init__(self, h: Callable[[torch.Tensor], torch.Tensor], names=None):
        self.h = h
        self.names = names

    def slack(self, x: torch.Tensor) -> torch.Tensor:
        return self.h(x)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth violation energy ``1/2 ||relu(h)||^2`` -- the Level-3 potential."""
        return 0.5 * torch.relu(self.h(x)).pow(2).sum(dim=(1, 2))

    @torch.no_grad()
    def violation(self, x: torch.Tensor) -> dict:
        s = torch.relu(self.h(x.to(torch.float64))).detach().cpu()
        any_viol = (s > 1e-9).any(dim=(1, 2)) if s.dim() == 3 else (s > 1e-9).any(dim=1)
        return {
            "ineq_rate": float(any_viol.to(torch.float64).mean()),
            "ineq_mean": float(s.sum() / max(1, s.shape[0])),
            "ineq_max": float(s.max()) if s.numel() else 0.0,
        }
