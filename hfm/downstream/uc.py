"""Downstream decision evaluation: two-stage stochastic unit commitment.

This is the test that decides whether better scenarios are *useful*, and it is
the evaluation the stochastic-operations literature actually uses, rather than a
bespoke RL reward.  The protocol is:

  1. For a test day, a generator produces S scenarios of net load.
  2. Solve a two-stage stochastic DC unit commitment MILP: first-stage binary
     commitments ``u[g,h]`` shared across scenarios, second-stage dispatch and
     load shedding per scenario.
  3. *Freeze the commitment* and re-dispatch against the **realised** day.
     Report the realised cost, unserved energy, and the regret against a
     perfect-foresight commitment.

Only step 3 touches the truth, so the comparison across scenario generators is
honest out-of-sample.  Solved with HiGHS through ``scipy.optimize.milp`` /
``linprog`` -- no commercial solver required.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import csr_matrix, eye, hstack, vstack, lil_matrix

__all__ = ["UCProblem", "solve_stochastic_uc", "evaluate_commitment"]


@dataclasses.dataclass
class UCProblem:
    """A compact DC unit-commitment instance derived from a MATPOWER case."""
    pmax: np.ndarray          # (G,) MW
    pmin: np.ndarray          # (G,) MW
    c_lin: np.ndarray         # (G,) $/MWh
    c_nl: np.ndarray          # (G,) $/h no-load cost when committed
    c_su: np.ndarray          # (G,) $ start-up cost
    ramp: np.ndarray          # (G,) MW/h
    voll: float = 5000.0      # $/MWh value of lost load
    spill_cost: float = 50.0  # $/MWh over-generation / curtailment penalty
    T: int = 24

    @property
    def G(self) -> int:
        return self.pmax.shape[0]

    @staticmethod
    def from_grid(grid, n_gen: Optional[int] = None, voll: float = 5000.0,
                  ramp_frac: float = 0.4, nl_frac: float = 0.05,
                  su_per_mw: float = 60.0) -> "UCProblem":
        """Build from a :class:`~hfm.grid.network.DCGrid`.

        Generators are kept in merit order; optionally aggregated to ``n_gen``
        units by merging adjacent merit-order units, which keeps the MILP small
        enough to solve for every test day and every baseline.
        """
        act = np.where(grid.gen_active & (grid.pmax > 0))[0]
        pmax, pmin, c = grid.pmax[act], np.maximum(grid.pmin[act], 0.0), grid.cost_lin[act]
        order = np.argsort(c)
        pmax, pmin, c = pmax[order], pmin[order], c[order]
        if n_gen is not None and n_gen < pmax.size:
            groups = np.array_split(np.arange(pmax.size), n_gen)
            pmax = np.array([pmax[g].sum() for g in groups])
            pmin = np.array([pmin[g].sum() for g in groups])
            c = np.array([np.average(c[g], weights=np.maximum(grid.pmax[act][order][g], 1e-6))
                          for g in groups])
        return UCProblem(
            pmax=pmax, pmin=pmin, c_lin=c,
            c_nl=nl_frac * c * pmax, c_su=su_per_mw * pmax * 0.01 + 100.0,
            ramp=ramp_frac * pmax, voll=voll,
        )


def _build_dispatch_block(prob: UCProblem, S: int):
    """Index helpers for the second-stage variables."""
    G, T = prob.G, prob.T
    n_p = G * T * S
    n_shed = T * S
    n_spill = T * S
    return G, T, n_p, n_shed, n_spill


def solve_stochastic_uc(prob: UCProblem, scenarios: np.ndarray, weights=None,
                        mip_gap: float = 5e-3, time_limit: float = 60.0):
    """Two-stage stochastic UC.

    ``scenarios``: ``(S, T)`` net-load trajectories in MW.
    Returns ``dict`` with the first-stage commitment ``u`` of shape ``(G, T)``.
    """
    S, T = scenarios.shape
    G = prob.G
    assert T == prob.T
    w = np.full(S, 1.0 / S) if weights is None else np.asarray(weights) / np.sum(weights)

    # variable layout: u (G*T binary) | su (G*T) | p (G*T*S) | shed (T*S) | spill (T*S)
    n_u = G * T
    n_su = G * T
    n_p = G * T * S
    n_sh = T * S
    n_sp = T * S
    N = n_u + n_su + n_p + n_sh + n_sp
    o_u, o_su, o_p, o_sh, o_sp = 0, n_u, n_u + n_su, n_u + n_su + n_p, n_u + n_su + n_p + n_sh

    c = np.zeros(N)
    c[o_u:o_u + n_u] = np.repeat(prob.c_nl, T)
    c[o_su:o_su + n_su] = np.repeat(prob.c_su, T)
    c[o_p:o_p + n_p] = np.tile(np.repeat(prob.c_lin, T), S) * np.repeat(w, G * T)
    c[o_sh:o_sh + n_sh] = prob.voll * np.repeat(w, T)
    c[o_sp:o_sp + n_sp] = prob.spill_cost * np.repeat(w, T)

    def iu(g, t): return o_u + g * T + t
    def isu(g, t): return o_su + g * T + t
    def ip(s, g, t): return o_p + s * G * T + g * T + t
    def ish(s, t): return o_sh + s * T + t
    def isp(s, t): return o_sp + s * T + t

    # Sparse (row, col, val) accumulation: the dense matrix would be ~0.7 GB.
    r_idx, c_idx, vals, cons_lo, cons_hi = [], [], [], [], []
    _row = [0]

    def add(coefs, lo, hi):
        for i, v in coefs:
            r_idx.append(_row[0]); c_idx.append(i); vals.append(v)
        cons_lo.append(lo); cons_hi.append(hi); _row[0] += 1

    # power balance per scenario-hour
    for s in range(S):
        for t in range(T):
            add([(ip(s, g, t), 1.0) for g in range(G)] + [(ish(s, t), 1.0), (isp(s, t), -1.0)],
                scenarios[s, t], scenarios[s, t])
    # capacity coupling p <= pmax * u ; p >= pmin * u
    for s in range(S):
        for g in range(G):
            for t in range(T):
                add([(ip(s, g, t), 1.0), (iu(g, t), -prob.pmax[g])], -np.inf, 0.0)
                add([(ip(s, g, t), 1.0), (iu(g, t), -prob.pmin[g])], 0.0, np.inf)
    # ramping per scenario
    for s in range(S):
        for g in range(G):
            for t in range(1, T):
                add([(ip(s, g, t), 1.0), (ip(s, g, t - 1), -1.0)], -prob.ramp[g], prob.ramp[g])
    # start-up logic  su >= u_t - u_{t-1}
    # start-up logic:  su[g,t] >= u[g,t] - u[g,t-1]   (u[g,-1] treated as 0)
    for g in range(G):
        for t in range(T):
            prev = [(iu(g, t - 1), 1.0)] if t > 0 else []
            add([(isu(g, t), 1.0), (iu(g, t), -1.0)] + prev, 0.0, np.inf)

    A = csr_matrix((vals, (r_idx, c_idx)), shape=(_row[0], N))
    lc = LinearConstraint(A, np.array(cons_lo), np.array(cons_hi))
    lb = np.zeros(N); ub = np.full(N, np.inf)
    ub[o_u:o_u + n_u] = 1.0
    ub[o_su:o_su + n_su] = 1.0
    integrality = np.zeros(N); integrality[o_u:o_u + n_u] = 1

    res = milp(c=c, constraints=[lc], bounds=Bounds(lb, ub), integrality=integrality,
               options={"mip_rel_gap": mip_gap, "time_limit": time_limit, "disp": False})
    if res.x is None:
        u = np.ones((G, T))
        return {"u": u, "obj": np.inf, "status": res.status, "success": False}
    u = np.round(res.x[o_u:o_u + n_u].reshape(G, T))
    return {"u": u, "obj": float(res.fun), "status": res.status, "success": True}


def evaluate_commitment(prob: UCProblem, u: np.ndarray, realised: np.ndarray):
    """Re-dispatch a *frozen* commitment against the realised net load (LP)."""
    G, T = prob.G, prob.T
    n_p, n_sh, n_sp = G * T, T, T
    N = n_p + n_sh + n_sp
    o_p, o_sh, o_sp = 0, n_p, n_p + n_sh
    c = np.concatenate([np.repeat(prob.c_lin, T),
                        np.full(T, prob.voll), np.full(T, prob.spill_cost)])
    Aeq = np.zeros((T, N)); beq = realised.copy()
    for t in range(T):
        for g in range(G):
            Aeq[t, o_p + g * T + t] = 1.0
        Aeq[t, o_sh + t] = 1.0
        Aeq[t, o_sp + t] = -1.0
    ri, ci, vv, b_ub, nr = [], [], [], [], 0
    for g in range(G):
        for t in range(1, T):
            for sgn in (1.0, -1.0):
                ri += [nr, nr]; ci += [o_p + g * T + t, o_p + g * T + t - 1]
                vv += [sgn, -sgn]; b_ub.append(prob.ramp[g]); nr += 1
    A_ub = csr_matrix((vv, (ri, ci)), shape=(nr, N)); b_ub = np.array(b_ub)
    bounds = []
    for g in range(G):
        for t in range(T):
            bounds.append((prob.pmin[g] * u[g, t], prob.pmax[g] * u[g, t]))
    bounds += [(0, None)] * (T + T)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=csr_matrix(Aeq), b_eq=beq,
                  bounds=bounds, method="highs")
    if not res.success:
        return {"cost": float(prob.voll * realised.sum()), "shed": float(realised.sum()),
                "spill": 0.0, "success": False}
    x = res.x
    commit_cost = float((prob.c_nl[:, None] * u).sum())
    su = np.maximum(u[:, 1:] - u[:, :-1], 0).sum(axis=1) + u[:, 0]
    start_cost = float((prob.c_su * su).sum())
    return {
        "cost": float(res.fun) + commit_cost + start_cost,
        "fuel_cost": float(res.fun),
        "shed": float(x[o_sh:o_sh + T].sum()),
        "spill": float(x[o_sp:o_sp + T].sum()),
        "success": True,
    }
