"""MATPOWER case parsing and power-system physics (DC and AC).

Self-contained: no MATPOWER/pandapower/PyPSA dependency, so the benchmark can be
reproduced from a bare Python environment plus the public ``.m`` case files.

Conventions follow MATPOWER (Zimmerman, Murillo-Sanchez & Thomas, 2011):
buses are indexed internally 0..nb-1, injections are in MW (per-unit on
``baseMVA`` where stated), and branch flows are "from"-end active flows in MW.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ["MatpowerCase", "load_case", "DCGrid"]

# MATPOWER column indices (0-based)
BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN = range(13)
GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN = range(10)
F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX = range(13)

REF_BUS, PV_BUS, PQ_BUS = 3, 2, 1


def _parse_matrix(text: str, name: str) -> np.ndarray:
    """Extract ``mpc.<name> = [ ... ];`` into a float array."""
    pat = re.compile(rf"mpc\.{name}\s*=\s*\[(.*?)\];", re.S)
    m = pat.search(text)
    if m is None:
        return np.zeros((0, 0))
    body = m.group(1)
    body = re.sub(r"%.*", "", body)                       # strip comments
    rows = []
    for line in body.split(";"):
        line = line.strip()
        if not line:
            continue
        toks = line.replace(",", " ").split()
        vals = []
        for tok in toks:
            try:
                vals.append(float(tok))
            except ValueError:
                vals.append(np.nan)
        if vals:
            rows.append(vals)
    width = max(len(r) for r in rows)
    out = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


@dataclasses.dataclass
class MatpowerCase:
    name: str
    baseMVA: float
    bus: np.ndarray
    gen: np.ndarray
    branch: np.ndarray
    gencost: np.ndarray

    @property
    def nb(self) -> int:
        return self.bus.shape[0]

    @property
    def nl(self) -> int:
        return self.branch.shape[0]

    @property
    def ng(self) -> int:
        return self.gen.shape[0]


def load_case(path: str | Path) -> MatpowerCase:
    path = Path(path)
    text = path.read_text()
    bm = re.search(r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)\s*;", text)
    baseMVA = float(bm.group(1)) if bm else 100.0
    case = MatpowerCase(
        name=path.stem,
        baseMVA=baseMVA,
        bus=_parse_matrix(text, "bus"),
        gen=_parse_matrix(text, "gen"),
        branch=_parse_matrix(text, "branch"),
        gencost=_parse_matrix(text, "gencost"),
    )
    return case


class DCGrid:
    """DC power-flow model: incidence, susceptance, PTDF, and a DC-OPF solver.

    The affine set ``{(p, f) : 1^T p = 0, f = PTDF p}`` is exactly the set of
    Kirchhoff-consistent operating points under the DC approximation, and is the
    Level-1 feasible set used throughout the paper.
    """

    def __init__(self, case: MatpowerCase, slack: Optional[int] = None):
        self.case = case
        nb, nl = case.nb, case.nl
        self.nb, self.nl = nb, nl

        # external -> internal bus numbering
        ext = case.bus[:, BUS_I].astype(int)
        self.ext2int = {int(e): i for i, e in enumerate(ext)}
        self.int2ext = ext

        f = np.array([self.ext2int[int(b)] for b in case.branch[:, F_BUS]])
        t = np.array([self.ext2int[int(b)] for b in case.branch[:, T_BUS]])
        self.fbus, self.tbus = f, t

        status = np.nan_to_num(case.branch[:, BR_STATUS], nan=1.0)
        x = case.branch[:, BR_X].copy()
        tap = case.branch[:, TAP].copy()
        tap = np.where(np.isnan(tap) | (tap == 0.0), 1.0, tap)
        b_series = status / (x * tap)                     # per-unit susceptance
        self.b_series = b_series

        # branch-bus incidence A: (nl, nb), +1 at "from", -1 at "to"
        A = np.zeros((nl, nb))
        A[np.arange(nl), f] = 1.0
        A[np.arange(nl), t] = -1.0
        self.A_inc = A

        self.Bf = (b_series[:, None] * A)                 # f_pu = Bf @ theta
        self.Bbus = A.T @ self.Bf                         # (nb, nb)

        if slack is None:
            ref = np.where(case.bus[:, BUS_TYPE] == REF_BUS)[0]
            slack = int(ref[0]) if ref.size else 0
        self.slack = slack
        self.noslack = np.array([i for i in range(nb) if i != slack])

        Bred = self.Bbus[np.ix_(self.noslack, self.noslack)]
        self.Bred_inv = np.linalg.inv(Bred)

        # PTDF: f_MW = PTDF @ p_MW  (columns sum-invariant; slack column is 0)
        H = np.zeros((nl, nb))
        H[:, self.noslack] = self.Bf[:, self.noslack] @ self.Bred_inv
        self.PTDF = H

        rate = case.branch[:, RATE_A].copy()
        rate = np.where(np.isnan(rate) | (rate <= 0.0), np.inf, rate)
        self.rate_a = rate                                # MW thermal limits

        # generators
        self.gen_bus = np.array([self.ext2int[int(b)] for b in case.gen[:, GEN_BUS]])
        gstat = np.nan_to_num(case.gen[:, GEN_STATUS], nan=1.0) > 0
        self.gen_active = gstat
        self.pmax = np.nan_to_num(case.gen[:, PMAX], nan=0.0)
        self.pmin = np.nan_to_num(case.gen[:, PMIN], nan=0.0)
        self.pd_base = np.nan_to_num(case.bus[:, PD], nan=0.0)   # MW nominal load

        self.cost_lin, self.cost_quad, self.cost_const = self._gen_cost()

    # ------------------------------------------------------- N-1 contingency
    def outage(self, branch: int) -> "DCGrid":
        """Return the grid with one branch switched out.

        Operators run scenarios against contingency topologies constantly, and a
        contingency changes the PTDF -- hence the *constraint set* -- without
        changing the data-generating process for demand or weather. This is the
        setting in which a constraint baked into the model's coordinates at
        training time becomes stale, while a projector can simply be swapped.
        """
        import copy
        case = copy.deepcopy(self.case)
        case.branch = case.branch.copy()
        case.branch[branch, BR_STATUS] = 0.0
        return DCGrid(case, slack=self.slack)

    def is_bridge(self, branch: int) -> bool:
        """True if removing the branch disconnects the network (islanding),
        in which case the DC model has no unique solution and we skip it."""
        import scipy.sparse.csgraph as csg
        from scipy.sparse import coo_matrix
        keep = [j for j in range(self.nl) if j != branch]
        r = np.concatenate([self.fbus[keep], self.tbus[keep]])
        c = np.concatenate([self.tbus[keep], self.fbus[keep]])
        A = coo_matrix((np.ones(len(r)), (r, c)), shape=(self.nb, self.nb))
        n_comp, _ = csg.connected_components(A, directed=False)
        return n_comp > 1

    # ------------------------------------------------------------------ cost
    def _gen_cost(self):
        gc = self.case.gencost
        ng = self.case.ng
        lin = np.ones(ng) * 20.0
        quad = np.zeros(ng)
        const = np.zeros(ng)
        if gc.size == 0:
            return lin, quad, const
        for i in range(min(ng, gc.shape[0])):
            model = gc[i, 0]
            n = int(gc[i, 3]) if not np.isnan(gc[i, 3]) else 0
            coef = gc[i, 4 : 4 + n]
            if model == 2:                                # polynomial
                if n == 3:
                    quad[i], lin[i], const[i] = coef[0], coef[1], coef[2]
                elif n == 2:
                    lin[i], const[i] = coef[0], coef[1]
                elif n >= 1:
                    lin[i] = coef[-2] if n >= 2 else coef[0]
            elif model == 1 and n >= 2:                   # piecewise linear
                pts = coef.reshape(-1, 2)
                dp = np.diff(pts[:, 0])
                dc = np.diff(pts[:, 1])
                lin[i] = float(np.mean(dc / np.maximum(dp, 1e-9)))
        return lin, quad, const

    # ------------------------------------------------------------ operators
    def flows(self, p_mw: np.ndarray) -> np.ndarray:
        """Branch active flows (MW) from nodal net injections (MW)."""
        return p_mw @ self.PTDF.T

    def theta(self, p_mw: np.ndarray) -> np.ndarray:
        """Bus voltage angles (rad) with the slack fixed at zero."""
        p_pu = p_mw / self.case.baseMVA
        th = np.zeros_like(p_pu)
        th[..., self.noslack] = p_pu[..., self.noslack] @ self.Bred_inv.T
        return th

    # -------------------------------------------------------------- DC-OPF
    def dcopf(self, pd_mw: np.ndarray, pmax: Optional[np.ndarray] = None,
              enforce_limits: bool = True, slack_cost: float = 5000.0):
        """Least-cost DC dispatch for one hour.

        Solved as an LP with HiGHS (``scipy.optimize.linprog``).  Load shedding
        and over-generation enter as high-cost slack variables so the problem is
        always feasible; their use is reported.

        Returns ``dict`` with ``pg`` (MW), ``p_inj`` (MW), ``f`` (MW), ``cost``,
        ``shed``, ``spill``, ``status``.
        """
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix, hstack, vstack

        ng, nb, nl = self.case.ng, self.nb, self.nl
        act = np.where(self.gen_active)[0]
        na = act.size
        pmax_a = (self.pmax if pmax is None else pmax)[act]
        pmin_a = np.minimum(self.pmin[act], pmax_a)

        # vars: [pg (na), shed (nb), spill (nb)]
        c = np.concatenate([self.cost_lin[act], np.full(nb, slack_cost), np.full(nb, slack_cost)])

        # power balance: sum(pg) + sum(shed) - sum(spill) = sum(pd)
        Aeq = np.concatenate([np.ones(na), np.ones(nb), -np.ones(nb)])[None, :]
        beq = np.array([pd_mw.sum()])

        bounds = ([(lo, hi) for lo, hi in zip(pmin_a, pmax_a)]
                  + [(0.0, float(max(v, 0.0))) for v in pd_mw]
                  + [(0.0, None)] * nb)

        A_ub, b_ub = None, None
        if enforce_limits and np.isfinite(self.rate_a).any():
            Mg = np.zeros((nb, na))
            Mg[self.gen_bus[act], np.arange(na)] = 1.0
            # p_inj = Mg pg + shed - spill - pd   =>   f = PTDF p_inj
            Hg = self.PTDF @ Mg
            Hs = self.PTDF
            Hblock = np.concatenate([Hg, Hs, -Hs], axis=1)     # (nl, nvar)
            f0 = -self.PTDF @ pd_mw
            lim = self.rate_a.copy()
            fin = np.isfinite(lim)
            A_ub = np.concatenate([Hblock[fin], -Hblock[fin]], axis=0)
            b_ub = np.concatenate([lim[fin] - f0[fin], lim[fin] + f0[fin]])

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=Aeq, b_eq=beq,
                      bounds=bounds, method="highs")
        if not res.success:
            res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
        xsol = res.x if res.x is not None else np.zeros(na + 2 * nb)
        pg = np.zeros(ng)
        pg[act] = xsol[:na]
        shed = xsol[na : na + nb]
        spill = xsol[na + nb :]
        p_inj = np.zeros(nb)
        np.add.at(p_inj, self.gen_bus, pg)
        p_inj = p_inj - pd_mw + shed - spill
        return {
            "pg": pg, "p_inj": p_inj, "f": self.PTDF @ p_inj,
            "cost": float(self.cost_lin[act] @ xsol[:na]),
            "shed": float(shed.sum()), "spill": float(spill.sum()),
            "status": res.status, "success": bool(res.success),
        }


def build_ybus(case: MatpowerCase, grid: DCGrid) -> np.ndarray:
    """Complex bus admittance matrix including shunts and transformer taps."""
    nb, nl = case.nb, case.nl
    br = case.branch
    status = np.nan_to_num(br[:, BR_STATUS], nan=1.0)
    r, x, b = br[:, BR_R], br[:, BR_X], np.nan_to_num(br[:, BR_B], nan=0.0)
    ys = status / (r + 1j * x)
    tap = np.nan_to_num(br[:, TAP], nan=1.0)
    tap = np.where(tap == 0.0, 1.0, tap)
    shift = np.deg2rad(np.nan_to_num(br[:, SHIFT], nan=0.0))
    tapc = tap * np.exp(1j * shift)

    Ytt = ys + 1j * b / 2.0
    Yff = Ytt / (tapc * np.conj(tapc))
    Yft = -ys / np.conj(tapc)
    Ytf = -ys / tapc

    Y = np.zeros((nb, nb), dtype=complex)
    f, t = grid.fbus, grid.tbus
    np.add.at(Y, (f, f), Yff)
    np.add.at(Y, (f, t), Yft)
    np.add.at(Y, (t, f), Ytf)
    np.add.at(Y, (t, t), Ytt)
    Ysh = (np.nan_to_num(case.bus[:, GS], nan=0.0)
           + 1j * np.nan_to_num(case.bus[:, BS], nan=0.0)) / case.baseMVA
    Y[np.arange(nb), np.arange(nb)] += Ysh
    return Y
