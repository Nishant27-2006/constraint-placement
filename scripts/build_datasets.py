"""Build and cache the benchmark suites.  Run once; everything else loads the cache."""
from __future__ import annotations
import argparse, json, os, pickle, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hfm.data.eia930 import load_ba, verify_identities, to_daily_blocks
from hfm.data.build import build_measured_suite, build_grid_suite, build_ac_suite
from hfm.grid.network import load_case, DCGrid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
BAS = {"CISO": -8, "ERCO": -6, "PJM": -5, "MISO": -6, "SWPP": -6, "BPAT": -8}
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
# EIA changed the BALANCE schema from 2024 H2 (solar/wind split into "with" and
# "without Integrated Battery Storage").  Rather than splice two definitions of
# the same channel we use the homogeneous record that ends just before it.
END = "2024-07-01"
MAX_NAN_FRAC = 0.01
CORE = ["demand", "coal", "gas", "nuclear", "oil", "hydro", "solar", "wind", "other"]


def build_measured(verbose=True):
    blocks, per_ba_dates, diag = {}, {}, {}
    for ba, shift in BAS.items():
        df = load_ba(os.path.join(RAW, "eia930"), ba, adjusted=True, years=YEARS)
        df = df.loc[:END]
        diag[ba] = verify_identities(df)
        # a channel a BA never reports is structurally absent, not missing data
        keep = [c for c in CORE if c in df.columns and df[c].isna().mean() < MAX_NAN_FRAC]
        dropped = [c for c in CORE if c in df.columns and c not in keep]
        df = df[keep]
        X, dates, names = to_daily_blocks(df, tz_shift_hours=shift)
        blocks[ba] = (X, names)
        per_ba_dates[ba] = dates
        if verbose:
            print(f"  {ba:5s} days={len(dates):5d} channels={names}"
                  + (f"  [dropped: {dropped}]" if dropped else ""))
    common = sorted(set.intersection(*[set(d.tolist()) for d in per_ba_dates.values()]))
    common = np.array(common, dtype="datetime64[D]")
    aligned = {}
    for ba in BAS:
        sel = np.isin(per_ba_dates[ba], common)
        order = np.argsort(per_ba_dates[ba][sel])
        aligned[ba] = (blocks[ba][0][sel][order], blocks[ba][1])
    if verbose:
        print(f"  common complete days across {len(BAS)} BAs: {len(common)} "
              f"({common.min()} .. {common.max()})")
    ds = build_measured_suite(aligned, common)
    ds.meta["identity_diagnostics"] = {k: {kk: vv for kk, vv in v.items()
                                           if kk != "fuels_present"} for k, v in diag.items()}
    return ds


def build_grid(case="pglib_opf_case118_ieee", ba="CISO", n_days=None, verbose=True,
               include_flows=True, workers=6):
    grid = DCGrid(load_case(os.path.join(RAW, f"{case}.m")))
    df = load_ba(os.path.join(RAW, "eia930"), ba, adjusted=True, years=YEARS)
    keep = [c for c in ["demand", "solar", "wind"] if c in df.columns]
    X, dates, names = to_daily_blocks(df.loc[:END][keep], tz_shift_hours=BAS[ba])
    if n_days:
        X, dates = X[:n_days], dates[:n_days]
    demand = X[:, :, names.index("demand")]
    solar = X[:, :, names.index("solar")]
    wind = X[:, :, names.index("wind")]
    scf = np.clip(solar / max(solar.max(), 1e-9), 0, 1)
    wcf = np.clip(wind / max(wind.max(), 1e-9), 0, 1)
    if verbose:
        print(f"  grid suite from {ba}: {len(dates)} days, case {case} "
              f"(nb={grid.nb}, nl={grid.nl})")
    ds = build_grid_suite(grid, demand, scf, wcf, dates, include_flows=include_flows,
                          verbose=verbose, workers=workers)
    ds.meta["case"] = case
    ds.meta["source_ba"] = ba
    return ds, grid


def build_ac(case="case57", ba="CISO", n_days=None, verbose=True):
    """Suite C.  A smaller network than Suite B because the AC tangent
    projector is a dense (2n x 4n) Jacobian per hour."""
    from hfm.grid.network import load_case as _lc
    c = _lc(os.path.join(RAW, f"{case}.m"))
    grid = DCGrid(c)
    df = load_ba(os.path.join(RAW, "eia930"), ba, adjusted=True, years=YEARS)
    keep = [x for x in ["demand", "solar", "wind"] if x in df.columns]
    X, dates, names = to_daily_blocks(df.loc[:END][keep], tz_shift_hours=BAS[ba])
    if n_days:
        X, dates = X[:n_days], dates[:n_days]
    dem = X[:, :, names.index("demand")]
    sol = X[:, :, names.index("solar")]
    wnd = X[:, :, names.index("wind")]
    scf = np.clip(sol / max(sol.max(), 1e-9), 0, 1)
    wcf = np.clip(wnd / max(wnd.max(), 1e-9), 0, 1)
    if verbose:
        print(f"  ac suite from {ba}: {len(dates)} days, case {case} (nb={c.nb})")
    return build_ac_suite(c, grid, dem, scf, wcf, dates, verbose=verbose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="measured",
                    choices=["measured", "grid", "grid_noflow", "ac", "all"])
    ap.add_argument("--n-days", type=int, default=None)
    ap.add_argument("--no-flows", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.which in ("measured", "all"):
        print("[measured suite]")
        t0 = time.time(); ds = build_measured(); print(f"  built in {time.time()-t0:.1f}s")
        print("  X", ds.X.shape, "codim/hour", ds.constraint.codim // ds.T,
              "dof/hour", ds.constraint.dof // ds.T)
        print("  self-consistency:", {k: f"{v:.3e}" for k, v in ds.verify(tol=1e-6).items()})
        with open(os.path.join(OUT, "measured.pkl"), "wb") as f: pickle.dump(ds, f)
        print("  saved ->", os.path.join(OUT, "measured.pkl"))
    if a.which in ("grid", "all"):
        print("[grid suite]")
        t0 = time.time()
        ds, grid = build_grid(n_days=a.n_days, include_flows=not a.no_flows)
        print(f"  built in {time.time()-t0:.1f}s")
        print("  X", ds.X.shape, "codim/hour", ds.constraint.codim // ds.T,
              "dof/hour", ds.constraint.dof // ds.T)
        print("  self-consistency:", {k: f"{v:.3e}" for k, v in ds.verify(tol=1e-4).items()})
        with open(os.path.join(OUT, "grid.pkl"), "wb") as f: pickle.dump(ds, f)
        print("  saved ->", os.path.join(OUT, "grid.pkl"))


    if a.which in ("grid_noflow", "all"):
        # Same intrinsic dimension (107 dof/hour), far lower codimension: dropping
        # the 186 branch-flow channels removes 186 dimensions AND the 186
        # constraints that determine them.  This is the benchmark's controlled
        # axis -- codimension varied with the data's true dof held fixed.
        print("[grid suite, no flow channels]")
        t0 = time.time()
        ds, grid = build_grid(n_days=a.n_days, include_flows=False)
        print(f"  built in {time.time()-t0:.1f}s")
        print("  X", ds.X.shape, "codim/hour", ds.constraint.codim // ds.T,
              "dof/hour", ds.constraint.dof // ds.T)
        print("  self-consistency:", {k: f"{v:.3e}" for k, v in ds.verify(tol=1e-4).items()})
        with open(os.path.join(OUT, "grid_noflow.pkl"), "wb") as f: pickle.dump(ds, f)
        print("  saved ->", os.path.join(OUT, "grid_noflow.pkl"))

    if a.which in ("ac", "all"):
        print("[ac suite]")
        t0 = time.time(); ds = build_ac(n_days=a.n_days); print(f"  built in {time.time()-t0:.1f}s")
        print("  X", ds.X.shape, "manifold codim/hour", ds.manifold.n_out,
              "kept", ds.meta["days_kept"], "of", ds.meta["days_total"], "days")
        print(f"  NR residual on kept data: {ds.meta['nr_residual']:.3e} p.u.")
        with open(os.path.join(OUT, "ac.pkl"), "wb") as f: pickle.dump(ds, f)
        print("  saved ->", os.path.join(OUT, "ac.pkl"))


if __name__ == "__main__":
    main()
