"""Zero-shot transfer to an N-1 contingency topology.

An outage changes the PTDF, hence the *constraint set*, without changing the
data-generating process for demand or weather.  Operators face this constantly.
The question is which constraint-handling route survives it **without retraining**:

  * a projector can simply be swapped for the contingency one (ours, and PCFM);
  * a soft penalty was fitted to the base residual and has no notion of the new one;
  * a nullspace chart or a DC3 completion was fixed before training, so the
    network's outputs are coordinates in a chart that no longer describes the
    feasible set --- we evaluate both "keep the stale chart" and "swap in the new
    chart", since neither is obviously the intended reading.

The headline claim is about *feasibility*, not fidelity: every method's
distribution degrades under a topology it never saw, but only some can still
emit Kirchhoff-consistent states for the new network.
"""
from __future__ import annotations

import argparse, json, os, pickle, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hfm.grid.network import load_case, DCGrid
from hfm.grid.constraints import build_blockdiag_affine
from hfm.data.build import _dispatch_day
from hfm.eval.metrics import all_metrics
from experiments.run_main import method_zoo, sample_chunked

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def contingency_constraint(grid, D, nb, nl, zero_inj, T):
    """Kirchhoff constraints for a (possibly outaged) topology."""
    rows = []
    bal = np.zeros(D); bal[:nb] = 1.0
    rows.append((bal, 0.0))
    for i in zero_inj:
        c = np.zeros(D); c[i] = 1.0
        rows.append((c, 0.0))
    for j in range(nl):
        c = np.zeros(D); c[nb + j] = 1.0; c[:nb] = -grid.PTDF[j]
        rows.append((c, 0.0))
    return build_blockdiag_affine([rows] * T, D, T)


def rebuild_truth(grid_out, ds, te, meta, workers=12, limit=0):
    """Re-dispatch the held-out days on the outaged network."""
    import concurrent.futures as cf
    nb = grid_out.nb
    w = grid_out.pd_base / max(grid_out.pd_base.sum(), 1e-9)
    raw = meta["raw_inputs"]
    args = [(grid_out, w, raw["lt"][i], raw["scf"][i], raw["wcf"][i],
             np.array(meta["solar_buses"]), np.array(meta["wind_buses"]),
             np.array(raw["s_share"]), np.array(raw["w_share"]),
             raw["scap"], raw["wcap"]) for i in range(te.start, te.stop)]
    if limit:
        args = args[:limit]
    P = np.zeros((len(args), 24, nb))
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (pi, _) in enumerate(ex.map(_dispatch_day, args, chunksize=8)):
            P[i] = pi
    P = P - P.sum(axis=2, keepdims=True) / nb
    return np.concatenate([P, P @ grid_out.PTDF.T], axis=2)


_CHART_ATTRS = ("basis", "x_part", "c_free", "c_dep", "c_Ainv", "c_Af", "c_b",
                "aff_s", "aff_dev")


def _snapshot(gen):
    return {k: getattr(gen, k) for k in _CHART_ATTRS if hasattr(gen, k)}


def _restore(gen, snap):
    for k, v in snap.items():
        setattr(gen, k, v)
    if getattr(gen, "model", None) is not None and gen.model.affine is not None:
        gen.model.affine = snap.get("aff_dev", gen.model.affine)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-contingencies", type=int, default=8)
    ap.add_argument("--methods", nargs="*", default=[
        "HFM (ours)", "FM+PCFM", "FM+posthoc", "FM+penalty(100)", "FM+reduced",
        "FM+DC3", "FM", "DDPM"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-test-days", type=int, default=0,
                    help="cap held-out days (smoke testing / runtime); 0 = all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "transfer.jsonl"))
    a = ap.parse_args()

    with open(os.path.join(ROOT, "data", "processed", "grid.pkl"), "rb") as f:
        ds = pickle.load(f)
    if "raw_inputs" not in ds.meta:
        print("ERROR: grid.pkl lacks raw_inputs; rebuild with the updated builder.")
        sys.exit(2)
    grid = DCGrid(load_case(os.path.join(ROOT, "data", "raw", f"{ds.meta['case']}.m")))
    tr, va, te = ds.split()
    nb, nl = ds.meta["nb"], ds.meta["nl"]
    zero_inj = ds.meta["zero_injection_buses"]

    cands = [j for j in range(nl) if not grid.is_bridge(j)]
    shift = [(j, float(np.abs(grid.outage(j).PTDF - grid.PTDF).max())) for j in cands[:60]]
    shift.sort(key=lambda z: -z[1])
    chosen = [j for j, _ in shift[:a.n_contingencies]]
    print("contingencies:", chosen, flush=True)

    Xtr = np.concatenate([ds.X[tr], ds.X[va]], 0).astype(np.float32)
    Ctr = np.concatenate([ds.C[tr], ds.C[va]], 0).astype(np.float32)
    Cte = ds.C[te].astype(np.float32)
    if a.max_test_days:
        Cte = Cte[:a.max_test_days]

    for seed in a.seeds:
        zoo = method_zoo(ds.T, ds.D, ds.C.shape[1], ds.constraint, ds.ineq,
                         a.device, a.epochs, seed, 50, 256, 4)
        for name in a.methods:
            factory, family, ours = zoo[name]
            gen = factory()
            t0 = time.time(); gen.fit(Xtr, Ctr); t_fit = time.time() - t0
            base_chart = _snapshot(gen)
            print(f"[{name}] trained in {t_fit:.0f}s", flush=True)
            for j in chosen:
                g_out = grid.outage(j)
                con_new = contingency_constraint(g_out, ds.D, nb, nl, zero_inj, ds.T)
                Xtruth = rebuild_truth(g_out, ds, te, ds.meta, workers=a.workers,
                                       limit=a.max_test_days).astype(np.float32)
                for mode in (["stale", "swapped"] if name in ("FM+reduced", "FM+DC3")
                             else ["swapped"]):
                    g2 = gen
                    if mode == "swapped" and hasattr(gen, "aff_dev"):
                        mu = torch.tensor(gen.mu, dtype=torch.float64)
                        sd = torch.tensor(gen.sd, dtype=torch.float64)
                        g2.aff_s = con_new.standardise(mu, sd)
                        g2.aff_dev = g2.aff_s.to(a.device, torch.float32)
                        if g2.model.affine is not None:
                            g2.model.affine = g2.aff_dev
                        # A chart-based model decodes through a basis fixed at
                        # training time.  "Swapped" means we rebuild that chart
                        # for the contingency and decode the *unchanged* network
                        # output through it.  The result is exactly feasible --
                        # any point of the new chart is -- but the latent code
                        # now means something different, so the question the
                        # experiment really asks is what happens to fidelity,
                        # not to feasibility.
                        if gen.constraint == "reduced":
                            g2.basis = g2.aff_s.null_basis().to(a.device, torch.float32)
                            g2.x_part = g2.aff_s.x_part.to(a.device, torch.float32)
                        elif gen.constraint == "completion":
                            comp = g2.aff_s.completion()
                            g2.c_free = torch.tensor(np.stack(comp["free"]), device=a.device)
                            g2.c_dep = torch.tensor(np.stack(comp["dep"]), device=a.device)
                            g2.c_Ainv = torch.tensor(np.stack(comp["Ad_inv"]),
                                                     device=a.device, dtype=torch.float32)
                            g2.c_Af = torch.tensor(np.stack(comp["Af"]),
                                                   device=a.device, dtype=torch.float32)
                            g2.c_b = torch.tensor(np.stack(comp["b_act"]),
                                                  device=a.device, dtype=torch.float32)
                    ens, nfe, fl, extra = sample_chunked(g2, Cte, a.M, seed)
                    if mode == "swapped" and gen.constraint in ("reduced", "completion"):
                        _restore(gen, base_chart)
                    v = con_new.violation(torch.tensor(ens.reshape(-1, ds.T, ds.D)[:3000]))
                    m = all_metrics(ens, Xtruth, pooled_real=Xtruth)
                    rec = {"method": name, "seed": seed, "branch": int(j), "mode": mode,
                           "nfe": int(nfe), "ours": ours, **v,
                           "energy_score": m["energy_score"], "crps": m["crps"],
                           "variogram_score": m["variogram_score"]}
                    with open(a.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"   br{j:3d} {mode:8s} eq_max={v['eq_max']:.3e} "
                          f"eq_rmse={v['eq_rmse']:.3e} ES={m['energy_score']:.4g}", flush=True)
            del gen


if __name__ == "__main__":
    main()
