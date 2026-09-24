"""Suite C: generative modelling on the nonlinear AC power-flow manifold.

Two questions:
 1. Does tangential projection plus Gauss-Newton retraction keep samples on the
    manifold, and how does the residual scale with step count?
 2. Do the drift rates of Proposition 2 actually hold for a *learned* field?
    The constants depend on v_theta, not only on g, so we measure rather than assert.
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hfm.eval.metrics import energy_score, crps_ensemble
from hfm.models.generators import FlowMatchingGen
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--M", type=int, default=25)
    ap.add_argument("--device", default="cpu")   # autograd Jacobians: CPU is safer
    ap.add_argument("--max-test-days", type=int, default=0)
    ap.add_argument("--steps-grid", nargs="*", type=int, default=[5, 10, 20, 50, 100])
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--retract-iters", type=int, default=3,
                    help="Gauss-Newton budget per retraction (NonlinearManifold.retract_iters)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "ac.jsonl"))
    a = ap.parse_args()

    # The manifold path solves damped normal equations (cond ~1e7) and takes
    # autograd Jacobians; on MPS this silently produces non-finite weights and
    # the whole variant samples as NaN.  Verified: cpu -> 0%% NaN, mps -> 100%%.
    # The pipeline scripts pass --device mps for every stage, so override here
    # rather than trusting the caller.
    if a.device == "mps":
        print("[ac] WARNING: --device mps is unsafe for the manifold path "
              "(non-finite weights -> NaN samples); falling back to cpu", flush=True)
        a.device = "cpu"

    with open(os.path.join(ROOT, "data", "processed", "ac.pkl"), "rb") as f:
        ds = pickle.load(f)
    man = ds.manifold
    man.retract_iters = a.retract_iters
    tr, va, te = ds.split()
    Xtr = np.concatenate([ds.X[tr], ds.X[va]], 0).astype(np.float32)
    Ctr = np.concatenate([ds.C[tr], ds.C[va]], 0).astype(np.float32)
    Cte, obs = ds.C[te].astype(np.float32), ds.X[te].astype(np.float32)
    if a.max_test_days:
        Cte, obs = Cte[:a.max_test_days], obs[:a.max_test_days]
    print(f"[ac] X={ds.X.shape} manifold codim={man.n_out}/hour "
          f"NR residual on data {ds.meta['nr_residual']:.2e} retract_iters={man.retract_iters}", flush=True)

    variants = [("FM", "none", 0), ("FM+retract-final", "none", -1),
                ("Manifold-FM (ours)", "manifold", 1)]
    if a.methods:
        variants = [v for v in variants if v[0] in a.methods]
    for label, constraint, retract in variants:
        gen = FlowMatchingGen(ds.T, ds.D, ds.C.shape[1], affine=None, manifold=man,
                              constraint=constraint, device=a.device, epochs=a.epochs,
                              hidden=192, depth=3, batch=64, seed=0)
        gen.retract_every = max(retract, 0)
        gen.fit(Xtr, Ctr)
        for steps in a.steps_grid:
            outs = []
            for b in range(0, Cte.shape[0], 8):
                s, nfe, fl = gen.sample(Cte[b:b + 8], a.M, seed=b, steps=steps)
                outs.append(np.asarray(s, np.float32))
            ens = np.concatenate(outs, 0)
            if retract == -1:
                # retract the WHOLE ensemble, so the energy score below is measured
                # on the same samples the violation metric sees
                shp = ens.shape
                flat = torch.tensor(ens.reshape(-1, ds.T, ds.D), dtype=torch.float64)
                ens = man.retract(flat, iters=a.retract_iters).numpy().astype(np.float32).reshape(shp)
            xt = torch.tensor(ens.reshape(-1, ds.T, ds.D)[:1500], dtype=torch.float64)
            v = man.violation(xt)
            rec = {"method": label, "steps": steps, "nfe": int(nfe),
                   "nl_max": v["nl_max"], "nl_rmse": v["nl_rmse"],
                   "energy_score": energy_score(ens, obs), "crps": crps_ensemble(ens, obs),
                   "ours": constraint == "manifold",
                   "retract_iters": a.retract_iters}
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  {label:20s} steps={steps:4d} |g|_max={v['nl_max']:.3e} "
                  f"rmse={v['nl_rmse']:.3e} ES={rec['energy_score']:.5g}", flush=True)
        del gen


if __name__ == "__main__":
    main()
