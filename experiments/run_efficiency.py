"""Accuracy-vs-compute frontier, with compute measured three ways.

Primary axis is **NFE** and **analytic FLOPs**: exactly countable integers that
reproduce on any machine.  Wall-clock latency is secondary.  Measured energy is
tertiary, explicitly whole-system and device-specific, taken from SMC battery
telemetry with an idle baseline subtracted -- and it is only recorded when the
machine is actually on battery, because otherwise the gauge reports charging
current and the number would be meaningless.

We claim no novelty for adaptive stepping; `dopri5` at matched tolerance is the
reference, not a straw man.
"""
from __future__ import annotations

import argparse, json, os, pickle, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hfm.eval.metrics import energy_score, crps_ensemble
from hfm.eval.energy import idle_baseline, measure_energy, system_power_w
from experiments.run_main import method_zoo, sample_chunked

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="grid")
    ap.add_argument("--methods", nargs="*", default=["HFM (ours)", "FM", "FM+PCFM"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--energy-days", type=int, default=16)
    ap.add_argument("--max-test-days", type=int, default=0,
                    help="cap held-out days (smoke testing); 0 = use all")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "efficiency.jsonl"))
    a = ap.parse_args()

    with open(os.path.join(ROOT, "data", "processed", f"{a.suite}.pkl"), "rb") as f:
        ds = pickle.load(f)
    tr, va, te = ds.split()
    Xtr = np.concatenate([ds.X[tr], ds.X[va]], 0).astype(np.float32)
    Ctr = np.concatenate([ds.C[tr], ds.C[va]], 0).astype(np.float32)
    Cte = ds.C[te].astype(np.float32)
    obs = ds.X[te].astype(np.float32)
    if a.max_test_days:
        Cte, obs = Cte[:a.max_test_days], obs[:a.max_test_days]

    grid_fixed = [("euler", k) for k in (2, 5, 10, 20, 50, 100)] \
               + [("heun", k) for k in (2, 5, 10, 25, 50)] \
               + [("rk4", k) for k in (2, 5, 10, 25)]
    grid_adapt = [("dopri5", tol) for tol in (1e-2, 1e-3, 1e-4, 1e-5)]

    idle = idle_baseline(4.0)
    print(f"idle power baseline: {idle if idle is None else round(idle,2)} W "
          f"(None => on AC, energy not recorded)", flush=True)

    for seed in a.seeds:
        zoo = method_zoo(ds.T, ds.D, ds.C.shape[1], ds.constraint, ds.ineq,
                         a.device, a.epochs, seed, 50, 256, 4)
        for name in a.methods:
            factory, family, ours = zoo[name]
            gen = factory(); gen.fit(Xtr, Ctr)
            print(f"[{name}] trained", flush=True)
            for solver, param in grid_fixed + grid_adapt:
                kw = ({"solver": "dopri5", "rtol": param, "atol": param / 10, "steps": 1}
                      if solver == "dopri5" else {"solver": solver, "steps": param})
                t0 = time.time()
                outs, nfe, fl = [], 0, 0
                for b in range(0, Cte.shape[0], 16):
                    s, n, f = gen.sample(Cte[b:b + 16], a.M, seed=seed + b, **kw)
                    outs.append(np.asarray(s, np.float32)); nfe, fl = n, f
                ens = np.concatenate(outs, 0)
                wall = time.time() - t0
                es = energy_score(ens, obs); cr = crps_ensemble(ens, obs)
                v = ds.constraint.violation(torch.tensor(ens.reshape(-1, ds.T, ds.D)[:2000]))
                emeas = {}
                if idle is not None:
                    emeas = measure_energy(
                        lambda: gen.sample(Cte[:a.energy_days], a.M, seed=seed, **kw),
                        a.energy_days * a.M, idle_power_w=idle, repeats=3)
                rec = {"method": name, "seed": seed, "solver": solver,
                       "param": param, "nfe": int(nfe),
                       "flops_per_scenario": int(fl),
                       "energy_score": es, "crps": cr, "eq_max": v["eq_max"],
                       "wall_s": wall,
                       "latency_ms_per_scenario": 1e3 * wall / (ens.shape[0] * ens.shape[1]),
                       "ours": ours, **emeas}
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"  {solver:7s} p={param:<8} NFE={nfe:4d} ES={es:.5g} "
                      f"eq_max={v['eq_max']:.2e}", flush=True)
            del gen


if __name__ == "__main__":
    main()
