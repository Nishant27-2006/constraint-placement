"""Does constraint-exactness change the *decision*?

Scenario generators are not deployed to score well on CRPS; they are deployed to
inform a scheduling decision.  We therefore run every method's scenarios through
a two-stage stochastic unit commitment and score the resulting commitment on the
**realised** day.  Only the final re-dispatch touches the truth, so the
comparison across generators is honest out-of-sample.

Reference points:
  * *perfect foresight* -- commit knowing the realised net load (lower bound);
  * *deterministic* -- commit on the ensemble mean (the standard industry
    straw man, and a strong reminder that scenarios exist for a reason).

Protocol controls, both of which reviewers reliably ask about:
  * the scenario count fed to the optimiser (S) is identical for every method;
  * the reduction from M generated members to S is the same fixed-seed random
    subset for every method, so we compare scenario *quality*, not budget or
    reduction strategy.
"""
from __future__ import annotations

import argparse, json, os, pickle, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hfm.grid.network import load_case, DCGrid
from hfm.downstream.uc import UCProblem, solve_stochastic_uc, evaluate_commitment
from experiments.run_main import method_zoo, sample_chunked

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _one_day(args):
    prob, scen, real = args
    r = solve_stochastic_uc(prob, scen)
    return evaluate_commitment(prob, r["u"], real)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="measured")
    ap.add_argument("--channel", default="CISO.netload")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--S", type=int, default=20)
    ap.add_argument("--n-days", type=int, default=60)
    ap.add_argument("--n-gen", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "downstream.jsonl"))
    a = ap.parse_args()

    with open(os.path.join(ROOT, "data", "processed", f"{a.suite}.pkl"), "rb") as f:
        ds = pickle.load(f)
    tr, va, te = ds.split()
    ch = ds.channels.index(a.channel)

    grid = DCGrid(load_case(os.path.join(ROOT, "data", "raw", "pglib_opf_case118_ieee.m")))
    prob = UCProblem.from_grid(grid, n_gen=a.n_gen)
    # scale the fleet so peak capacity is a realistic reserve margin over peak net load
    peak = float(ds.X[tr, :, ch].max())
    k = 1.25 * peak / prob.pmax.sum()
    prob.pmax = prob.pmax * k; prob.pmin = prob.pmin * k
    prob.ramp = prob.ramp * k; prob.c_nl = prob.c_nl * k
    print(f"UC fleet scaled by {k:.2f}: capacity {prob.pmax.sum():.0f} MW vs peak "
          f"net load {peak:.0f} MW ({a.n_gen} units)", flush=True)

    day_idx = np.linspace(te.start, te.stop - 1, a.n_days).astype(int)
    real = ds.X[day_idx, :, ch]
    Cte = ds.C[day_idx].astype(np.float32)
    rng = np.random.default_rng(0)
    pick = rng.choice(a.M, size=a.S, replace=False)   # same reduction for every method

    import concurrent.futures as cf
    def run_set(tag, scen_per_day, seed=0):
        t0 = time.time()
        args = [(prob, scen_per_day[i], real[i]) for i in range(len(day_idx))]
        outs = []
        with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(_one_day, args, chunksize=1):
                outs.append(r)
        cost = float(np.mean([o["cost"] for o in outs]))
        shed = float(np.mean([o["shed"] for o in outs]))
        rec = {"method": tag, "seed": seed, "mean_cost": cost, "mean_shed_mwh": shed,
               "frac_days_shed": float(np.mean([o["shed"] > 1e-6 for o in outs])),
               "solve_s": time.time() - t0, "S": a.S, "n_days": len(day_idx)}
        return rec

    # Resume support: this sweep is many hours long and the file is rewritten
    # whole after each seed, so a restart without this would clobber the seeds
    # already finished.  Reload them and skip the (method, seed) pairs we have.
    recs, done = [], set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            recs.append(r)
            done.add((r["method"], r.get("seed")))
    pf = next((r for r in recs if r["method"] == "PerfectForesight"), None)
    if pf is None:
        pf = run_set("PerfectForesight", real[:, None, :].repeat(a.S, axis=1))
        recs.append(pf)
    else:
        print("  PerfectForesight reused from previous run", flush=True)
    if recs and done:
        print(f"  resuming: {len(done)} (method, seed) pairs already done", flush=True)
    print(f"  PerfectForesight cost={pf['mean_cost']:.0f} shed={pf['mean_shed_mwh']:.2f}", flush=True)

    zoo = method_zoo(ds.T, ds.D, ds.C.shape[1], ds.constraint, ds.ineq,
                     a.device, a.epochs, a.seeds[0], 50, 256, 4)
    names = a.methods or list(zoo.keys())
    for seed in a.seeds:
        zoo = method_zoo(ds.T, ds.D, ds.C.shape[1], ds.constraint, ds.ineq,
                         a.device, a.epochs, seed, 50, 256, 4)
        for name in names:
            if (name, seed) in done:
                print(f"  skip {name} seed={seed} (done)", flush=True); continue
            factory, family, ours = zoo[name]
            gen = factory()
            Xtr = np.concatenate([ds.X[tr], ds.X[va]], 0).astype(np.float32)
            Ctr = np.concatenate([ds.C[tr], ds.C[va]], 0).astype(np.float32)
            gen.fit(Xtr, Ctr)
            ens, _, _, _ = sample_chunked(gen, Cte, a.M, seed)
            scen = ens[:, :, :, ch][:, pick, :]              # (n_days, S, 24)
            rec = run_set(name, scen, seed); rec["ours"] = ours
            rec["regret_pct"] = 100.0 * (rec["mean_cost"] / pf["mean_cost"] - 1.0)
            recs.append(rec)
            print(f"  {name:18s} cost={rec['mean_cost']:.0f} "
                  f"regret={rec['regret_pct']:+.2f}% shed={rec['mean_shed_mwh']:.2f} "
                  f"days_shed={rec['frac_days_shed']*100:.0f}%", flush=True)
            det = ens[:, :, :, ch].mean(axis=1)[:, None, :].repeat(a.S, axis=1)
            rd = run_set(name + " [det-mean]", det, seed)
            rd["regret_pct"] = 100.0 * (rd["mean_cost"] / pf["mean_cost"] - 1.0)
            recs.append(rd)
            del gen, ens
        with open(a.out, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
