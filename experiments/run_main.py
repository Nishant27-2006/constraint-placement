"""Main benchmark: every generator, every seed, one protocol.

Protocol (fixed across all methods, following the evaluation shape established
by Dumas et al., Applied Energy 305:117871, 2022):

  * chronological 70/10/20 train/val/test split -- never random, because days
    are strongly autocorrelated and a random split leaks the future;
  * the same ensemble size M for every method, so we measure scenario *quality*
    and not scenario *budget*;
  * scoring rules against the realised held-out day, plus pooled distributional
    distances, calibration, physical violation, and exact compute accounting;
  * identical backbone, width, depth and optimiser for every neural method.
"""
from __future__ import annotations

import argparse, json, os, pickle, sys, time, traceback
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hfm.eval.metrics import all_metrics
from hfm.models.generators import (GaussianCopulaGen, KNNResampleGen, CVAEGen, CGANGen,
                                   DDPMGen, FlowMatchingGen, NormalizingFlowGen)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def method_zoo(T, D, cond_dim, affine, ineq, device, epochs, seed, steps, hidden, depth):
    """Name -> factory.  `ours` marks the rows this paper contributes."""
    nk = dict(T=T, D=D, cond_dim=cond_dim, affine=affine, ineq=ineq, device=device,
              epochs=epochs, seed=seed, hidden=hidden, depth=depth)
    fm = lambda **kw: (lambda: FlowMatchingGen(nk["T"], nk["D"], nk["cond_dim"],
                                               affine=affine, ineq=ineq, device=device,
                                               epochs=epochs, seed=seed, hidden=hidden,
                                               depth=depth, steps=steps, **kw))
    return {
        "GaussianCopula":   (lambda: GaussianCopulaGen(), "classical", False),
        "kNN-Historical":   (lambda: KNNResampleGen(), "classical", False),
        "cVAE":             (lambda: CVAEGen(T, D, cond_dim, affine=affine, device=device,
                                             epochs=epochs, seed=seed, hidden=hidden,
                                             depth=depth), "deep", False),
        "cWGAN-GP":         (lambda: CGANGen(T, D, cond_dim, affine=affine, device=device,
                                             epochs=max(1, epochs // 3), seed=seed,
                                             hidden=hidden, depth=depth), "deep", False),
        "NormFlow-RealNVP": (lambda: NormalizingFlowGen(T, D, cond_dim, affine=affine,
                                                        device=device, epochs=epochs,
                                                        seed=seed, hidden=hidden,
                                                        depth=depth), "deep", False),
        "DDPM":             (lambda: DDPMGen(T, D, cond_dim, affine=affine, device=device,
                                             epochs=epochs, seed=seed, hidden=hidden,
                                             depth=depth, n_steps=200), "deep", False),
        "FM":               (fm(constraint="none"), "flow", False),
        "FM+posthoc":       (fm(constraint="posthoc"), "flow", False),
        # The soft-penalty baseline is swept rather than guessed; the main table
        # reports its best lambda, which is the generous reading for that baseline.
        "FM+penalty(1)":    (fm(constraint="penalty", penalty_weight=1.0), "flow", False),
        "FM+penalty(10)":   (fm(constraint="penalty", penalty_weight=10.0), "flow", False),
        "FM+penalty(100)":  (fm(constraint="penalty", penalty_weight=100.0), "flow", False),
        "FM+penalty(1000)": (fm(constraint="penalty", penalty_weight=1000.0), "flow", False),
        "FM+PCFM":          (fm(constraint="pcfm"), "flow", False),
        "FM+DC3":           (fm(constraint="completion"), "flow", False),
        "FM+reduced":       (fm(constraint="reduced"), "flow", False),
        "HFM (ours)":       (fm(constraint="projected"), "flow", True),
    }


def sample_chunked(gen, C, M, seed, days_per_chunk=16):
    """Sample in chunks of test days to bound peak memory; NFE/FLOPs are
    per-scenario and identical across chunks."""
    outs, nfe, fl, extra = [], 0, 0, 0
    for a in range(0, C.shape[0], days_per_chunk):
        s, n, f = gen.sample(C[a:a + days_per_chunk], M, seed=seed + a)
        outs.append(np.asarray(s, dtype=np.float32))
        nfe, fl = n, f
        extra = getattr(gen, "extra_solves", 0)
    return np.concatenate(outs, 0), nfe, fl, extra


def run_one(name, factory, ds, tr, va, te, M, seed, device):
    X, C = ds.X, ds.C
    Xtr = np.concatenate([X[tr], X[va]], 0)
    Ctr = np.concatenate([C[tr], C[va]], 0)
    gen = factory()
    t0 = time.time()
    gen.fit(Xtr.astype(np.float32), Ctr.astype(np.float32))
    t_fit = time.time() - t0
    t0 = time.time()
    ens, nfe, fl, extra = sample_chunked(gen, C[te].astype(np.float32), M, seed)
    t_sample = time.time() - t0

    obs = X[te].astype(np.float32)

    # Guard: a diverged sampler must not be silently written into the results.
    # Anything this far outside the data's own range is a numerical failure, not
    # a bad model, and recording it would poison every aggregate downstream.
    if not np.isfinite(ens).all():
        raise FloatingPointError(f"{name}: non-finite samples")
    scale = float(np.abs(Xtr).max())
    smax = float(np.abs(ens).max())
    if smax > 1e4 * max(scale, 1e-9):
        raise FloatingPointError(
            f"{name}: sampler diverged (max|sample|={smax:.3e} vs data scale {scale:.3e})")

    res = all_metrics(ens, obs, pooled_real=Xtr.astype(np.float32))
    res.update(ds.constraint.violation(torch.tensor(ens.reshape(-1, ds.T, ds.D)[:4000])))
    if ds.ineq is not None:
        res.update(ds.ineq.violation(torch.tensor(ens.reshape(-1, ds.T, ds.D)[:4000])))
    n_scen = ens.shape[0] * ens.shape[1]
    res.update({
        "method": name, "seed": seed,
        "nfe": int(nfe), "flops_per_scenario": int(fl),
        "extra_projections": int(extra),
        "train_s": t_fit, "sample_s": t_sample,
        "latency_ms_per_scenario": 1e3 * t_sample / max(1, n_scen),
        "n_params": int(getattr(gen, "n_params", 0)),
        "n_test_days": int(obs.shape[0]), "M": int(M),
    })
    del gen, ens
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="measured")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with open(os.path.join(ROOT, "data", "processed", f"{a.suite}.pkl"), "rb") as f:
        ds = pickle.load(f)
    tr, va, te = ds.split()
    print(f"[{a.suite}] X={ds.X.shape} codim/hr={ds.constraint.codim//ds.T} "
          f"dof/hr={ds.constraint.dof//ds.T} train={tr.stop-tr.start} test={te.stop-te.start}",
          flush=True)

    out_path = a.out or os.path.join(ROOT, "results", f"main_{a.suite}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line); done.add((r["method"], r["seed"]))
            except Exception:
                pass

    for seed in a.seeds:
        zoo = method_zoo(ds.T, ds.D, ds.C.shape[1], ds.constraint, ds.ineq,
                         a.device, a.epochs, seed, a.steps, a.hidden, a.depth)
        names = a.methods or list(zoo.keys())
        for name in names:
            if (name, seed) in done:
                print(f"  skip {name} seed={seed} (done)", flush=True); continue
            factory, family, ours = zoo[name]
            try:
                t0 = time.time()
                r = run_one(name, factory, ds, tr, va, te, a.M, seed, a.device)
                r["family"] = family; r["ours"] = ours; r["suite"] = a.suite
                r["wall_s"] = time.time() - t0
                with open(out_path, "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"  {name:18s} seed={seed} ES={r['energy_score']:.4g} "
                      f"CRPS={r['crps']:.4g} eq_max={r['eq_max']:.2e} "
                      f"NFE={r['nfe']} {r['wall_s']:.0f}s", flush=True)
            except Exception as e:
                print(f"  {name:18s} seed={seed} FAILED: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()


if __name__ == "__main__":
    main()
