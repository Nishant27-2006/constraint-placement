"""Export per-suite geometry (dimension, constraint rank, codimension) to JSON.

Run once after `scripts/build_datasets.py`.  The result record and the paper
tables read this file rather than re-loading the (large) processed pickles, so
that no geometric constant is ever typed by hand.
"""
from __future__ import annotations
import json, os, pickle, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "results", "suite_meta.json")


def main():
    out = {}
    for name in ["measured", "grid", "grid_noflow", "ac"]:
        p = os.path.join(ROOT, "data", "processed", f"{name}.pkl")
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            ds = pickle.load(f)
        D = int(ds.X.shape[-1])
        cs = ds.constraint
        rank = int(np.asarray(cs.rank).reshape(-1)[0]) if cs is not None else 0
        rec = {"D": D, "T": int(ds.X.shape[1]), "n_days": int(ds.X.shape[0]),
               "affine_rank_per_hour": rank, "affine_dof_per_hour": D - rank,
               "codim_frac": rank / D, "units": ds.units}
        man = getattr(ds, "manifold", None)
        if man is not None:
            rec["manifold_codim_per_hour"] = int(man.n_out)
            rec["codim_frac"] = float(man.n_out) / D
        # channel scale spread, which is what an orthogonal projector is blind to
        X = ds.X.reshape(-1, D)
        s = X.std(axis=0)
        s = s[s > 1e-12]
        rec["scale_ratio_max_over_median"] = float(s.max() / np.median(s))
        out[name] = rec
        print(f"  {name:12s} D={D:4d} codim={rec['codim_frac']:.3f} "
              f"scale-ratio={rec['scale_ratio_max_over_median']:.1f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
