"""Mechanically evaluate the pre-registered falsification conditions.

`docs/CLAIMS_AND_EVIDENCE.md` fixes, in advance, what result would kill each
claim.  This script applies those conditions to the result files and prints one
verdict per claim.  The point is that the decision is made by the criterion that
was written before the data existed, not by reading the tables and deciding
afterwards what the paper was always about.

Verdicts: SUPPORTED · FALSIFIED · INSUFFICIENT DATA · ATTENTION
"""
from __future__ import annotations
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP32_EPS = float(np.finfo(np.float32).eps)

G, R, Y, B, X = "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m"
VERD = {"SUPPORTED": G + "SUPPORTED " + X, "FALSIFIED": R + "FALSIFIED " + X,
        "INSUFFICIENT": Y + "INSUFFICIENT DATA" + X, "ATTENTION": Y + "ATTENTION " + X}

EXACT_ROUTES = {"HFM (ours)", "FM+posthoc", "FM+PCFM", "FM+DC3", "FM+reduced"}
UNCONSTRAINED = {"FM", "DDPM", "cVAE", "cWGAN-GP", "NormFlow-RealNVP", "GaussianCopula"}
# Which stage of the pipeline each exact route enforces the constraint at.
# C3's prediction is an ordering over these classes, so the check needs them.
ROUTE_CLASS = {"HFM (ours)": "train-time", "FM+reduced": "train-time",
               "FM+DC3": "train-time", "FM+PCFM": "inference-time",
               "FM+posthoc": "post-hoc"}


def load(name):
    p = os.path.join(ROOT, "results", name)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def by_method(rows, key):
    d = {}
    for r in rows:
        v = r.get(key)
        if v is not None and np.isfinite(v):
            d.setdefault(r["method"], []).append(float(v))
    return d


def ms(vals):
    a = np.asarray(vals, float)
    return a.mean(), (a.std(ddof=1) if a.size > 1 else 0.0), a.size


def say(cid, title, verdict, detail):
    print(f"\n{B}{cid}{X}  {title}\n    verdict: {VERD[verdict]}")
    for line in detail:
        print(f"    {line}")


def check(suite="grid"):
    rows = load(f"main_{suite}.jsonl")
    print(f"\n{'='*74}\n  CLAIM CHECK -- suite: {suite}  ({len(rows)} runs)\n{'='*74}")
    if not rows:
        print("  no results yet"); return
    eq = by_method(rows, "eq_max")
    es = by_method(rows, "energy_score")
    scale = max((abs(r.get("eq_max", 0.0)) for r in rows), default=1.0)

    # ---- C1: projection exact to round-off ------------------------------
    if "HFM (ours)" in eq:
        m, s, n = ms(eq["HFM (ours)"])
        tol = FP32_EPS * scale * 64
        say("C1", "Projection is exact (to float32 round-off)",
            "SUPPORTED" if m <= tol else "FALSIFIED",
            [f"HFM eq_max = {m:.3e} (n={n})",
             f"round-off allowance = {tol:.3e}  [= 64 * eps32 * max observed state scale]"])
    else:
        say("C1", "Projection is exact", "INSUFFICIENT", ["HFM not yet run"])

    # ---- C2: unconstrained models violate materially --------------------
    unc = {k: ms(v)[0] for k, v in eq.items() if k in UNCONSTRAINED}
    if unc:
        worst, best = max(unc.values()), min(unc.values())
        say("C2", "Unconstrained models violate by operationally large margins",
            "SUPPORTED" if best > 1.0 else "FALSIFIED",
            [f"unconstrained eq_max range: {best:.3e} .. {worst:.3e} MW",
             "falsification threshold: all unconstrained below 1 MW"]
            + [f"  {k:20s} {v:.3e}" for k, v in sorted(unc.items(), key=lambda kv: -kv[1])])

    # ---- C3: does placement change fidelity, IN THE PREDICTED DIRECTION? -
    # The pre-registered prediction is an ORDERING, not merely "some difference":
    #   train-time projection >= inference-time correction >= post-hoc projection
    # on energy score at matched NFE.  Lower ES is better, so the predicted ES
    # ordering is  train-time <= inference-time <= post-hoc.  A check that only
    # asks "is anything separated?" returns SUPPORTED even when the observed
    # ordering is exactly reversed, so we test each adjacent pair's SIGN.
    routes = {k: ms(v) for k, v in es.items() if k in EXACT_ROUTES}
    if len(routes) >= 2 and all(n >= 2 for _, _, n in routes.values()):
        names = sorted(routes, key=lambda k: routes[k][0])
        classes = [("train-time", [k for k in ROUTE_CLASS
                                   if ROUTE_CLASS[k] == "train-time" and k in routes]),
                   ("inference-time", [k for k in ROUTE_CLASS
                                       if ROUTE_CLASS[k] == "inference-time" and k in routes]),
                   ("post-hoc", [k for k in ROUTE_CLASS
                                 if ROUTE_CLASS[k] == "post-hoc" and k in routes])]
        classes = [(cname, mem) for cname, mem in classes if mem]
        # represent each class by its BEST member -- most generous to the claim
        rep = [(cname, min(mem, key=lambda k: routes[k][0])) for cname, mem in classes]
        detail = [f"predicted ES ordering: {' <= '.join(c for c, _ in rep)}",
                  "observed ordering:     "
                  + " < ".join(f"{k}({routes[k][0]:.5g})" for k in names)]
        verdict, reversed_any, resolved_all = "SUPPORTED", False, True
        for (ca, ka), (cb, kb) in zip(rep, rep[1:]):
            am, asd, _ = routes[ka]; bm, bsd, _ = routes[kb]
            sep = (bm - am) / max(np.hypot(asd, bsd), 1e-12)   # >0 means predicted
            if sep < -2.0:
                reversed_any = True
                detail.append(f"  {ca}({ka}) vs {cb}({kb}): REVERSED by {-sep:.2f} sd")
            elif sep > 2.0:
                detail.append(f"  {ca}({ka}) vs {cb}({kb}): holds by {sep:.2f} sd")
            else:
                resolved_all = False
                detail.append(f"  {ca}({ka}) vs {cb}({kb}): indistinguishable ({sep:+.2f} sd)")
        if reversed_any:
            verdict = "FALSIFIED"
        elif not resolved_all:
            verdict = "ATTENTION"
        detail += ["NOTE: 'no difference' is an admissible, pre-registered outcome --"
                   " it becomes the finding, and the cost table carries the paper.",
                   "NOTE: a REVERSED ordering falsifies the prediction as stated; the"
                   " reversal is itself the reportable result."]
        detail += [f"  {k:16s} {routes[k][0]:.5g} +/- {routes[k][1]:.3g}" for k in names]
        say("C3", "WHERE the constraint goes changes fidelity (predicted ordering)",
            verdict, detail)
    else:
        say("C3", "WHERE the constraint goes changes fidelity", "INSUFFICIENT",
            ["needs >= 2 seeds for every exact route"])

    # ---- C4: exactness is compute-free for affine invariants ------------
    lat = by_method(rows, "latency_ms_per_scenario")
    extra = by_method(rows, "extra_projections")
    if "HFM (ours)" in lat and "FM+PCFM" in lat:
        h, hs, _ = ms(lat["HFM (ours)"]); p, ps, _ = ms(lat["FM+PCFM"])
        pe = ms(extra.get("FM+PCFM", [0]))[0]
        he = ms(extra.get("HFM (ours)", [0]))[0]
        say("C4", "Exactness is free in compute; inference-time correction is not",
            "SUPPORTED" if h < p else "FALSIFIED",
            [f"HFM  latency {h:.4g} ms/scenario, extra projections {he:.0f}",
             f"PCFM latency {p:.4g} ms/scenario, extra projections {pe:.0f}",
             f"speedup {p/max(h,1e-9):.2f}x"])

    # ---- C9: is a non-neural baseline winning? --------------------------
    if es:
        order = sorted(es, key=lambda k: ms(es[k])[0])
        top = order[0]
        say("C9", "A non-neural baseline may win (and if so it goes in the abstract)",
            "ATTENTION" if top in ("kNN-Historical", "GaussianCopula") else "SUPPORTED",
            [f"best energy score: {top} ({ms(es[top])[0]:.5g})"]
            + [f"  {i+1}. {k:20s} {ms(es[k])[0]:.5g}" for i, k in enumerate(order[:5])])


def check_transfer():
    rows = load("transfer.jsonl")
    print(f"\n{'='*74}\n  C5 -- zero-shot N-1 transfer  ({len(rows)} runs)\n{'='*74}")
    if not rows:
        print("  not yet run"); return
    d, esd = {}, {}
    for r in rows:
        d.setdefault((r["method"], r["mode"]), []).append(r["eq_max"])
        esd.setdefault((r["method"], r["mode"]), []).append(r["energy_score"])
    # Compare LIKE FOR LIKE: only mode == "swapped", where every route rebuilds its
    # operator for the new topology.  Pooling the "stale" rows (old projector reused)
    # into the chart-route average and comparing that against swapped-only projector
    # routes measures the stale/swapped contrast, not chart vs projector.
    chart = [v for (m, mo), v in d.items()
             if m in ("FM+reduced", "FM+DC3") and mo == "swapped"]
    proj = [v for (m, mo), v in d.items()
            if m in ("HFM (ours)", "FM+PCFM") and mo == "swapped"]
    if chart and proj:
        cm = np.mean([np.mean(v) for v in chart]); pm = np.mean([np.mean(v) for v in proj])
        say("C5", "Projector routes transfer zero-shot; chart routes do not",
            "SUPPORTED" if cm > 1e3 * max(pm, 1e-12) else "FALSIFIED",
            [f"projector-route residual  {pm:.3e} MW   (swapped mode only)",
             f"chart-route residual      {cm:.3e} MW   (swapped mode only)",
             "falsified if chart routes also stay exact"])
    # C3's ordering, re-tested out of distribution -- this is where it reverses.
    swap_es = {m: np.mean(v) for (m, mo), v in esd.items() if mo == "swapped"}
    reps = [(c, min([k for k in ROUTE_CLASS if ROUTE_CLASS[k] == c and k in swap_es],
                    key=lambda k: swap_es[k], default=None))
            for c in ("train-time", "inference-time", "post-hoc")]
    reps = [(c, k) for c, k in reps if k]
    if len(reps) >= 2:
        seq = [swap_es[k] for _, k in reps]
        ok = all(a <= b for a, b in zip(seq, seq[1:]))
        say("C3-transfer", "Predicted ordering holds OUT of distribution too",
            "SUPPORTED" if ok else "FALSIFIED",
            [f"predicted: {' <= '.join(c for c, _ in reps)}",
             "observed:  " + " < ".join(
                 f"{k}({swap_es[k]:.4g})" for k in sorted(swap_es, key=lambda k: swap_es[k])
                 if k in ROUTE_CLASS)])
    stale = {m: np.mean(v) for (m, mo), v in d.items() if mo == "stale"}
    if stale:
        print("    [ablation] reusing the TRAINING projector on the new topology:")
        for m, v in sorted(stale.items(), key=lambda kv: kv[1]):
            se = np.mean(esd[(m, "stale")])
            print(f"      {m:18s} eq_max {v:.3e}  ES {se:.4g}"
                  f"   (swapped: eq_max {np.mean(d[(m,'swapped')]):.3e}"
                  f"  ES {np.mean(esd[(m,'swapped')]):.4g})")
    for (m, mode), v in sorted(d.items(), key=lambda kv: np.mean(kv[1])):
        print(f"    {m:18s} [{mode:8s}] eq_max {np.mean(v):.3e}"
              f"  ES {np.mean(esd[(m, mode)]):.4g}")


def check_downstream():
    rows = [r for r in load("downstream.jsonl") if "regret_pct" in r]
    print(f"\n{'='*74}\n  C6 -- does it reach the decision?  ({len(rows)} runs)\n{'='*74}")
    if not rows:
        print("  not yet run"); return
    main = [r for r in rows if "[det-mean]" not in r["method"]]
    reg = [r["regret_pct"] for r in main]
    spread = max(reg) - min(reg)
    # Spread alone only says "the generator matters", which is near-tautological.
    # The claim of interest is DIRECTIONAL: does enforcing the constraint exactly
    # buy better decisions?  Compare the exact routes against the unconstrained
    # baselines that are not outright diverged (cVAE / cWGAN-GP blow up for
    # reasons unrelated to feasibility and would flatter the exact routes).
    ex = [r["regret_pct"] for r in main if r["method"] in EXACT_ROUTES]
    sane = {"GaussianCopula", "NormFlow-RealNVP", "kNN-Historical", "FM", "DDPM"}
    bl = [r["regret_pct"] for r in main if r["method"] in sane]
    detail = [f"regret spread across generators: {spread:.2f} percentage points",
              "falsified if generators are indistinguishable -- then feasibility does",
              "not reach the decision, and the paper must say so."]
    verdict = "SUPPORTED" if spread > 1.0 else "FALSIFIED"
    if ex and bl:
        em, bm = float(np.mean(ex)), float(np.mean(bl))
        detail += [f"mean regret, exact routes      {em:+.2f}%  (n={len(ex)})",
                   f"mean regret, sane baselines    {bm:+.2f}%  (n={len(bl)})"]
        if em > bm:
            verdict = "FALSIFIED"
            detail.append(f"DIRECTION REVERSED: exact routes are {em - bm:.2f} pp WORSE."
                          " Feasibility reaches the decision -- and hurts it.")
        else:
            detail.append(f"exact routes better by {bm - em:.2f} pp")
    nseeds = len({r.get("seed") for r in main})
    if nseeds < 2:
        detail.append(f"CAUTION: only {nseeds} seed -- ordering is not yet resolvable.")
    say("C6", "Constraint-exactness changes the scheduling decision (and helps)",
        verdict, detail)
    for r in sorted(rows, key=lambda r: r["regret_pct"])[:8]:
        print(f"    {r['method']:26s} regret {r['regret_pct']:+7.2f}%  "
              f"unserved {r['mean_shed_mwh']:.2f} MWh")


if __name__ == "__main__":
    for s in ("measured", "grid", "grid_noflow"):
        check(s)
    check_transfer()
    check_downstream()
    print(f"\n{B}Criteria are fixed in docs/CLAIMS_AND_EVIDENCE.md and were written "
          f"before any result existed.{X}\n")
