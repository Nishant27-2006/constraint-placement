"""The paper's figures, as publication-grade SVG built from `results/*.jsonl`.

Every figure: position encoding on a common scale, colourblind-safe Okabe-Ito
hues with a redundant marker shape, direct labels where they fit, an explicit
annotation of the insight, and ARIA title/desc for screen readers.
"""
from __future__ import annotations
import json, math, os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from svgfig import Figure, Scale, SERIES, OKABE, SHAPES, LabelPlacer, text_box

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
T_CRIT_9 = 2.262
SUITES = [("grid_noflow", "grid-noflow"), ("measured", "measured"), ("grid", "grid")]


def load(n):
    p = os.path.join(RES, n)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def paired_ci(rows, a, b, key="energy_score"):
    """Paired %-difference with a 95% CI. Returns (pct, lo, hi, t, n)."""
    by = defaultdict(dict)
    for r in rows:
        by[r["method"]][r["seed"]] = r
    seeds = sorted(set(by.get(a, {})) & set(by.get(b, {})))
    if len(seeds) < 2:
        return None
    d = np.array([by[a][s][key] - by[b][s][key] for s in seeds], float)
    base = np.mean([by[b][s][key] for s in seeds])
    se = d.std(ddof=1) / math.sqrt(len(d))
    t = d.mean() / se if se > 0 else float("inf")
    f = 100.0 / base
    return d.mean() * f, (d.mean() - T_CRIT_9 * se) * f, (d.mean() + T_CRIT_9 * se) * f, t, len(d)


def mean_of(rows, m, k="energy_score"):
    v = [r[k] for r in rows if r["method"] == m and r.get(k) is not None]
    return float(np.mean(v)) if v else float("nan")


# ===================================================================== FIG 1
def fig_signflip(meta, mains):
    """The central result: paired effect vs codimension, with 95% CIs."""
    METHODS = [("HFM (ours)", "train-time projection", SERIES[0], SHAPES[0]),
               ("FM+reduced", "nullspace chart", SERIES[1], SHAPES[1]),
               ("FM+DC3", "DC3 completion", SERIES[2], SHAPES[2]),
               ("FM+PCFM", "inference-time", SERIES[3], SHAPES[3])]
    data = {}
    for name, _, _, _ in METHODS:
        pts = []
        for s, _lab in SUITES:
            r = paired_ci(mains[s], name, "FM")
            if r:
                pts.append((meta[s]["codim_frac"], *r))
        data[name] = sorted(pts)

    allv = [v for pts in data.values() for p in pts for v in (p[2], p[3])]
    lo, hi = min(allv), max(allv)
    F = Figure(880, 470, 74, 178, 34, 62,
               "Effect of constraint placement versus codimension",
               "Paired change in energy score against unconstrained flow matching for "
               "four exact constraint routes, across three benchmark suites ordered by "
               "constraint codimension. Train-time projection crosses zero: it is "
               "significantly worse on one suite and significantly better on another.",
               "f1")
    sx = Scale(0.0, 0.72, F.x0, F.x1)
    sy = Scale(lo, hi, F.y1, F.y0, pad=0.12)
    F.frame()
    F.grid_y(sy, sy.ticks(6), lambda v: f"{v:+.0f}%")
    F.rule(sy, 0.0, "no effect")

    # suite positions get a tick + a two-line label (name, codimension)
    for s, lab in SUITES:
        c = meta[s]["codim_frac"]
        x = sx(c)
        F.o.append(f'<line x1="{x:.2f}" y1="{F.y1}" x2="{x:.2f}" y2="{F.y1+5}" '
                   f'class="grid"/>')
        F.label(x, F.y1 + 20, lab, "var(--muted)", "middle", "tick")
        F.label(x, F.y1 + 34, f"codim {c:.3f}", "var(--muted)", "middle", "note")

    # shaded band: "worse" half-plane, so the sign is readable without colour
    yz = sy(0.0)
    F.o.append(f'<rect x="{F.x0}" y="{F.y0}" width="{F.x1-F.x0}" '
               f'height="{max(yz-F.y0,0):.2f}" fill="var(--bad)" opacity=".05"/>')
    F.label(F.x0 + 8, F.y0 + 16, "worse than no constraint", "var(--bad)", "start", "note")

    lp = LabelPlacer(F)
    lp.reserve_text(F.x1 - 6, sy(0.0) - 7, "no effect", "rule-lbl", "end")
    lp.reserve_text(F.x0 + 8, F.y0 + 16, "worse than no constraint", "note")
    lp.reserve_text(F.x0 + 8, F.y1 - 12,
                    "Filled = significant at 5% (paired, 10 seeds). Bars are 95% CIs.",
                    "note")
    for name, pretty, col, shp in METHODS:
        pts = data[name]
        F.path([(sx(p[0]), sy(p[1])) for p in pts], col, 2.2, op=.9)
        for c, pct, clo, chi, t, n in pts:
            x = sx(c)
            F.errbar_v(x, sy(clo), sy(chi), col)
            sig = abs(t) > T_CRIT_9
            F.mark(shp, x, sy(pct), col, 6.0 if sig else 5.0, filled=sig,
                   title=f"{name} on codim {c:.3f}: {pct:+.2f}% "
                         f"(95% CI {clo:+.2f} to {chi:+.2f}, t={t:+.2f}, n={n})")
            lp.reserve_mark(x, sy(pct), 8)
            lp.reserve_box([x - 5, sy(chi) - 4, x + 5, sy(clo) + 4])
    # direct labels at the right-hand end, de-collided against everything placed
    RING = [(13, 4, "start"), (13, -11, "start"), (13, 18, "start"),
            (13, -24, "start"), (13, 31, "start"), (13, -37, "start"),
            (13, 44, "start")]
    for name, pretty, col, shp in sorted(METHODS, key=lambda m: data[m[0]][-1][1]):
        pts = data[name]
        lp.place(sx(pts[-1][0]), sy(pts[-1][1]), pretty, col, "series-lbl", ring=RING)

    F.axis_titles("Constraint codimension  r / D  →",
                  "Δ energy score vs unconstrained FM")
    F.note(F.x0 + 8, F.y1 - 12,
           ["Filled = significant at 5% (paired, 10 seeds). Bars are 95% CIs."],
           cls="note")
    return F.done()


# ===================================================================== FIG 2
def fig_pareto(mains):
    """Feasibility vs fidelity on one plane: the penalty family is dominated."""
    rows = mains["grid"]
    base = mean_of(rows, "FM")
    pens = [(lam, mean_of(rows, f"FM+penalty({lam})"),
             mean_of(rows, f"FM+penalty({lam})", "eq_max")) for lam in (1, 10, 100, 1000)]
    exact = [(n, mean_of(rows, n), mean_of(rows, n, "eq_max")) for n in
             ("HFM (ours)", "FM+reduced", "FM+DC3", "FM+PCFM", "FM+posthoc")]
    unc = ("FM", base, mean_of(rows, "FM", "eq_max"))

    F = Figure(880, 470, 74, 150, 34, 66,
               "Constraint violation versus distributional fidelity",
               "Worst-case constraint violation on a log axis against energy score for "
               "the penalty family, the exact routes and unconstrained flow matching. "
               "The exact routes occupy the lower-left corner; every penalty setting is "
               "dominated on both axes simultaneously.", "f2")
    xs = [p[2] for p in pens] + [p[2] for p in exact] + [unc[2]]
    ys = [p[1] for p in pens] + [p[1] for p in exact] + [unc[1]]
    sx = Scale(min(xs), max(xs), F.x0, F.x1, log=True, pad=0.06)
    sy = Scale(min(ys), max(ys), F.y1, F.y0, pad=0.10)
    F.frame()
    F.grid_y(sy, sy.ticks(6), lambda v: f"{v:.0f}")

    def logfmt(v):
        e = int(round(math.log10(v)))
        return {0: "1", 1: "10", 2: "100", 3: "1k"}.get(e, f"1e{e}")
    F.grid_x(sx, sx.ticks(), logfmt)

    # the region every exact route occupies
    band_r = sx(1e-2)
    F.o.append(f'<rect x="{F.x0+1}" y="{F.y0+1}" width="{band_r-F.x0:.2f}" '
               f'height="{F.y1-F.y0-2:.2f}" fill="{OKABE["green"]}" opacity=".055"/>')
    F.o.append(f'<line x1="{band_r:.2f}" y1="{F.y0+1}" x2="{band_r:.2f}" '
               f'y2="{F.y1-1}" stroke="{OKABE["green"]}" stroke-width="1.1" '
               f'stroke-dasharray="4 4" opacity=".55"/>')
    F.label(band_r - 8, F.y1 - 10, "exactly feasible", OKABE["green"], "end", "note")

    lp = LabelPlacer(F)
    lp.reserve_text(F.x0 + 8, F.y0 + 18, "Raising \u03bb walks the family up,", "note")
    lp.reserve_text(F.x0 + 8, F.y0 + 32, "never left \u2014 fidelity is spent,", "note")
    lp.reserve_text(F.x0 + 8, F.y0 + 46, "feasibility is not bought.", "note")
    lp.reserve_text(band_r - 8, F.y1 - 10, "exactly feasible", "note", "end")

    col_p, col_e = SERIES[3], SERIES[0]
    F.path([(sx(p[2]), sy(p[1])) for p in pens], col_p, 1.8, dash="4 4", op=.8)
    for lam, es, eq in pens:
        F.mark("diamond", sx(eq), sy(es), col_p, 6.0,
               title=f"penalty lambda={lam}: ES {es:.4g}, violation {eq:.3g} MW")
        lp.reserve_mark(sx(eq), sy(es), 8)
    F.mark("cross", sx(unc[2]), sy(unc[1]), "var(--muted)", 6.0,
           title=f"unconstrained FM: ES {unc[1]:.4g}, violation {unc[2]:.3g} MW")
    lp.reserve_mark(sx(unc[2]), sy(unc[1]), 8)
    for i, (n, es, eq) in enumerate(exact):
        ours = n == "HFM (ours)"
        F.mark(SHAPES[i % len(SHAPES)], sx(eq), sy(es), col_e if ours else SERIES[1],
               6.5 if ours else 5.2,
               title=f"{n}: ES {es:.4g}, violation {eq:.3g} MW")
        lp.reserve_mark(sx(eq), sy(es), 8)

    # labels last, so placement sees every mark
    for lam, es, eq in pens:
        lp.place(sx(eq), sy(es), f"\u03bb={lam}", col_p, "lbl")
    lp.place(sx(unc[2]), sy(unc[1]), "FM (unconstrained)", "var(--muted)", "lbl")
    order = sorted(exact, key=lambda e: e[1])
    for n, es, eq in order:
        ours = n == "HFM (ours)"
        lp.place(sx(eq), sy(es), n + (" \u2605" if ours else ""),
                 col_e if ours else "var(--muted)",
                 "series-lbl" if ours else "lbl")

    F.axis_titles("Worst-case violation  ||Ax−b||∞  (MW, log scale)  →",
                  "Energy score  ↓")
    F.note(F.x0 + 8, F.y0 + 18,
           ["Raising λ walks the family up,", "never left — fidelity is spent,",
            "feasibility is not bought."], cls="note")
    return F.done()


# ===================================================================== FIG 3
def fig_regret(down, measured):
    """Decision value tracks calibration, not the proper score."""
    reg, cov, esm = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in down:
        if r.get("regret_pct") is not None and "[det-mean]" not in r["method"]:
            reg[r["method"]].append(r["regret_pct"])
    for r in measured:
        cov[r["method"]].append(r["cov90"]); esm[r["method"]].append(r["energy_score"])
    EX = {"HFM (ours)", "FM+reduced", "FM+DC3", "FM+PCFM", "FM+posthoc"}
    pts = [(float(np.mean(cov[m])), float(np.mean(reg[m])), m, m in EX)
           for m in reg if m in cov]
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    zs = np.array([float(np.mean(esm[p[2]])) for p in pts])
    r_cov = float(np.corrcoef(xs, ys)[0, 1]); sp = spearmanr(xs, ys)
    r_es = float(np.corrcoef(zs, ys)[0, 1]); sp_es = spearmanr(zs, ys)

    F = Figure(880, 520, 76, 132, 34, 66,
               "Unit-commitment cost regret versus scenario calibration",
               "Scatter of two-stage stochastic unit-commitment cost regret against "
               "90 percent interval coverage for sixteen generators. Regret falls "
               "steeply as coverage rises; the exactly-feasible routes are the "
               "sharpest and the most expensive.", "f3")
    sx = Scale(xs.min(), xs.max(), F.x0, F.x1, pad=0.09)
    sy = Scale(0, ys.max(), F.y1, F.y0, pad=0.06)
    F.frame()
    F.grid_y(sy, sy.ticks(6), lambda v: f"{v:.0f}%")
    F.grid_x(sx, sx.ticks(5), lambda v: f"{v:.2f}", show_lines=False)

    a, b = np.polyfit(xs, ys, 1)
    F.path([(sx(xs.min()), sy(a * xs.min() + b)), (sx(xs.max()), sy(a * xs.max() + b))],
           "var(--muted)", 1.6, dash="6 5", op=.85)

    lp = LabelPlacer(F, bounds=(F.x0 - 58, F.y0 - 24, F.w - 6, F.y1 + 24))
    # reserve the two annotation blocks before anything else competes for space
    lp.reserve_box([F.x0 + 10, F.y0 + 6, F.x0 + 200, F.y0 + 44])
    stat_lines = [
        f"regret vs coverage   r = {r_cov:+.3f}  (\u03c1 = {sp.statistic:+.2f}, "
        f"p = {sp.pvalue:.3f})",
        f"regret vs energy score   r = {r_es:+.3f}  (\u03c1 = {sp_es.statistic:+.2f}, "
        f"p = {sp_es.pvalue:.2f}, n.s.)",
        "Feasibility does not reach the decision; calibration does."]
    lp.reserve_box([F.x1 - 400, F.y0 + 6, F.x1 - 6, F.y0 + 52])
    lp.reserve_box([sx(xs.max()) - 90, sy(a * xs.max() + b) - 16,
                    sx(xs.max()) + 6, sy(a * xs.max() + b) + 2])

    for x, y, name, ex in pts:
        F.mark("circle" if ex else "square", sx(x), sy(y),
               SERIES[0] if ex else SERIES[1], 6.4 if ex else 5.4, filled=ex,
               title=f"{name}: regret {y:.1f}%, coverage {x:.3f}")
        lp.reserve_mark(sx(x), sy(y), 8.5)

    NAMED = ["HFM (ours)", "FM+reduced", "FM+DC3", "FM+PCFM", "FM+posthoc",
             "GaussianCopula", "NormFlow-RealNVP", "kNN-Historical", "cWGAN-GP",
             "cVAE", "DDPM", "FM"]
    unplaced = []
    for name in NAMED:
        m = next((q for q in pts if q[2] == name), None)
        if not m:
            continue
        x, y, _, ex = m
        ok = lp.place(sx(x), sy(y), name, SERIES[0] if ex else "var(--muted)",
                      "series-lbl" if ex else "lbl")
        if not ok:
            unplaced.append(name)

    F.legend([("circle", SERIES[0], "exactly feasible", True),
              ("square", SERIES[1], "all other generators", False)],
             F.x0 + 16, F.y0 + 20)
    F.label(sx(xs.max()) - 6, sy(a * xs.max() + b) - 8, "least squares",
            "var(--muted)", "end", "note")
    F.note(F.x1 - 8, F.y0 + 18, stat_lines, anchor="end", cls="note")
    F.axis_titles("90% interval coverage (calibration)  →",
                  "Unit-commitment cost regret  ↓")
    return F.done()


# ===================================================================== FIG 4
def fig_frontier(eff):
    """Accuracy per function evaluation."""
    SER = [("HFM (ours)", SERIES[0], SHAPES[0]), ("FM", SERIES[1], SHAPES[1]),
           ("FM+PCFM", SERIES[3], SHAPES[3])]
    data = {}
    for name, _, _ in SER:
        pts = sorted({(r["nfe"], r["energy_score"]) for r in eff
                      if r["method"] == name and r["solver"] in ("euler", "dopri5")})
        agg = {}
        for n, e in pts:
            agg[n] = min(e, agg.get(n, 1e18))
        data[name] = sorted(agg.items())
    allx = [n for v in data.values() for n, _ in v]
    ally = [e for v in data.values() for _, e in v]

    F = Figure(880, 450, 74, 168, 34, 62,
               "Energy score versus number of function evaluations",
               "Accuracy-versus-cost frontier on the grid suite. Train-time projection "
               "reaches a better score at twenty function evaluations than unconstrained "
               "flow matching reaches at five hundred and forty-eight.", "f4")
    sx = Scale(min(allx), max(allx), F.x0, F.x1, log=True, pad=0.05)
    sy = Scale(min(ally), max(ally), F.y1, F.y0, pad=0.08)
    F.frame()
    F.grid_y(sy, sy.ticks(6), lambda v: f"{v:.0f}")
    F.grid_x(sx, [2, 5, 10, 20, 50, 100, 200, 500], lambda v: f"{v:g}")
    lp = LabelPlacer(F)
    for name, col, shp in SER:
        pts = data[name]
        F.path([(sx(n), sy(e)) for n, e in pts], col, 2.2, op=.9)
        for n, e in pts:
            F.mark(shp, sx(n), sy(e), col, 5.2,
                   title=f"{name}: NFE {n}, ES {e:.4g}")
            lp.reserve_mark(sx(n), sy(e), 7)
    for name, col, shp in SER:
        n, e = data[name][-1]
        lp.place(sx(n), sy(e), name, col, "series-lbl",
                 ring=[(13, 4, "start"), (13, -11, "start"), (13, 18, "start"),
                       (13, -24, "start"), (13, 31, "start")])

    fm = data["FM"]; hf = data["HFM (ours)"]
    fm_best_n, fm_best_e = min(fm, key=lambda p: p[1])
    cand = [p for p in hf if p[1] <= fm_best_e]
    if cand:
        cn, ce = min(cand, key=lambda p: p[0])
        F.o.append(f'<line x1="{F.x0}" y1="{sy(fm_best_e):.2f}" x2="{F.x1}" '
                   f'y2="{sy(fm_best_e):.2f}" class="rule" stroke-dasharray="4 4"/>')
        # vertical drops mark the two budgets being compared
        for nn in (cn, fm_best_n):
            F.o.append(f'<line x1="{sx(nn):.2f}" y1="{sy(fm_best_e):.2f}" '
                       f'x2="{sx(nn):.2f}" y2="{F.y1:.2f}" stroke="var(--muted)" '
                       f'stroke-width="1" stroke-dasharray="2 4" opacity=".65"/>')
        F.arrow(sx(fm_best_n), F.y1 - 14, sx(cn) + 7, F.y1 - 14)
        F.label((sx(cn) + sx(fm_best_n)) / 2, F.y1 - 20,
                f"{fm_best_n/max(cn,1):.0f}\u00d7 fewer evaluations, better score",
                "var(--muted)", "middle", "note")
        F.label(sx(cn), F.y1 - 2, f"NFE {cn}", "var(--muted)", "middle", "note")
        F.label(sx(fm_best_n), F.y1 - 2, f"NFE {fm_best_n}", "var(--muted)",
                "middle", "note")
    F.axis_titles("Function evaluations per scenario (NFE, log scale)  →",
                  "Energy score  ↓")
    return F.done()


# ===================================================================== FIG 5
def fig_ranking(mains, suite="grid", label="grid"):
    """All sixteen methods on one common scale: a ranked dot plot."""
    rows = mains[suite]
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r["energy_score"])
    stats = sorted(((m, float(np.mean(v)),
                     float(np.std(v, ddof=1)) / math.sqrt(len(v)) if len(v) > 1 else 0.0)
                    for m, v in by.items()), key=lambda p: p[1])
    EX = {"HFM (ours)", "FM+reduced", "FM+DC3", "FM+PCFM", "FM+posthoc"}
    PEN = {f"FM+penalty({l})" for l in (1, 10, 100, 1000)}
    n = len(stats)
    row_h = 23
    F = Figure(880, 60 + n * row_h + 58, 168, 56, 42, 58,
               f"Energy score by method on the {label} suite",
               f"Ranked dot plot of mean energy score with standard-error bars for "
               f"{n} generators on the {label} suite, lower is better.", "f5")
    xs = [s[1] + s[2] for s in stats] + [s[1] - s[2] for s in stats]
    sx = Scale(min(xs), max(xs), F.x0, F.x1, pad=0.05)
    F.frame()
    for v in sx.ticks(6):
        F.o.append(f'<line x1="{sx(v):.2f}" y1="{F.y0}" x2="{sx(v):.2f}" '
                   f'y2="{F.y1}" class="grid"/>')
        F.label(sx(v), F.y1 + 20, f"{v:g}", "var(--muted)", "middle", "tick")
    for i, (m, mu, se) in enumerate(stats):
        y = F.y0 + 14 + i * row_h
        ours = m == "HFM (ours)"
        fam = SERIES[0] if ours else (SERIES[1] if m in EX else
                                      (SERIES[3] if m in PEN else "var(--muted)"))
        shp = "circle" if m in EX else ("diamond" if m in PEN else "square")
        F.o.append(f'<line x1="{F.x0}" y1="{y:.2f}" x2="{sx(mu):.2f}" y2="{y:.2f}" '
                   f'stroke="{fam}" stroke-width="1" opacity=".25"/>')
        if se > 0:
            F.o.append(f'<line x1="{sx(mu-se):.2f}" y1="{y:.2f}" '
                       f'x2="{sx(mu+se):.2f}" y2="{y:.2f}" stroke="{fam}" '
                       f'stroke-width="1.8" opacity=".8"/>')
        F.mark(shp, sx(mu), y, fam, 5.6 if ours else 4.8,
               filled=(m in EX or m in PEN),
               title=f"{m}: ES {mu:.4g} ± {se:.3g} (s.e.)")
        F.label(F.x0 - 12, y + 4, m, fam if (ours or m in EX) else "var(--muted)",
                "end", "series-lbl" if ours else "lbl")
    F.axis_titles("Energy score  ↓  (mean ± s.e. over 10 seeds)", "")
    F.legend([("circle", SERIES[1], "exact", True),
              ("diamond", SERIES[3], "penalty", True),
              ("square", "var(--muted)", "baseline", False)],
             F.x1 - 96, F.y0 + 4, dy=16)
    return F.done()
