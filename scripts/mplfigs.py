"""Publication figures in matplotlib/seaborn, built from `results/*.jsonl`.

House style, applied to every panel:
  * DejaVu Sans, seaborn whitegrid, despined axes, horizontal gridlines only.
  * ColorBrewer Dark2 / cubehelix, both colourblind safe, with a distinct marker
    shape per series so nothing depends on hue alone.
  * Value labels on the marks, and one annotation per panel naming the insight
    rather than restating the axis.
Each figure is written to figures/ as PDF (for LaTeX), PNG (for the web) and
SVG (crisp on the project page).
"""
from __future__ import annotations
import json, math, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")
T_CRIT_9 = 2.262
SUITES = [("grid_noflow", "grid-noflow"), ("measured", "measured"), ("grid", "grid")]

# ColorBrewer Dark2: colourblind safe, prints legibly in greyscale.
C = {"ours": "#D95F02", "chart": "#1B9E77", "dc3": "#7570B3", "infer": "#E7298A",
     "base": "#666666", "pen": "#A6761D", "grid": "#CFCFCF"}
INK = "#1A1A1A"
MUTED = "#5A5A5A"


def style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlepad": 14,
        "axes.labelsize": 10.5, "axes.labelcolor": INK, "axes.edgecolor": "#BBBBBB",
        "axes.labelpad": 7, "text.color": INK,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": "#E4E4E4", "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def load(n):
    p = os.path.join(RES, n)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def paired_ci(rows, a, b, key="energy_score"):
    by = defaultdict(dict)
    for r in rows:
        by[r["method"]][r["seed"]] = r
    seeds = sorted(set(by.get(a, {})) & set(by.get(b, {})))
    if len(seeds) < 2:
        return None
    d = np.array([by[a][s][key] - by[b][s][key] for s in seeds], float)
    base = np.mean([by[b][s][key] for s in seeds])
    se = d.std(ddof=1) / math.sqrt(len(d))
    f = 100.0 / base
    return (d.mean() * f, (d.mean() - T_CRIT_9 * se) * f,
            (d.mean() + T_CRIT_9 * se) * f, d.mean() / se, len(d))


def mean_of(rows, m, k="energy_score"):
    v = [r[k] for r in rows if r["method"] == m and r.get(k) is not None]
    return float(np.mean(v)) if v else float("nan")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  figures/{name}.{{pdf,png,svg}}")


def note(ax, text, xy, xytext, color=MUTED, arrow=True, size=9.5, ha="left"):
    """Annotate the insight, with a light connector when it helps."""
    ax.annotate(text, xy=xy, xytext=xytext, textcoords="data", fontsize=size,
                color=color, ha=ha, va="center", zorder=6,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0,
                                shrinkA=4, shrinkB=6,
                                connectionstyle="arc3,rad=0.12") if arrow else None)


# ============================================================== FIG 1 lollipop
def fig_ranking(mains, suite="grid", label="grid"):
    rows = mains[suite]
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r["energy_score"])
    st = sorted(((m, float(np.mean(v)),
                  float(np.std(v, ddof=1)) / math.sqrt(len(v)))
                 for m, v in by.items()), key=lambda p: p[1], reverse=True)
    EX = {"HFM (ours)", "FM+reduced", "FM+DC3", "FM+PCFM", "FM+posthoc"}
    PEN = {f"FM+penalty({l})" for l in (1, 10, 100, 1000)}

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    names = [s[0] for s in st]
    y = np.arange(len(st))
    for i, (m, mu, se) in enumerate(st):
        col = C["ours"] if m == "HFM (ours)" else (
            C["chart"] if m in EX else (C["pen"] if m in PEN else C["base"]))
        ax.hlines(i, 0, mu, color=col, alpha=.30, lw=1.6, zorder=1)
        ax.errorbar(mu, i, xerr=se, fmt="none", ecolor=col, elinewidth=1.7,
                    capsize=2.5, alpha=.9, zorder=2)
        mk = "o" if m in EX else ("D" if m in PEN else "s")
        ax.plot(mu, i, mk, ms=8.5 if m == "HFM (ours)" else 6.6, color=col,
                mec="white", mew=1.1, zorder=3)
        ax.text(mu + (st[0][1] * .012), i, f"{mu:.0f}", va="center", ha="left",
                fontsize=9, color=col if m in EX or m in PEN else MUTED,
                fontweight="bold" if m == "HFM (ours)" else "normal", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    for t, (m, _, _) in zip(ax.get_yticklabels(), st):
        if m == "HFM (ours)":
            t.set_color(C["ours"]); t.set_fontweight("bold")
        elif m in EX:
            t.set_color(C["chart"])
    lo = min(s[1] for s in st); hi = max(s[1] for s in st)
    ax.set_xlim(lo - (hi - lo) * .10, hi + (hi - lo) * .10)
    ax.set_xlabel("Energy score  (mean ± s.e. over 10 seeds, lower is better)")
    ax.set_title(f"Every generator on one scale: the {label} suite", loc="left")
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True, bottom=False)
    knn = next(i for i, s in enumerate(st) if s[0] == "kNN-Historical")
    note(ax, "a non-neural baseline places\nsecond of sixteen",
         (st[knn][1], knn), (lo + (hi - lo) * .22, knn - 2.2), color=MUTED)
    # the legend sits in the empty middle-right band, clear of every lollipop
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=C["ours"], label="ours"),
        Line2D([], [], marker="o", ls="", color=C["chart"], label="other exact routes"),
        Line2D([], [], marker="D", ls="", color=C["pen"], label="soft penalty"),
        Line2D([], [], marker="s", ls="", color=C["base"], label="baseline")],
        loc="center right", bbox_to_anchor=(1.0, 0.36), ncol=1, labelspacing=.5)
    save(fig, "fig1_ranking")


# ============================================================ FIG 2 sign flip
def fig_signflip(meta, mains):
    M = [("HFM (ours)", "train-time projection", C["ours"], "o"),
         ("FM+reduced", "nullspace chart", C["chart"], "s"),
         ("FM+DC3", "DC3 completion", C["dc3"], "^"),
         ("FM+PCFM", "inference-time", C["infer"], "D")]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    xs = [meta[s]["codim_frac"] for s, _ in SUITES]
    ax.axhline(0, color=INK, lw=1.2, ls=(0, (5, 4)), alpha=.65, zorder=1)
    # fix the y-range from the data first, so the shaded half-plane cannot
    # drag the limits out and flatten every series into a sliver
    allv = []
    for name, _, _, _ in M:
        for s, _ in SUITES:
            r = paired_ci(mains[s], name, "FM")
            if r:
                allv += [r[1], r[2]]
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.30
    ax.set_ylim(lo - pad * 0.45, hi + pad)
    ax.axhspan(0, hi + pad, color="#D95F02", alpha=.05, zorder=0)
    for name, pretty, col, mk in M:
        pts = []
        for s, _ in SUITES:
            r = paired_ci(mains[s], name, "FM")
            if r:
                pts.append((meta[s]["codim_frac"], *r))
        pts.sort()
        x = [p[0] for p in pts]; y = [p[1] for p in pts]
        err = np.array([[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]])
        ax.errorbar(x, y, yerr=err, color=col, lw=2.0, marker=mk, ms=7,
                    mec="white", mew=1.1, capsize=3.2, elinewidth=1.5, zorder=3,
                    label=pretty)
        for px, py, _, _, t, _ in pts:
            if abs(t) <= T_CRIT_9:
                ax.plot(px, py, mk, ms=7, color="white", mec=col, mew=1.8, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lab}\ncodim {meta[s]['codim_frac']:.3f}"
                        for s, lab in SUITES])
    ax.set_xlim(0.02, 0.72)
    ax.set_ylabel("Δ energy score vs unconstrained FM  (%)")
    ax.set_xlabel("Constraint codimension  r / D")
    ax.set_title("The sign of the constraint effect flips with codimension", loc="left")
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:+.0f}%")
    hfm = [paired_ci(mains[s], "HFM (ours)", "FM") for s, _ in SUITES]
    y0, y1 = ax.get_ylim()
    ax.text(0.715, y1 - (y1 - y0) * .045, "worse than no constraint", fontsize=9,
            color="#B4531F", va="top", ha="right")
    # both annotations name the orange series, so anchor them on it exactly
    note(ax, f"significantly worse\n({hfm[1][0]:+.2f}%, t = {hfm[1][3]:+.2f})",
         (0.304, hfm[1][0]), (0.395, y1 - (y1 - y0) * .30), color=C["ours"])
    note(ax, f"significantly better\n({hfm[2][0]:+.2f}%, t = {hfm[2][3]:+.2f})",
         (0.648, hfm[2][0]), (0.425, y0 + (y1 - y0) * .10), color=C["ours"])
    ax.legend(loc="upper left", ncol=1, bbox_to_anchor=(0.005, 0.985),
              handlelength=2.4, labelspacing=.45)
    sns.despine(ax=ax)
    ax.grid(axis="x", visible=False)
    ax.text(0.02, -0.215, "Hollow marker = not significant at the 5% level. "
            "Bars are 95% CIs, paired over 10 seeds.",
            transform=ax.transAxes, fontsize=8.8, color=MUTED)
    save(fig, "fig2_signflip")


# ============================================================== FIG 3 pareto
def fig_pareto(mains):
    rows = mains["grid"]
    pens = [(l, mean_of(rows, f"FM+penalty({l})"),
             mean_of(rows, f"FM+penalty({l})", "eq_max")) for l in (1, 10, 100, 1000)]
    ex = [(n, mean_of(rows, n), mean_of(rows, n, "eq_max")) for n in
          ("HFM (ours)", "FM+reduced", "FM+DC3", "FM+PCFM", "FM+posthoc")]
    fm = ("FM", mean_of(rows, "FM"), mean_of(rows, "FM", "eq_max"))

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.set_xscale("log")
    ax.axvspan(1e-5, 1e-2, color=C["chart"], alpha=.06, zorder=0)
    ax.plot([p[2] for p in pens], [p[1] for p in pens], ls=(0, (4, 3)),
            color=C["pen"], lw=1.6, zorder=2)
    ax.plot([p[2] for p in pens], [p[1] for p in pens], "D", ms=8,
            color=C["pen"], mec="white", mew=1.1, zorder=3, label="soft penalty")
    for l, es, eq in pens:
        dx, ha = ((-12, "right") if l == 1 else (11, "left"))
        ax.annotate(f"λ={l}", (eq, es), textcoords="offset points",
                    xytext=(dx, 0), va="center", ha=ha, fontsize=9.2,
                    color=C["pen"])
    for i, (n, es, eq) in enumerate(ex):
        ours = n == "HFM (ours)"
        ax.plot(eq, es, "o", ms=10 if ours else 7, color=C["ours"] if ours else C["chart"],
                mec="white", mew=1.2, zorder=4,
                label="train-time projection (ours)" if ours else
                      ("other exact routes" if i == 1 else None))
    ax.plot(fm[2], fm[1], "X", ms=10, color=C["base"], mec="white", mew=1.2,
            zorder=4, label="unconstrained FM")
    ax.annotate("FM (unconstrained)", (fm[2], fm[1]), textcoords="offset points",
                xytext=(0, -18), ha="center", fontsize=9.2, color=C["base"])
    hf = ex[0]
    ax.annotate("ours", (hf[2], hf[1]), textcoords="offset points", xytext=(12, -3),
                fontsize=9.6, color=C["ours"], fontweight="bold")
    ax.set_xlabel("Worst-case violation  ‖Ax−b‖∞   (MW, log scale)")
    ax.set_ylabel("Energy score  (lower is better)")
    ax.set_title("Penalties are dominated on both axes at once", loc="left")
    note(ax, "raising λ walks the family up,\nnever left",
         (pens[3][2], pens[3][1]), (2.5e-2, pens[3][1] * .955), color=C["pen"])
    ax.text(1.4e-5, ax.get_ylim()[1] * .985, "exactly feasible", fontsize=9,
            color=C["chart"], va="top")
    ax.legend(loc="upper left", bbox_to_anchor=(0.035, 0.87), labelspacing=.45)
    sns.despine(ax=ax)
    ax.grid(axis="x", visible=False)
    save(fig, "fig3_pareto")


# ============================================================== FIG 4 regret
def fig_regret(down, measured):
    reg, cov = defaultdict(list), defaultdict(list)
    for r in down:
        if r.get("regret_pct") is not None and "[det-mean]" not in r["method"]:
            reg[r["method"]].append(r["regret_pct"])
    for r in measured:
        cov[r["method"]].append(r["cov90"])
    esm = defaultdict(list)
    for r in measured:
        esm[r["method"]].append(r["energy_score"])
    EX = {"HFM (ours)", "FM+reduced", "FM+DC3", "FM+PCFM", "FM+posthoc"}
    pts = [(float(np.mean(cov[m])), float(np.mean(reg[m])), m, m in EX)
           for m in reg if m in cov]
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    zs = np.array([float(np.mean(esm[p[2]])) for p in pts])
    r_cov = float(np.corrcoef(xs, ys)[0, 1]); sp = spearmanr(xs, ys)
    r_es = float(np.corrcoef(zs, ys)[0, 1]); sp_es = spearmanr(zs, ys)

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    # regret spans 16% to 285%, so two outliers otherwise squash the cluster
    # that carries the result; a log axis spreads it without hiding anything
    ax.set_yscale("log")
    a, b = np.polyfit(xs, np.log10(ys), 1)
    gx = np.linspace(xs.min() - .03, xs.max() + .03, 60)
    ax.plot(gx, 10 ** (a * gx + b), ls=(0, (5, 4)), color=MUTED, lw=1.4, zorder=1,
            label="least squares (log)")
    for x, y, n, e in pts:
        ax.plot(x, y, "o" if e else "s", ms=9 if e else 7,
                color=C["ours"] if e else "white",
                mec=C["ours"] if e else C["base"], mew=1.6, zorder=3,
                label=("exactly feasible" if e else "all other generators")
                if n in ("HFM (ours)", "GaussianCopula") else None)
    LBL = {"HFM (ours)": (-11, -11), "FM+reduced": (-11, 9), "FM+DC3": (-11, -2),
           "FM+PCFM": (-11, -11), "FM+posthoc": (11, 6),
           "GaussianCopula": (-10, 9), "NormFlow-RealNVP": (11, -9),
           "kNN-Historical": (11, 7), "cWGAN-GP": (11, 0), "cVAE": (11, 0),
           "DDPM": (11, 6), "FM": (-11, -12)}
    for x, y, n, e in pts:
        if n in LBL:
            dx, dy = LBL[n]
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(dx, dy),
                        fontsize=8.8, ha="left" if dx > 0 else "right",
                        color=C["ours"] if e else MUTED,
                        fontweight="bold" if n == "HFM (ours)" else "normal")
    ax.set_xlabel("90% interval coverage  (calibration)")
    ax.set_ylabel("Unit-commitment cost regret  (%)")
    ax.set_title("Decision value tracks calibration, not feasibility", loc="left")
    ax.set_yticks([20, 30, 50, 75, 100, 150, 200, 300])
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")
    ax.minorticks_off()
    ax.text(0.985, 0.97,
            f"regret vs coverage      r = {r_cov:+.3f}  (ρ = {sp.statistic:+.2f}, "
            f"p = {sp.pvalue:.3f})\n"
            f"regret vs energy score  r = {r_es:+.3f}  (ρ = {sp_es.statistic:+.2f}, "
            f"p = {sp_es.pvalue:.2f}, n.s.)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=MUTED,
            family="DejaVu Sans")
    ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.02), labelspacing=.45)
    sns.despine(ax=ax)
    ax.grid(axis="x", visible=False)
    save(fig, "fig4_regret")


# ============================================================ FIG 5 frontier
def fig_frontier(eff):
    SER = [("HFM (ours)", C["ours"], "o"), ("FM", C["base"], "s"),
           ("FM+PCFM", C["infer"], "D")]
    data = {}
    for n, _, _ in SER:
        agg = {}
        for r in eff:
            if r["method"] == n and r["solver"] in ("euler", "dopri5"):
                agg[r["nfe"]] = min(r["energy_score"], agg.get(r["nfe"], 1e18))
        data[n] = sorted(agg.items())
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.set_xscale("log")
    for n, col, mk in SER:
        x = [p[0] for p in data[n]]; y = [p[1] for p in data[n]]
        ax.plot(x, y, marker=mk, color=col, lw=2.0, ms=6.5, mec="white", mew=1.0,
                label=n, zorder=3)
    fm_n, fm_e = min(data["FM"], key=lambda p: p[1])
    cn, ce = min([p for p in data["HFM (ours)"] if p[1] <= fm_e], key=lambda p: p[0])
    ax.axhline(fm_e, color=MUTED, lw=1.1, ls=(0, (4, 4)), alpha=.8, zorder=1)
    ax.annotate("", xy=(cn, fm_e), xytext=(fm_n, fm_e),
                arrowprops=dict(arrowstyle="<|-", color=INK, lw=1.3,
                                shrinkA=2, shrinkB=2))
    # park the callout in the empty upper band so it clears every curve
    y0_, y1_ = ax.get_ylim()
    ax.annotate(f"{fm_n/cn:.0f}× fewer evaluations\nfor a better score",
                xy=(math.sqrt(cn * fm_n), fm_e),
                xytext=(math.sqrt(cn * fm_n), y0_ + (y1_ - y0_) * 0.40),
                textcoords="data", ha="center", va="bottom", fontsize=9.8,
                color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0, shrinkA=3,
                                shrinkB=4, connectionstyle="arc3,rad=0.0"))
    for nn, lab in ((cn, f"NFE {cn}"), (fm_n, f"NFE {fm_n}")):
        ax.axvline(nn, color=MUTED, lw=0.9, ls=(0, (2, 4)), alpha=.7, zorder=1)
        ax.annotate(lab, xy=(nn, ax.get_ylim()[0]), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8.8,
                    color=MUTED)
    ax.set_xlabel("Function evaluations per scenario  (NFE, log scale)")
    ax.set_ylabel("Energy score  (lower is better)")
    ax.set_title("Exactness is free in function evaluations", loc="left")
    ax.legend(loc="upper right")
    sns.despine(ax=ax)
    ax.grid(axis="x", visible=False)
    save(fig, "fig5_frontier")


def main():
    style()
    meta = json.load(open(os.path.join(RES, "suite_meta.json")))
    mains = {s: load(f"main_{s}.jsonl") for s, _ in SUITES}
    print("writing figures:")
    fig_ranking(mains)
    fig_signflip(meta, mains)
    fig_pareto(mains)
    fig_regret(load("downstream.jsonl"), mains["measured"])
    fig_frontier(load("efficiency.jsonl"))


if __name__ == "__main__":
    main()
