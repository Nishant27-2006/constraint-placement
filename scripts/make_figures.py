"""Generate every paper figure from the result files.

Design rules followed throughout (and the reason each is not a matter of taste):
  * Categorical palette validated for colour-vision deficiency -- worst adjacent
    pair dE 9.1 (protan), normal-vision floor 19.6.  Not eyeballed.
  * Colour follows the *method*, fixed order, never cycled and never re-assigned
    when a subset is plotted, so a method keeps its colour across every figure.
  * Never a dual y-axis.  Where two quantities of different scale matter
    (accuracy and compute) they get two panels, not two scales.
  * Every series is direct-labelled or in a legend, so identity is never carried
    by colour alone -- required because three palette slots fall below 3:1
    contrast on white.
  * Serif type matching the document; recessive grid; log axes wherever the
    quantity spans orders of magnitude (constraint residuals span ~13).
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "paper", "figs")
os.makedirs(FIGS, exist_ok=True)

# validated categorical order -- see scripts/validate_palette.js report in docs
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#c3c2b7"
OURS = "#2a78d6"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "grid.color": GRID, "grid.linewidth": 0.4, "grid.alpha": 0.7,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(name):
    p = os.path.join(ROOT, "results", name)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def agg(rows, key, by="method"):
    d = {}
    for r in rows:
        if key in r and r[key] is not None and np.isfinite(r[key]):
            d.setdefault(r[by], []).append(r[key])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in d.items()}


def pending(name, note):
    """A visibly-marked placeholder.  Never substitute a real figure for a
    missing one: a wrong plot in a paper is worse than an obvious gap."""
    out = os.path.join(FIGS, f"{name}.pdf")
    if os.path.exists(out) and os.path.getsize(out) > 6000:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False,
                               ls="--", lw=1.0, color="#e34948",
                               transform=ax.transAxes))
    ax.text(0.5, 0.56, "FIGURE PENDING", ha="center", va="center",
            fontsize=11, color="#e34948", weight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.38, note, ha="center", va="center", fontsize=7,
            color=INK2, transform=ax.transAxes, wrap=True)
    fig.savefig(out); plt.close(fig)
    print(f"  wrote figs/{name}.pdf (PENDING placeholder)")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote figs/{name}.pdf")


# ---------------------------------------------------------------- F1 violation
def fig_violation(suite="grid"):
    rows = load(f"main_{suite}.jsonl")
    if not rows:
        pending(f"violation_{suite}", f"run experiments/run_main.py --suite {suite}"); return
    st = agg(rows, "eq_max")
    ours = {r["method"] for r in rows if r.get("ours")}
    items = sorted(st.items(), key=lambda kv: kv[1][0])
    names = [k for k, _ in items]
    vals = np.array([max(v[0], 1e-16) for _, v in items])
    fig, ax = plt.subplots(figsize=(5.4, 0.26 * len(names) + 0.9))
    colors = [OURS if n in ours else INK2 for n in names]
    ax.barh(range(len(names)), vals, color=colors, height=0.62, zorder=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xscale("log")
    ax.set_xlabel(r"constraint residual $\|Ax-b\|_\infty$  (MW, log scale)")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=16))
    ax.grid(axis="x", zorder=0)
    ax.invert_yaxis()
    eps = np.finfo(np.float32).eps
    ax.axvline(eps * float(np.nanmax([r.get("eq_max", 0) for r in rows]) or 1.0),
               color=INK, ls=":", lw=0.8, zorder=4)
    for i, (v, n) in enumerate(zip(vals, names)):
        ax.text(v * 1.4, i, f"{v:.1e}", va="center", fontsize=6, color=INK2)
    ax.set_title(f"Physical-constraint violation, {suite} suite "
                 f"(mean over seeds; dotted line = float32 round-off)", loc="left")
    save(fig, f"violation_{suite}")


# ------------------------------------------------------- F2 accuracy vs compute
def fig_frontier():
    rows = load("efficiency.jsonl")
    if not rows:
        pending("frontier", "run experiments/run_efficiency.py"); return
    methods = sorted({r["method"] for r in rows})
    cmap = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for ax, xk, xl in ((axes[0], "nfe", "network evaluations per scenario (NFE)"),
                       (axes[1], "flops_per_scenario", "FLOPs per scenario")):
        for m in methods:
            sub = sorted([r for r in rows if r["method"] == m], key=lambda r: r[xk])
            if not sub:
                continue
            x = [r[xk] for r in sub]; y = [r["energy_score"] for r in sub]
            ax.plot(x, y, "-o", color=cmap[m], lw=1.6, ms=3.5, label=m, zorder=3)
        ax.set_xscale("log"); ax.set_xlabel(xl); ax.grid(zorder=0)
    axes[0].set_ylabel("energy score  (lower is better)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Accuracy-compute frontier", x=0.02, ha="left", fontsize=9)
    save(fig, "frontier")


# ------------------------------------------------------------- F3 N-1 transfer
def fig_transfer():
    rows = load("transfer.jsonl")
    if not rows:
        pending("transfer", "run experiments/run_transfer.py"); return
    keys, vals = [], []
    d = {}
    for r in rows:
        d.setdefault(f"{r['method']}\n({r['mode']})", []).append(max(r["eq_max"], 1e-16))
    for k, v in sorted(d.items(), key=lambda kv: np.mean(kv[1])):
        keys.append(k); vals.append(np.mean(v))
    ours = [i for i, k in enumerate(keys) if "ours" in k]
    fig, ax = plt.subplots(figsize=(5.6, 0.34 * len(keys) + 0.9))
    colors = [OURS if i in ours else INK2 for i in range(len(keys))]
    ax.barh(range(len(keys)), vals, color=colors, height=0.6, zorder=3)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=6.5)
    ax.set_xscale("log"); ax.invert_yaxis(); ax.grid(axis="x", zorder=0)
    ax.set_xlabel(r"residual against the *contingency* constraint $\|A'x-b'\|_\infty$ (MW)")
    ax.set_title("Zero-shot transfer to an N-1 topology never seen in training", loc="left")
    save(fig, "transfer")


# ---------------------------------------------------------------- F4 downstream
def fig_downstream():
    rows = [r for r in load("downstream.jsonl") if "regret_pct" in r]
    if not rows:
        pending("downstream", "run experiments/run_downstream.py"); return
    rows = sorted(rows, key=lambda r: r["regret_pct"])
    names = [r["method"] for r in rows]
    vals = [r["regret_pct"] for r in rows]
    colors = [OURS if r.get("ours") else INK2 for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 0.28 * len(names) + 0.9))
    ax.barh(range(len(names)), vals, color=colors, height=0.62, zorder=3)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6.5)
    ax.invert_yaxis(); ax.grid(axis="x", zorder=0)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("out-of-sample cost regret vs perfect foresight (%)")
    ax.set_title("Does constraint-exactness reach the scheduling decision?", loc="left")
    save(fig, "downstream")


# -------------------------------------------------------------------- F5 AC
def fig_ac():
    rows = load("ac.jsonl")
    if not rows:
        pending("ac_drift", "run experiments/run_ac.py"); return
    methods = sorted({r["method"] for r in rows})
    cmap = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)}
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for m in methods:
        sub = sorted([r for r in rows if r["method"] == m], key=lambda r: r["steps"])
        ax.plot([r["steps"] for r in sub], [max(r["nl_max"], 1e-17) for r in sub],
                "-o", color=cmap[m], lw=1.6, ms=3.5, label=m, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(zorder=0)
    ax.set_xlabel("integration steps $K$")
    ax.set_ylabel(r"AC manifold residual $\|g(x)\|_\infty$ (p.u.)")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("Nonlinear manifold: drift vs. retraction", loc="left")
    save(fig, "ac_drift")


# ------------------------------------------------- F6 theorem check (self-contained)
def fig_theorem():
    """Residual vs. number of integrator steps for a projected field.

    Self-contained: recomputed here rather than read from a results file, so the
    figure verifying Theorem 1(iv) cannot silently go stale.
    """
    import torch
    from hfm.grid.constraints import AffineConstraintSet
    torch.manual_seed(0)
    T, M, D = 24, 6, 32
    A = torch.randn(T, M, D, dtype=torch.float64)
    b = torch.randn(T, M, dtype=torch.float64)
    C = AffineConstraintSet(A, b)
    ks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for dt, lbl, col in ((torch.float64, "float64", PALETTE[0]),
                         (torch.float32, "float32", PALETTE[1])):
        res = []
        for K in ks:
            x = C.project_point(torch.randn(32, T, D, dtype=torch.float64)).to(dt)
            Ck = C.to("cpu", dt)
            for _ in range(K):
                x = x + (1.0 / K) * Ck.project_tangent(torch.randn_like(x))
            res.append(max(C.violation(x)["eq_max"], 1e-18))
        ax.plot(ks, res, "-o", color=col, lw=1.6, ms=3.5, label=lbl, zorder=3)
        if dt is torch.float32:
            # sqrt(K) guide: round-off accumulates as a random walk, which is
            # the *only* growth present -- the exact-arithmetic statement of
            # Theorem 1(iv) is the flat float64 trace.
            k0, r0 = ks[0], res[0]
            ax.plot(ks, [r0 * (k / k0) ** 0.5 for k in ks], ls=":", lw=1.0,
                    color=INK2, zorder=2, label=r"$\propto\sqrt{K}$ (round-off)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(zorder=0)
    ax.set_xlabel("integrator steps $K$")
    ax.set_ylabel(r"$\|Ax-b\|_\infty$")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("Invariant drift vs. integrator steps", loc="left")
    save(fig, "theorem_drift")


def fig_codim():
    """Violation under two codimensions at fixed intrinsic dimension."""
    lo = load("main_grid_noflow.jsonl"); hi = load("main_grid.jsonl")
    if not lo or not hi:
        pending("codim", "run experiments/run_main.py on grid and grid_noflow"); return
    a, b = agg(lo, "eq_max"), agg(hi, "eq_max")
    methods = sorted(set(a) & set(b), key=lambda m: b[m][0])
    x = np.arange(len(methods)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 0.30 * len(methods) + 1.2))
    ax.barh(x - w/2, [max(a[m][0], 1e-16) for m in methods], height=w,
            color=PALETTE[0], label="codim 11 (no flow channels)", zorder=3)
    ax.barh(x + w/2, [max(b[m][0], 1e-16) for m in methods], height=w,
            color=PALETTE[1], label="codim 197 (with flow channels)", zorder=3)
    ax.set_yticks(x); ax.set_yticklabels(methods, fontsize=6.5)
    ax.set_xscale("log"); ax.invert_yaxis(); ax.grid(axis="x", zorder=0)
    ax.set_xlabel(r"constraint residual $\|Ax-b\|_\infty$ (MW)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.set_title("Codimension varied 11 -> 197 at fixed intrinsic dimension (107 dof/hour)",
                 loc="left", fontsize=8)
    save(fig, "codim")


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    print("[figures]")
    fig_theorem()
    for s in ("measured", "grid", "grid_noflow"):
        fig_violation(s)
    fig_codim()
    fig_frontier(); fig_transfer(); fig_downstream(); fig_ac()
