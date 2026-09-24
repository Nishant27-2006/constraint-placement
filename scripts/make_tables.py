"""Generate every LaTeX table from the result files.

No number is ever typed into the paper by hand: `paper/tables/*.tex` is
regenerated from `results/*.jsonl` and \\input{} by main.tex.  Values are
mean +/- std across seeds; the best entry per column is bolded automatically.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "tables")
os.makedirs(OUT, exist_ok=True)


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def placeholder(name, note):
    """Emit a compilable stub so the paper always builds, and so a missing
    result is visible in the PDF rather than silently absent."""
    p = os.path.join(OUT, f"{name}.tex")
    if os.path.exists(p) and os.path.getsize(p) > 400:
        return
    safe = note.replace("\\", "").replace("_", r"\_").replace("&", r"\&")
    body = ["\\begin{tabular}{l}", "\\toprule",
            r"\emph{[pending: " + safe + r"]} \\", "\\bottomrule",
            "\\end{tabular}"]
    open(p, "w").write("\n".join(body) + "\n")
    print(f"  wrote {name}.tex (placeholder)")


def agg(rows, key, by="method"):
    d = {}
    for r in rows:
        d.setdefault(r[by], []).append(r.get(key, np.nan))
    return {k: (float(np.nanmean(v)), float(np.nanstd(v)), len(v)) for k, v in d.items()}


def fmt(x, sig=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e5):
        return f"\\num{{{x:.{sig-1}e}}}"
    return f"{x:.{sig}g}"


def main_table(suite):
    rows = load(os.path.join(ROOT, "results", f"main_{suite}.jsonl"))
    if not rows:
        placeholder(f"main_{suite}", f"awaiting experiments/run_main.py --suite {suite}"); return
    cols = [("energy_score", "ES $\\downarrow$", -1), ("crps", "CRPS $\\downarrow$", -1),
            ("variogram_score", "VS $\\downarrow$", -1),
            ("reliability_index", "RI $\\downarrow$", -1),
            ("cov90", "cov$_{90}$", 0), ("eq_max", "$\\|Ax-b\\|_\\infty$ $\\downarrow$", -1),
            ("nfe", "NFE", None), ("flops_per_scenario", "FLOPs/scen", None)]
    stats = {c: agg(rows, c) for c, _, _ in cols}
    methods = sorted({r["method"] for r in rows},
                     key=lambda m: stats["energy_score"].get(m, (1e18,))[0])
    best = {}
    for c, _, d in cols:
        if d == -1:
            vals = {m: stats[c][m][0] for m in methods if m in stats[c] and np.isfinite(stats[c][m][0])}
            if vals: best[c] = min(vals, key=vals.get)
    lines = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
             "Method & " + " & ".join(h for _, h, _ in cols) + " \\\\", "\\midrule"]
    for m in methods:
        ours = any(r["method"] == m and r.get("ours") for r in rows)
        label = ("\\textbf{" + m.replace("(ours)", "").strip() + "} (ours)") if ours else m
        cells = []
        for c, _, _ in cols:
            if m not in stats[c]:
                cells.append("--"); continue
            mu, sd, n = stats[c][m]
            s = fmt(mu) if c in ("nfe", "flops_per_scenario") else \
                f"{fmt(mu)}\\,\\tiny{{$\\pm$ {fmt(sd,2)}}}"
            if best.get(c) == m:
                s = "\\textbf{" + s + "}"
            cells.append(s)
        lines.append(label.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    p = os.path.join(OUT, f"main_{suite}.tex")
    open(p, "w").write("\n".join(lines) + "\n")
    print(f"  wrote {p} ({len(methods)} methods, {len(rows)} runs)")


def transfer_table():
    rows = load(os.path.join(ROOT, "results", "transfer.jsonl"))
    if not rows:
        placeholder("transfer", "awaiting experiments/run_transfer.py"); return
    d = {}
    for r in rows:
        d.setdefault((r["method"], r["mode"]), []).append((r["eq_max"], r["eq_rmse"], r["energy_score"]))
    lines = ["\\begin{tabular}{llrrr}", "\\toprule",
             "Method & chart & $\\|A'x-b'\\|_\\infty$ & $\\|A'x-b'\\|_{\\mathrm{RMS}}$ & ES \\\\",
             "\\midrule"]
    for (m, mode), v in sorted(d.items(), key=lambda kv: np.mean([z[0] for z in kv[1]])):
        arr = np.array(v)
        lines.append(f"{m.replace('_','\\_')} & {mode} & {fmt(arr[:,0].mean())} & "
                     f"{fmt(arr[:,1].mean())} & {fmt(arr[:,2].mean())} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "transfer.tex"), "w").write("\n".join(lines) + "\n")
    print(f"  wrote transfer.tex ({len(d)} rows)")


def downstream_table():
    rows = load(os.path.join(ROOT, "results", "downstream.jsonl"))
    if not rows:
        placeholder("downstream", "awaiting experiments/run_downstream.py"); return
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "Scenario source & cost (\\$) & regret (\\%) & unserved (MWh) & days shed \\\\",
             "\\midrule"]
    for r in sorted(rows, key=lambda r: r["mean_cost"]):
        lines.append(f"{r['method'].replace('_','\\_')} & {r['mean_cost']:,.0f} & "
                     f"{r.get('regret_pct', 0.0):+.2f} & {r['mean_shed_mwh']:.2f} & "
                     f"{100*r['frac_days_shed']:.0f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "downstream.tex"), "w").write("\n".join(lines) + "\n")
    print(f"  wrote downstream.tex ({len(rows)} rows)")


def codim_table():
    """Codimension varied (11 -> 197 constraints/hour) with the data's intrinsic
    dimension held fixed at 107 dof/hour.  Adding the 186 branch-flow channels
    adds 186 dimensions *and* the 186 constraints that determine them, so any
    change in score is attributable to the structure a model must respect and
    not to a change in the underlying distribution."""
    lo = load(os.path.join(ROOT, "results", "main_grid_noflow.jsonl"))
    hi = load(os.path.join(ROOT, "results", "main_grid.jsonl"))
    if not lo or not hi:
        placeholder("codim", "awaiting run_main.py on suites grid and grid_noflow"); return
    a, b = agg(lo, "eq_max"), agg(hi, "eq_max")
    ea, eb = agg(lo, "energy_score"), agg(hi, "energy_score")
    methods = sorted(set(a) & set(b), key=lambda m: b[m][0])
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "& \\multicolumn{2}{c}{codim 11 (no flows)} & \\multicolumn{2}{c}{codim 197 (flows)} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Method & $\\|Ax-b\\|_\\infty$ & ES & $\\|Ax-b\\|_\\infty$ & ES \\\\",
             "\\midrule"]
    for m in methods:
        lines.append(f"{m.replace('_','X')} & {fmt(a[m][0])} & {fmt(ea[m][0])} & "
                     f"{fmt(b[m][0])} & {fmt(eb[m][0])} \\\\".replace('X', chr(92)+'_'))
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "codim.tex"), "w").write("\n".join(lines) + "\n")
    print(f"  wrote codim.tex ({len(methods)} methods)")


if __name__ == "__main__":
    print("[tables]")
    for s in ("measured", "grid", "grid_noflow"):
        main_table(s)
    codim_table()
    transfer_table()
    downstream_table()
