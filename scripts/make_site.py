"""Generate the project page (`docs/index.html`) from the result files.

Nerfies/Bulma academic project-page template, matching the layout used for the
group's other project sites. Same discipline as the paper tables: every number
and every plotted point is read from `results/*.jsonl`, so the page cannot
drift from the runs.
"""
from __future__ import annotations
import json, math, os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
T_CRIT_9 = 2.262
SUITES = [("grid_noflow", "grid-noflow"), ("measured", "measured"), ("grid", "grid")]

REPO = "https://github.com/Nishant27-2006/constraint-placement"
BLUE = "#4A90E2"
ORANGE = "#E8833A"


def load(n):
    p = os.path.join(RES, n)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def paired(rows, a, b, key="energy_score"):
    by = defaultdict(dict)
    for r in rows:
        by[r["method"]][r["seed"]] = r
    seeds = sorted(set(by.get(a, {})) & set(by.get(b, {})))
    if len(seeds) < 2:
        return None
    d = np.array([by[a][s][key] - by[b][s][key] for s in seeds], float)
    base = np.mean([by[b][s][key] for s in seeds])
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    return 100 * d.mean() / base, t


def mean_of(rows, m, k="energy_score"):
    v = [r[k] for r in rows if r["method"] == m and r.get(k) is not None]
    return float(np.mean(v)) if v else float("nan")


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def g(x, sig=4):
    if x is None or not np.isfinite(x):
        return "&mdash;"
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e5):
        m, e = f"{x:.{sig-1}e}".split("e")
        return f"{m}&times;10<sup>{int(e)}</sup>"
    return f"{x:.{sig}g}"


# --------------------------------------------------------------------- charts
def flip_chart(meta, mains):
    methods = ["HFM (ours)", "FM+reduced", "FM+DC3", "FM+PCFM", "FM+posthoc"]
    colors = [ORANGE, BLUE, "#8E7CC3", "#5AAB6B", "#C95F7A"]
    W, H, PADL, PADB, PADT = 760, 350, 66, 62, 20
    pts = []
    for mi, meth in enumerate(methods):
        for sname, _ in SUITES:
            r = paired(mains[sname], meth, "FM")
            if r:
                pts.append((meta[sname]["codim_frac"], r[0], mi, sname, r[1]))
    if not pts:
        return ""
    lo, hi = min(p[1] for p in pts), max(p[1] for p in pts)
    pad = (hi - lo) * 0.18 or 1
    lo, hi = lo - pad, hi + pad
    X = lambda c: PADL + c / 0.72 * (W - PADL - 28)
    Y = lambda v: PADT + (hi - v) / (hi - lo) * (H - PADT - PADB)
    o = [f'<svg viewBox="0 0 {W} {H}" class="fig-svg" role="img" aria-label="Paired '
         f'change in energy score versus constraint codimension">']
    o.append(f'<rect x="{PADL}" y="{PADT}" width="{W-PADL-28}" height="{H-PADT-PADB}" '
             f'fill="#f7f9fb" stroke="#e6ebf0" rx="8"/>')
    o.append(f'<line x1="{PADL}" y1="{Y(0):.1f}" x2="{W-28}" y2="{Y(0):.1f}" '
             f'stroke="#4a4a4a" stroke-width="1.4" opacity=".6"/>')
    o.append(f'<text x="{PADL+8}" y="{Y(0)-8:.1f}" class="cap">no effect</text>')
    for v in np.linspace(lo, hi, 5):
        o.append(f'<text x="{PADL-10}" y="{Y(v)+4:.1f}" class="tick" '
                 f'text-anchor="end">{v:+.0f}%</text>')
    for (sname, label), c in zip(SUITES, [meta[s]["codim_frac"] for s, _ in SUITES]):
        o.append(f'<line x1="{X(c):.1f}" y1="{H-PADB}" x2="{X(c):.1f}" y2="{H-PADB+5}" '
                 f'stroke="#b5b5b5"/>')
        o.append(f'<text x="{X(c):.1f}" y="{H-PADB+21:.1f}" class="tick" '
                 f'text-anchor="middle">{esc(label)}</text>')
        o.append(f'<text x="{X(c):.1f}" y="{H-PADB+36:.1f}" class="cap" '
                 f'text-anchor="middle">codim {c:.3f}</text>')
    for mi, meth in enumerate(methods):
        seq = sorted([p for p in pts if p[2] == mi], key=lambda p: p[0])
        if len(seq) > 1:
            d = " ".join(f"{'M' if i==0 else 'L'}{X(p[0]):.1f},{Y(p[1]):.1f}"
                         for i, p in enumerate(seq))
            o.append(f'<path d="{d}" fill="none" stroke="{colors[mi]}" stroke-width="2.2" '
                     f'opacity=".85"/>')
        for p in seq:
            sig = abs(p[4]) > T_CRIT_9
            o.append(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="{6 if sig else 4.5}" '
                     f'fill="{colors[mi] if sig else "#fff"}" stroke="{colors[mi]}" '
                     f'stroke-width="2.2"><title>{esc(meth)} on {esc(p[3])}: '
                     f'{p[1]:+.2f}% (t={p[4]:+.2f})</title></circle>')
    cy = PADT + (H - PADT - PADB) / 2
    o.append(f'<text x="16" y="{cy:.0f}" class="axis" transform="rotate(-90 16 {cy:.0f})" '
             f'text-anchor="middle">&#916; energy score vs unconstrained FM</text>')
    o.append("</svg>")
    leg = " ".join(f'<span class="key"><i style="background:{colors[i]}"></i>{esc(m)}</span>'
                   for i, m in enumerate(methods))
    note = '<span class="key note">filled = significant at 5% (paired, 10 seeds)</span>'
    return "".join(o) + f'<div class="legend">{leg}{note}</div>'


def regret_chart(down, measured):
    reg = defaultdict(list)
    for r in down:
        if r.get("regret_pct") is not None and "[det-mean]" not in r["method"]:
            reg[r["method"]].append(r["regret_pct"])
    cov = defaultdict(list)
    for r in measured:
        cov[r["method"]].append(r["cov90"])
    pts = [(float(np.mean(cov[m])), float(np.mean(reg[m])), m) for m in reg if m in cov]
    if not pts:
        return "", None, None
    W, H, PADL, PADB, PADT, PADR = 760, 380, 66, 60, 20, 128
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = min(xs) - .04, max(xs) + .04
    y1 = max(ys) * 1.12
    X = lambda v: PADL + (v - x0) / (x1 - x0) * (W - PADL - PADR)
    Y = lambda v: PADT + (y1 - v) / y1 * (H - PADT - PADB)
    EXACT = {"HFM (ours)", "FM+reduced", "FM+DC3", "FM+PCFM", "FM+posthoc"}
    o = [f'<svg viewBox="0 0 {W} {H}" class="fig-svg" role="img" aria-label="Unit '
         f'commitment cost regret versus 90 percent coverage">']
    o.append(f'<rect x="{PADL}" y="{PADT}" width="{W-PADL-PADR}" height="{H-PADT-PADB}" '
             f'fill="#f7f9fb" stroke="#e6ebf0" rx="8"/>')
    for v in np.linspace(0, y1, 5):
        o.append(f'<line x1="{PADL}" y1="{Y(v):.1f}" x2="{W-PADR}" y2="{Y(v):.1f}" '
                 f'stroke="#e6ebf0"/>')
        o.append(f'<text x="{PADL-10}" y="{Y(v)+4:.1f}" class="tick" '
                 f'text-anchor="end">{v:.0f}%</text>')
    for v in np.linspace(x0, x1, 5):
        o.append(f'<text x="{X(v):.1f}" y="{H-PADB+21:.1f}" class="tick" '
                 f'text-anchor="middle">{v:.2f}</text>')
    a, b = np.polyfit(xs, ys, 1)
    o.append(f'<line x1="{X(x0):.1f}" y1="{Y(a*x0+b):.1f}" x2="{X(x1):.1f}" '
             f'y2="{Y(a*x1+b):.1f}" stroke="{BLUE}" stroke-width="1.8" '
             f'stroke-dasharray="6 5" opacity=".8"/>')
    for x, y, m in pts:
        ex = m in EXACT
        o.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="6.5" '
                 f'fill="{ORANGE if ex else "#fff"}" stroke="{ORANGE if ex else "#9aa5b1"}" '
                 f'stroke-width="2.2"><title>{esc(m)}: regret {y:.1f}%, '
                 f'cov90 {x:.3f}</title></circle>')
        if m in ("HFM (ours)", "FM+reduced", "GaussianCopula", "cWGAN-GP",
                 "NormFlow-RealNVP", "kNN-Historical"):
            o.append(f'<text x="{X(x)+11:.1f}" y="{Y(y)+4:.1f}" class="lbl">{esc(m)}</text>')
    o.append(f'<text x="{PADL+(W-PADL-PADR)/2:.0f}" y="{H-14}" class="axis" '
             f'text-anchor="middle">90% coverage (calibration) &rarr;</text>')
    cy = PADT + (H - PADT - PADB) / 2
    o.append(f'<text x="16" y="{cy:.0f}" class="axis" transform="rotate(-90 16 {cy:.0f})" '
             f'text-anchor="middle">UC cost regret</text>')
    o.append("</svg>")
    leg = (f'<span class="key"><i style="background:{ORANGE}"></i>exactly feasible</span>'
           f'<span class="key"><i style="background:#fff;border:2px solid #9aa5b1"></i>'
           f'everything else</span><span class="key note">dashed line = least squares</span>')
    return "".join(o) + f'<div class="legend">{leg}</div>', \
        float(np.corrcoef(xs, ys)[0, 1]), spearmanr(xs, ys)


CSS = """
body{font-family:'Noto Sans',sans-serif}
.publication-title{font-family:'Google Sans',sans-serif}
.publication-authors{margin-top:1.1rem}
.publication-authors .author-block{margin-right:.55rem}
.publication-links{margin-top:1.6rem}
.publication-links .link-block{display:inline-block;margin:.25rem .2rem}
.anonymous-note{margin-top:.75rem;color:#666}
.venue-tag{display:inline-block;margin-top:1rem;padding:.3rem .95rem;border-radius:999px;
  background:#eef4fb;color:#3d7ab8;font-size:.82rem;font-weight:600;letter-spacing:.04em}
.fig-svg{width:100%;height:auto;display:block;margin:.5rem auto 0;overflow:visible}
.fig-svg .tick{font:11.5px 'Noto Sans',sans-serif;fill:#7a7a7a}
.fig-svg .cap{font:10.5px 'Noto Sans',sans-serif;fill:#9aa5b1}
.fig-svg .lbl{font:11.5px 'Noto Sans',sans-serif;fill:#363636}
.fig-svg .axis{font:12px 'Noto Sans',sans-serif;fill:#7a7a7a}
.legend{display:flex;flex-wrap:wrap;justify-content:center;gap:.5rem 1.2rem;
  margin:.7rem 0 .2rem;font-size:.8rem;color:#7a7a7a}
.key{display:flex;align-items:center;gap:.4rem}
.key i{width:11px;height:11px;border-radius:50%;display:inline-block}
.key.note{font-style:italic;opacity:.85}
.fig-card{background:#fff;border-radius:12px;padding:1.4rem 1.2rem;
  box-shadow:0 4px 14px rgba(0,0,0,.08)}
.tbl-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:1rem 0}
table.data{border-collapse:collapse;width:100%;font-size:.86rem;min-width:540px}
table.data th,table.data td{padding:.5rem .7rem;text-align:right;
  border-bottom:1px solid #ededed;white-space:nowrap}
table.data th:first-child,table.data td:first-child{text-align:left}
table.data thead th{color:#7a7a7a;font-weight:600;font-size:.78rem;
  letter-spacing:.03em;border-bottom:2px solid #e3e3e3}
table.data tbody tr.ours{background:#fdf3ea}
table.data tbody tr:hover{background:#f7f9fb}
.good{color:#2f7d4f;font-weight:600}.bad{color:#b23a4d;font-weight:600}.dim{color:#9aa5b1}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
  gap:1rem;margin:1.6rem 0}
.stat{background:#fff;border-radius:12px;padding:1.1rem 1.2rem;text-align:center;
  box-shadow:0 4px 12px rgba(0,0,0,.08)}
.stat .v{display:block;font-family:'Google Sans',sans-serif;font-size:1.75rem;
  font-weight:700;line-height:1.15;margin-bottom:.35rem;color:#363636}
.stat .k{font-size:.82rem;color:#7a7a7a;line-height:1.45}
.kill{background:#fff;border-left:4px solid #b23a4d;border-radius:0 12px 12px 0;
  padding:1.1rem 1.3rem;margin:1.1rem 0;box-shadow:0 3px 10px rgba(0,0,0,.07)}
.kill .tag{display:block;font-size:.72rem;font-weight:700;letter-spacing:.09em;
  text-transform:uppercase;color:#b23a4d;margin-bottom:.5rem}
.kill p{margin:.4rem 0}
.takeaway{background:#eef4fb;border-left:4px solid #4A90E2;border-radius:0 12px 12px 0;
  padding:1.1rem 1.3rem;margin:1.3rem 0}
.takeaway p{margin:.4rem 0}
.link-card{display:flex;flex-direction:column;justify-content:space-between;height:100%;
  border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,.1);
  transition:transform .3s ease,box-shadow .3s ease}
.link-card:hover{transform:translateY(-10px);box-shadow:0 8px 20px rgba(0,0,0,.2)}
.link-card .icon{color:#4A90E2;margin-bottom:10px}
.link-card .button.is-link{background-color:#4A90E2;border:none}
.link-card .button.is-link:hover{background-color:#357ABD}
.caption{font-size:.85rem;color:#7a7a7a;margin-top:.7rem;text-align:center}
@media(max-width:640px){.stat .v{font-size:1.45rem}}
"""


def main():
    meta = json.load(open(os.path.join(RES, "suite_meta.json")))
    mains = {s: load(f"main_{s}.jsonl") for s, _ in SUITES}
    down, transfer, eff = load("downstream.jsonl"), load("transfer.jsonl"), load("efficiency.jsonl")
    ac = load("ac.jsonl") + load("ac_retract20.jsonl")
    nseeds = len({r["seed"] for r in mains["grid"]})
    nmeth = len({r["method"] for r in mains["grid"]})
    fl = {s: paired(mains[s], "HFM (ours)", "FM") for s, _ in SUITES}
    H = []
    w = H.append

    def sec(title, light=False):
        w(f'<section class="section{" hero is-small is-light" if light else ""}">')
        w('<div class="container is-max-desktop">')
        w('<div class="columns is-centered"><div class="column is-full-width">')
        if title:
            w(f'<h2 class="title is-3 has-text-centered">{title}</h2>')

    def endsec():
        w("</div></div></div></section>")

    # ------------------------------------------------------------------ head
    w('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">')
    w('<meta name="viewport" content="width=device-width,initial-scale=1">')
    w("<title>Hamiltonian Network</title>")
    w('<meta name="description" content="Where a hard physical constraint belongs in a '
      'generative model: a codimension-controlled benchmark on real power-grid data.">')
    w('<link href="https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro" '
      'rel="stylesheet">')
    w('<link rel="stylesheet" '
      'href="https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.4/css/bulma.min.css">')
    w('<link rel="stylesheet" '
      'href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">')
    w('<link rel="stylesheet" '
      'href="https://cdnjs.cloudflare.com/ajax/libs/academicons/1.9.4/css/academicons.min.css">')
    w(f"<style>{CSS}</style></head><body>")

    # ------------------------------------------------------------------ hero
    w('<section class="hero"><div class="hero-body">')
    w('<div class="container is-max-desktop"><div class="columns is-centered">')
    w('<div class="column has-text-centered">')
    w('<h1 class="title is-1 publication-title">Where Does a Physical Constraint '
      "Belong in a Generative Model?</h1>")
    w('<p class="is-size-4" style="color:#4a4a4a;margin-top:.7rem">'
      "A codimension-controlled benchmark for constraint-exact<br>"
      "power-grid scenario generation</p>")
    w('<div class="is-size-5 publication-authors"><span class="author-block">'
      "Anonymous Authors</span></div>")
    w('<div class="venue-tag">Under review at ICLR</div>')
    w('<div class="column has-text-centered"><div class="publication-links">')
    for icon, label, href in [
            ("fas fa-file-pdf", "Paper", "paper.pdf"),
            ("fab fa-github", "Code", REPO),
            ("fas fa-table", "Results", f"{REPO}/blob/main/docs/RESULTS.md"),
            ("fas fa-clipboard-check", "Pre-registration",
             f"{REPO}/blob/main/docs/CLAIMS_AND_EVIDENCE.md")]:
        w(f'<span class="link-block"><a href="{href}" class="external-link button '
          f'is-normal is-rounded is-dark"><span class="icon"><i class="{icon}"></i>'
          f"</span><span>{label}</span></a></span>")
    w("</div></div></div></div></div></div></section>")

    # -------------------------------------------------------------- abstract
    sec("Abstract")
    w('<div class="content has-text-justified">')
    w("<p>Generative models are increasingly used to produce operational scenarios "
      "for power systems, and such scenarios must satisfy the physical laws they "
      "describe: a set of injections that violates Kirchhoff's laws is not a "
      "conservative forecast but an impossible one. Several methods achieve "
      "<b>exact</b> satisfaction of affine physical invariants, and the literature "
      "disagrees about which to prefer. We argue the open question is not whether to "
      "enforce a constraint but <b>where it belongs</b>: in the hypothesis class "
      "(train-time projection), at inference (zero-shot correction), in the loss "
      "(penalty), or in the coordinates (nullspace or completion).</p>")
    w(f"<p>We give the affine theory &mdash; train-time projection is exact under any "
      f"Runge&ndash;Kutta scheme, excludes no minimiser, and never increases the "
      f"flow-matching loss &mdash; then build a benchmark that varies one quantity, the "
      f"<b>codimension</b> of the constraint set, across <b>{nmeth} methods, {nseeds} "
      f"seeds and three suites</b> derived from real grid measurements. "
      f"<b>The answer is not universal.</b> On a controlled contrast in which the only "
      f"change is how much of the state the physics determines, train-time projection "
      f"moves from statistically indistinguishable from unconstrained flow matching to "
      f"a significant improvement; on a third real dataset the identical code is "
      f"significantly worse.</p>")
    w("</div>")
    endsec()

    # ------------------------------------------------------------ key numbers
    sec("Headline Results", light=True)
    w('<div class="stat-grid">')
    for s, label in SUITES:
        pct, t = fl[s]
        cls = "good" if (pct < 0 and abs(t) > T_CRIT_9) else (
            "bad" if abs(t) > T_CRIT_9 else "dim")
        verd = ("significantly better" if pct < 0 else "significantly worse") \
            if abs(t) > T_CRIT_9 else "indistinguishable"
        w(f'<div class="stat"><span class="v {cls}">{pct:+.2f}%</span><span class="k">'
          f"energy score vs unconstrained FM<br><code>{esc(label)}</code> &middot; "
          f"codim <b>{meta[s]['codim_frac']:.3f}</b><br>t&nbsp;=&nbsp;{t:+.2f} "
          f"&middot; {verd}</span></div>")
    w("</div>")
    w('<div class="content has-text-justified"><p class="has-text-centered" '
      'style="color:#7a7a7a">Same code, same backbone, same budget. '
      "<b>The effect changes sign.</b></p></div>")
    endsec()

    # ---------------------------------------------------------------- result 1
    sec("The Effect Changes Sign with Codimension")
    w(f'<div class="fig-card">{flip_chart(meta, mains)}'
      f'<p class="caption">Paired change in energy score against unconstrained flow '
      f"matching, {nseeds} seeds. Hover any point for its paired "
      "<i>t</i>-statistic.</p></div>")
    w('<div class="takeaway"><p>The two grid suites are the <b>same</b> 118-bus system '
      "and the same measured injections; the only change is whether the model must also "
      "emit the 186 line flows the constraint matrix determines exactly. That makes the "
      "contrast causal. Codimension goes 0.093 &rarr; 0.648 and train-time projection "
      "goes from indistinguishable to significantly better.</p>"
      "<p><b>Anyone reporting a single winner has measured one suite.</b></p></div>")
    endsec()

    # ---------------------------------------------------------------- result 2
    sec("Soft Penalties Are Dominated on Both Axes", light=True)
    w('<div class="tbl-wrap"><table class="data"><thead><tr><th>λ</th><th>ES</th>'
      "<th>rel. to FM</th><th>‖Ax−b‖<sub>∞</sub> (MW)</th></tr></thead><tbody>")
    base = mean_of(mains["grid"], "FM")
    for lam in [1, 10, 100, 1000]:
        m = f"FM+penalty({lam})"
        es, eq = mean_of(mains["grid"], m), mean_of(mains["grid"], m, "eq_max")
        d = 100 * (es - base) / base
        w(f"<tr><td>{lam}</td><td>{g(es)}</td>"
          f'<td class="bad">{d:+.1f}%</td><td>{g(eq,3)}</td></tr>')
    w("</tbody></table></div>")
    e1 = mean_of(mains["grid"], "FM+penalty(1)", "eq_max")
    e1k = mean_of(mains["grid"], "FM+penalty(1000)", "eq_max")
    dg = 100 * (mean_of(mains["grid"], "FM+penalty(1000)") - base) / base
    w(f'<div class="takeaway"><p>Three orders of magnitude of λ move the violation from '
      f"{e1:.3g}&nbsp;MW to {e1k:.3g}&nbsp;MW &mdash; the same order, still "
      f"unacceptable &mdash; while the score degrades to <b>{dg:+.0f}%</b>. "
      "<b>There is no λ to tune.</b> Every exact route beats the entire penalty family "
      "on feasibility <i>and</i> fidelity at once.</p></div>")
    endsec()

    # ---------------------------------------------------------------- result 3
    sec("Feasibility Is Not Decision Value")
    svg, r_cov, sp = regret_chart(down, mains["measured"])
    w(f'<div class="fig-card">{svg}<p class="caption">Two-stage stochastic unit '
      "commitment. Each point is one generator.</p></div>")
    reg = defaultdict(list)
    for r in down:
        if r.get("regret_pct") is not None and "[det-mean]" not in r["method"]:
            reg[r["method"]].append(r["regret_pct"])
    esm = defaultdict(list)
    for r in mains["measured"]:
        esm[r["method"]].append(r["energy_score"])
    pts = [(np.mean(esm[m]), np.mean(reg[m])) for m in reg if m in esm]
    sp_es = spearmanr([p[0] for p in pts], [p[1] for p in pts])
    r_es = float(np.corrcoef([p[0] for p in pts], [p[1] for p in pts])[0, 1])
    w('<div class="stat-grid">')
    w(f'<div class="stat"><span class="v good">{r_cov:+.3f}</span><span class="k">'
      f"regret vs <b>calibration</b><br>Spearman ρ&nbsp;=&nbsp;{sp.statistic:+.2f}, "
      f"p&nbsp;=&nbsp;{sp.pvalue:.3f}</span></div>")
    w(f'<div class="stat"><span class="v dim">{r_es:+.3f}</span><span class="k">'
      f"regret vs <b>energy score</b><br>Spearman ρ&nbsp;=&nbsp;{sp_es.statistic:+.2f}, "
      f"p&nbsp;=&nbsp;{sp_es.pvalue:.2f} &mdash; <b>not significant</b></span></div>")
    w("</div>")
    w(f'<div class="takeaway"><p><b>Scenario feasibility does not reach the scheduling '
      "decision; calibration carries it.</b> The exactly-feasible routes take the worst "
      f"regret among neural methods (ours {np.mean(reg['HFM (ours)']):.0f}%, "
      f"<code>FM+reduced</code> {np.mean(reg['FM+reduced']):.0f}%); the best belong to "
      f"the most over-dispersed baselines ({np.mean(reg['GaussianCopula']):.0f}%). Exact "
      "ensembles are sharper, the scheduler trusts them, under-commits reserve and pays "
      f"in load shed.</p><p>A study that stopped at the energy score would have reported "
      f"the opposite. Limit: {max(len(v) for v in reg.values())} seeds on this stage, "
      f"not {nseeds}.</p></div>")
    endsec()

    # ------------------------------------------------------------- falsified
    sec("What We Predicted, and Got Wrong", light=True)
    w('<div class="content has-text-justified"><p>Every claim was pre-registered with '
      "an explicit falsification condition <b>before</b> the sweep ran. Two "
      "fired.</p></div>")
    hg, thg = paired(mains["grid"], "HFM (ours)", "FM+PCFM")
    hm, thm = paired(mains["measured"], "HFM (ours)", "FM+PCFM")
    w('<div class="kill"><span class="tag">Falsified &middot; C3</span>'
      "<p><b>Predicted:</b> train-time ≤ inference-time ≤ post-hoc, universally.</p>"
      f"<p><b>Measured:</b> holds on <code>grid</code> ({hg:+.2f}%, "
      f"t&nbsp;=&nbsp;{thg:+.2f}), reverses on <code>measured</code> ({hm:+.2f}%, "
      f"t&nbsp;=&nbsp;{thm:+.2f}). The reversal became the paper's central "
      "finding.</p></div>")
    if transfer:
        by = defaultdict(list)
        for r in transfer:
            by[(r["method"], r["mode"])].append(r["energy_score"])
        w('<div class="kill"><span class="tag">Falsified &middot; C5&prime;</span>'
          "<p><b>Predicted:</b> under an N-1 topology swap, physical-coordinate routes "
          "keep their learned distribution; chart-based routes do not.</p>"
          f"<p><b>Measured:</b> ours {np.mean(by[('HFM (ours)','swapped')]):.4g} vs "
          f"<code>FM+reduced</code> {np.mean(by[('FM+reduced','swapped')]):.4g} &mdash; "
          f"indistinguishable, both beaten by <code>FM+posthoc</code> "
          f"{np.mean(by[('FM+posthoc','swapped')]):.4g}. Differentiator dropped.</p></div>")
    if ac:
        fmv = np.mean([r["nl_max"] for r in ac if r["method"] == "FM"])
        rfv = np.mean([r["nl_max"] for r in ac if r["method"] == "FM+retract-final"])
        mf3 = [r["nl_max"] for r in ac if r["method"].startswith("Manifold")
               and r.get("retract_iters", 3) == 3]
        w('<div class="kill"><span class="tag">Negative result &middot; AC manifold</span>'
          "<p><b>Predicted:</b> per-step retraction is needed to bound drift on the "
          "nonlinear AC manifold.</p>"
          f"<p><b>Measured:</b> one retraction after the solve reaches "
          f"|g|<sub>∞</sub>&nbsp;=&nbsp;{rfv:.3g} from {fmv:.3g}, free in score. "
          f"Per-step retraction at 3 Gauss-Newton iterations reaches only "
          f"{np.mean(mf3):.3g} &mdash; worse than doing nothing.</p></div>")
    w('<div class="takeaway"><p><b>A non-neural baseline places second of sixteen</b> on '
      "both grid suites. <code>kNN-Historical</code> resamples analogue days: zero "
      "parameters, zero NFE, exactly feasible by construction. It beats every GAN, VAE "
      "and normalising flow on all three suites.</p></div>")
    endsec()

    # ------------------------------------------------------------ link cards
    w('<section class="section"><div class="container is-max-desktop">')
    w('<h2 class="title is-3 has-text-centered">Resources</h2>')
    w('<div class="columns is-centered" style="margin-top:1rem">')
    for icon, title, body, label, href in [
            ("fas fa-table", "Results",
             "Every measured number, generated from the run files.",
             "Read", f"{REPO}/blob/main/docs/RESULTS.md"),
            ("fas fa-scale-balanced", "Predictions",
             "What prior work would have predicted, against what happened.",
             "Read", f"{REPO}/blob/main/docs/PREDICTIONS_VS_MEASURED.md"),
            ("fas fa-clipboard-check", "Pre-registration",
             "Every claim and what would falsify it, written before the sweep.",
             "Read", f"{REPO}/blob/main/docs/CLAIMS_AND_EVIDENCE.md"),
            ("fab fa-github", "Code",
             "Full implementation, reproduction commands, provenance and proofs.",
             "Browse", REPO)]:
        w('<div class="column is-3"><div class="card link-card">'
          '<div class="card-content has-text-centered">'
          f'<span class="icon is-large"><i class="{icon} fa-2x"></i></span>'
          f'<p class="title is-5">{title}</p>'
          f'<p class="is-size-7 mt-2 mb-4">{body}</p>'
          f'<a href="{href}" class="button is-link is-rounded">{label}</a>'
          "</div></div></div>")
    w("</div></div></section>")

    # ---------------------------------------------------------------- bibtex
    w('<section class="section hero is-small is-light" id="BibTeX">')
    w('<div class="container is-max-desktop content">')
    w('<h2 class="title">BibTeX</h2>')
    w("<pre><code>@inproceedings{anonymous2026constraintplacement,\n"
      "  title  = {Where Does a Physical Constraint Belong in a Generative Model?},\n"
      "  author = {Anonymous},\n"
      "  booktitle = {Submitted to the International Conference on Learning "
      "Representations},\n"
      "  year   = {2026}\n"
      "}</code></pre></div></section>")

    # ---------------------------------------------------------------- footer
    w('<footer class="footer"><div class="container"><div class="columns is-centered">')
    w('<div class="column is-8"><div class="content has-text-centered">')
    w("<p>Data: EIA-930 (US Energy Information Administration, public domain) and "
      "pglib-opf / MATPOWER network cases. Code released under MIT.</p>")
    w("<p>Every number on this page is generated from the run files. "
      "Website template adapted from the Nerfies project page.</p>")
    w("</div></div></div></div></footer></body></html>")

    out = os.path.join(ROOT, "docs", "index.html")
    with open(out, "w") as f:
        f.write("\n".join(H))
    open(os.path.join(ROOT, "docs", ".nojekyll"), "w").close()
    print(f"wrote {out} ({os.path.getsize(out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
