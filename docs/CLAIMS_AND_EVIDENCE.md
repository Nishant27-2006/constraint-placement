# Claims, evidence, and falsification conditions

This is the contract for the paper. Every claim we intend to make is listed with
(a) the experiment that establishes it, (b) the artefact it produces, and
(c) **the result that would falsify it**. Claims whose falsification condition
fires are cut from the paper, not softened.

This file is written *before* the sweep so that the acceptance criteria cannot
drift to fit whatever comes out.

Status key: `VERIFIED` measured already · `PENDING` queued · `AT RISK` known threat

---

## C1 — Projection onto an affine invariant is exact, free, and bias-free
**Status:** `VERIFIED` (theory) + `VERIFIED` (numerics) + `PENDING` (at scale)

* **Theory.** Theorem 1, `paper/sec_theory.tex` and `docs/PROOFS.md`. Four parts:
  path feasibility, no bias at the optimum, monotone loss improvement by
  Pythagoras, exactness under any Runge–Kutta scheme.
* **Numerics already in hand.** Residual `1.0e-14` after projection, unchanged
  after 50 Euler steps; identical across euler/heun/rk4/dopri5; `float64` gives
  `2.4e-14` where `float32` gives `7.4e-06`.
* **At scale.** `results/main_{measured,grid}.jsonl` → column `eq_max`.
* **Falsified if:** the residual for `HFM (ours)` grows with step count, or
  exceeds `float32` round-off relative to state magnitude, or the projected
  model's training loss is systematically *above* the unprojected model's.

## C2 — Unconstrained generative models violate grid physics by operationally large margins
**Status:** `VERIFIED` (smoke) → `PENDING` (full)

* Smoke test, Suite A, 15 epochs: unconstrained FM `eq_max = 2.47e+05 MW`;
  Gaussian copula `4.78e+04 MW`. HFM `4.44e-02 MW`.
* **Falsified if:** after full training the unconstrained baselines' violations
  fall to operationally negligible levels (say < 1 MW). Then the constraint
  machinery solves a non-problem and the paper must say so.

## C3 — *Where* the constraint goes changes distributional fidelity, not just feasibility
**Status:** `PENDING` — **this is the paper's central empirical question**

Six routes to the same feasible set, identical backbone and budget:
`FM+penalty(λ)` · `FM+posthoc` · `FM+PCFM` (reimpl.) · `FM+DC3` · `FM+reduced` ·
`HFM (ours)`.

* **Prediction.** Train-time projection ≥ inference-time correction ≥ post-hoc
  projection on energy score and variogram score at matched NFE, because the
  projected network spends no capacity on the `rank(A)` directions that are
  known analytically.
* **Artefact.** Main results table, `results/main_*.jsonl`.
* **Falsified if:** the methods are statistically indistinguishable across 3
  seeds. **If that happens we report it as the finding** — "for affine
  invariants the choice does not matter, use whichever is cheapest" is a useful
  negative result and the cost table then carries the paper.

## C4 — Exactness is free in compute for affine invariants; inference-time correction is not
**Status:** `VERIFIED` (mechanism) → `PENDING` (measured)

* Mechanism: `Π_V` is one cached `D×D` matrix per hour → one matvec per network
  evaluation, **zero** extra NFE. PCFM inserts a projection inside every step:
  measured **51 extra projections** for a 50-step solve.
* **Artefact.** Columns `nfe`, `flops_per_scenario`, `extra_projections`,
  `latency_ms_per_scenario`.
* **Falsified if:** the measured latency of HFM is not materially below PCFM's
  at equal NFE and equal residual.

## C5 — Zero-shot transfer to an N-1 contingency topology
**Status:** `FALSIFIED AS ORIGINALLY STATED (2026-09-18)` → **restated below**

### What was claimed, and what killed it
The original C5 predicted that only projector-based routes could stay exact when
an outage changes the PTDF, and pre-registered: *"Falsified if chart-based
methods remain exact under the swapped constraint."*

They remain exact. Measured on contingency branch 2 (smoke configuration):

| method | chart | $\|A'x-b'\|_\infty$ (MW) |
|---|---|---|
| HFM (ours) | swapped projector | 4.18e-04 |
| FM+reduced | stale | 6.57e+01 |
| FM+reduced | **swapped** | **6.83e-05** |
| FM+DC3 | stale | 7.50e+01 |
| FM+DC3 | **swapped** | **8.98e-05** |

This is not an empirical accident, it is structural: any point of the form
`x = x_p' + N' c` lies in the new feasible set for *every* `c`, so rebuilding the
chart makes a chart-based generator exactly feasible regardless of what the
network emits. It is the same 1-Lipschitz-projection logic as Proposition 3, and
it should have been anticipated when that proposition was written.

Nor is there a cost advantage: rebuilding a projector and rebuilding a nullspace
basis are the same decomposition of the same matrix.

### C5′ (restated) — the surviving question is *fidelity*, not feasibility
**Status:** `PENDING`

Feasibility under topology change is available to every exact route. What differs
is what the model's coordinates *mean* after the swap:

* **Physical coordinates (ours, PCFM).** The network's inputs and outputs are
  MW at a bus. An outage changes the projector, not the semantics, so the learned
  field still means what it meant.
* **Chart coordinates (reduced, DC3).** Each latent dimension is defined by the
  old basis. Swapping the chart reinterprets the trained network's output in a
  different coordinate system: the samples are feasible but the distribution they
  induce is no longer the learned one.

* **Falsified if:** under the swapped chart, `FM+reduced` and `FM+DC3` match the
  projector routes on energy score and variogram score against the contingency
  ground truth (3 seeds, 8 contingencies). If they match, the paper reports that
  the four exact routes are interchangeable under topology change and drops the
  transfer differentiator entirely.

* **Note on evidence quality.** The table above is from a 2-epoch smoke run, so
  its ES values carry no information. Only the feasibility column is meaningful
  there, because feasibility is structural and training-independent.

## C6 — Constraint-exactness changes the scheduling decision, not just the score
**Status:** `PENDING` — **the "does it matter" test; nobody in this literature has run it**

Two-stage stochastic UC; commitment frozen; scored on the realised day. Harness
already shown to discriminate: +141.42 % regret for a point forecast vs +0.413 %
for a good scenario set.

* **Artefact.** `results/downstream.jsonl`.
* **Falsified if:** out-of-sample cost regret is indistinguishable across
  generators. Then scenario feasibility does not reach the decision, which is
  itself worth reporting and would substantially deflate the paper's
  significance claim. We would say so plainly.

## C7 — Efficiency should be reported as NFE/FLOPs, not device joules
**Status:** `VERIFIED` (position) → `PENDING` (frontier)

* NFE and analytic FLOPs are exactly countable and hardware-independent.
  Measured energy is reported as a *secondary*, explicitly whole-system,
  device-specific figure from SMC telemetry — never as the headline.
* **We do not claim** the draft's "3.1x energy reduction". Adaptive
  step-size generative sampling already exists (`Gotta Go Fast`,
  torchdiffeq/dopri5), so no novelty is claimed for adaptivity itself; we report
  the frontier against `dopri5` at matched tolerance.
* **Falsified if:** our adaptive schedule does not beat a plain `dopri5` at
  matched tolerance — in which case we report `dopri5` as the recommendation.

## C8 — Physical priors must be verified against data, not assumed
**Status:** `VERIFIED`

Two priors the original draft asserted are **false in the real record**:
1. EIA's published `Net Generation` ≠ its published fuel sum; residuals to
   `3.3e+04 MW` (PJM 2023). Fraction of hours exactly consistent: 0 %.
2. Reported solar is **not** zero at night — CISO 2023 has 3,326 negative
   night-time values, minimum `−87 MW`.

Both are reported; neither is imposed. This is a methodological contribution in
its own right and costs nothing to defend because it is a measurement.

## C9 — A non-neural baseline may win
**Status:** `AT RISK`, and stays in the paper either way

`kNN-Historical` resamples analogue days: it is exactly constraint-satisfying by
construction (`eq_max = 0.00e+00`) and at 15 epochs beat every neural method
(ES `1.33e+05` vs FM `2.15e+05`). At 300 epochs it still led the two methods
scored so far.

* **Commitment:** if it still leads after the full sweep, it goes in the
  abstract. A benchmark paper that hides its strongest baseline is worthless.

---

## Claims we have *withdrawn* from the draft

| Draft claim | Why withdrawn |
|---|---|
| "62 % reduction in physical-constraint violations" | No code existed; and the honest comparison is ~0 % violation vs several exact methods, not a percentage reduction |
| "3.1x lower inference energy" | Measured on a GPU never run; adaptive sampling is not novel (`Gotta Go Fast`, dopri5) |
| "18.4 % higher RL reward" | No environment existed; replaced by a stochastic-UC cost-regret evaluation, which is the established protocol |
| "Hamiltonian Flow Matching" (name) | Collides with Holderrieth et al., NeurIPS 2024, which uses it for something else; our method is not symplectic |
| "first to enforce hard physical constraints in flow matching" | PCFM (NeurIPS 2025) does exactly this, zero-shot, for arbitrary nonlinear constraints |
| Theorem 1 as stated ("as λ→∞ the penalty → 0") | Vacuous; replaced with a substantive, proved statement |
| Eq. (2) projection formula | Ill-typed — projected onto the span of a single vector called an "orthonormal complement"; would have destroyed the flow-matching target |
| Night-time-solar-is-zero constraint | Measured false in the data |
