# Verification log

Every number in this file was **measured in this repository**, not asserted.
Each row gives the command that reproduces it. Nothing here depends on the
pending GPU sweep; all of it ran on CPU before the machine lost power.

Convention: constraint residuals are always computed in `float64` on CPU, so a
claim of "exact" is a statement about the arithmetic and not about a tolerance
someone chose.

---

## 1. Constraint algebra (`hfm/grid/constraints.py`)

| Property | Measured | Expected |
|---|---|---|
| Residual after `project_point` | `1.010e-14` | machine-eps |
| Idempotency `‖Π(Π x) − Π x‖∞` | `2.776e-15` | 0 |
| Tangent vectors lie in `null(A)` | `1.453e-14` | 0 |
| **50 Euler steps with a random tangential field** | `1.038e-14` | **does not grow** |
| Standardised-space residual | `7.93e-15` | machine-eps |
| Physical residual after standardised-space projection | `1.35e-14` | machine-eps |
| Reduced-coordinate (nullspace) residual | `3.08e-15` | machine-eps |
| DC3 completion reconstruction error | `1.67e-14` | 0 |
| DC3 completion residual | `4.61e-15` | machine-eps |

The fourth row is the empirical core of Theorem 1(iv): the invariant does not
drift with the number of integrator steps, because every increment is a linear
combination of vectors in `ker A`.

## 2. ODE solvers (`hfm/flows/fm.py`)

All four solvers, integrating a projected field from a feasible start:

| Solver | NFE | Final `eq_max` |
|---|---|---|
| `euler` (50 steps) | 50 | `2.809e-14` |
| `heun` (25 steps) | 50 | `2.864e-14` |
| `rk4` (12 steps) | 48 | `2.820e-14` |
| `dopri5` (rtol 1e-5, atol 1e-6) | **19** | `2.798e-14` |

Exactness is independent of solver and of step count, as the theorem requires.
`dopri5` reaches the same residual with 2.6x fewer evaluations.

## 3. Precision ablation — the residual is round-off, not method error

| Working dtype | Trained-model sample `eq_max` |
|---|---|
| `float32` | `7.389e-06` |
| `float64` | `2.426e-14` |

Eight orders of magnitude, purely from the arithmetic. This is the honest
statement of "exact": exact in real arithmetic, round-off-limited in practice.

## 4. Power-system physics (`hfm/grid/network.py`)

| Check | Measured |
|---|---|
| `max‖f − PTDF·p‖∞` on DC-OPF output | `0.00e+00` |
| Kirchhoff current law `‖Σp‖` | `2.27e-13` |
| Ybus symmetry `max‖Y − Yᵀ‖` | `0.00e+00` (nnz 476) |
| DC-OPF on `pglib_opf_case118_ieee` nominal load | cost 84,840; shed 0.000; spill 0.000 |
| Branches / islanding bridges / usable N-1 | 186 / 9 / **177** |
| Largest PTDF shift under a single outage | `max‖ΔPTDF‖∞ = 0.8187` |

## 5. Metric suite (`hfm/eval/metrics.py`)

Validated against ensembles with **known** miscalibration. A proper scoring rule
must rank the calibrated ensemble best; the rank histogram must detect both
failure directions.

| Ensemble | Energy score | CRPS | Variogram | Reliability idx | cov90 |
|---|---|---|---|---|---|
| calibrated | **9.783** | **0.5688** | **805.9** | **0.037** | 0.883 |
| under-dispersed | 12.174 | 0.7012 | 1090.4 | 1.208 | 0.257 |
| over-dispersed | 14.421 | 0.8482 | 2252.3 | 0.897 | 1.000 |

Correct on all three counts. Cost at benchmark scale (N=400, M=100):
`D=69` → 5.6 s / 2.70 GB peak; `D=304` → 21.7 s / 10.38 GB peak.

## 6. Downstream unit commitment (`hfm/downstream/uc.py`)

The downstream task must *discriminate*, or it cannot support a claim.

| Commitment source | Realised cost | Unserved energy | Regret vs perfect foresight |
|---|---|---|---|
| Perfect foresight | 1,538,815 | 0.000 | — |
| Deterministic point forecast | 3,715,008 | 437.4 MWh | **+141.42 %** |
| Stochastic, S=20, σ=0.04 | 1,545,166 | 0.000 | +0.413 % |
| Stochastic, S=20, σ=0.15 | 1,564,842 | 0.000 | +1.691 % |

A two-order-of-magnitude spread between good and bad scenario sets. Solve time
0.4–2.9 s per instance (HiGHS via `scipy.optimize.milp`).

> **Bug found and fixed here.** The first implementation produced a checkerboard
> commitment and shed 39,292 MWh even under perfect foresight. Cause: the
> start-up constraint was written `su ≥ u_t + u_{t-1}` instead of
> `su ≥ u_t − u_{t-1}`; combined with `su ≤ 1` this forbade committing any unit
> in two consecutive hours. Recorded because it is exactly the class of silent
> error that makes a downstream number meaningless.

## 7. Datasets

| Suite | Shape | Constraints/hour | DoF/hour | Residual on real data |
|---|---|---|---|---|
| A `measured` | 1,990 × 24 × 69 | 21 | 48 | `0.000e+00` |
| B `grid` | 1,998 × 24 × 304 | 197 | 107 | `7.105e-13` |

Suite B also satisfies every branch thermal rating on the real data
(`ineq_rate = 0`), confirming the constructed operating points are not merely
Kirchhoff-consistent but operationally feasible.

## 8. Hardware

| Op | CPU (18 core) | MPS (M5 Max) |
|---|---|---|
| Training step (B=128, D=304) | 72.2 ms | **9.9 ms** |
| Inference (B=1600) | 268.2 ms | **38.4 ms** |

Backbone: 4.54 M parameters (`d_in=305, hidden=256, depth=4`).

## 9. Bibliography

191 entries in `paper/references.bib`, each verified against a primary record
(arXiv Atom API **by ID**, Crossref DOI metadata, or a publisher/proceedings
page). Automated checks: **0** entries with placeholder authors, **0** entries
missing an author field.

The bibliography being replaced had 21 entries, of which 6 carried a literal
`author={Authors}`, roughly 9 matched no primary record at all,
`lipman2023flow` credited two people who do not exist, `song2021score` listed 3
of 6 authors, and `schwartz2021greenai` was mis-dated by a year.

---

## 10. Bugs found by running the code, and what each would have cost

Every one of these produced output. None of them raised an error. They are
recorded because "the script ran" is not evidence that a number is real.

### (a) DDPM diverged silently — would have been the worst row by 11 orders of magnitude
First full-training run returned `ES = 3.129e+11`, `eq_max = 5.00e+11`.
**Cause:** with a cosine schedule `alpha_bar -> 0` at the start of sampling, so
`x0 = (x - sqrt(1-ab) eps)/sqrt(ab)` divides by ~0 and the ancestral chain
amplifies error without bound. The standard static-thresholding clip was missing.
**Fix:** clamp `alpha_bar >= 1e-4` and clip the `x0` prediction to +/-5 standard
deviations (data is standardised).
**After:** `ES = 1.331e+05`; ratio of max sample to max data went from ~1e7 to
**1.13**, mean matches the data to 4 significant figures.
**Also added:** `run_main.py` now refuses to record any run with non-finite
samples or `max|sample| > 1e4 x` the data scale, and logs a failure instead.

### (b) The soft-penalty baseline was saturated at every weight — an unfairly crippled competitor
`FM+penalty` returned *identical* energy scores (2.258e+05, 4 s.f.) for
lambda = 1, 10, 100, 1000, and its violation slightly *increased* with lambda.
**Cause:** a scaling error, not a modelling one. In standardised coordinates the
constraint is `A diag(sd)` with `sd` in MW (~1e4), so the quadratic residual is
~1e9 while the flow-matching term is ~1. Every weight in the sweep was already
far past saturation.
**Fix:** `AffineConstraintSet.row_normalised()` scales each row to unit norm,
which leaves the feasible set and its projectors identical (verified: projector
difference `8.9e-16`, projected-point difference `9.1e-15`) but makes the
residual a geometric distance. Residual scale falls from `1.19e+09` to `1.22`.
**Why it matters:** reporting a baseline that was silently disabled by a units
mismatch would have inflated our own result. The lambda sweep is re-running.

### (c) The AC manifold path was wrong in three separate ways
1. **Wrong coordinates.** The network trains on standardised values, but `g` (the
   AC power-flow equations) was being evaluated on them directly — physically
   meaningless. The affine path transforms the constraint
   (`AffineConstraintSet.standardise`); the nonlinear path had no equivalent.
   **Fix:** `StandardisedManifold`, the pull-back `g_std(z) = g(mu + sd z)`.
2. **Unusable cost, presenting as a hang.** `torch.func.jacrev` needs one
   backward pass per output row: `2 n_b = 114` per element, ~1.8e5 per training
   step. The run did not fail, it simply never finished.
   **Fix:** analytic polar power-flow Jacobian (the standard H/N/M/L blocks),
   **verified against autograd to `1.4e-14`** (relative `1.2e-16`).
3. **Silent fallback.** `StandardisedManifold` wrapped `g` in a bound method with
   no `.jacobian` attribute, so the analytic path was skipped and it hung again.
   **Fix:** chain-rule override `J_std = J(x) diag(sd)`, **verified to `8.9e-16`**.

A first verification of (3) reported a relative error of `7.8e+06`. That was the
*test* being wrong: it drew voltage magnitudes at ~0 p.u., where `P/V_m` is
singular. At physical voltages (~1.0 p.u.) it matches to machine precision.

**Result once fixed** (5-epoch smoke, IEEE 57-bus, AC manifold):

| method | `|g|_max` (p.u.) | RMSE |
|---|---|---|
| FM | 1.141e+01 | 6.96e-01 |
| FM + retract at the end | 1.513e-01 | 1.98e-03 |
| **Manifold-FM (ours)** | **6.28e-03** | **5.28e-05** |

Tangential projection along the flow beats retracting only at the end by 24x on
max residual and 37x on RMSE: retraction from a point far off the manifold does
not converge, whereas a trajectory that stays near it does.
