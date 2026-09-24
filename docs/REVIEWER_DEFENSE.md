# Anticipated reviewer objections, and where each is answered

Written before the results exist, so the experiments are designed to answer the
objections rather than the objections rewritten to fit the results. Ordered by
how likely they are to sink the paper.

---

### R1. "PCFM already does exact hard constraints on flow models. What is new?"
**Severity: fatal if unanswered.** Utkarsh et al., NeurIPS 2025, arXiv:2506.04171.

*Answer.* We concede the claim entirely and cite it on page 1. What we contribute
is not exactness but the **comparison**: PCFM places the constraint at inference
on a pretrained model; we ask whether placing it in the hypothesis class does
better, and at what cost. We reimplement PCFM as a first-class baseline
(`hfm/flows/correctors.py`) and report it at matched NFE. Its per-step endpoint
projection is measured — **51 extra projections for a 50-step solve** — against
our zero extra network evaluations. If PCFM matches us on fidelity, the paper's
answer is "put it at inference, it is cheaper to deploy", and we say so.

### R2. "Why not just project the samples at the end?"
**Severity: high — and the objection is mathematically correct.**

*Answer.* Proposition 3 in `docs/PROOFS.md` proves the referee's point *for*
them: orthogonal projection is 1-Lipschitz and fixes the data law, so
$W_r(\Pi_\# p_\theta, q) \le W_r(p_\theta, q)$ — post-hoc projection can never
increase Wasserstein distance. We state this ourselves rather than hope it is
missed. It means Theorem 1 does **not** prove train-time projection is better,
and the paper says so explicitly. The hypothesised mechanism is optimisation —
the network stops spending capacity on the 197-of-304 directions per hour whose
answer is closed-form — and that is an empirical question, pre-registered as
claim C3 with "no difference" as an admissible outcome.

### R3. "Generating branch flows is artificial — they are a function of injections."
**Severity: medium.**

*Answer.* Correct, and deliberate. It is what makes the benchmark's key axis
possible: adding the 186 flow channels adds 186 dimensions *and* 186
constraints, so the **intrinsic dimension is fixed at 107/hour while codimension
moves 11 → 197**. Every other benchmark confounds these. We also report the
injections-only variant. Separately, operator scenario files genuinely do carry
both components and aggregates, and a model that generates them independently
emits internally inconsistent files.

### R4. "Your baselines are weak / you omitted normalizing flows."
**Severity: high.** Dumas et al., *Applied Energy* 305:117871 (2022) already
benchmarks NF vs GAN vs VAE vs copula with scoring rules *and* a downstream task.

*Answer.* We follow that protocol and include a conditional RealNVP, cWGAN-GP
(the Chen et al. TPWRS 2018 family), cVAE, DDPM, Gaussian copula, and
kNN-historical resampling, plus six constraint-handling variants of flow
matching — 16 methods. All neural methods share one backbone, optimiser and
schedule. The soft-penalty baseline is **swept** over λ ∈ {1,10,100,1000} and
reported at its best λ.

### R5. "kNN-historical resampling beats everything, so the deep models are pointless."
**Severity: medium, and currently live.** At 15 epochs it led every neural
method (ES 1.33e+05 vs FM 2.15e+05) and is exactly feasible by construction.

*Answer.* We keep it, and if it still leads after the full sweep it goes in the
abstract (claim C9). Its known limitation — it can only emit days that occurred,
so it cannot extrapolate to unseen conditions and its support is finite — is
tested directly by the N-1 transfer experiment, where the historical record
contains no days for the contingency topology.

### R6. "Metrics are gameable / the energy score is insensitive to correlations."
**Severity: medium.**

*Answer.* We report the variogram score (Scheuerer & Hamill, MWR 143:1321–1334)
precisely because the energy score is weakly sensitive to misspecified
dependence, plus rank histograms and interval coverage for calibration, plus the
structure diagnostics of Cramer et al. (IEEE Access 10:8194–8207, 2022):
autocorrelation, ramp-rate distribution, power spectral density. The metric
suite was validated against ensembles with *known* miscalibration before use
(`docs/VERIFICATION_LOG.md` §5).

### R7. "The energy/sustainability claims are unfalsifiable hardware anecdotes."
**Severity: medium — and this is why the draft's 3.1× claim is gone.**

*Answer.* Primary efficiency metrics are **NFE** and **analytic FLOPs**: exactly
countable integers, hardware-independent, reproducible anywhere. Measured energy
is reported only as a secondary, explicitly whole-system, device-specific figure
from SMC telemetry with an idle baseline subtracted, and never as a headline. We
claim no novelty for adaptive stepping (`Gotta Go Fast`, `dopri5` predate it) and
benchmark against `dopri5` at matched tolerance.

### R8. "Your data is semi-synthetic."
**Severity: medium.**

*Answer.* Suite A is **entirely measured** — no network model, no dispatch
simulation — and carries 21 exact identities per hour. Suite B's stochasticity
is the same measured record; only the network and the nodal allocation are
standard-practice modelling, stated plainly in `docs/DATA_PROVENANCE.md`. We do
not claim Suite B reconstructs a real system.

### R9. "Constraints are 'exact' only to 1e-6."
**Severity: low, but must be pre-empted.**

*Answer.* That is `float32` round-off, not method error. The identical
experiment in `float64` gives `2.4e-14`, and the residual does not grow with step
count (flat at `1e-14` over 50 Euler steps). Both numbers are reported, and all
residual *reporting* is done in `float64`.

### R10. "Theorem 1 is trivial."
**Severity: low.**

*Answer.* Parts (i) and (iv) are elementary and we say so — that is the point,
since it is why the method needs no tolerance and no inner solve. The content is
(ii) + (iii): the optimum is already tangential, so the restriction is
*bias-free* rather than a relaxation, and projection is therefore a free
variance-reducing restriction of the hypothesis class. The draft's original
"Theorem 1" ("as λ→∞ the penalty →0") was genuinely vacuous and has been removed.

### R11. "Single network, single region, n=1 dataset."
**Severity: medium.**

*Answer.* Three suites (measured / DC grid / AC grid), six balancing
authorities, 1,990–1,998 days, 3 seeds, and 8 contingency topologies. OPSD
(European) is available as an out-of-distribution split. We do not claim
generalisation beyond what is run.
