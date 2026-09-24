# Execution plan

One command runs everything:

```bash
cd /path/to/hamiltonian_network
bash scripts/run_everything.sh          # resumable; safe to re-run after any interruption
```

Progress: `tail -f logs/run_everything.log`. Every stage is **idempotent** — a
stage that already produced its artefact is skipped, and `run_main.py` skips any
`(method, seed)` already present in its `.jsonl`. An interruption costs at most
the single run in flight.

---

## Hard prerequisite

**The machine must be on AC power.** The sweep saturates the GPU for hours. A
watcher (`scripts/resume_on_power.sh`) polls for AC and starts automatically;
`run_everything.sh` also refuses to start on battery below 60 %.

---

## Stage order, runtimes, and acceptance criteria

Runtimes are measured extrapolations from the benchmarks in
`docs/VERIFICATION_LOG.md` (MPS: 9.9 ms/train-step, 38.4 ms/inference at B=1600).

### Stage 0 — rebuild Suite B  ·  ~2 min
```bash
python scripts/build_datasets.py --which grid
```
**Why this is not optional:** `grid.pkl` on disk was built *before* the
`raw_inputs` field was added to the builder. `run_transfer.py` needs those raw
driving series to re-dispatch the held-out days on an outaged network, and will
exit with status 2 without them.

*Accept if:* `X = (1998, 24, 304)`, `codim/hour = 197`, `dof/hour = 107`,
self-consistency `eq_max < 1e-9`, and `meta['raw_inputs']` is present.

### Stage 1 — main benchmark  ·  ~4–5 h
```bash
python -u experiments/run_main.py --suite measured --seeds 0 1 2 --M 100 --epochs 300
python -u experiments/run_main.py --suite grid     --seeds 0 1 2 --M 100 --epochs 300
```
16 methods x 3 seeds x 2 suites = 96 runs. Produces `results/main_*.jsonl`.

*Accept if:* every `(method, seed)` row is present; `HFM (ours)` `eq_max` is at
`float32` round-off; unconstrained baselines' `eq_max` is orders of magnitude
larger; no method silently returns NaN.

**Fairness controls already encoded:** identical backbone (24-step temporal
transformer, width 256, depth 4), identical optimiser and schedule, identical
`M=100` for every method, chronological 70/10/20 split, strictly causal
covariates, and the soft-penalty baseline **swept** over λ ∈ {1, 10, 100, 1000}
with its best λ reported — the generous reading for that baseline.

### Stage 2 — N-1 zero-shot transfer  ·  ~1.5 h
```bash
python -u experiments/run_transfer.py --n-contingencies 8 --M 50 --epochs 300
```
8 contingencies chosen by largest PTDF shift, from the 177 non-islanding
branches. Ground truth is re-dispatched per contingency.

*Accept if:* projector routes stay exact under the swapped constraint;
chart-based routes are evaluated in **both** the stale-chart and swapped-chart
readings. If chart routes also stay exact, claim **C5 is withdrawn**.

### Stage 3 — downstream unit commitment  ·  ~1 h
```bash
python -u experiments/run_downstream.py --S 20 --n-days 60 --M 100
```
Commitment solved on generated scenarios, frozen, scored on the realised day.
Scenario count and the M→S reduction are identical for every method.

*Accept if:* `PerfectForesight` has zero unserved energy and the lowest cost
(sanity), and the spread across generators exceeds seed noise. If not,
**C6 is withdrawn** and the paper says feasibility does not reach the decision.

### Stage 4 — efficiency frontier  ·  ~45 min
Sweep `steps ∈ {2,5,10,20,50,100}` x `solver ∈ {euler, heun, rk4}` plus `dopri5`
at `rtol ∈ {1e-2 … 1e-5}`; record score vs NFE vs FLOPs vs measured energy.

*Accept if:* the frontier is monotone and `dopri5` is on or near it. We claim no
novelty for adaptivity; the comparison is against `dopri5` at matched tolerance.

### Stage 5 — AC nonlinear manifold  ·  ~1.5 h
Suite C: tangential projection + Gauss–Newton retraction. Verify the drift rates
of Proposition 2 numerically (`O(h^p)` without retraction, bounded with).

*Accept if:* measured drift exponents match the predicted orders. If they do
not, the proposition is reported as **not empirically confirmed**.

### Stage 6 — tables, figures, PDF  ·  ~20 min
```bash
python scripts/make_tables.py && python scripts/make_figures.py
cd paper && latexmk -pdf main.tex
```
All tables and figures are generated **from the `.jsonl` files** — no number is
ever typed into the LaTeX by hand. Mean ± std over 3 seeds throughout.

---

## Total: ~10–11 hours unattended.

## If something fails
* A stage crashes → re-run `scripts/run_everything.sh`; completed work is kept.
* A single method crashes → `run_main.py` catches it, logs the traceback, and
  continues. The method appears as absent in the results and is reported as
  failed rather than omitted.
* Power lost → the watcher restarts the whole chain on reconnection.

## What is *not* automated
Reading the results and deciding which claims survive `docs/CLAIMS_AND_EVIDENCE.md`.
That is the one step that must not be automated, because the falsification
conditions only mean something if they are applied honestly.
