# Where does a physical constraint belong in a generative model?

Benchmark and reference implementation for constraint placement in generative
scenario models of power grids.

A generative model of grid scenarios should respect the physical laws its
samples are supposed to satisfy. Several routes achieve *exact* satisfaction.
The open question is **where the constraint belongs** — and the answer turns out
not to be universal.

| route | exact? | zero-shot to a new constraint? | extra NFE |
|---|---|---|---|
| soft penalty | no | no | 0 |
| post-hoc projection | yes | yes | 1 |
| inference-time correction (PCFM) | yes | yes | projection every step |
| reduced coordinates / DC3 completion | yes | chart must be rebuilt | 0 |
| tangential projection at train time (ours) | yes | yes | **0** |

## Headline findings

1. **The sign of the effect flips with codimension.** On the controlled
   contrast — the same 118-bus system, same measured injections, the only
   change being whether the model must also emit the 186 line flows that the
   constraint matrix determines exactly — train-time projection goes from
   statistically indistinguishable from unconstrained flow matching
   (codim 0.093) to significantly better (codim 0.648). On a third, real
   dataset it is significantly *worse*. Anyone reporting a single winner has
   measured one suite.
2. **Soft penalties are dominated on both axes.** Raising λ three orders of
   magnitude leaves the constraint violation at the same order while
   monotonically destroying the score. Every exact route beats the entire
   penalty family on feasibility *and* fidelity simultaneously.
3. **Feasibility is not decision value.** In two-stage stochastic unit
   commitment, cost regret is essentially uncorrelated with the energy score
   and strongly correlated with 90% coverage. Exactly-feasible ensembles are
   sharper, the scheduler trusts them, under-commits reserve, and pays in load
   shed. A study that stopped at the proper scoring rule would have reported
   the opposite conclusion with confidence.
4. **Exactness is free in NFE** for affine invariants: one cached matvec, zero
   extra function evaluations, against 51 extra projections for inference-time
   correction on a 50-step solve.
5. **A non-neural baseline places second of sixteen** on both grid suites.

Two of our own predictions were pre-registered with explicit falsification
conditions before the sweep ran, and both were falsified by our own data. They
are reported as falsified. See [`docs/RESULTS.md`](docs/RESULTS.md) and
[`docs/PREDICTIONS_VS_MEASURED.md`](docs/PREDICTIONS_VS_MEASURED.md).

## Quick start

```bash
pip install -r requirements.txt
bash scripts/download_data.sh          # public data, no API key (~500 MB)
python scripts/build_datasets.py --which all
bash scripts/run_everything.sh         # full pipeline; idempotent and resumable
```

Nothing in this repository reports a number that was not produced by code in
it. `results/` is not committed — it is what running the pipeline creates.
See [`REPRODUCE.md`](REPRODUCE.md) for exact commands, runtimes and hardware.

## Benchmark

Three affine suites, built from real public data, differing in the fraction of
each hour's state that the physics fixes analytically:

| suite | source | days | D/hour | constraints/hour | codim |
|---|---|---:|---:|---:|---:|
| `grid_noflow` | EIA-930 on pglib IEEE 118-bus | 1998 | 118 | 11 | 0.093 |
| `measured` | EIA-930, 6 US balancing authorities | 1990 | 69 | 21 | 0.304 |
| `grid` | as `grid_noflow`, plus 186 line flows | 1998 | 304 | 197 | 0.648 |

Plus a nonlinear suite (`ac`, IEEE 57-bus AC power-flow manifold, 114 nonlinear
equations per hour) and an N-1 contingency transfer suite.

16 generators × 10 seeds × 3 suites, identical backbone, budget and data.

## Layout

```
hfm/grid/        constraint algebra, MATPOWER parser, PTDF, DC-OPF, N-1 outages
hfm/flows/       flow matching, 4 ODE solvers with exact NFE counting, PCFM corrector
hfm/models/      shared backbone + 16 scenario generators
hfm/eval/        13-metric suite, power meter
hfm/downstream/  two-stage stochastic unit commitment (HiGHS via scipy)
hfm/data/        EIA-930 loader, benchmark construction
experiments/     run_main, run_transfer, run_downstream, run_efficiency, run_ac
scripts/         data build, sweep drivers, table/figure/record generation
docs/            claims, proofs, provenance, results
```

## Documentation

| file | what it is |
|---|---|
| [`docs/RESULTS.md`](docs/RESULTS.md) | every measured result, generated from the run files |
| [`docs/PREDICTIONS_VS_MEASURED.md`](docs/PREDICTIONS_VS_MEASURED.md) | what the prior state of the art would have predicted, against what happened |
| [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md) | every claim, its experiment, and **what would falsify it** — written before the sweep |
| [`docs/PROOFS.md`](docs/PROOFS.md) | Theorem 1 and Prop. 2 in full, plus Prop. 3 — the result that argues *against* us |
| [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) | sources, licences, and two physical priors the data falsifies |
| [`docs/REVIEWER_DEFENSE.md`](docs/REVIEWER_DEFENSE.md) | anticipated objections and where each is answered |

## Notes on scope

* We claim no novelty for hard constraints in flow matching (PCFM, NeurIPS
  2025) or for adaptive-step sampling (`Gotta Go Fast`, `dopri5`).
* Two physical priors commonly assumed in this setting are **false in the real
  record** and are therefore not imposed. See `docs/DATA_PROVENANCE.md`.
* Efficiency is reported as NFE and analytic FLOPs, which are exactly countable
  and hardware-independent — not as device joules.

## Licence

MIT. See [`LICENSE`](LICENSE).
