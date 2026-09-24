# Reproduction

Every number in `docs/RESULTS.md` comes from `results/*.jsonl`, which this
repository does not ship. The commands below produce them.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12. The reported sweep ran on an Apple M-series laptop with the
PyTorch `mps` backend; everything runs on CPU and CUDA without modification.
The unit-commitment stage uses HiGHS through `scipy.optimize.milp`, so no
commercial solver is required.

## Data

```bash
bash scripts/download_data.sh          # ~500 MB, public, no API key
python scripts/build_datasets.py --which all
python scripts/export_suite_meta.py
```

Sources: EIA-930 hourly balancing-authority operating data (2019-01-02 to
2024-06-30, six US balancing authorities) and the pglib-opf IEEE 118-bus and
MATPOWER case57 networks. Licences and per-source detail in
`docs/DATA_PROVENANCE.md`.

`build_datasets.py` is deterministic. Two checks it enforces, both of which
matter for the reported numbers:

* AC suite days that fail Newton-Raphson convergence are **dropped**, rather
  than letting solver luck select the training set (1979 of 1998 days kept).
* Renewables are curtailed when they exceed demand; without this the slack bus
  absorbs the surplus and the AC solve diverges.

## The sweep

One idempotent, resumable command:

```bash
bash scripts/run_everything.sh
```

Re-running skips any `(method, seed)` pair already present in the result files,
so an interrupted sweep resumes exactly where it stopped. The stages
individually:

```bash
# 1. main sweep: 16 methods x 10 seeds x 3 suites
python -u experiments/run_main.py --suite measured    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --M 100 --epochs 300 --steps 50 --hidden 256 --depth 4 --device mps
python -u experiments/run_main.py --suite grid        --seeds 0 1 2 3 4 5 6 7 8 9 ...
python -u experiments/run_main.py --suite grid_noflow --seeds 0 1 2 3 4 5 6 7 8 9 ...

# 2. N-1 zero-shot topology transfer (8 contingencies)
python -u experiments/run_transfer.py

# 3. two-stage stochastic unit commitment
python -u experiments/run_downstream.py --seeds 0 1 2 3 4 5 6 7 8 9 \
    --S 20 --n-days 60 --M 100 --epochs 300 --device mps

# 4. NFE / FLOPs frontier
python -u experiments/run_efficiency.py

# 5. nonlinear AC manifold
python -u experiments/run_ac.py --epochs 250
python -u experiments/run_ac.py --methods "Manifold-FM (ours)" --retract-iters 20 \
    --epochs 250 --out results/ac_retract20.jsonl
```

Note on stage 5: the manifold path takes autograd Jacobians and solves damped
normal equations with condition number around 1e7. On `mps` this silently
produces non-finite weights and the whole variant samples as NaN; `run_ac.py`
overrides `--device mps` to CPU for that reason. Verified: cpu 0% NaN,
mps 100% NaN.

## Records and tables

```bash
python scripts/check_claims.py            # pre-registered claims vs measured
python scripts/make_results_record.py     # docs/RESULTS.md
python scripts/make_prediction_ledger.py  # docs/PREDICTIONS_VS_MEASURED.md
python scripts/make_tables.py             # paper/tables/*.tex
python scripts/make_figures.py            # paper/figs/*
```

No number is typed into the record or the paper by hand; all five scripts read
`results/*.jsonl`. A missing result produces a visibly-marked `[pending]` table
and a `FIGURE PENDING` plot, never a substituted figure.

## Runtimes

Measured on the machine above. The main sweep dominates.

| stage | wall clock |
|---|---|
| data download + build | ~40 min (network-bound) |
| main sweep, 3 suites x 10 seeds | ~6 h |
| transfer | ~25 min |
| downstream UC | ~2 h (MILP-bound) |
| efficiency frontier | ~15 min |
| AC suite | ~1 h (CPU-bound by construction) |

## Seeds and determinism

Seeds 0–9 are passed explicitly to every stage and control model
initialisation, the flow-matching noise draw and the sampling noise. Data
splits are fixed by date, not by seed, so every method sees the identical
train/validation/test partition. Paired statistics in `docs/RESULTS.md` pair on
seed.

## Known limitation in the shipped results

The downstream unit-commitment stage completed 3 seeds, not 10. The ordering of
the extremes is stable across them; the middle of that table is not resolved.
This is stated wherever the stage is reported.
