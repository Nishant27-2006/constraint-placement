#!/usr/bin/env bash
# Master pipeline.  Idempotent and resumable: re-running skips completed stages.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
mkdir -p logs results figs
LOG=logs/run_everything.log
exec > >(tee -a "$LOG") 2>&1
echo "================ run_everything $(date -u +%FT%TZ) ================"

# ---- power guard -------------------------------------------------------
EXT=$(ioreg -rn AppleSmartBattery | grep -c '"ExternalConnected" = Yes' || true)
PCT=$(pmset -g batt | grep -Eo '[0-9]+%' | head -1 | tr -d '%'); PCT=${PCT:-0}
if [ "$EXT" -lt 1 ] && [ "$PCT" -lt 60 ]; then
  echo "REFUSING TO START: on battery at ${PCT}%.  Connect AC power."
  echo "(scripts/resume_on_power.sh will start this automatically when you do.)"
  exit 1
fi
echo "power ok (external=$EXT battery=${PCT}%)"

stage () { echo; echo "---------- STAGE $1 : $2 ----------"; date -u +%FT%TZ; }

# ---- 0. datasets -------------------------------------------------------
stage 0 "rebuild Suite B with raw_inputs (needed by transfer)"
python - <<'PY' || REBUILD=1
import pickle,sys
try:
    d=pickle.load(open('data/processed/grid.pkl','rb'))
    sys.exit(0 if 'raw_inputs' in d.meta else 1)
except Exception: sys.exit(1)
PY
if [ $? -ne 0 ]; then
  python scripts/build_datasets.py --which grid
else
  echo "grid.pkl already has raw_inputs - skipping"
fi
[ -f data/processed/measured.pkl ] || python scripts/build_datasets.py --which measured
[ -f data/processed/ac.pkl ]       || python scripts/build_datasets.py --which ac
[ -f data/processed/grid_noflow.pkl ] || python scripts/build_datasets.py --which grid_noflow

# ---- 1. main sweep -----------------------------------------------------
stage 1 "main benchmark (16 methods x 3 seeds x 3 suites)"
for SUITE in measured grid grid_noflow; do
  python -u experiments/run_main.py --suite $SUITE --seeds 0 1 2 --M 100 \
      --epochs 300 --steps 50 --hidden 256 --depth 4 --device mps
done

# ---- 2. N-1 transfer ---------------------------------------------------
stage 2 "N-1 zero-shot topology transfer"
python -u experiments/run_transfer.py --n-contingencies 8 --M 50 --epochs 300 --device mps

# ---- 3. downstream UC --------------------------------------------------
stage 3 "stochastic unit-commitment decision value"
python -u experiments/run_downstream.py --S 20 --n-days 60 --M 100 --epochs 300 --device mps

# ---- 4. efficiency frontier -------------------------------------------
stage 4 "NFE / FLOPs / energy frontier"
[ -f experiments/run_efficiency.py ] && \
  python -u experiments/run_efficiency.py --device mps || echo "(stage 4 script pending)"

# ---- 5. AC manifold ----------------------------------------------------
stage 5 "AC nonlinear manifold suite"
[ -f experiments/run_ac.py ] && \
  python -u experiments/run_ac.py --device mps || echo "(stage 5 script pending)"

# ---- 6. claim check, tables, figures, pdf -----------------------------
stage 6 "pre-registered claim check"
python scripts/check_claims.py | tee logs/claim_check.txt

stage 6b "tables, figures, paper"
[ -f scripts/make_tables.py ]  && python scripts/make_tables.py
[ -f scripts/make_figures.py ] && python scripts/make_figures.py
# TeX Live lives in a different place on every machine; add yours if latexmk
# is not already on PATH.
command -v latexmk >/dev/null || for d in "$HOME"/texlive/*/bin/* /usr/local/texlive/*/bin/*; do
  [ -x "$d/latexmk" ] && export PATH="$d:$PATH" && break
done
if [ -f paper/main.tex ]; then (cd paper && latexmk -pdf -interaction=nonstopmode main.tex); fi

echo; echo "================ RUN_EVERYTHING_DONE $(date -u +%FT%TZ) ================"
