#!/usr/bin/env bash
# Phase 2: 10-seed sweep on clean hardware + the two repaired stages + paper.
# Idempotent: run_main.py skips (method, seed) pairs already in the jsonl.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
LOG=logs/phase2.log
exec > >(tee -a "$LOG") 2>&1
echo "================ phase2 $(date -u +%FT%TZ) ================"

EXT=$(ioreg -rn AppleSmartBattery | grep -c '"ExternalConnected" = Yes' || true)
PCT=$(pmset -g batt | grep -Eo '[0-9]+%' | head -1 | tr -d '%'); PCT=${PCT:-0}
if [ "$EXT" -lt 1 ] && [ "$PCT" -lt 60 ]; then
  echo "REFUSING TO START: on battery at ${PCT}%.  Connect AC power."; exit 1
fi
echo "power ok (external=$EXT battery=${PCT}%)"
stage () { echo; echo "---------- PHASE2 STAGE $1 : $2 ----------"; date -u +%FT%TZ; }

stage 1 "main sweep, seeds 0-9, 3 suites (refills 22 removed rows + adds seeds 3-9)"
for SUITE in grid measured grid_noflow; do
  python -u experiments/run_main.py --suite $SUITE --seeds 0 1 2 3 4 5 6 7 8 9 --M 100 \
      --epochs 300 --steps 50 --hidden 256 --depth 4 --device mps
done

stage 3 "stochastic unit-commitment decision value (IndexError fixed)"
python -u experiments/run_downstream.py --S 20 --n-days 60 --M 100 --epochs 300 --device mps

stage 5 "AC nonlinear manifold suite (retraction now applied to the scored ensemble)"
rm -f results/ac.jsonl
python -u experiments/run_ac.py --device mps

stage 6 "claim check, tables, figures, paper"
python scripts/check_claims.py | tee logs/claim_check.txt
[ -f scripts/make_tables.py ]  && python scripts/make_tables.py
[ -f scripts/make_figures.py ] && python scripts/make_figures.py
# TeX Live lives in a different place on every machine; add yours if latexmk
# is not already on PATH.
command -v latexmk >/dev/null || for d in "$HOME"/texlive/*/bin/* /usr/local/texlive/*/bin/*; do
  [ -x "$d/latexmk" ] && export PATH="$d:$PATH" && break
done
if [ -f paper/main.tex ]; then (cd paper && latexmk -pdf -interaction=nonstopmode main.tex); fi

echo; echo "================ PHASE2_DONE $(date -u +%FT%TZ) ================"
