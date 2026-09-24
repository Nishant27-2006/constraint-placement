#!/usr/bin/env bash
# Phase 3: re-run the two experiments invalidated by bugs.
#  - AC with per-step retraction actually happening (hfm/flows/fm.py on_step)
#  - downstream UC at 10 seeds, so the C6 reversal is testable
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
LOG=logs/phase3.log
exec > >(tee -a "$LOG") 2>&1
echo "================ phase3 $(date -u +%FT%TZ) ================"

EXT=$(ioreg -rn AppleSmartBattery | grep -c '"ExternalConnected" = Yes' || true)
PCT=$(pmset -g batt | grep -Eo '[0-9]+%' | head -1 | tr -d '%'); PCT=${PCT:-0}
if [ "$EXT" -lt 1 ] && [ "$PCT" -lt 60 ]; then
  echo "REFUSING TO START: on battery at ${PCT}%.  Connect AC power."; exit 1
fi
echo "power ok (external=$EXT battery=${PCT}%)"
stage () { echo; echo "---------- PHASE3 STAGE $1 : $2 ----------"; date -u +%FT%TZ; }

stage A "AC manifold suite, per-step retraction (was: retracted once after the solve)"
cp -f results/ac.jsonl results/ac_prefix_bug.jsonl 2>/dev/null || true
rm -f results/ac.jsonl
python -u experiments/run_ac.py

stage B "downstream UC at 10 seeds (was n=1; C6 reversal needs seeds)"
cp -f results/downstream.jsonl results/downstream_1seed.jsonl 2>/dev/null || true
rm -f results/downstream.jsonl
python -u experiments/run_downstream.py --seeds 0 1 2 3 4 5 6 7 8 9 \
    --S 20 --n-days 60 --M 100 --epochs 300 --device mps

stage C "claim check, tables, figures, paper"
python scripts/check_claims.py | tee logs/claim_check.txt
[ -f scripts/make_tables.py ]  && python scripts/make_tables.py
[ -f scripts/make_figures.py ] && python scripts/make_figures.py
# TeX Live lives in a different place on every machine; add yours if latexmk
# is not already on PATH.
command -v latexmk >/dev/null || for d in "$HOME"/texlive/*/bin/* /usr/local/texlive/*/bin/*; do
  [ -x "$d/latexmk" ] && export PATH="$d:$PATH" && break
done
if [ -f paper/main.tex ]; then (cd paper && latexmk -pdf -interaction=nonstopmode main.tex); fi

echo; echo "================ PHASE3_DONE $(date -u +%FT%TZ) ================"
