set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
for SUITE in measured grid; do
  echo "################ SUITE=$SUITE ################"
  python -u experiments/run_main.py --suite $SUITE --seeds 0 1 2 --M 100 \
      --epochs 300 --steps 50 --hidden 256 --depth 4 --device mps
done
echo "MAIN_SWEEP_DONE"
