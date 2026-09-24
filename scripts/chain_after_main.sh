set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "[chain] waiting for the in-flight main sweep to finish..."
while pgrep -f "run_all_main.sh|run_main.py" > /dev/null; do sleep 30; done
echo "[chain] main sweep finished at $(date -u +%FT%TZ); running full pipeline"
exec bash scripts/run_everything.sh
