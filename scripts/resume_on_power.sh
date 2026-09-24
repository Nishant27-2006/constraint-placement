set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "[watcher] waiting for AC power (or battery >= 60%)..."
WAITED=0
while :; do
  EXT=$(ioreg -rn AppleSmartBattery 2>/dev/null | grep -c '"ExternalConnected" = Yes' || true)
  PCT=$(pmset -g batt 2>/dev/null | grep -Eo '[0-9]+%' | head -1 | tr -d '%')
  PCT=${PCT:-0}
  if [ "$EXT" -ge 1 ] || [ "$PCT" -ge 60 ]; then
    echo "[watcher] power OK (external=$EXT battery=${PCT}%) after ${WAITED}s -> resuming sweep"
    break
  fi
  sleep 20; WAITED=$((WAITED+20))
  if [ $((WAITED % 300)) -eq 0 ]; then echo "[watcher] still waiting (${WAITED}s, battery ${PCT}%, external=$EXT)"; fi
  if [ "$WAITED" -ge 21600 ]; then echo "[watcher] gave up after 6h"; exit 3; fi
done
exec bash scripts/run_all_main.sh
