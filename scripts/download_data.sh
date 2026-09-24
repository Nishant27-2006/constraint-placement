set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$ROOT/data/raw"
mkdir -p "$R/eia930" "$R/opsd"
echo "=== EIA-930 six-month BALANCE files (public, no key) ==="
for Y in 2019 2020 2021 2022 2023 2024; do
  for H in Jan_Jun Jul_Dec; do
    F="$R/eia930/EIA930_BALANCE_${Y}_${H}.csv"
    [ -s "$F" ] && { echo "have ${Y}_${H}"; continue; }
    U="https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_${Y}_${H}.csv"
    C=$(curl -sL --max-time 900 -o "$F" -w "%{http_code}" "$U")
    S=$(wc -c < "$F" 2>/dev/null || echo 0)
    echo "EIA930 ${Y}_${H}  http=$C  bytes=$S"
    [ "$C" != "200" ] && rm -f "$F"
  done
done
echo "=== OPSD 60-min time series (public, no key) ==="
F="$R/opsd/time_series_60min_singleindex.csv"
if [ ! -s "$F" ]; then
  C=$(curl -sL --max-time 900 -o "$F" -w "%{http_code}" \
    "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv")
  echo "OPSD http=$C bytes=$(wc -c < "$F")"
fi
rm -f "$R/eia930/EIA930_BALANCE_2024_Jan_Jun.head8M.csv" "$R/opsd/opsd_time_series_60min.head6M.csv"
echo "=== TOTALS ==="; du -sh "$R"/eia930 "$R"/opsd; ls -la "$R"/eia930 | tail -20
echo "DOWNLOAD_DONE"
