#!/usr/bin/env bash
# Deterministic layer of the camera-ready paper: regenerates every table from files committed to the repository.
# No network access and no model calls. Run from the repository root.
#
#   bash scripts/run_camera_ready.sh            regenerate data/v2.1 final file, results/cr/*.csv, tables/*.tex and figures/*.{png,pdf}
#   bash scripts/run_camera_ready.sh --fresh    also delete cached registry-model predictions first (full refit)
#   bash scripts/run_camera_ready.sh --check    regenerate into a scratch copy and diff every tables/*.tex against the
#                                               committed version; exits non-zero on any difference
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-}"
TABLES="seeds main lmruns arch contamination exposure coverage op lmrevision blindness design robust fg seen level mol lmarea lmsel"

if [[ "$MODE" == "--check" ]]; then
  REF=".check_ref"; rm -rf "$REF"; mkdir -p "$REF/figures"; cp tables/*.tex "$REF/"; cp figures/*.png figures/*.pdf "$REF/figures/"
fi
if [[ "$MODE" == "--fresh" || "$MODE" == "--check" ]]; then
  rm -f results/cr/registry_preds_*.csv.gz
fi

python scripts/cr/finalize_v21.py
for t in $TABLES; do
  start=$(date +%s)
  python "scripts/cr/tables/tab_${t}.py" > "results/cr/log_${t}.txt" 2>&1 || { echo "FAILED: tab_${t}.py (see results/cr/log_${t}.txt)"; exit 1; }
  echo "tab_${t}.py  $(( $(date +%s) - start )) s"
done

bash scripts/cr/figures/make_figures.sh > results/cr/log_figures.txt 2>&1 || { echo "FAILED: make_figures.sh (see results/cr/log_figures.txt)"; exit 1; }
echo "make_figures.sh"

if [[ "$MODE" == "--check" ]]; then
  status=0
  for f in "$REF"/*.tex; do
    b=$(basename "$f")
    if cmp -s "$f" "tables/$b"; then echo "identical  tables/$b"; else echo "DIFFERS    tables/$b"; diff "$f" "tables/$b" | head -20; status=1; fi
  done
  for f in "$REF"/figures/*.png "$REF"/figures/*.pdf; do
    b=$(basename "$f")
    if cmp -s "$f" "figures/$b"; then echo "identical  figures/$b"; else echo "DIFFERS    figures/$b"; status=1; fi
  done
  exit $status
fi
