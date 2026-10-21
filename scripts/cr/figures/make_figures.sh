#!/usr/bin/env bash
# Render the four camera-ready figures from repository files (no network, no LM calls, deterministic).
#   bash scripts/cr/figures/make_figures.sh
# Outputs figures/{fig2_benchmark_skill,fig3_interventional_forest,fig4_reliability,figA_probes}.{png,pdf} (300 dpi PNG + vector PDF).
# fig4 refits the cross-fitted temporal regime (scripts/cr/tables/_conformal.py, about one minute on a laptop).
set -euo pipefail
cd "$(dirname "$0")/../../.."
for f in fig2_benchmark_skill fig3_interventional_forest fig4_reliability figA_probes; do
  python "scripts/cr/figures/fig_${f}.py"
done
