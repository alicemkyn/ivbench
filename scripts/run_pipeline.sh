#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
import os, sys
sys.path.insert(0, 'src')
os.chdir('data')
exec(open('../src/ivbench/harvest.py').read())          # -> trials_raw.jsonl.gz
PY
python - <<'PY'
import sys, gzip, json, pandas as pd
sys.path.insert(0, 'src')
from ivbench import parse as pt, build as bb
rows = []
with gzip.open('data/trials_raw.jsonl.gz', 'rt') as f:
    for line in f:
        r = pt.parse_study(json.loads(line))
        if r: rows += r
ARM, CON, card = bb.build(pd.DataFrame(rows))
CON['ctrl_kind'] = CON['ctrl_role']
ARM.to_csv('data/benchmark_arms.csv.gz', index=False); CON.to_csv('data/benchmark_contrasts.csv.gz', index=False)
card['contrast_trials_final'] = int(CON['nct_id'].nunique())
json.dump(card, open('data/datacard.json', 'w'), indent=1, default=str)
print(json.dumps({k: card[k] for k in ('trials_parsed', 'arm_trials', 'contrast_rows_final', 'contrast_trials_final')}))
PY
