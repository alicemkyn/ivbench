"""Supplementary check (not part of the random re-audit): the same deterministic + model-assisted pairing review applied to
every v2.1 TEST contrast that is not identical to v2 (re-paired, relabelled, new pair in a v2 trial), before new
language-model calls are spent on them. Run in the Claude Science python kernel:
    exec(open('audit_repaired_test_v21.py').read()); A = run_repaired(host)
Outputs: audit_samples_repaired_test_v21.csv, audit_records_repaired_test_v21.jsonl.gz, audit_results_repaired_test_v21.csv"""
import json, gzip, os
from audit_v21 import *
from fetch_ctgov_meta import fetch_ids
from fetch_audit_records_v21 import MODULES

def run_repaired(host):
    C = pd.read_csv(os.path.join(REPO, 'data', 'v2.1', 'benchmark_contrasts_v2.1.csv.gz'), low_memory=False); C['row_id'] = np.arange(len(C))
    S = C[(C.split_time == 'test') & (C.v21_status != 'kept')].copy(); S['sample'] = 'v21_test_not_kept'
    S.to_csv('audit_samples_repaired_test_v21.csv', index=False)
    recs = fetch_ids(sorted(S.nct_id.unique()), fields=','.join(MODULES), batch=50)
    with gzip.open('audit_records_repaired_test_v21.jsonl.gz', 'wt') as f:
        for s in recs: f.write(json.dumps(s, separators=(',', ':')) + '\n')
    return run_v21(host, samples='audit_samples_repaired_test_v21.csv', records='audit_records_repaired_test_v21.jsonl.gz',
                   out='audit_results_repaired_test_v21.csv')
