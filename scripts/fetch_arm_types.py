"""Re-fetch armsInterventionsModule + designModule from ClinicalTrials.gov API v2 for every trial in the released
benchmark arm table (data/benchmark_arms.csv.gz), to recover armGroupType for results groups that the v2 parser did not
match to a registered arm group, and interventionModel for all trials.
Usage: python fetch_arm_types.py <benchmark_arms.csv.gz> <out.jsonl.gz>"""
import sys, json, gzip, time
import pandas as pd
from fetch_ctgov_meta import fetch_ids
FIELDS = ','.join(['protocolSection.identificationModule.nctId', 'protocolSection.designModule',
                   'protocolSection.armsInterventionsModule'])
if __name__ == '__main__':
    ids = sorted(pd.read_csv(sys.argv[1], usecols=['nct_id']).nct_id.unique())
    t0 = time.time(); S = fetch_ids(ids, fields=FIELDS, batch=200)
    with gzip.open(sys.argv[2], 'wt') as f:
        for s in S: f.write(json.dumps(s, separators=(',', ':')) + '\n')
    got = {s['protocolSection']['identificationModule']['nctId'] for s in S}
    print(len(ids), len(S), len(set(ids) - got), f'{time.time()-t0:.0f}s')
