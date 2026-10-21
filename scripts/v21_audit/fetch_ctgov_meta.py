"""Re-fetch status/design/references modules for all benchmark trials from ClinicalTrials.gov API v2 (Oct 2026).
Usage: python fetch_ctgov_meta.py <benchmark_contrasts.csv.gz> <out.jsonl.gz>"""
import sys, json, gzip, time, urllib.parse, urllib.request
import pandas as pd
BASE = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = ','.join(['protocolSection.identificationModule.nctId', 'protocolSection.statusModule',
                   'protocolSection.designModule', 'protocolSection.referencesModule'])

def get(params):
    url = BASE + '?' + urllib.parse.urlencode(params)
    for a in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'python-urllib'}), timeout=120) as r:
                return json.load(r)
        except Exception as e:
            if a == 4: raise
            time.sleep(3 + 5 * a)

def fetch_ids(ids, fields=FIELDS, batch=200):
    out = []
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]; token = None
        while True:
            p = {'filter.ids': ','.join(chunk), 'fields': fields, 'pageSize': '1000'}
            if token: p['pageToken'] = token
            r = get(p); out += r['studies']; token = r.get('nextPageToken')
            if not token: break
    return out

if __name__ == '__main__':
    C = pd.read_csv(sys.argv[1], low_memory=False)
    ids = sorted(C.nct_id.unique())
    t0 = time.time(); S = fetch_ids(ids)
    with gzip.open(sys.argv[2], 'wt') as f:
        for s in S: f.write(json.dumps(s, separators=(',', ':')) + '\n')
    got = {s['protocolSection']['identificationModule']['nctId'] for s in S}
    print(len(ids), len(S), len(got), len(set(ids) - got), f'{time.time()-t0:.0f}s')
