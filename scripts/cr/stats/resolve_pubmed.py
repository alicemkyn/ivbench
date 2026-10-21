"""Resolve PubMed publication dates (esummary) for CT.gov-linked references. Usage: python resolve_pubmed.py meta.jsonl.gz contrasts.csv.gz out.csv"""
import sys, json, gzip, time, urllib.parse, urllib.request
import pandas as pd
ES = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

def esummary(pmids):
    res = {}
    for i in range(0, len(pmids), 200):
        data = urllib.parse.urlencode({'db': 'pubmed', 'id': ','.join(pmids[i:i+200]), 'retmode': 'json'}).encode()
        for a in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(ES, data=data), timeout=120) as r:
                    j = json.load(r); break
            except Exception:
                if a == 4: raise
                time.sleep(2 + 3 * a)
        for u in j['result'].get('uids', []):
            d = j['result'][u]
            res[u] = dict(pubdate=d.get('pubdate', ''), epubdate=d.get('epubdate', ''), sortpubdate=d.get('sortpubdate', ''),
                          pmc_hist=';'.join(f"{h.get('pubstatus')}={h.get('date')}" for h in d.get('history', []) if h.get('pubstatus') in ('pubmed', 'entrez', 'medline')),
                          source=d.get('source', ''), error=d.get('error', ''))
        time.sleep(0.4)
    return res

if __name__ == '__main__':
    rows = []
    for line in gzip.open(sys.argv[1], 'rt'):
        s = json.loads(line); ps = s['protocolSection']; nct = ps['identificationModule']['nctId']
        for ref in ps.get('referencesModule', {}).get('references', []):
            rows.append(dict(nct_id=nct, ref_type=ref.get('type'), pmid=str(ref.get('pmid', '') or ''), citation=ref.get('citation', '')[:300]))
    R = pd.DataFrame(rows)
    pm = sorted({p for p in R.loc[(R.ref_type == 'RESULT') & (R.pmid != ''), 'pmid']})
    t0 = time.time(); info = esummary(pm)
    I = pd.DataFrame.from_dict(info, orient='index'); I.index.name = 'pmid'; I = I.reset_index()
    R = R.merge(I, on='pmid', how='left')
    R.to_csv(sys.argv[3], index=False)
    print(len(R), (R.ref_type == 'RESULT').sum(), len(pm), len(info), f'{time.time()-t0:.0f}s')
