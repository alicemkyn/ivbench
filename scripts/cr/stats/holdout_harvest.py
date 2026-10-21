"""Step 2: post-cutoff holdout harvest (Oct 2026). Same registry query as src/ivbench/harvest.py (completed, interventional,
phase 2/3, results posted) restricted to primary completion >= 2024-01-01, plus referencesModule. Trials already in the
benchmark (8,011 nct_ids) are excluded. Contrasts are built with the unchanged ivbench parse.parse_study + build.build.
Outputs: holdout_raw_oct2026.jsonl.gz, holdout_counts.csv, holdout_contrasts.csv.gz, holdout_arms.csv.gz"""
import json, gzip, time, urllib.parse, urllib.request
from common import *
from ivbench import parse as pt, build as bb
from resolve_pubmed import esummary
BASE = "https://clinicaltrials.gov/api/v2/studies"
MODULES = ['protocolSection.identificationModule', 'protocolSection.statusModule', 'protocolSection.sponsorCollaboratorsModule',
           'protocolSection.descriptionModule', 'protocolSection.conditionsModule', 'protocolSection.designModule',
           'protocolSection.armsInterventionsModule', 'protocolSection.outcomesModule', 'protocolSection.eligibilityModule',
           'resultsSection.participantFlowModule', 'resultsSection.baselineCharacteristicsModule', 'resultsSection.outcomeMeasuresModule',
           'derivedSection.conditionBrowseModule', 'derivedSection.interventionBrowseModule', 'protocolSection.referencesModule']

def harvest(min_pcd='2024-01-01'):
    out, token = [], None
    while True:
        q = {'filter.overallStatus': 'COMPLETED', 'aggFilters': 'results:with,studyType:int,phase:2 3', 'pageSize': '500',
             'fields': ','.join(MODULES), 'filter.advanced': f'AREA[PrimaryCompletionDate]RANGE[{min_pcd},MAX]', 'countTotal': 'true'}
        if token: q['pageToken'] = token
        with urllib.request.urlopen(urllib.request.Request(BASE + '?' + urllib.parse.urlencode(q), headers={'User-Agent': 'python-urllib'}), timeout=180) as r:
            j = json.load(r)
        out += j['studies']; token = j.get('nextPageToken')
        if not token: return out

if __name__ == '__main__':
    S = harvest()
    with gzip.open('holdout_raw_oct2026.jsonl.gz', 'wt') as f:
        for s in S: f.write(json.dumps(s, separators=(',', ':')) + '\n')
    bench = set(load_contrasts().nct_id)
    recs = []
    for s in S:
        ps = s['protocolSection']; st = ps['statusModule']; nct = ps['identificationModule']['nctId']
        refs = [r for r in ps.get('referencesModule', {}).get('references', []) if r.get('type') == 'RESULT']
        recs.append(dict(nct_id=nct, in_benchmark=nct in bench, pcd_dt=todate(st.get('primaryCompletionDateStruct', {}).get('date')),
                         rfp_dt=todate(st.get('resultsFirstPostDateStruct', {}).get('date')),
                         result_pmids=';'.join(str(r.get('pmid')) for r in refs if r.get('pmid')),
                         result_cites='||'.join(r.get('citation', '')[:200] for r in refs if not r.get('pmid'))))
    H = pd.DataFrame(recs)
    pm = sorted({p for x in H.result_pmids for p in x.split(';') if p})
    info = esummary(pm) if pm else {}
    def earliest(row):
        ds = []
        for p in [p for p in row.result_pmids.split(';') if p]:
            d = info.get(p, {}); cand = [pm_date(d.get('pubdate')), pm_date(d.get('epubdate'))]
            cand += [pd.to_datetime(x.split('=')[1][:10], errors='coerce') for x in (d.get('pmc_hist') or '').split(';') if '=' in x]
            cand = [c for c in cand if pd.notna(c)]; ds.append(min(cand) if cand else pd.NaT)
        for c in [c for c in row.result_cites.split('||') if c]:
            m = re.search(r'\b(20[0-2]\d)\b', c); ds.append(pd.Timestamp(int(m.group(1)), 1, 1) if m else pd.NaT)
        return ds
    H['result_pub_dates'] = H.apply(earliest, axis=1)
    rows = []
    for mdl, b in list(BOUND.items()) + [('haiku4.5_reliable', RELI['haiku4.5'])]:
        nopub = H.result_pub_dates.map(lambda ds: not any((pd.isna(d) or d < b) for d in ds))
        H[f'holdout_{mdl}'] = (~H.in_benchmark) & (H.pcd_dt >= b) & (H.rfp_dt >= b) & nopub
        rows.append(dict(model=mdl, boundary=str(b.date()), registry_pcd_ge_b=int((H.pcd_dt >= b).sum()),
                         not_in_benchmark=int(((H.pcd_dt >= b) & ~H.in_benchmark).sum()),
                         plus_rfp_ge_b=int(((H.pcd_dt >= b) & ~H.in_benchmark & (H.rfp_dt >= b)).sum()),
                         three_condition_holdout_trials=int(H[f'holdout_{mdl}'].sum())))
    H.drop(columns=['result_pub_dates']).to_csv('holdout_trial_flags.csv', index=False)
    # build contrasts with the unchanged pipeline on all non-benchmark trials (pcd >= 2024-01-01)
    keep = set(H.loc[~H.in_benchmark, 'nct_id'])
    arm_rows = []
    for s in S:
        if s['protocolSection']['identificationModule']['nctId'] in keep:
            r = pt.parse_study(s)
            if r: arm_rows += r
    ARM, CON, card = bb.build(pd.DataFrame(arm_rows)); CON['ctrl_kind'] = CON['ctrl_role']
    for mdl in list(BOUND) + ['haiku4.5_reliable']:
        CON[f'holdout_{mdl}'] = CON.nct_id.map(H.set_index('nct_id')[f'holdout_{mdl}'])
    CON.to_csv('holdout_contrasts.csv.gz', index=False); ARM.to_csv('holdout_arms.csv.gz', index=False)
    json.dump(card, open('holdout_datacard.json', 'w'), indent=1, default=str)
    for r in rows:
        m = r['model']; c = CON[CON[f'holdout_{m}'] == True]
        for fam in ['binary', 'continuous']:
            cf = c[c.family == fam]; r[f'{fam}_contrasts'] = len(cf); r[f'{fam}_trials'] = cf.nct_id.nunique()
    pd.DataFrame(rows).to_csv('holdout_counts.csv', index=False)
    print(len(S), H.in_benchmark.sum()); print(pd.DataFrame(rows).to_string())
