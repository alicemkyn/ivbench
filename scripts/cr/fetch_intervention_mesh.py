"""Fetch trial-level intervention MeSH terms (derivedSection.interventionBrowseModule.meshes) from the ClinicalTrials.gov
API v2 for every binary-family trial in benchmark v2 and v2.1. Writes data/cr/intervention_mesh_terms.csv (nct_id, mesh_ids,
mesh_terms; '|'-separated). Used by scripts/cr/tables/tab_mol.py (mesh / union keys). Network required; the table scripts read
the saved CSV only. The released file was fetched on 2026-10-17.
  python scripts/cr/fetch_intervention_mesh.py"""
import os
import pandas as pd, requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
API = 'https://clinicaltrials.gov/api/v2/studies'

if __name__ == '__main__':
    ids = set()
    for p in ['data/benchmark_contrasts.csv.gz', 'data/v2.1/benchmark_contrasts_v2.1_final.csv.gz']:
        C = pd.read_csv(os.path.join(ROOT, p), low_memory=False); ids |= set(C.loc[C.family == 'binary', 'nct_id'])
    ids, out = sorted(ids), {}
    for i in range(0, len(ids), 200):
        r = requests.get(API, params={'filter.ids': ','.join(ids[i:i + 200]), 'pageSize': 1000,
                                      'fields': 'protocolSection.identificationModule.nctId,derivedSection.interventionBrowseModule.meshes'}, timeout=60)
        r.raise_for_status()
        for s in r.json()['studies']:
            out[s['protocolSection']['identificationModule']['nctId']] = s.get('derivedSection', {}).get('interventionBrowseModule', {}).get('meshes', [])
    assert len(out) == len(ids), (len(out), len(ids))
    pd.DataFrame([dict(nct_id=k, mesh_ids='|'.join(m['id'] for m in v), mesh_terms='|'.join(m['term'] for m in v))
                  for k, v in sorted(out.items())]).to_csv(os.path.join(ROOT, 'data', 'cr', 'intervention_mesh_terms.csv'), index=False)
    print(f'wrote data/cr/intervention_mesh_terms.csv ({len(out)} trials)')
