"""Step 1: cutoff-corrected contamination strata (three-condition uncontaminated set) and LM IS by stratum.
Inputs: ctgov_meta_oct2026.jsonl.gz (fetch_ctgov_meta.py), ctgov_references_pubmed.csv (resolve_pubmed.py).
Outputs: contamination_trial_flags.csv, contamination_counts.csv, contamination_lm_is.csv"""
from common import *
C = load_contrasts(); TE = test_sets(C); L = load_lm(TE); M = load_meta(); RR = load_result_refs()
allb = [(k, v, 'training-data cutoff') for k, v in BOUND.items()] + [(k, v, 'reliable-knowledge cutoff') for k, v in RELI.items()]
# per-trial flags
FL = M[['nct_id', 'pcd', 'pcd_type', 'rfp']].copy()
nres = RR.groupby('nct_id').size(); FL['n_result_refs'] = FL.nct_id.map(nres).fillna(0).astype(int)
FL['earliest_result_pub'] = FL.nct_id.map(RR.groupby('nct_id').pub_earliest.min())
FL['n_result_refs_undated'] = FL.nct_id.map(RR[RR.pub_earliest.isna()].groupby('nct_id').size()).fillna(0).astype(int)
for k, v, kind in allb:
    tag = k if kind.startswith('training') else k + '_reliable'
    FL[f'uncont_{tag}'] = FL.nct_id.map(contamination_flags(M, RR, v).uncont)
FL['in_test_binary'] = FL.nct_id.isin(TE['binary'].nct_id); FL['in_test_continuous'] = FL.nct_id.isin(TE['continuous'].nct_id)
FL.to_csv('contamination_trial_flags.csv', index=False)
rows, lrows = [], []
LMMAP = {'sonnet5': [('binary', 'sonnet_bin', 'claude-sonnet-5'), ('continuous', 'sonnet_cont', 'claude-sonnet-5')],
         'haiku4.5': [('binary', 'haiku_bin', 'claude-haiku-4-5-20251001')], 'llama3.1': []}
for k, b, kind in allb:
    F = contamination_flags(M, RR, b)
    for fam, te in TE.items():
        f = F.loc[te.nct_id]
        for cond, m in [('pcd>=b', f.c_pcd), ('rfp>=b', f.c_rfp), ('no RESULT pub <b', f.c_nopub), ('pcd>=b & rfp>=b', f.c_pcd & f.c_rfp), ('uncontaminated (all three)', f.uncont)]:
            m = m.to_numpy(bool)
            rows.append(dict(model=k, cutoff_kind=kind, boundary=str(b.date()), family=fam, condition=cond, n_contrasts=int(m.sum()),
                             n_trials=int(te.nct_id[m].nunique()), n_test_contrasts=len(te), n_test_trials=te.nct_id.nunique()))
    for fam, key, mid in LMMAP[k]:
        D = L[key]; f = F.loc[D.nct_id]
        for stratum, m in [('uncontaminated', f.uncont.to_numpy(bool)), ('complement', ~f.uncont.to_numpy(bool)),
                           ('pcd>=b', f.c_pcd.to_numpy(bool)), ('pcd<b', ~f.c_pcd.to_numpy(bool))]:
            d = D[m]
            r = dict(IS=np.nan, ci_low=np.nan, ci_high=np.nan) if len(d) == 0 else dict(zip(['IS', 'ci_low', 'ci_high'], skill_ci(d.truth, d.point_A, d.point_B, d.nct_id, 2000)))
            lrows.append(dict(model=k, model_id=mid, cutoff_kind=kind, boundary=str(b.date()), family=fam, stratum=stratum, **r, n=len(d),
                              n_trials=d.nct_id.nunique(), call_dates='2026-09-10/11', note='CI undefined for <2 trials; n<30 not interpretable' if len(d) < 30 else ''))
pd.DataFrame(rows).to_csv('contamination_counts.csv', index=False)
pd.DataFrame(lrows).to_csv('contamination_lm_is.csv', index=False)
print(pd.DataFrame(rows).query("condition=='uncontaminated (all three)'")[['model', 'cutoff_kind', 'family', 'n_contrasts', 'n_trials']].to_string())
print(pd.DataFrame(lrows).round(3).to_string())
print('RESULT refs:', len(RR), 'undated:', int(RR.pub_earliest.isna().sum()))
