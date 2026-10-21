"""Step 3a: design fields (allocation x intervention model) for the 8,011 benchmark trials from the Oct 2026 re-fetch, and
IS on randomized parallel-group contrasts (registry seeds 0-4 with shared trial resamples; LM paired files).
Outputs: design_counts.csv, design_subset_is.csv, design_trial_fields.csv"""
import pickle
from common import *  # shared_boot_mean lives in common.py
C = load_contrasts(); TE = test_sets(C); L = load_lm(TE); M = load_meta().set_index('nct_id')
cache = pickle.load(open('registry_preds_cache.pkl', 'rb')); P = cache['P']
C['allocation_now'] = C.nct_id.map(M.allocation_now).fillna('MISSING'); C['intervention_model'] = C.nct_id.map(M.intervention_model).fillna('MISSING')
M[['allocation_now', 'intervention_model', 'masking_now', 'phases', 'pcd', 'rfp']].reset_index().to_csv('design_trial_fields.csv', index=False)
rows = []
for scope, D in [('all_8011_trials', C), ('train', C[C.split_time == 'train']), ('test', C[C.split_time == 'test'])]:
    for fam in ['all', 'binary', 'continuous']:
        d = D if fam == 'all' else D[D.family == fam]
        g = d.groupby(['allocation_now', 'intervention_model']).agg(n_contrasts=('nct_id', 'size'), n_trials=('nct_id', 'nunique')).reset_index()
        g['scope'] = scope; g['family'] = fam; g['frac_contrasts'] = g.n_contrasts / len(d); rows.append(g)
DC = pd.concat(rows)[['scope', 'family', 'allocation_now', 'intervention_model', 'n_contrasts', 'n_trials', 'frac_contrasts']]
DC.to_csv('design_counts.csv', index=False)
agree = (C.drop_duplicates('nct_id').allocation.fillna('MISSING') == C.drop_duplicates('nct_id').allocation_now).mean()

res = []
for fam in ['binary', 'continuous']:
    te = TE[fam]; y, g = te.delta.to_numpy(), te.nct_id.to_numpy()
    rp = ((te.nct_id.map(M.allocation_now) == 'RANDOMIZED') & (te.nct_id.map(M.intervention_model) == 'PARALLEL')).to_numpy()
    for sub, m in [('all test', np.ones(len(te), bool)), ('randomized parallel-group', rp), ('excluded (not randomized parallel)', ~rp)]:
        yy, gg = y[m], g[m]
        s0 = skill_ci(yy, P[(fam, 'full', 0)][m], P[(fam, 'blind_ctrl', 0)][m], gg, 1000)
        per = [me.skill(yy, P[(fam, 'full', s)][m], P[(fam, 'blind_ctrl', s)][m]) for s in range(5)]
        lo5, hi5 = shared_boot_mean(yy, [P[(fam, 'full', s)][m] for s in range(5)], [P[(fam, 'blind_ctrl', s)][m] for s in range(5)], gg)
        base = dict(family=fam, subset=sub, n=int(m.sum()), n_trials=int(pd.Series(gg).nunique()), drop_frac_vs_test=1 - m.mean())
        res.append(dict(base, arm='registry GBT seed 0', IS=s0[0], ci_low=s0[1], ci_high=s0[2]))
        res.append(dict(base, arm='registry GBT 5-seed mean (shared resamples)', IS=float(np.mean(per)), ci_low=lo5, ci_high=hi5, seed_sd=float(np.std(per, ddof=1))))
        for key, name in [('sonnet_bin', 'claude-sonnet-5'), ('haiku_bin', 'claude-haiku-4-5-20251001'), ('sonnet_cont', 'claude-sonnet-5')]:
            D = L[key]
            if (fam == 'binary') != key.endswith('bin'): continue
            s = skill_ci(D.truth.to_numpy()[m], D.point_A.to_numpy()[m], D.point_B.to_numpy()[m], gg, 2000)
            res.append(dict(base, arm=name, IS=s[0], ci_low=s[1], ci_high=s[2]))
R = pd.DataFrame(res); R.to_csv('design_subset_is.csv', index=False)
rp_all = (C.allocation_now == 'RANDOMIZED') & (C.intervention_model == 'PARALLEL')
print('allocation agreement (benchmark vs refetch, per trial):', round(agree, 4))
print('drop fraction full benchmark:', round(1 - rp_all.mean(), 4), 'trials kept', C[rp_all].nct_id.nunique(), 'of', C.nct_id.nunique())
for fam in ['binary', 'continuous']:
    print(fam, 'test drop', round(1 - rp_all[C.split_time.eq('test') & C.family.eq(fam)].mean(), 4), 'train drop', round(1 - rp_all[C.split_time.eq('train') & C.family.eq(fam)].mean(), 4))
print(DC[(DC.scope == 'all_8011_trials') & (DC.family == 'all')].to_string())
print(R.round(3).to_string())
