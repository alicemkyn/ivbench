"""Step 6: protocol-time registry model. Same GBT/featurizer as ivbench.specs, with results-derived fields removed from BOTH
the FULL and BLIND specs: numerics n_treat, n_ctrl, age_mean, frac_female, year_posted; results-group text treat_group,
treat_group_desc, ctrl_group, ctrl_group_desc. Five seeds, both families; IS = skill(FULL | BLIND) with 1,000 trial resamples
(seed 0); five-seed mean with shared resamples. Outputs: protocol_time_is.csv, protocol_time_preds.npz"""
from common import *
from ivbench.specs import SPECS, run_models
RESULTS_NUM = ['n_treat', 'n_ctrl', 'age_mean', 'frac_female', 'year_posted']
RESULTS_TXT = ['treat_group', 'treat_group_desc', 'ctrl_group', 'ctrl_group_desc']
PT = {k: dict(text=[c for c in v['text'] if c not in RESULTS_TXT], cat=list(v['cat']), num=[c for c in v['num'] if c not in RESULTS_NUM])
      for k, v in SPECS.items() if k in ('blind_ctrl', 'full')}
if __name__ == '__main__':
    C = load_contrasts(); TE = test_sets(C); rows, store = [], {}
    for fam in ['binary', 'continuous']:
        tr, te = train_set(C, fam), TE[fam]; y, g = te.delta.to_numpy(), te.nct_id.to_numpy(); ref = np.full(len(y), tr.delta.mean())
        Pf, Pb = [], []
        for s in range(5):
            P = run_models(tr, te, ['blind_ctrl', 'full'], seed=s, specs=PT); Pf.append(P['full']); Pb.append(P['blind_ctrl'])
            store[f'{fam}_full_s{s}'] = P['full']; store[f'{fam}_blind_ctrl_s{s}'] = P['blind_ctrl']
            sk = skill_ci(y, P['full'], P['blind_ctrl'], g, 1000)
            rows.append(dict(family=fam, seed=s, model='protocol-time full', reference='protocol-time blind_ctrl', IS=sk[0], ci_low=sk[1], ci_high=sk[2],
                             full_skill_vs_mean=me.skill(y, P['full'], ref), blind_skill_vs_mean=me.skill(y, P['blind_ctrl'], ref), n=len(y), n_trials=len(np.unique(g))))
            print(fam, s, round(sk[0], 4), flush=True)
        per = [me.skill(y, a, b) for a, b in zip(Pf, Pb)]; lo, hi = shared_boot_mean(y, Pf, Pb, g)
        rows.append(dict(family=fam, seed='mean of 5 (shared resamples)', model='protocol-time full', reference='protocol-time blind_ctrl', IS=float(np.mean(per)),
                         ci_low=lo, ci_high=hi, seed_sd=float(np.std(per, ddof=1)), n=len(y), n_trials=len(np.unique(g))))
    pd.DataFrame(rows).to_csv('protocol_time_is.csv', index=False); np.savez('protocol_time_preds.npz', **store)
    print(pd.DataFrame(rows).round(3).to_string())
