"""Registry-model IS (seed 0 and 5-seed shared-resample mean) inside and outside each model's three-condition uncontaminated
set (reference only: the registry model has no pretraining contamination; this shows what a date stratum alone does).
Output: contamination_registry_is.csv"""
import pickle
from common import *
C = load_contrasts(); TE = test_sets(C); M = load_meta(); RR = load_result_refs(); P = pickle.load(open('registry_preds_cache.pkl', 'rb'))['P']; rows = []
for mdl, b in list(BOUND.items()) + [('haiku4.5_reliable', RELI['haiku4.5'])]:
    F = contamination_flags(M, RR, b)
    for fam in ['binary', 'continuous']:
        te = TE[fam]; y, g = te.delta.to_numpy(), te.nct_id.to_numpy(); u = F.loc[te.nct_id].uncont.to_numpy(bool)
        for st, m in [('uncontaminated', u), ('complement', ~u)]:
            if m.sum() < 2: rows.append(dict(model=mdl, family=fam, stratum=st, n=int(m.sum()))); continue
            s0 = skill_ci(y[m], P[(fam, 'full', 0)][m], P[(fam, 'blind_ctrl', 0)][m], g[m], 1000)
            Fs = [P[(fam, 'full', s)][m] for s in range(5)]; Bs = [P[(fam, 'blind_ctrl', s)][m] for s in range(5)]
            lo, hi = shared_boot_mean(y[m], Fs, Bs, g[m])
            rows.append(dict(model=mdl, boundary=str(b.date()), family=fam, stratum=st, n=int(m.sum()), n_trials=len(np.unique(g[m])), IS_seed0=s0[0], ci_low=s0[1], ci_high=s0[2],
                             IS_5seed_mean=float(np.mean([me.skill(y[m], a, c) for a, c in zip(Fs, Bs)])), ci5_low=lo, ci5_high=hi))
pd.DataFrame(rows).to_csv('contamination_registry_is.csv', index=False); print(pd.DataFrame(rows).round(3).to_string())
