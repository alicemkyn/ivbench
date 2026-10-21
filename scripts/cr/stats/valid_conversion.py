"""Step 4: valid-conversion subsets, IS on them (registry seed 0 / 5-seed shared-resample mean; LM paired files), and the
sampling-noise ceiling recomputed on each subset (same formulas as scripts/run_positive_control.py::noise_ceiling).
Outputs: valid_conversion_is.csv, valid_conversion_noise_ceiling.csv, valid_conversion_flags.csv"""
import pickle
from common import *
from fetch_audit_records import binary_participant
C = load_contrasts(); TE = test_sets(C); L = load_lm(TE); M = load_meta().set_index('nct_id')
P = pickle.load(open('registry_preds_cache.pkl', 'rb'))['P']

def sampling_var(te, family):
    n1, n2 = te.n_treat.to_numpy(float), te.n_ctrl.to_numpy(float)
    if family == 'binary':
        p1 = np.clip(te.y_treat.to_numpy(float), 0.5 / n1, 1 - 0.5 / n1); p2 = np.clip(te.y_ctrl.to_numpy(float), 0.5 / n2, 1 - 0.5 / n2)
        return p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
    d = te.delta.to_numpy(float); return (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))

def subsets(fam, te):
    rand = (te.allocation == 'RANDOMIZED').to_numpy()
    rp = ((te.nct_id.map(M.allocation_now) == 'RANDOMIZED') & (te.nct_id.map(M.intervention_model) == 'PARALLEL')).to_numpy()
    if fam == 'binary':
        v = binary_participant(te)
        return {'all test': np.ones(len(te), bool), 'participant-level unit': v, 'participant-level unit & randomized': v & rand,
                'participant-level unit & randomized parallel': v & rp}
    mean = (te.param_type == 'MEAN').to_numpy(); ls = te.param_type.isin(['MEAN', 'LEAST_SQUARES_MEAN']).to_numpy()
    return {'all test': np.ones(len(te), bool), 'strict: arithmetic MEAN (SD/SE/CI)': mean, 'strict & randomized': mean & rand,
            'strict & randomized parallel': mean & rp, 'sensitivity: MEAN or LS MEAN & randomized': ls & rand,
            'sensitivity: MEAN or LS MEAN & randomized parallel': ls & rp}

LMK = {'binary': [('sonnet_bin', 'claude-sonnet-5'), ('haiku_bin', 'claude-haiku-4-5-20251001')], 'continuous': [('sonnet_cont', 'claude-sonnet-5')]}
rows, nrows, flags = [], [], []
for fam in ['binary', 'continuous']:
    te = TE[fam]; tr = train_set(C, fam); y, g = te.delta.to_numpy(), te.nct_id.to_numpy(); sv_all = sampling_var(te, fam)
    SS = subsets(fam, te)
    fl = te[['cid', 'nct_id']].copy()
    for name, m in SS.items():
        fl[name] = m
        yy, gg = y[m], g[m]; base = dict(family=fam, subset=name, n=int(m.sum()), n_trials=int(pd.Series(gg).nunique()), frac_of_test=float(m.mean()))
        s0 = skill_ci(yy, P[(fam, 'full', 0)][m], P[(fam, 'blind_ctrl', 0)][m], gg, 1000)
        per = [me.skill(yy, P[(fam, 'full', s)][m], P[(fam, 'blind_ctrl', s)][m]) for s in range(5)]
        lo5, hi5 = shared_boot_mean(yy, [P[(fam, 'full', s)][m] for s in range(5)], [P[(fam, 'blind_ctrl', s)][m] for s in range(5)], gg)
        rows.append(dict(base, arm='registry GBT seed 0', IS=s0[0], ci_low=s0[1], ci_high=s0[2]))
        rows.append(dict(base, arm='registry GBT 5-seed mean (shared resamples)', IS=float(np.mean(per)), ci_low=lo5, ci_high=hi5, seed_sd=float(np.std(per, ddof=1))))
        for key, mid in LMK[fam]:
            D = L[key]; s = skill_ci(D.truth.to_numpy()[m], D.point_A.to_numpy()[m], D.point_B.to_numpy()[m], gg, 2000)
            rows.append(dict(base, arm=mid, IS=s[0], ci_low=s[1], ci_high=s[2]))
        # noise ceiling on the subset
        sv = sv_all[m]; tot = float(np.var(yy)); mse_mean = float(np.mean((yy - tr.delta.mean()) ** 2))
        rng = np.random.default_rng(0); u = np.unique(gg); idx = {t: np.flatnonzero(gg == t) for t in u}; bo = []
        for _ in range(1000):
            ii = np.concatenate([idx[t] for t in rng.choice(u, len(u), replace=True)]); bo.append(1 - sv[ii].mean() / np.var(yy[ii]))
        nb = dict(base, mean_sampling_var=float(sv.mean()), total_var_delta=tot, mse_train_mean=mse_mean, reliability=1 - sv.mean() / tot,
                  reliability_ci_low=float(np.percentile(bo, 2.5)), reliability_ci_high=float(np.percentile(bo, 97.5)),
                  ceiling_skill_vs_mean=1 - sv.mean() / mse_mean, frac_noise_dominated=float(np.mean(sv > tot)))
        refs = [('registry blind_ctrl seed 0', P[(fam, 'blind_ctrl', 0)][m]), ('registry full seed 0', P[(fam, 'full', 0)][m])]
        refs += [(f'{mid} BLIND (B)', L[key].point_B.to_numpy()[m]) for key, mid in LMK[fam]]
        for rname, rp_ in refs:
            mse_ref = float(np.mean((yy - rp_) ** 2))
            nrows.append(dict(nb, reference=rname, mse_reference=mse_ref, skill_vs_mean_reference=1 - mse_ref / mse_mean,
                              max_interventional_skill_given_ref=1 - sv.mean() / mse_ref))
    flags.append(fl)
pd.DataFrame(rows).to_csv('valid_conversion_is.csv', index=False)
pd.DataFrame(nrows).to_csv('valid_conversion_noise_ceiling.csv', index=False)
pd.concat(flags).to_csv('valid_conversion_flags.csv', index=False)
print(pd.DataFrame(rows).round(3).to_string())
print(pd.DataFrame(nrows)[['family', 'subset', 'n', 'reliability', 'reliability_ci_low', 'reliability_ci_high', 'ceiling_skill_vs_mean', 'reference', 'max_interventional_skill_given_ref']].round(3).to_string())
