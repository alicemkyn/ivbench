"""Language-model interventional skill on the v2.1 test sets restricted to contrasts whose cid and content are unchanged
from v2 (v21_status == 'kept'). IS = skill(truth, FULL, BLIND) = 1 - MSE(A)/MSE(B); CI = ivbench.models.boot_skill,
trial-level, 2,000 resamples, seed 0. Inputs: results/llm/llm_v2_paired.csv, llm_haiku_paired.csv, llm_cont_paired.csv
(point_A = FULL, point_B = BLIND) and lm_wave1_merged.csv (pA / pB = protocol-only A / B; point_A / point_B = original).
Outputs: is_lm_v21.csv, is_lm_removed_by_reason_v21.csv (uses data/v2.1/v2_removed_by_reason_v2.1.csv)."""
import os
from common import *
LMF = {'sonnet_binary': ('llm_v2_paired.csv', 'binary'), 'haiku_binary': ('llm_haiku_paired.csv', 'binary'),
       'sonnet_continuous': ('llm_cont_paired.csv', 'continuous')}

def lm_row(name, D, a, b, sub, fam):
    y, g = D.truth.to_numpy(), D.nct_id.to_numpy()
    s = me.skill(y, D[a].to_numpy(), D[b].to_numpy()); lo, hi = me.boot_skill(y, D[a].to_numpy(), D[b].to_numpy(), g, n=2000, seed=0)
    return dict(arm=name, family=fam, test_set=sub, full_col=a, blind_col=b, IS=s, ci_low=lo, ci_high=hi, n=len(D), n_trials=len(np.unique(g)))

if __name__ == '__main__':
    C21 = pd.read_csv(os.path.join(REPO, 'data', 'v2.1', 'benchmark_contrasts_v2.1.csv.gz'), low_memory=False)
    kept = {f: set(C21.loc[(C21.family == f) & (C21.split_time == 'test') & (C21.v21_status == 'kept'), 'cid']) for f in ['binary', 'continuous']}
    t21 = C21.dropna(subset=['cid']).set_index('cid').delta
    rows = []
    for name, (fn, fam) in LMF.items():
        D = pd.read_csv(os.path.join(REPO, 'results', 'llm', fn))
        rows.append(lm_row(name, D, 'point_A', 'point_B', 'v2_test_full', fam))
        Dk = D[D.cid.isin(kept[fam])]
        assert np.allclose(Dk.truth.to_numpy(), t21.loc[Dk.cid].to_numpy())
        rows.append(lm_row(name, Dk, 'point_A', 'point_B', 'v21_test_kept', fam))
    LW = pd.read_csv('lm_wave1_merged.csv'); assert np.allclose(LW.truth, LW.t2)
    for nm, a, b in [('sonnet_binary_protocol_only', 'pA', 'pB'), ('sonnet_binary_original_wave1file', 'point_A', 'point_B')]:
        rows.append(lm_row(nm, LW, a, b, 'v2_test_full', 'binary'))
        rows.append(lm_row(nm, LW[LW.cid.isin(kept['binary'])], a, b, 'v21_test_kept', 'binary'))
    pd.DataFrame(rows).to_csv('is_lm_v21.csv', index=False)
    X = pd.read_csv(os.path.join(REPO, 'data', 'v2.1', 'v2_removed_by_reason_v2.1.csv')); X = X[X.v2_cid.notna()]; res = []
    for name, (fn, fam) in LMF.items():
        D = pd.read_csv(os.path.join(REPO, 'results', 'llm', fn)).merge(X[['v2_cid', 'reason']], left_on='cid', right_on='v2_cid')
        for rsn, d in D.groupby('reason'):
            res.append(dict(arm=name, reason=rsn, n=len(d), IS=me.skill(d.truth.values, d.point_A.values, d.point_B.values),
                            mse_A=float(np.mean((d.truth - d.point_A) ** 2)), mse_B=float(np.mean((d.truth - d.point_B) ** 2))))
    pd.DataFrame(res).to_csv('is_lm_removed_by_reason_v21.csv', index=False)
    print(pd.DataFrame(rows).round(4).to_string())
