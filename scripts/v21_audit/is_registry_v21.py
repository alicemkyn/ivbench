"""Registry-model interventional skill (IS = skill(FULL | BLIND+ctrl), ivbench.specs.run_models, GBT) on v2 and v2.1.
Test sets: v2 full; v2.1 full; v2.1 restricted to contrasts whose cid and content are unchanged from v2 ('kept').
Models are trained on the train split of the same benchmark version (year_posted < 2023). Seeds 0-4; seed-0 CI by
ivbench.models.boot_skill (1,000 trial resamples, seed 0); 5-seed mean CI by common.shared_boot_mean (one shared set of
1,000 trial resamples applied to every seed). Output: is_registry_v21.csv, registry_preds_v21.csv.gz"""
import os, sys
from common import *
from ivbench.specs import run_models
V21 = os.path.join(REPO, 'data', 'v2.1', 'benchmark_contrasts_v2.1.csv.gz')
SEEDS = [0, 1, 2, 3, 4]

def main():
    C2 = load_contrasts(); C21 = pd.read_csv(V21, low_memory=False)
    TE2 = test_sets(C2)
    rows, preds = [], []
    for fam in ['binary', 'continuous']:
        for ver, C in [('v2', C2), ('v2.1', C21)]:
            tr = train_set(C, fam)
            te = TE2[fam] if ver == 'v2' else C[(C.family == fam) & (C.split_time == 'test')].reset_index(drop=True)
            P = {s: run_models(tr, te, ['blind_ctrl', 'full'], seed=s) for s in SEEDS}
            if ver == 'v2':
                kept = set(C21.loc[(C21.family == fam) & (C21.split_time == 'test') & (C21.v21_status == 'kept'), 'cid'])
                subsets = {'v2_test_full': np.ones(len(te), bool), 'v2_test_restricted_to_v21_kept': te.cid.isin(kept).to_numpy()}
            else:
                subsets = {'v21_test_full': np.ones(len(te), bool), 'v21_test_kept': (te.v21_status == 'kept').to_numpy(),
                           'v21_test_repaired_or_new': (te.v21_status != 'kept').to_numpy()}
            for s in SEEDS:
                preds.append(pd.DataFrame(dict(version=ver, family=fam, seed=s, cid=te.cid, nct_id=te.nct_id, truth=te.delta,
                                               pred_blind_ctrl=P[s]['blind_ctrl'], pred_full=P[s]['full'])))
            for name, m in subsets.items():
                y, g = te.delta.to_numpy()[m], te.nct_id.to_numpy()[m]
                for s in SEEDS:
                    sk, lo, hi = skill_ci(y, P[s]['full'][m], P[s]['blind_ctrl'][m], g, 1000, seed=0)
                    rows.append(dict(train_version=ver, test_set=name, family=fam, seed=s, IS=sk, ci_low=lo if s == 0 else np.nan,
                                     ci_high=hi if s == 0 else np.nan, n=int(m.sum()), n_trials=int(len(np.unique(g)))))
                    print(ver, name, fam, s, round(sk, 4), flush=True)
                mean = float(np.mean([me.skill(y, P[s]['full'][m], P[s]['blind_ctrl'][m]) for s in SEEDS]))
                lo, hi = shared_boot_mean(y, [P[s]['full'][m] for s in SEEDS], [P[s]['blind_ctrl'][m] for s in SEEDS], g, n=1000, seed=0)
                rows.append(dict(train_version=ver, test_set=name, family=fam, seed='mean_0-4', IS=mean, ci_low=lo, ci_high=hi,
                                 n=int(m.sum()), n_trials=int(len(np.unique(g)))))
    pd.DataFrame(rows).to_csv('is_registry_v21.csv', index=False)
    pd.concat(preds).to_csv('registry_preds_v21.csv.gz', index=False)

if __name__ == '__main__':
    main()
