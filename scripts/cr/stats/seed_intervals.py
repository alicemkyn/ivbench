"""Step 5 (seeds): five-seed mean interventional skill with ONE trial-bootstrap interval computed on shared resamples
(the same 1,000 trial resamples, seed 0, applied to every seed's predictions), plus per-seed values and the seed-to-seed SD.
Output: seed_intervals.csv"""
import pickle
from common import *
C = load_contrasts(); TE = test_sets(C); P = pickle.load(open('registry_preds_cache.pkl', 'rb'))['P']; rows = []
for fam in ['binary', 'continuous']:
    te = TE[fam]; y, g = te.delta.to_numpy(), te.nct_id.to_numpy()
    F = [P[(fam, 'full', s)] for s in range(5)]; B = [P[(fam, 'blind_ctrl', s)] for s in range(5)]
    for s in range(5):
        sk = skill_ci(y, F[s], B[s], g, 1000); rows.append(dict(family=fam, seed=str(s), IS=sk[0], ci_low=sk[1], ci_high=sk[2], n=len(y), n_trials=len(np.unique(g))))
    per = [me.skill(y, a, b) for a, b in zip(F, B)]; lo, hi = shared_boot_mean(y, F, B, g, n=1000, seed=0)
    rows.append(dict(family=fam, seed='mean of 5 (shared resamples)', IS=float(np.mean(per)), ci_low=lo, ci_high=hi, seed_sd=float(np.std(per, ddof=1)),
                     seed_min=float(min(per)), seed_max=float(max(per)), n=len(y), n_trials=len(np.unique(g))))
pd.DataFrame(rows).to_csv('seed_intervals.csv', index=False); print(pd.DataFrame(rows).round(4).to_string())
