"""Fit blind_ctrl/full registry models (both families, seeds 0-4) on the original training split; cache test predictions."""
import pickle
from registry_models import *
C = load_contrasts(); TE = test_sets(C)
HO = pd.read_csv('holdout_contrasts.csv.gz', low_memory=False)
P, PH = {}, {}
for fam in ['binary', 'continuous']:
    Fm = fit_family(C, fam)
    for k, f in Fm.items():
        P[(fam,) + k] = f(TE[fam]); h = HO[HO.family == fam]
        PH[(fam,) + k] = f(h) if len(h) else np.array([])
    # keep seed-0 predictors for the swap test
    pickle.dump({k: v for k, v in Fm.items()}, open(f'registry_predictors_{fam}.pkl', 'wb')) if False else None
pickle.dump(dict(P=P, PH=PH), open('registry_preds_cache.pkl', 'wb'))
for fam in ['binary', 'continuous']:
    y, g = TE[fam].delta.to_numpy(), TE[fam].nct_id.to_numpy()
    for s in range(5):
        print(fam, s, round(me.skill(y, P[(fam, 'full', s)], P[(fam, 'blind_ctrl', s)]), 4))
