"""Step 7: registry swap test. For every test contrast, swap all treatment-side and comparator-side fields (intervention names,
intervention descriptions, protocol-arm descriptions, results-group titles and descriptions, analysed n), keep everything else
(endpoint, population, design, ctrl_kind, treat_intervention_types, which have no comparator-side counterpart), and predict with
the trained FULL and BLIND (blind_ctrl) models (seed 0 primary; seeds 1-4 reported). Note: the featurizer concatenates all text
fields into one TF-IDF document, so swapping text columns leaves the text representation unchanged up to field order.
Outputs: swap_test.csv, swap_test_predictions.csv"""
import pickle
from registry_models import *
PAIRS = [('treat_intervention', 'ctrl_intervention'), ('treat_intervention_desc', 'ctrl_intervention_desc'), ('treat_arm_desc', 'ctrl_arm_desc'),
         ('treat_group', 'ctrl_group'), ('treat_group_desc', 'ctrl_group_desc'), ('n_treat', 'n_ctrl'), ('treat_arm_matched', 'ctrl_arm_matched')]
def swap(df):
    d = df.copy()
    for a, b in PAIRS: d[a], d[b] = df[b].values, df[a].values
    return d
if __name__ == '__main__':
    C = load_contrasts(); TE = test_sets(C); P = pickle.load(open('registry_preds_cache.pkl', 'rb'))['P']; rows, preds = [], []
    for fam in ['binary', 'continuous']:
        te = TE[fam]; ts = swap(te); Fm = fit_family(C, fam)
        for s in range(5):
            for nm in ['full', 'blind_ctrl']:
                f = Fm[(nm, s)]; po = f(te); ps = f(ts)
                assert np.allclose(po, P[(fam, nm, s)]), 'predictor mismatch'
                nz = np.abs(po) > 1e-12
                rows.append(dict(family=fam, seed=s, model=nm, n=len(te), sign_reversal_frac=float(np.mean(np.sign(ps[nz]) == -np.sign(po[nz]))),
                                 corr_swapped_vs_negated=float(np.corrcoef(ps, -po)[0, 1]), corr_swapped_vs_original=float(np.corrcoef(ps, po)[0, 1]),
                                 mean_shift=float(np.mean(ps - po)), mean_abs_shift=float(np.mean(np.abs(ps - po))), mean_abs_pred=float(np.mean(np.abs(po))),
                                 ideal_mean_shift=float(np.mean(-2 * po)), frac_unchanged=float(np.mean(np.abs(ps - po) < 1e-12))))
                if s == 0: preds.append(pd.DataFrame(dict(family=fam, cid=te.cid, nct_id=te.nct_id, model=nm, pred_orig=po, pred_swapped=ps, truth=te.delta)))
    R = pd.DataFrame(rows); R.to_csv('swap_test.csv', index=False); pd.concat(preds).to_csv('swap_test_predictions.csv', index=False)
    pd.set_option('display.width', 250); print(R.round(4).to_string())
