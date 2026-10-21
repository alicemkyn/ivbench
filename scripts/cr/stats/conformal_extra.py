"""Step 5 (registry part): re-run the two calibration recipes of scripts/run_shift.py (identical code path, seed 0) exposing
calibration residuals, then add
  (1) a trial-level valid split-conformal variant: calibrate on ONE randomly chosen contrast per calibration trial
      (R=200 draws), evaluate coverage on all test contrasts and on one random contrast per test trial;
      the same one-per-trial rule applied to the cross-fitted OOF residuals;
  (2) trial-bootstrap (1,000 resamples over nct_id, seed 0) coverage intervals for every row (Wilson is not used);
  (3) abstention thresholds chosen on calibration data (OOF predictions/sigma of the training pool) and evaluated on test.
Outputs: conformal_coverage_extra.csv, conformal_abstention_registry.csv"""
import sys, os
from common import *
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import run_shift as rs
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
ALPHA, SEED, R = 0.10, 0, 200

def qhat(res, alpha=ALPHA):
    n = len(res); return float(np.quantile(np.abs(res), min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)))

def one_per_trial(groups, rng):
    """Index of one uniformly chosen row per trial."""
    s = pd.Series(np.arange(len(groups))).groupby(groups).apply(lambda x: x.iloc[rng.integers(len(x))])
    return s.to_numpy()

def regime(name, pool, test):
    Xtr, Xte, _, _ = me.featurize(pool, test, rs.FULL_TXT, cat_cols=rs.FULL_CAT, num_cols=rs.POP_NUM)
    ytr, yte = pool.delta.to_numpy(), test.delta.to_numpy(); gtr, gte = pool.nct_id.to_numpy(), test.nct_id.to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED); i_fit, i_cal = next(gss.split(Xtr, ytr, gtr))
    m1, p_te = me.fit_gbt(Xtr[i_fit], ytr[i_fit], Xte, seed=SEED); res_cal = ytr[i_cal] - m1.predict(Xtr[i_cal])
    s1 = rs.fit_sigma(Xtr[i_fit], np.abs(ytr[i_fit] - m1.predict(Xtr[i_fit]))); sig_cal, sig_te = rs.sig(s1, Xtr[i_cal]), rs.sig(s1, Xte)
    folds = list(GroupKFold(n_splits=5).split(Xtr, ytr, gtr)); oof = np.zeros_like(ytr)
    for k, (a, b) in enumerate(folds): _, oof[b] = me.fit_gbt(Xtr[a], ytr[a], Xtr[b], seed=SEED + k)
    oof_res = ytr - oof; oof_sig = np.zeros_like(ytr)
    for k, (a, b) in enumerate(folds): oof_sig[b] = rs.sig(rs.fit_sigma(Xtr[a], np.abs(oof_res[a]), seed=SEED + k), Xtr[b])
    mF, p_te2 = me.fit_gbt(Xtr, ytr, Xte, seed=SEED); sig_te2 = rs.sig(rs.fit_sigma(Xtr, np.abs(oof_res)), Xte)
    return dict(name=name, yte=yte, gte=gte, split=dict(p=p_te, res_cal=res_cal, sig_cal=sig_cal, sig_te=sig_te, g_cal=gtr[i_cal]),
                crossfit=dict(p=p_te2, res_cal=oof_res, sig_cal=oof_sig, sig_te=sig_te2, g_cal=gtr, oof_pred=oof, y_cal=ytr))

def coverage_rows(Rg):
    rows = []; y, g = Rg['yte'], Rg['gte']
    for rec in ['split', 'crossfit']:
        d = Rg[rec]
        for interval in ['constant', 'normalized']:
            scale_te = np.ones(len(y)) if interval == 'constant' else d['sig_te']
            scale_cal = np.ones(len(d['res_cal'])) if interval == 'constant' else d['sig_cal']
            q = qhat(d['res_cal'] / scale_cal); cov = np.abs(y - d['p']) <= q * scale_te
            rows.append(dict(regime=Rg['name'], recipe=rec, interval=interval, calibration='all calibration contrasts (published)', coverage=cov.mean(),
                             width=float(np.mean(2 * q * scale_te)), n_test=len(y), n_test_trials=len(np.unique(g)), n_cal=len(d['res_cal']),
                             **rs.boot_cov(cov, g), eval_set='all test contrasts'))
            # trial-level variant: one contrast per calibration trial; evaluated on all test contrasts and on one per test trial
            rng = np.random.default_rng(1); covs_all, covs_one, widths, qs = [], [], [], []
            for r in range(R):
                ic = one_per_trial(d['g_cal'], rng); qq = qhat(d['res_cal'][ic] / scale_cal[ic]); qs.append(qq)
                c = np.abs(y - d['p']) <= qq * scale_te; covs_all.append(c.mean()); widths.append(np.mean(2 * qq * scale_te))
                it = one_per_trial(g, rng); covs_one.append(c[it].mean())
            # trial-bootstrap interval for the first draw (a single realised procedure) on all test contrasts
            ic = one_per_trial(d['g_cal'], np.random.default_rng(1)); qq = qhat(d['res_cal'][ic] / scale_cal[ic]); c0 = np.abs(y - d['p']) <= qq * scale_te
            b = rs.boot_cov(c0, g)
            rows.append(dict(regime=Rg['name'], recipe=rec, interval=interval, calibration='one random contrast per calibration trial (trial-level exchangeable)',
                             coverage=float(np.mean(covs_all)), coverage_draw1=float(c0.mean()), coverage_lo=b['coverage_lo'], coverage_hi=b['coverage_hi'],
                             coverage_draws_p2_5=float(np.percentile(covs_all, 2.5)), coverage_draws_p97_5=float(np.percentile(covs_all, 97.5)),
                             width=float(np.mean(widths)), n_test=len(y), n_test_trials=len(np.unique(g)), n_cal=len(ic), eval_set='all test contrasts', n_draws=R))
            rows.append(dict(regime=Rg['name'], recipe=rec, interval=interval, calibration='one random contrast per calibration trial (trial-level exchangeable)',
                             coverage=float(np.mean(covs_one)), coverage_draws_p2_5=float(np.percentile(covs_one, 2.5)), coverage_draws_p97_5=float(np.percentile(covs_one, 97.5)),
                             width=float(np.mean(widths)), n_test=len(np.unique(g)), n_test_trials=len(np.unique(g)), n_cal=len(ic), eval_set='one random contrast per test trial', n_draws=R))
    return rows

def abstention_rows(Rg, targets=(0.10, 0.08), fracs=(0.10, 0.25, 0.50)):
    """Thresholds on sigma chosen on calibration data: cross-fitted OOF predictions and OOF sigma on the training pool."""
    d = Rg['crossfit']; y, g, p, s = Rg['yte'], Rg['gte'], d['p'], d['sig_te']
    ae_cal = np.abs(d['y_cal'] - d['oof_pred']); sc = d['sig_cal']; o = np.argsort(sc); cum = np.cumsum(ae_cal[o]) / np.arange(1, len(o) + 1)
    rows = []
    def ev(cut, rule, tgt):
        m = s <= cut; return dict(regime=Rg['name'], recipe='crossfit', rule=rule, target=tgt, sigma_cut=float(cut),
                                  answered_frac_test=float(m.mean()), answered_trials_frac_test=len(np.unique(g[m])) / len(np.unique(g)),
                                  mae_answered_test=float(np.mean(np.abs(y - p)[m])) if m.any() else np.nan, mae_all_test=float(np.mean(np.abs(y - p))),
                                  n_test=len(y))
    for t in targets:
        ok = np.where(cum <= t)[0]; cut = np.sort(sc)[ok.max()] if len(ok) else -np.inf
        r = ev(cut, 'largest calibration fraction with OOF MAE <= target', t); r['answered_frac_cal'] = float((ok.max() + 1) / len(sc)) if len(ok) else 0.0; rows.append(r)
    for f in fracs:
        cut = np.quantile(sc, f); r = ev(cut, f'sigma quantile {f:.2f} of calibration OOF sigma', f); r['answered_frac_cal'] = f; rows.append(r)
    return rows

if __name__ == '__main__':
    b = rs.load(); trp = b[b.split_time == 'train'].reset_index(drop=True); tep = b[b.split_time == 'test'].reset_index(drop=True)
    i_pool, i_ho = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(trp, groups=trp.nct_id))
    regs = [('in_distribution', trp.iloc[i_pool].reset_index(drop=True), trp.iloc[i_ho].reset_index(drop=True)), ('temporal_2023plus', trp, tep)]
    regs += [(f'loao_{a}', b[b.area != a].reset_index(drop=True), b[b.area == a].reset_index(drop=True)) for a in sorted(b.area.unique())]
    cov, ab = [], []
    for name, pool, test in regs:
        Rg = regime(name, pool, test); cov += coverage_rows(Rg)
        if name in ('in_distribution', 'temporal_2023plus'):
            ab += abstention_rows(Rg)
            np.savez(f'conformal_preds_{name}.npz', y=Rg['yte'], g=Rg['gte'], p_crossfit=Rg['crossfit']['p'], sig_crossfit=Rg['crossfit']['sig_te'], p_split=Rg['split']['p'])
        print('done', name, flush=True)
    pd.DataFrame(cov).to_csv('conformal_coverage_extra.csv', index=False); pd.DataFrame(ab).to_csv('conformal_abstention_registry.csv', index=False)
