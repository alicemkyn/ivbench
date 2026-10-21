"""Split conformal under distribution shift: binary contrasts, FULL v2 feature set.
Recipe (i)  'split'  : 50/50 trial-grouped split of the training pool; point model on proper-train;
                       constant width from (1-alpha) quantile of |cal residual|; normalized variant uses a
                       sigma model fit on proper-train IN-SAMPLE |residuals| (valid but scale-compressed).
Recipe (ii) 'crossfit': 5-fold GroupKFold over the training pool; out-of-fold residuals for every training
                       row; sigma model cross-fitted (sigma for fold k trained on OOF |res| of other folds);
                       q from all n_train OOF (normalized) residuals; final point + sigma models refit on the
                       whole pool.  n_cal = whole training pool.
"""
import numpy as np, pandas as pd, importlib.util, json
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

HELP = "/Users/alicemkoyun/.claude-science/orgs/c41cd831-fc31-43d7-9abd-94c51ae7e22f/artifacts/proj_788ff34a2c46/3e2c1452-f3ad-4f63-b109-88c291f66172/vf90e14b8_models_eval.py"
DATA = "/Users/alicemkoyun/.claude-science/orgs/c41cd831-fc31-43d7-9abd-94c51ae7e22f/artifacts/proj_788ff34a2c46/3304d446-e25d-4962-a645-b3e968b135b0/v8bfc76b1_benchmark_contrasts.csv.gz"
spec = importlib.util.spec_from_file_location("me", HELP); me = importlib.util.module_from_spec(spec); spec.loader.exec_module(me)

ALPHA = 0.10
SEED = 0
ENDPOINT_TXT = ['outcome_title', 'outcome_desc', 'outcome_timeframe', 'conditions']
INTERV_TXT = ['treat_intervention', 'treat_intervention_desc', 'treat_arm_desc', 'treat_group', 'treat_group_desc',
              'ctrl_intervention', 'ctrl_arm_desc', 'ctrl_group', 'ctrl_group_desc']
POP_CAT = ['phase', 'sponsor_class', 'allocation', 'masking', 'purpose', 'area', 'sex']
POP_NUM = ['n_treat', 'n_ctrl', 'age_mean', 'frac_female', 'year_posted']
FULL_TXT = ENDPOINT_TXT + INTERV_TXT
FULL_CAT = POP_CAT + ['ctrl_kind', 'treat_intervention_types']
SIG_FLOOR = 0.01
GRID = np.round(np.linspace(0.05, 1.0, 20), 3)
REPORT_COV = [0.10, 0.25, 0.50, 0.75, 1.00]


def load():
    df = pd.read_csv(DATA, low_memory=False)
    b = df[df.family == 'binary'].reset_index(drop=True)
    return b


def fit_sigma(X, absres, seed=SEED):
    m, _ = me.fit_gbt(X, absres, X[:1], seed=seed)
    return m


def sig(m, X):
    return np.clip(m.predict(X), SIG_FLOOR, None)


def run_regime(name, pool, test):
    """pool/test: DataFrames. Returns list of coverage rows + per-test-row predictions dict."""
    Xtr, Xte, _, _ = me.featurize(pool, test, FULL_TXT, cat_cols=FULL_CAT, num_cols=POP_NUM)
    ytr, yte = pool.delta.to_numpy(), test.delta.to_numpy()
    gtr, gte = pool.nct_id.to_numpy(), test.nct_id.to_numpy()
    rows, preds = [], {}

    # ---- recipe (i): 50/50 trial-grouped split -----------------------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    i_fit, i_cal = next(gss.split(Xtr, ytr, gtr))
    m1, p_te = me.fit_gbt(Xtr[i_fit], ytr[i_fit], Xte, seed=SEED)
    p_cal = m1.predict(Xtr[i_cal]); res_cal = ytr[i_cal] - p_cal
    q_const = me.split_conformal(res_cal, ALPHA)
    s1 = fit_sigma(Xtr[i_fit], np.abs(ytr[i_fit] - m1.predict(Xtr[i_fit])))
    sig_cal, sig_te = sig(s1, Xtr[i_cal]), sig(s1, Xte)
    q_norm = me.normalized_conformal(res_cal, sig_cal, ALPHA)
    for interval, lo, hi in [('constant', p_te - q_const, p_te + q_const),
                             ('normalized', p_te - q_norm * sig_te, p_te + q_norm * sig_te)]:
        cov = (yte >= lo) & (yte <= hi)
        rows.append(dict(regime=name, recipe='split', interval=interval, coverage=float(cov.mean()),
                         width=float(np.mean(hi - lo)), n_test=int(len(yte)), n_test_trials=int(len(np.unique(gte))),
                         n_cal=int(len(i_cal)), n_fit=int(len(i_fit)), q=float(q_const if interval == 'constant' else q_norm),
                         mae=float(np.mean(np.abs(yte - p_te))),
                         **boot_cov(cov, gte)))
    preds['split'] = dict(y=yte, pred=p_te, sigma=sig_te, groups=gte)

    # ---- recipe (ii): honest 5-fold cross-fit ----------------------------------------
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(Xtr, ytr, gtr))
    oof_pred = np.zeros_like(ytr)
    for k, (tr, ho) in enumerate(folds):
        m, oof_pred[ho] = me.fit_gbt(Xtr[tr], ytr[tr], Xtr[ho], seed=SEED + k)
    oof_res = ytr - oof_pred
    oof_sig = np.zeros_like(ytr)
    for k, (tr, ho) in enumerate(folds):
        s = fit_sigma(Xtr[tr], np.abs(oof_res[tr]), seed=SEED + k)
        oof_sig[ho] = sig(s, Xtr[ho])
    q_const2 = me.split_conformal(oof_res, ALPHA)
    q_norm2 = me.normalized_conformal(oof_res, oof_sig, ALPHA)
    mF, p_te2 = me.fit_gbt(Xtr, ytr, Xte, seed=SEED)
    sF = fit_sigma(Xtr, np.abs(oof_res))
    sig_te2 = sig(sF, Xte)
    for interval, lo, hi in [('constant', p_te2 - q_const2, p_te2 + q_const2),
                             ('normalized', p_te2 - q_norm2 * sig_te2, p_te2 + q_norm2 * sig_te2)]:
        cov = (yte >= lo) & (yte <= hi)
        rows.append(dict(regime=name, recipe='crossfit', interval=interval, coverage=float(cov.mean()),
                         width=float(np.mean(hi - lo)), n_test=int(len(yte)), n_test_trials=int(len(np.unique(gte))),
                         n_cal=int(len(ytr)), n_fit=int(len(ytr)), q=float(q_const2 if interval == 'constant' else q_norm2),
                         mae=float(np.mean(np.abs(yte - p_te2))),
                         **boot_cov(cov, gte)))
    preds['crossfit'] = dict(y=yte, pred=p_te2, sigma=sig_te2, groups=gte,
                             oof_sigma_rank_corr=float(pd.Series(oof_sig).corr(pd.Series(np.abs(oof_res)), method='spearman')))
    return rows, preds


def boot_cov(cov, groups, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups); idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        out.append(cov[idx].mean())
    return dict(coverage_lo=float(np.percentile(out, 2.5)), coverage_hi=float(np.percentile(out, 97.5)))


def selective(P, regime, recipe):
    d = me.selective_curve(P['y'], P['pred'], P['sigma'], grid=GRID)
    # fraction of trials retained at each coverage
    order = np.argsort(P['sigma'])
    d['n_trials'] = [int(len(np.unique(P['groups'][order[:k]]))) for k in d['n']]
    d['frac_trials'] = d['n_trials'] / len(np.unique(P['groups']))
    d['sigma_cut'] = [float(np.sort(P['sigma'])[k - 1]) for k in d['n']]
    d.insert(0, 'recipe', recipe); d.insert(0, 'regime', regime)
    return d


def operating(P, regime, recipe, targets=(0.10, 0.08)):
    order = np.argsort(P['sigma']); y, p, g = P['y'][order], P['pred'][order], P['groups'][order]
    ae = np.abs(y - p); cum_mae = np.cumsum(ae) / np.arange(1, len(ae) + 1)
    rows = []
    for t in targets:
        ok = np.where(cum_mae <= t)[0]
        k = int(ok.max() + 1) if len(ok) else 0
        rows.append(dict(regime=regime, recipe=recipe, target_mae=t, n_answered=k, frac_contrasts=k / len(y),
                         n_trials_answered=int(len(np.unique(g[:k]))) if k else 0,
                         frac_trials=(len(np.unique(g[:k])) / len(np.unique(g))) if k else 0.0,
                         mae_answered=float(cum_mae[k - 1]) if k else float('nan'),
                         sigma_cut=float(np.sort(P['sigma'])[k - 1]) if k else float('nan'),
                         mae_all=float(ae.mean())))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    b = load()
    train_period = b[b.split_time == 'train'].reset_index(drop=True)
    test_period = b[b.split_time == 'test'].reset_index(drop=True)
    cov_rows, sel, ops, extra = [], [], [], {}

    # (a) in-distribution: 20% trial-grouped holdout within training period
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    i_pool, i_ho = next(gss.split(train_period, groups=train_period.nct_id))
    r, P = run_regime('in_distribution', train_period.iloc[i_pool].reset_index(drop=True),
                      train_period.iloc[i_ho].reset_index(drop=True))
    cov_rows += r
    for rec in P:
        sel.append(selective(P[rec], 'in_distribution', rec)); ops.append(operating(P[rec], 'in_distribution', rec))
    extra['in_distribution_oof_sigma_spearman'] = P['crossfit']['oof_sigma_rank_corr']
    extra['in_distribution_test_sigma_spearman'] = float(pd.Series(P['crossfit']['sigma']).corr(
        pd.Series(np.abs(P['crossfit']['y'] - P['crossfit']['pred'])), method='spearman'))
    print('done (a)')

    # (b) temporal
    r, P = run_regime('temporal_2023plus', train_period, test_period)
    cov_rows += r
    for rec in P:
        sel.append(selective(P[rec], 'temporal_2023plus', rec)); ops.append(operating(P[rec], 'temporal_2023plus', rec))
    extra['temporal_oof_sigma_spearman'] = P['crossfit']['oof_sigma_rank_corr']
    extra['temporal_test_sigma_spearman'] = float(pd.Series(P['crossfit']['sigma']).corr(
        pd.Series(np.abs(P['crossfit']['y'] - P['crossfit']['pred'])), method='spearman'))
    pd.DataFrame(dict(nct_id=P['crossfit']['groups'], y=P['crossfit']['y'], pred=P['crossfit']['pred'],
                      sigma=P['crossfit']['sigma'])).to_csv('v2_temporal_test_predictions.csv', index=False)
    print('done (b)')

    # (c) leave-one-area-out: pool = all other areas (both periods), test = held-out area (both periods)
    for area in sorted(b.area.unique()):
        r, P = run_regime(f'loao_{area}', b[b.area != area].reset_index(drop=True), b[b.area == area].reset_index(drop=True))
        cov_rows += r
        print('done (c)', area)

    cov = pd.DataFrame(cov_rows); cov['alpha'] = ALPHA; cov['nominal'] = 1 - ALPHA
    cov.to_csv('v2_coverage_under_shift.csv', index=False)
    sel = pd.concat(sel, ignore_index=True); sel.to_csv('v2_selective_curves.csv', index=False)
    ops = pd.concat(ops, ignore_index=True); ops.to_csv('v2_operating_points.csv', index=False)
    json.dump(extra, open('handoff_extra.json', 'w'), indent=1)
    print(cov[['regime', 'recipe', 'interval', 'coverage', 'coverage_lo', 'coverage_hi', 'width', 'n_test', 'n_cal', 'mae']].round(3).to_string())
    print(sel[sel.coverage.isin(REPORT_COV)][['regime', 'recipe', 'coverage', 'mae', 'n', 'frac_trials']].round(3).to_string())
    print(ops.round(3).to_string())
    print(extra)
