"""Conformal-under-shift helpers shared by tab_coverage.py and tab_op.py.

Port of scripts/run_shift.py and scripts/cr/stats/conformal_extra.py (identical model code path, seed 0) to any benchmark
version. Binary family, FULL feature set (endpoint + intervention text, POP_CAT + ctrl_kind + treat_intervention_types).
Regimes: in_distribution (20% trial-grouped holdout of the training period), temporal_2023plus (train -> test split), and
leave-one-area-out (pool = all other areas, both periods). Recipes: 'split' (50/50 trial-grouped split of the pool) and
'crossfit' (5-fold GroupKFold out-of-fold residuals over the whole pool; final models refit on the pool).
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from _common import ROOT, me, load

sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import run_shift as rs  # noqa: E402  (fit_sigma, sig, boot_cov, operating, feature lists, ALPHA, SEED)

ALPHA, SEED = rs.ALPHA, rs.SEED


def qhat(res, alpha=ALPHA):
    n = len(res); return float(np.quantile(np.abs(res), min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)))


def one_per_trial(groups, rng):
    """Index of one uniformly chosen row per trial (trials in sorted order; one vectorised draw per trial)."""
    u, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind='stable'); sizes = np.bincount(inv); starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return order[starts + rng.integers(0, sizes)]


def binary(version):
    C = load(version); return C[C.family == 'binary'].reset_index(drop=True)


def regimes(version, which=None):
    b = binary(version)
    trp = b[b.split_time == 'train'].reset_index(drop=True); tep = b[b.split_time == 'test'].reset_index(drop=True)
    i_pool, i_ho = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(trp, groups=trp.nct_id))
    R = [('in_distribution', lambda: (trp.iloc[i_pool].reset_index(drop=True), trp.iloc[i_ho].reset_index(drop=True))),
         ('temporal_2023plus', lambda: (trp, tep))]
    R += [(f'loao_{a}', (lambda a=a: (b[b.area != a].reset_index(drop=True), b[b.area == a].reset_index(drop=True))))
          for a in sorted(b.area.unique())]
    return [(n, f) for n, f in R if which is None or n in which]


def fit_regime(name, pool, test):
    Xtr, Xte, _, _ = me.featurize(pool, test, rs.FULL_TXT, cat_cols=rs.FULL_CAT, num_cols=rs.POP_NUM)
    ytr, yte = pool.delta.to_numpy(), test.delta.to_numpy(); gtr, gte = pool.nct_id.to_numpy(), test.nct_id.to_numpy()
    i_fit, i_cal = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED).split(Xtr, ytr, gtr))
    m1, p_te = me.fit_gbt(Xtr[i_fit], ytr[i_fit], Xte, seed=SEED); p_cal = m1.predict(Xtr[i_cal])
    s1 = rs.fit_sigma(Xtr[i_fit], np.abs(ytr[i_fit] - m1.predict(Xtr[i_fit])))
    folds = list(GroupKFold(n_splits=5).split(Xtr, ytr, gtr)); oof = np.zeros_like(ytr)
    for k, (a, c) in enumerate(folds):
        _, oof[c] = me.fit_gbt(Xtr[a], ytr[a], Xtr[c], seed=SEED + k)
    oof_res = ytr - oof; oof_sig = np.zeros_like(ytr)
    for k, (a, c) in enumerate(folds):
        oof_sig[c] = rs.sig(rs.fit_sigma(Xtr[a], np.abs(oof_res[a]), seed=SEED + k), Xtr[c])
    _, p_te2 = me.fit_gbt(Xtr, ytr, Xte, seed=SEED); sig_te2 = rs.sig(rs.fit_sigma(Xtr, np.abs(oof_res)), Xte)
    return dict(name=name, yte=yte, gte=gte, n_pool=len(ytr),
                split=dict(p=p_te, res_cal=ytr[i_cal] - p_cal, sig_cal=rs.sig(s1, Xtr[i_cal]), sig_te=rs.sig(s1, Xte),
                           g_cal=gtr[i_cal], pred_cal=p_cal, y_cal=ytr[i_cal]),
                crossfit=dict(p=p_te2, res_cal=oof_res, sig_cal=oof_sig, sig_te=sig_te2, g_cal=gtr, pred_cal=oof, y_cal=ytr))
