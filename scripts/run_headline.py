"""Reproduce the headline numbers: level vs contrast skill, intervention-blind ablation, conformal, selective.

Usage:  python scripts/run_headline.py --contrasts data/benchmark_contrasts.csv.gz \
            --arms data/benchmark_arms.csv.gz --out results/ [--seed 0]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ivbench import models as me
from ivbench.specs import SPECS, ENDPOINT_TXT, INTERV_TXT, POP_CAT, POP_NUM
from sklearn.model_selection import GroupShuffleSplit

FEATURE_SETS = {'blind': SPECS['blind'], 'blind+ctrl': SPECS['blind_ctrl'], 'full': SPECS['full'],
                'intervention_only': SPECS['interv_only']}


def fit_set(tr, te, target, cfg, seed):
    Xtr, Xte, _, _ = me.featurize(tr, te, cfg['text'], cat_cols=cfg['cat'], num_cols=cfg['num'])
    _, p = me.fit_gbt(Xtr, tr[target].to_numpy(float), Xte, seed=seed)
    return p


def ablation(tr, te, target, seed, out_prefix):
    y = te[target].to_numpy(float)
    g = te['nct_id'].to_numpy()
    ref = np.full(len(y), tr[target].mean())
    P = {k: fit_set(tr, te, target, cfg, seed) for k, cfg in FEATURE_SETS.items()}
    rows = []
    for k, p in P.items():
        s = me.skill(y, p, ref); lo, hi = me.boot_skill(y, p, ref, g)
        rows.append(dict(feature_set=k, reference='global_mean', skill=s, ci_low=lo, ci_high=hi,
                         mae=float(np.mean(np.abs(y - p)))))
    for k in ('full', 'intervention_only'):
        s = me.skill(y, P[k], P['blind+ctrl']); lo, hi = me.boot_skill(y, P[k], P['blind+ctrl'], g)
        rows.append(dict(feature_set=k, reference='blind+ctrl', skill=s, ci_low=lo, ci_high=hi,
                         mae=float(np.mean(np.abs(y - P[k])))))
    df = pd.DataFrame(rows)
    df.to_csv(out_prefix + '_ablation.csv', index=False)
    return df, P, y, g


def conformal(tr, te, target, seed, out_prefix, alpha=0.1):
    cfg = FEATURE_SETS['full']
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    fi, ci = next(gss.split(tr, groups=tr['nct_id']))
    fit_df, cal_df = tr.iloc[fi], tr.iloc[ci]
    Xf, Xc, _, _ = me.featurize(fit_df, cal_df, cfg['text'], cat_cols=cfg['cat'], num_cols=cfg['num'])
    _, Xt, _, _ = me.featurize(fit_df, te, cfg['text'], cat_cols=cfg['cat'], num_cols=cfg['num'])
    yf, yc, yt = (d[target].to_numpy(float) for d in (fit_df, cal_df, te))
    m, pc = me.fit_gbt(Xf, yf, Xc, seed=seed); pt = m.predict(Xt)
    q = me.split_conformal(yc - pc, alpha)
    sig_m, _ = me.fit_gbt(Xc, np.abs(yc - pc), Xc, seed=seed + 1)
    sig_t = np.clip(sig_m.predict(Xt), 1e-3, None)
    curve = me.selective_curve(yt, pt, sig_t)
    curve.to_csv(out_prefix + '_selective.csv', index=False)
    return dict(alpha=alpha, coverage_constant=float(np.mean(np.abs(yt - pt) <= q)), width_constant=2 * q,
                mae=float(np.mean(np.abs(yt - pt))), n_test=int(len(yt)), n_cal=int(len(yc)), n_fit=int(len(yf)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contrasts', required=True); ap.add_argument('--arms', required=True)
    ap.add_argument('--out', default='results'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    C = pd.read_csv(a.contrasts); C = C[C['family'] == 'binary']
    tr, te = C[C['split_time'] == 'train'], C[C['split_time'] == 'test']
    abl, P, y, g = ablation(tr, te, 'delta', a.seed, os.path.join(a.out, 'contrast_binary'))
    cf = conformal(tr, te, 'delta', a.seed, os.path.join(a.out, 'contrast_binary'))
    head = dict(seed=a.seed, n_train=int(len(tr)), n_test=int(len(te)), n_test_trials=int(te['nct_id'].nunique()),
                interventional_skill=float(abl.query("feature_set=='full' and reference=='blind+ctrl'")['skill'].iloc[0]),
                interventional_ci=[float(abl.query("feature_set=='full' and reference=='blind+ctrl'")['ci_low'].iloc[0]),
                                   float(abl.query("feature_set=='full' and reference=='blind+ctrl'")['ci_high'].iloc[0])],
                conformal=cf)
    json.dump(head, open(os.path.join(a.out, 'headline.json'), 'w'), indent=1)
    print(json.dumps(head, indent=1))


if __name__ == '__main__':
    main()
