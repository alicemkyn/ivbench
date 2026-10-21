"""Table 14 (tab:op): answer-or-abstain operating points for the registry model, v2.1 binary.

  python scripts/cr/tables/tab_op.py [--version v2.1|v2]

Operating-point fix: the abstention threshold on the predicted error scale sigma is chosen on CALIBRATION data (split recipe:
the calibration half of the pool with the proper-train sigma model; crossfit recipe: out-of-fold predictions and out-of-fold
sigma over the whole pool) as the largest answered fraction whose calibration MAE does not exceed the target. It is then
applied unchanged to the evaluated (shifted) set, and the realised answered fraction and realised MAE there are reported.
For comparison, the published rule (threshold chosen on the evaluated set itself, run_shift.operating) is reported as
'evaluated set (oracle)'. Fixed-quantile rules (answer the 10/25/50% lowest-sigma calibration contrasts) are in the CSV.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *  # noqa: E402,F403
import _conformal as cf  # noqa: E402

TARGETS = (0.10, 0.08)


def op_rows(Rg):
    rows = []; y, g = Rg['yte'], Rg['gte']
    for rec in ['split', 'crossfit']:
        d = Rg[rec]; p, s = d['p'], d['sig_te']; ae = np.abs(y - p)
        sc = d['sig_cal']; o = np.argsort(sc); ae_cal_u = np.abs(d['y_cal'] - d['pred_cal']); ae_cal = ae_cal_u[o]; cum = np.cumsum(ae_cal) / np.arange(1, len(o) + 1)
        def ev(cut, rule, chosen_on, tgt, frac_cal):
            msk = s <= cut
            return dict(regime=Rg['name'], recipe=rec, rule=rule, threshold_chosen_on=chosen_on, target_mae=tgt, sigma_cut=float(cut),
                        answered_frac_cal=float(np.mean(sc <= cut)), nominal_frac_cal=frac_cal, n_cal=len(sc),
                        mae_cal_answered=float(ae_cal_u[sc <= cut].mean()) if (sc <= cut).any() else np.nan,
                        n_test=len(y), answered_frac_test=float(msk.mean()), answered_trials_frac_test=float(len(np.unique(g[msk])) / len(np.unique(g))),
                        mae_answered_test=float(ae[msk].mean()) if msk.any() else np.nan, mae_all_test=float(ae.mean()),
                        meets_target_on_test=bool(msk.any() and ae[msk].mean() <= tgt) if tgt < 1 else np.nan)
        # tie-aware: a threshold answers every contrast with sigma <= cut, so candidate cuts are the distinct calibration sigma
        # values and the calibration MAE is taken over all calibration contrasts at or below each (sigma has ties at SIG_FLOOR).
        ss = np.sort(sc); last = np.r_[np.flatnonzero(np.diff(ss) > 0), len(ss) - 1]
        for t in TARGETS:
            ok = last[cum[last] <= t]
            if len(ok):
                cut, fc = ss[ok.max()], (ok.max() + 1) / len(sc)
            else:
                cut, fc = -np.inf, 0.0
            r = ev(cut, 'largest calibration fraction with calibration MAE <= target (tie-aware)', 'calibration', t, fc)
            r.update(frac_sigma_at_floor_cal=float(np.mean(sc <= cf.rs.SIG_FLOOR)), frac_sigma_at_floor_test=float(np.mean(s <= cf.rs.SIG_FLOOR)))
            rows.append(r)
        for q in (0.10, 0.25, 0.50):
            rows.append(ev(np.quantile(sc, q), f'sigma quantile {q:.2f} of calibration sigma', 'calibration', np.nan, q))
        P = dict(y=y, pred=p, sigma=s, groups=g); orc = cf.rs.operating(P, Rg['name'], rec, targets=TARGETS)
        for _, r in orc.iterrows():
            rows.append(dict(regime=Rg['name'], recipe=rec, rule='largest evaluated fraction with evaluated MAE <= target',
                             threshold_chosen_on='evaluated set (oracle)', target_mae=r.target_mae, sigma_cut=r.sigma_cut, n_test=len(y),
                             answered_frac_test=r.frac_contrasts, answered_trials_frac_test=r.frac_trials, mae_answered_test=r.mae_answered,
                             mae_all_test=r.mae_all, meets_target_on_test=True if r.n_answered else np.nan))
    return rows


def _rows(T):
    return [f"{r.regime.replace('_', chr(92) + '_')} & {f(r.target_mae, 2)} & {'calibration' if r.k == 0 else 'evaluated (oracle)'} & "
            f"{f(r.answered_frac_cal) if r.k == 0 else '--'} & {f(r.answered_frac_test)} & {f(r.answered_trials_frac_test)} & "
            f"{f(r.mae_answered_test)} & {f(r.mae_all_test)} \\\\" for _, r in T.iterrows()]


HEAD = ("\\resizebox{\\linewidth}{!}{\\begin{tabular}{llllcccc}\\toprule regime & target MAE & threshold chosen on & "
        "answered (calibration) & answered (contrasts) & answered (trials) & realised MAE, answered & MAE if all answered \\\\ \\midrule\n")


def tex_of(R, version):
    T = R[R.target_mae.isin(TARGETS)].copy()
    T['k'] = T.threshold_chosen_on.map({'calibration': 0, 'evaluated set (oracle)': 1})
    T = T.sort_values(['regime', 'target_mae', 'k'], ascending=[True, False, True], kind='stable')
    T = T.sort_values('regime', key=lambda s: s.map({'in_distribution': 0, 'temporal_2023plus': 1}), kind='stable')
    C, S = T[T.recipe == 'crossfit'], T[T.recipe == 'split']
    cap = (f"Answer-or-abstain operating points for the registry model, cross-fitted recipe, {version} binary contrasts: answer the "
           "contrasts whose predicted error scale is below a threshold. \\emph{Calibration} rows choose the threshold on out-of-fold "
           "calibration residuals (largest answered fraction whose calibration MAE does not exceed the target) and report the answered "
           "fractions and the realised MAE when that threshold is applied unchanged to the evaluated set. \\emph{Evaluated (oracle)} rows "
           "choose the threshold on the evaluated set itself, as in the submitted version, and meet the target by construction. The "
           "split recipe is in Table~\\ref{tab:opsplit}.")
    main = ("\\begin{table}[h]\\centering\\small\n\\caption{" + cap + "}\n\\label{tab:op}\n" + HEAD + '\n'.join(_rows(C))
            + "\n\\bottomrule\\end{tabular}}\\end{table}\n")
    fl = R[(R.recipe == 'split')].frac_sigma_at_floor_test.dropna()
    cap2 = (f"Answer-or-abstain operating points, split recipe, {version}. The split recipe fits its error-scale model to in-sample "
            "residuals of the proper training half, which are compressed towards zero; the fitted scale therefore sits at its floor value "
            f"for most contrasts ({f(fl.min(), 2)} to {f(fl.max(), 2)} of the evaluated set across regimes), ties cannot be broken, and where "
            "the tied floor group alone exceeds the target no calibration threshold answers any contrast (answered fraction 0). The "
            "recipe is kept for comparability with the submitted version; the cross-fitted recipe (Table~\\ref{tab:op}) is the one the "
            "main text uses.")
    appx = ("\\begin{table}[h]\\centering\\small\n\\caption{" + cap2 + "}\n\\label{tab:opsplit}\n" + HEAD + '\n'.join(_rows(S))
            + "\n\\bottomrule\\end{tabular}}\\end{table}\n")
    with open(os.path.join(OUT_TEX, 'opsplit.tex'), 'w') as fh:
        fh.write(appx)
    return main


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--version', default='v2.1'); a = ap.parse_args()
    with Timer():
        rows = []
        for name, mk in cf.regimes(a.version, which=('in_distribution', 'temporal_2023plus')):
            pool, test_ = mk(); rows += op_rows(cf.fit_regime(name, pool, test_)); print('done', name, flush=True)
        R = pd.DataFrame(rows); R['version'] = a.version
        if a.version == 'v2.1':
            write('op', R, tex_of(R, a.version))
        else:
            os.makedirs(OUT_CSV, exist_ok=True); R.to_csv(os.path.join(OUT_CSV, 'op_v2.csv'), index=False)
        print(R[['regime', 'recipe', 'threshold_chosen_on', 'target_mae', 'answered_frac_cal', 'answered_frac_test', 'answered_trials_frac_test',
                 'mae_answered_test', 'mae_all_test']].round(3).to_string())
