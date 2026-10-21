"""Table 13 (tab:coverage): conformal coverage and width (nominal 90%) for the registry model under shift, v2.1 binary.

  python scripts/cr/tables/tab_coverage.py [--version v2.1|v2]

Port of scripts/run_shift.py + scripts/cr/stats/conformal_extra.py to v2.1 (see _conformal.py). For each regime, recipe
(split / crossfit) and interval (constant / normalized):
  * published calibration (all calibration contrasts): coverage on all test contrasts, 95% TRIAL-LEVEL BOOTSTRAP interval
    (1,000 resamples over nct_id, seed 0; run_shift.boot_cov; Wilson intervals are not used), mean width, MAE;
  * trial-level exchangeable calibration: one random contrast per calibration trial (R = 200 draws, rng seed 1), coverage on
    all test contrasts (mean over draws, 2.5/97.5 percentiles over draws) and on one random contrast per test trial.
Writes results/cr/coverage.csv (v2.1) or results/cr/coverage_v2.csv (v2 check against the published Table 13).
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *  # noqa: E402,F403
import _conformal as cf  # noqa: E402

R_DRAWS = 200


def coverage_rows(Rg):
    rows = []; y, g = Rg['yte'], Rg['gte']
    for rec in ['split', 'crossfit']:
        d = Rg[rec]
        for interval in ['constant', 'normalized']:
            st = np.ones(len(y)) if interval == 'constant' else d['sig_te']
            sc = np.ones(len(d['res_cal'])) if interval == 'constant' else d['sig_cal']
            q = cf.qhat(d['res_cal'] / sc); cov = np.abs(y - d['p']) <= q * st
            r = dict(regime=Rg['name'], recipe=rec, interval=interval, n_test=len(y), n_test_trials=len(np.unique(g)),
                     n_cal=len(d['res_cal']), q=q, coverage=float(cov.mean()), **cf.rs.boot_cov(cov, g),
                     width=float(np.mean(2 * q * st)), mae=float(np.mean(np.abs(y - d['p']))))
            rng = np.random.default_rng(1); ca, co, wd = [], [], []
            for _ in range(R_DRAWS):
                ic = cf.one_per_trial(d['g_cal'], rng); qq = cf.qhat(d['res_cal'][ic] / sc[ic])
                c = np.abs(y - d['p']) <= qq * st; ca.append(c.mean()); wd.append(np.mean(2 * qq * st))
                co.append(c[cf.one_per_trial(g, rng)].mean())
            r.update(trial_cal_n=len(np.unique(d['g_cal'])), trial_cal_coverage=float(np.mean(ca)),
                     trial_cal_coverage_p2_5=float(np.percentile(ca, 2.5)), trial_cal_coverage_p97_5=float(np.percentile(ca, 97.5)),
                     trial_cal_width=float(np.mean(wd)), trial_cal_coverage_one_per_test_trial=float(np.mean(co)),
                     trial_cal_one_per_test_trial_p2_5=float(np.percentile(co, 2.5)), trial_cal_one_per_test_trial_p97_5=float(np.percentile(co, 97.5)))
            rows.append(r)
    return rows


def tex_of(R, version):
    L = [f"{r.regime.replace('_', chr(92) + '_')} & {r.recipe} & {r.interval} & {r.n_test} & {f(r.coverage)} {ci(r.coverage_lo, r.coverage_hi)} & "
         f"{f(r.width)} & {f(r.mae)} & {f(r.trial_cal_coverage)} \\ci{{{f(r.trial_cal_coverage_p2_5)}}}{{{f(r.trial_cal_coverage_p97_5)}}} \\\\"
         for _, r in R.iterrows()]
    cap = (f"Conformal coverage and mean width (nominal 90\\%) for the registry model, {version} binary contrasts, both calibration "
           "recipes, all regimes. Coverage intervals are 95\\% trial-level bootstrap intervals (1{,}000 resamples over trials), not "
           "Wilson intervals. The last column recalibrates on one randomly chosen contrast per calibration trial, which restores "
           "trial-level exchangeability (mean over 200 draws, with the 2.5th--97.5th percentiles across draws); coverage of one random "
           "contrast per test trial is in \\texttt{results/cr/coverage.csv}.")
    return ("\\begin{table}[h]\\centering\\footnotesize\n\\caption{" + cap + "}\n\\label{tab:coverage}\n"
            "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lllrcccc}\\toprule regime & recipe & interval & $n$ & coverage & width & MAE & "
            "coverage, one contrast per calibration trial \\\\ \\midrule\n" + '\n'.join(L) + "\n\\bottomrule\\end{tabular}}\\end{table}\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--version', default='v2.1'); a = ap.parse_args()
    with Timer():
        rows = []
        for name, mk in cf.regimes(a.version):
            pool, test_ = mk(); rows += coverage_rows(cf.fit_regime(name, pool, test_)); print('done', name, flush=True)
        R = pd.DataFrame(rows); R['alpha'] = cf.ALPHA; R['version'] = a.version
        if a.version == 'v2.1':
            write('coverage', R, tex_of(R, a.version))
        else:
            os.makedirs(OUT_CSV, exist_ok=True); R.to_csv(os.path.join(OUT_CSV, 'coverage_v2.csv'), index=False)
        print(R[['regime', 'recipe', 'interval', 'n_test', 'coverage', 'coverage_lo', 'coverage_hi', 'width', 'mae', 'trial_cal_coverage']].round(3).to_string())
