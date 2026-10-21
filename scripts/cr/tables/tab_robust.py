"""Table tab:robust (paper Table 7): robustness of the registry-model null on benchmark v2.1.
Alternative temporal cuts (retrained), comparator / phase / rich-intervention-text strata (headline seed-0 models evaluated on
the stratum, and retrained within it). Reuses scripts/run_positive_control.py robustness() unchanged; trial-level bootstrap,
1,000 resamples, seed 0. v2 rows (recomputed, and checked against results/positive_control/v2_robustness.csv) go to the CSV only.
  python scripts/cr/tables/tab_robust.py"""
import os, sys, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common_b import ROOT, load, f, ci, n, write
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location('rpc', os.path.join(ROOT, 'scripts', 'run_positive_control.py'))
rpc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rpc)

LABEL = 'robust'
MODE = {'headline': 'headline', 'retrained': 'retrained', 'headline models evaluated on stratum': 'evaluated',
        'retrained within stratum': 'retrained'}


def setting_tex(s):
    if s.startswith('headline'):
        return 'headline (seed 0)'
    return s.replace('(arm matched & desc non-empty)', '(arm matched, desc non-empty)').replace('_', '\\_')


def main():
    out = []
    for version in ['v2.1', 'v2']:
        R = rpc.robustness(load(version), 1000, 0); R.insert(0, 'version', version); out.append(R)
    D = pd.concat(out, ignore_index=True)
    ref = pd.read_csv(os.path.join(ROOT, 'results', 'positive_control', 'v2_robustness.csv'))
    v2 = D[D.version == 'v2'].drop(columns='version').reset_index(drop=True)
    num = ref.select_dtypes('number').columns
    D['v2_reproduces_released'] = None
    D.loc[D.version == 'v2', 'v2_reproduces_released'] = np.isclose(v2[num].to_numpy(), ref[num].to_numpy(), rtol=0, atol=1e-12).all(1)
    T = D[D.version == 'v2.1']
    h = T.iloc[0]
    rows = '\n'.join(f"{setting_tex(r.setting)} & {MODE[r['mode']]} & {r.n_test} & {r.n_trials} & {f(r.blind_ctrl_skill)} & "
                     f"{f(r.full_skill)} & {f(r.interventional_skill)} {ci(r.ci_low, r.ci_high)} \\\\" for _, r in T.iterrows())
    tex = ("\\begin{table}[h]\\centering\\footnotesize\n"
           "\\caption{Robustness of the registry-model null on v2.1 (headline split: " f"{n(h.n_test)} contrasts / {n(h.n_trials)} trials): alternative temporal cuts (retrained), comparator/phase strata and the "
           "rich-intervention-text subset (headline models evaluated on the stratum, and retrained within it). Skill against the "
           "training mean; $\\IS$ = skill(\\full\\,|\\,\\blind) with 95\\% trial-level bootstrap intervals (1{,}000 resamples). "
           "v2 values in the released CSV.}\n\\label{tab:robust}\n"
           "\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrccc}\\toprule setting & mode & contrasts & trials & skill(\\blind) & "
           "skill(\\full) & $\\IS$ \\\\ \\midrule\n" + rows + "\n\\bottomrule\\end{tabular}}\\end{table}")
    write(LABEL, D, tex)
    print('v2 reproduces released v2_robustness.csv:', bool(D.v2_reproduces_released.dropna().astype(bool).all()))


if __name__ == '__main__':
    main()
