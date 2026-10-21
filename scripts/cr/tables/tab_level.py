"""Table tab:level (paper Table 11): arm-level task on benchmark v2.1 arms (data/v2.1/benchmark_arms_v2.1.csv.gz).
Each binary test arm's own event proportion is predicted (train: year_posted < 2023). v2.1 port: the arm role is the repaired
role_v21 and arms flagged as pooled / total groups (pooled_reason non-empty) are excluded. Models and bootstrap as in
scripts/run_positive_control.py level_task() (gradient boosting seed 0; trial-level bootstrap, 1,000 resamples, seed 0).
v2 rows (recomputed on data/benchmark_arms.csv.gz, checked against results/positive_control/v2_level_task.csv) go to the CSV only.
  python scripts/cr/tables/tab_level.py"""
import os, sys, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common_b import ROOT, f, ci, n, write
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location('rpc', os.path.join(ROOT, 'scripts', 'run_positive_control.py'))
rpc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rpc)
LABEL = 'level'


def main():
    A21 = pd.read_csv(os.path.join(ROOT, 'data', 'v2.1', 'benchmark_arms_v2.1.csv.gz'), low_memory=False)
    A21 = A21[A21.pooled_reason.isna()].copy(); A21['role'] = A21['role_v21']
    A2 = pd.read_csv(os.path.join(ROOT, 'data', 'benchmark_arms.csv.gz'), low_memory=False)
    out = []
    for version, A in [('v2.1', A21), ('v2', A2)]:
        R = rpc.level_task(A, 1000, 0); R.insert(0, 'version', version); out.append(R)
    D = pd.concat(out, ignore_index=True)
    ref = pd.read_csv(os.path.join(ROOT, 'results', 'positive_control', 'v2_level_task.csv'))
    v2 = D[D.version == 'v2'].reset_index(drop=True); num = ref.select_dtypes('number').columns
    D['v2_reproduces_released'] = None
    D.loc[D.version == 'v2', 'v2_reproduces_released'] = np.isclose(v2[num].to_numpy(), ref[num].to_numpy(), rtol=0, atol=1e-12).all(1)
    T = D[D.version == 'v2.1']
    rows = '\n'.join(f"{r.model.replace('_', chr(92) + '_')} & {r.reference.replace('_', chr(92) + '_')} & {f(r.skill)} {ci(r.ci_low, r.ci_high)} \\\\"
                     for _, r in T.iterrows())
    tex = ("\\begin{table}[h]\\centering\\small\n"
           f"\\caption{{Arm-level task on v2.1 (predicting each test arm's event proportion; {n(T.n_test.iat[0])} arms / {n(T.n_trials.iat[0])} trials; pooled or "
           "total groups excluded, repaired arm roles). ``blind\\_norole'' additionally omits the arm's role. 95\\% trial-level "
           "bootstrap intervals (1{,}000 resamples). v2 values in the released CSV.}\n" +
           "\\label{tab:level}\n\\begin{tabular}{llc}\\toprule model & reference & skill \\\\ \\midrule\n" + rows +
           "\n\\bottomrule\\end{tabular}\\end{table}")
    write(LABEL, D, tex)
    print('v2 reproduces released v2_level_task.csv:', bool(D.v2_reproduces_released.dropna().astype(bool).all()))


if __name__ == '__main__':
    main()
