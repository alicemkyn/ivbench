"""Table tab:lmsel (paper Table 16): language-model (Claude Sonnet 5) selective prediction on the v2.1 binary test set.
Contrasts are ranked by the model's own stated interval width (high - low) for the FULL prompt (condition A); MAE of the
FULL point prediction at answered fractions 0.10 / 0.25 / 0.50 / 0.75 / 1.00 (ivbench.models.selective_curve). Primary row:
protocol-only prompts; secondary row: posted-record (original) prompts. v2.1 predictions as in tab_lmarea.py. No LM is called.
v2 rows (recomputed; posted-record row reproduces results/llm/llm_v2_selective.csv) go to the CSV only.
  python scripts/cr/tables/tab_lmsel.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common_b import ROOT, LLM, load, split, lm_arm_v21, f, n, write, me
from tab_lmarea import v2_arm, PROMPT
import numpy as np, pandas as pd

LABEL = 'lmsel'
GRID = np.array([0.10, 0.25, 0.50, 0.75, 1.0])


def curve(y, point, low, high, version, arm):
    sc = me.selective_curve(y, point, high - low, grid=GRID)
    sc.insert(0, 'prompt', PROMPT[arm]); sc.insert(0, 'version', version)
    return sc


def main():
    out = []
    _, te = split(load('v2.1'), 'binary')
    for arm in ['protocol', 'posted']:
        L = lm_arm_v21(arm, te)
        out.append(curve(te.delta.to_numpy(), L.point_A.to_numpy(), L.low_A.to_numpy(), L.high_A.to_numpy(), 'v2.1', arm))
    for arm in ['protocol', 'posted']:
        D = v2_arm(arm)
        if arm == 'protocol':
            D = D.assign(low_A=D.low, high_A=D.high)
        out.append(curve(D.truth.to_numpy(), D.point_A.to_numpy(), D.low_A.to_numpy(), D.high_A.to_numpy(), 'v2', arm))
    O = pd.concat(out, ignore_index=True)
    ref = pd.read_csv(os.path.join(LLM, 'llm_v2_selective.csv'))
    m = (O.version == 'v2') & (O.prompt == 'posted-record')
    O.loc[m, 'v2_reproduces_released'] = np.isclose(O.loc[m, 'mae'].to_numpy(), ref.mae.to_numpy(), rtol=0, atol=1e-12) & (O.loc[m, 'n'].to_numpy() == ref.n.to_numpy())
    T = O[O.version == 'v2.1']
    rows = '\n'.join(f"{p} & " + ' & '.join(f(v) for v in T[T.prompt == p].mae) + ' \\\\' for p in ['protocol-only', 'posted-record'])
    tex = ("\\begin{table}[h]\\centering\\small\n"
           f"\\caption{{Selective prediction, language model (Claude Sonnet~5) condition A on the v2.1 binary test split "
           f"({n(len(te))} contrasts), ranked by the model's stated interval width: MAE at answered fractions 0.10 / 0.25 / 0.50 / "
           "0.75 / 1.00, protocol-only (headline) and posted-record prompts. v2 values in the released CSV.}\n\\label{tab:lmsel}\n"
           "\\begin{tabular}{lccccc}\\toprule prompt & 0.10 & 0.25 & 0.50 & 0.75 & 1.00 \\\\ \\midrule\n" + rows +
           "\n\\bottomrule\\end{tabular}\\end{table}")
    write(LABEL, O, tex)
    print('v2 posted-record row reproduces llm_v2_selective.csv:', bool(O.loc[m, 'v2_reproduces_released'].astype(bool).all()))


if __name__ == '__main__':
    main()
