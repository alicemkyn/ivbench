"""Table tab:lmarea (paper Table 15): language-model (Claude Sonnet 5) interventional skill by therapeutic area, v2.1 binary
test set (878 contrasts / 655 trials). Primary rows: protocol-only prompts; secondary rows: posted-record (original) prompts.
v2.1 predictions: 'kept' test rows use the archived v2 prediction by cid, repaired / added rows the v2.1 new-call prediction
(results/llm/cr/v21new_sonnet_binary_predictions.csv). skill(FULL) and skill(BLIND) are against the v2.1 training mean;
IS = skill(FULL | BLIND); trial bootstrap 2,000 resamples, seed 0. No LM is called.
v2 rows (recomputed; posted-record rows reproduce results/llm/llm_v2_by_area.csv) go to the CSV only.
  python scripts/cr/tables/tab_lmarea.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common_b import ROOT, LLM, load, split, lm_arm_v21, f, ci, n, write, me
import numpy as np, pandas as pd

LABEL = 'lmarea'
AREA_TEX = {'immune_inflam': 'immune/inflam', 'neuro_psych': 'neuro/psych'}
PROMPT = {'protocol': 'protocol-only', 'posted': 'posted-record'}


def v2_arm(arm):
    if arm == 'posted':
        return pd.read_csv(os.path.join(LLM, 'llm_v2_paired.csv'))
    A = pd.read_csv(os.path.join(LLM, 'cr', 'protocol_A_predictions.csv'))
    B = pd.read_csv(os.path.join(LLM, 'cr', 'protocol_B_predictions.csv')).set_index('cid').loc[A.cid]
    return A.assign(point_A=A.point, point_B=B.point.to_numpy())


def by_area(D, trmean, version, arm):
    rows = []
    for area, d in D.groupby('area'):
        y, g, r = d.truth.to_numpy(), d.nct_id.to_numpy(), np.full(len(d), trmean)
        a, b = d.point_A.to_numpy(), d.point_B.to_numpy()
        lo, hi = me.boot_skill(y, a, b, g, n=2000, seed=0)
        rows.append(dict(version=version, prompt=PROMPT[arm], area=area, n=len(d), n_trials=len(np.unique(g)), A_skill=me.skill(y, a, r),
                         B_skill=me.skill(y, b, r), interventional=me.skill(y, a, b), lo=lo, hi=hi))
    return rows


def main():
    rows = []
    C21 = load('v2.1'); tr, te = split(C21, 'binary')
    for arm in ['protocol', 'posted']:
        L = lm_arm_v21(arm, te)
        D = pd.DataFrame({'cid': te.cid, 'nct_id': te.nct_id, 'area': te.area, 'truth': te.delta, 'point_A': L.point_A, 'point_B': L.point_B})
        rows += by_area(D, tr.delta.mean(), 'v2.1', arm)
    tr2, _ = split(load('v2'), 'binary')
    for arm in ['protocol', 'posted']:
        rows += by_area(v2_arm(arm), tr2.delta.mean(), 'v2', arm)
    O = pd.DataFrame(rows)
    ref = pd.read_csv(os.path.join(LLM, 'llm_v2_by_area.csv'))
    m = (O.version == 'v2') & (O.prompt == 'posted-record')
    M = O[m].merge(ref, on='area', suffixes=('', '_rel'))
    O.loc[m, 'v2_reproduces_released'] = np.isclose(M[['A_skill', 'B_skill', 'interventional', 'lo', 'hi']].to_numpy(),
                                                    M[['A_skill_rel', 'B_skill_rel', 'interventional_rel', 'lo_rel', 'hi_rel']].to_numpy(), rtol=0, atol=1e-12).all(1)
    T = O[O.version == 'v2.1']
    body = []
    for p in ['protocol-only', 'posted-record']:
        body.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{p} prompts}}}} \\\\")
        body += [f"{AREA_TEX.get(r.area, r.area)} & {r.n} & {r.n_trials} & {f(r.A_skill)} & {f(r.B_skill)} & {f(r.interventional)} {ci(r.lo, r.hi)} \\\\"
                 for _, r in T[T.prompt == p].iterrows()]
        body.append('\\midrule')
    body = body[:-1]
    tex = ("\\begin{table}[h]\\centering\\small\n"
           f"\\caption{{Language model (Claude Sonnet~5) by therapeutic area, v2.1 binary test split ({n(len(te))} contrasts / "
           f"{n(te.nct_id.nunique())} trials). Protocol-only prompts (headline condition) and posted-record prompts. skill(\\full) and "
           "skill(\\blind) against the training mean; $\\IS$ with 95\\% trial-level bootstrap intervals (2{,}000 resamples). "
           "v2 values in the released CSV.}\n\\label{tab:lmarea}\n"
           "\\begin{tabular}{lrrccc}\\toprule area & contrasts & trials & skill(\\full) & skill(\\blind) & $\\IS$ \\\\ \\midrule\n"
           + '\n'.join(body) + "\n\\bottomrule\\end{tabular}\\end{table}")
    write(LABEL, O, tex)
    print('v2 posted-record rows reproduce llm_v2_by_area.csv:', bool(O.loc[m, 'v2_reproduces_released'].astype(bool).all()))


if __name__ == '__main__':
    main()
