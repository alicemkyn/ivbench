"""Table tab:fg (paper Table 9): feature-group gains over the design-only registry model, benchmark v2.1 binary temporal split.
Gradient boosting (ivbench.specs.fit_spec, seed 0). design-only = phase, sponsor class, allocation, masking, purpose, area,
sex. Each group is added on top of design-only; skill is reported against design-only and against the training mean, with
trial-level bootstrap intervals (1,000 resamples, seed 0). Groups: endpoint text = outcome title / description / time frame;
condition text = conditions; population covariates = mean age, fraction female, year posted; arm sizes = n_treat, n_ctrl;
control type = ctrl_kind; intervention text = the nine intervention text fields of ivbench.specs.INTERV_TXT. Confound probe =
sponsor class + phase + year posted.
The original v2 probe script is not part of the repository. Under these definitions only 2 of the 14 released v2 rows reproduce
exactly: 'design + population covariates' vs train_mean and the confound probe. The design-only model does not (0.058 vs 0.013
released), so every row referenced to design-only also differs (see the v2 rows of the CSV, which carry the released value and a
reproduction flag). The v2.1 table is therefore a re-derivation under the stated definitions.
  python scripts/cr/tables/tab_fg.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common_b import ROOT, load, split, f, ci, n, write, me
from ivbench.specs import fit_spec, eval_row, INTERV_TXT
import numpy as np, pandas as pd

LABEL = 'fg'
DES = ['phase', 'sponsor_class', 'allocation', 'masking', 'purpose', 'area', 'sex']
GROUPS = [('design + endpoint text', dict(text=['outcome_title', 'outcome_desc', 'outcome_timeframe'], cat=DES, num=[])),
          ('design + condition text', dict(text=['conditions'], cat=DES, num=[])),
          ('design + population covariates', dict(text=[], cat=DES, num=['age_mean', 'frac_female', 'year_posted'])),
          ('design + arm sizes', dict(text=[], cat=DES, num=['n_treat', 'n_ctrl'])),
          ('design + control type', dict(text=[], cat=DES + ['ctrl_kind'], num=[])),
          ('design + intervention text', dict(text=INTERV_TXT, cat=DES, num=[]))]
CONF = ('confound probe: sponsor_class+phase+year', dict(text=[], cat=['sponsor_class', 'phase'], num=['year_posted']))


def compute(C):
    tr, te = split(C, 'binary')
    y, g, mean = te.delta.to_numpy(), te.nct_id.to_numpy(), np.full(len(te), tr.delta.mean())
    base = dict(n_test=len(te), n_trials=te.nct_id.nunique())
    pd_ = fit_spec(tr, te, dict(text=[], cat=DES, num=[]), seed=0)
    rows = [dict(model='design-only', reference='train_mean', **dict(zip(['skill', 'ci_low', 'ci_high'], eval_row(y, pd_, mean, g))), **base)]
    for name, sp in GROUPS:
        p = fit_spec(tr, te, sp, seed=0)
        for refname, ref in [('design-only', pd_), ('train_mean', mean)]:
            rows.append(dict(model=name, reference=refname, **dict(zip(['skill', 'ci_low', 'ci_high'], eval_row(y, p, ref, g))), **base))
    p = fit_spec(tr, te, CONF[1], seed=0)
    rows.append(dict(model=CONF[0], reference='train_mean', **dict(zip(['skill', 'ci_low', 'ci_high'], eval_row(y, p, mean, g))), **base))
    return pd.DataFrame(rows)


def main():
    out = []
    for version in ['v2.1', 'v2']:
        R = compute(load(version)); R.insert(0, 'version', version); out.append(R)
    D = pd.concat(out, ignore_index=True)
    ref = pd.read_csv(os.path.join(ROOT, 'results', 'positive_control', 'v2_feature_groups.csv'))
    m = D.version == 'v2'
    D.loc[m, 'v2_released_skill'] = ref.skill.to_numpy()
    D.loc[m, 'v2_reproduces_released'] = np.isclose(D.loc[m, 'skill'].to_numpy(), ref.skill.to_numpy(), rtol=0, atol=1e-12)
    T = D[D.version == 'v2.1']
    tx = lambda s: s.replace('_', '\\_').replace('+', '$+$')
    rows = '\n'.join(f"{tx(r.model)} & {tx(r.reference)} & {f(r.skill)} {ci(r.ci_low, r.ci_high)} \\\\" for _, r in T.iterrows())
    tex = ("\\begin{table}[h]\\centering\\small\n"
           f"\\caption{{Feature-group gains (gradient boosting, seed 0, v2.1 binary test split, {n(T.n_test.iat[0])} contrasts / "
           f"{n(T.n_trials.iat[0])} trials). ``design-only'' = phase, sponsor class, allocation, masking, purpose, area, sex. "
           "95\\% trial-level bootstrap intervals (1{,}000 resamples). Group definitions follow \\texttt{ivbench.specs}; this table is computed afresh on v2.1 and supersedes the submitted version, whose probe code was not preserved.}\n"
           "\\label{tab:fg}\n\\begin{tabular}{llc}\\toprule model & reference & skill \\\\ \\midrule\n" + rows +
           "\n\\bottomrule\\end{tabular}\\end{table}")
    write(LABEL, D, tex)
    print(D[m][['model', 'reference', 'skill', 'v2_released_skill', 'v2_reproduces_released']].round(4).to_string())


if __name__ == '__main__':
    main()
