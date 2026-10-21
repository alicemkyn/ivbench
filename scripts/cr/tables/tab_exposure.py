"""Appendix table (tab:exposure): exposure test with Llama 3.1 70B Instruct (knowledge cutoff December 2023).

  python scripts/cr/tables/tab_exposure.py

IS = skill(A | B) with 2,000 trial resamples (seed 0), protocol-only and posted-record prompts, on v2.1 and v2.
Uncontaminated: data/cr/contamination_trial_flags.csv column uncont_llama3.1 (primary completion and first results posting
on or after 2024-01-01, no earlier linked RESULT publication).
Vocabulary flagged: LEAKY regex on treat_group_desc + ' ' + ctrl_group_desc (ivbench.llm_prompts._s).
Molecule seen: any ivbench.specs.norm_name element of treat_intervention occurs among the norm_name elements of
treat_intervention over the v2 TRAIN split of both families (binary and continuous). The approved draft used the v2 training
names for both the v2 and the v2.1 rows, and that definition is kept here; the CSV also gives the v2.1 rows with v2.1 training
names (molecule_names='v2.1 train').
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *  # noqa: E402,F403
from ivbench.specs import norm_name  # noqa: E402
from ivbench.llm_prompts import _s  # noqa: E402

LEAKY = re.compile(r'(?i)\b(achiev|respond|response rate|remission|improv|reduc|increas|decreas|p\s*[<=]|significan|efficac|superior|'
                   r'non-?inferior|outcome|result|percent|%)\w*')
FLAGS = os.path.join(ROOT, 'data', 'cr', 'contamination_trial_flags.csv')


def train_names(C):
    return {n for x in C.loc[C.split_time == 'train', 'treat_intervention'] for n in norm_name(x)}


def rows_for(version, C, names, names_tag):
    te = test(C, 'binary', version); fl = pd.read_csv(FLAGS).set_index('nct_id')['uncont_llama3.1']
    P, Q = lm_arm('llama_protocol', version, C=C), lm_arm('llama_posted', version, C=C)
    u = fl.loc[te.nct_id].to_numpy(bool)
    leak = np.array([bool(LEAKY.search(_s(a) + ' ' + _s(b))) for a, b in zip(te.treat_group_desc, te.ctrl_group_desc)])
    seen = np.array([any(n in names for n in norm_name(x)) for x in te.treat_intervention])
    R = []
    def add(stratum, msk, D, prompt):
        d = D[msk]; s = skill_ci(d.truth, d.A, d.B, d.nct_id, 2000)
        R.append(dict(set=version, stratum=stratum, prompt=prompt, IS=s[0], IS_lo=s[1], IS_hi=s[2], n=int(msk.sum()),
                      trials=int(d.nct_id.nunique()), molecule_names=names_tag))
    al = np.ones(len(te), bool)
    for st, ms in (('all', al), ('uncontaminated', u), ('complement', ~u)):
        add(st, ms, P, 'protocol-only'); add(st, ms, Q, 'posted-record')
    for st, ms in (('uncontaminated', u), ('complement', ~u)):
        add(f'{st}, vocabulary unflagged', ms & ~leak, Q, 'posted-record'); add(f'{st}, vocabulary flagged', ms & leak, Q, 'posted-record')
        add(f'{st}, molecule seen', ms & seen, P, 'protocol-only'); add(f'{st}, molecule unseen', ms & ~seen, P, 'protocol-only')
    # difference uncontaminated minus contaminated (complement), protocol-only, independent trial strata
    for D, prompt in ((P, 'protocol-only'), (Q, 'posted-record')):
        y, a, b, g = D.truth.to_numpy(), D.A.to_numpy(), D.B.to_numpy(), D.nct_id.to_numpy()
        def diff(i):
            uu = u[i]
            return me.skill(y[i][uu], a[i][uu], b[i][uu]) - me.skill(y[i][~uu], a[i][~uu], b[i][~uu])
        lo, hi = boot_stat(diff, g, 2000)
        R.append(dict(set=version, stratum='difference, uncontaminated minus complement', prompt=prompt, IS=diff(np.arange(len(y))),
                      IS_lo=lo, IS_hi=hi, n=len(y), trials=int(len(np.unique(g))), molecule_names=names_tag))
    return R


def tex_of(R):
    T = R[(R.molecule_names == 'v2 train') & ~R.stratum.str.startswith('difference')]
    L = [f"{r.set} & {r.stratum} & {r.prompt} & {m(r.IS, 2)} {ci(r.IS_lo, r.IS_hi, 2)} & {r.n} ({r.trials}) \\\\" for _, r in T.iterrows()]
    return ("\\begin{table}[h]\\centering\\footnotesize\n\\caption{Exposure test with Llama~3.1 70B (knowledge cutoff December 2023). "
            "Interventional skill with 95\\% trial-level bootstrap\nintervals (2{,}000 resamples). Uncontaminated: primary completion and "
            "first results posting on or after 2024-01-01 and no\nearlier linked results publication. ``Molecule seen'': a normalised "
            "treatment name occurs among treatment arms in the registry\ntraining split (a proxy for prior knowledge, not a measure of "
            "pretraining exposure). ``Flagged'': results-group descriptions\nmatch an outcome-vocabulary pattern.}\\label{tab:exposure}\n"
            "\\begin{tabular}{llccc}\\toprule\nset & stratum / split & prompt & $\\IS$ & $n$ (trials) \\\\ \\midrule\n"
            + '\n'.join(L) + "\n\\bottomrule\\end{tabular}\\end{table}\n")


if __name__ == '__main__':
    with Timer():
        C21, C2 = load('v2.1'), load('v2'); N2, N21 = train_names(C2), train_names(C21)
        R = pd.DataFrame(rows_for('v2.1', C21, N2, 'v2 train') + rows_for('v2', C2, N2, 'v2 train') + rows_for('v2.1', C21, N21, 'v2.1 train'))
        write('exposure', R, tex_of(R))
        print(R.round(3).to_string())
