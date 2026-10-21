"""Table 5 (tab:contamination): interventional skill inside and outside each model's three-condition uncontaminated set, v2.1.

  python scripts/cr/tables/tab_contamination.py

Ported to v2.1 from scripts/cr/stats/contamination_strata.py and contamination_registry_is.py. Per-trial flags are read from
data/cr/contamination_trial_flags.csv (CT.gov / PubMed fetch of Oct 2026, treated as data): a trial is uncontaminated for a
model with boundary b when its primary completion and first results posting are both on or after b and no linked RESULT
publication predates b (undated publications count as before b). Boundaries: training-data cutoffs Llama 3.1 2024-01-01,
Haiku 4.5 2025-08-01, Sonnet 5 2026-02-01; reliable-knowledge cutoffs Haiku 4.5 2025-03-01, Sonnet 5 2026-02-01.
LM IS: 2,000 trial resamples (seed 0); registry GBT IS: five-seed mean with one shared set of 1,000 trial resamples (the
registry model has no pretraining exposure; its row shows what the date stratum alone does). CI undefined for < 2 trials.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *  # noqa: E402,F403

FLAGS = os.path.join(ROOT, 'data', 'cr', 'contamination_trial_flags.csv')
BOUNDS = [('Llama 3.1 70B', 'llama3.1', 'training cutoff', '2024-01-01', ('llama_protocol', 'llama_posted')),
          ('Claude Haiku 4.5', 'haiku4.5', 'training cutoff', '2025-08-01', ('haiku_protocol', 'haiku_posted')),
          ('Claude Haiku 4.5', 'haiku4.5_reliable', 'reliable-knowledge cutoff', '2025-03-01', ('haiku_protocol', 'haiku_posted')),
          ('Claude Sonnet 5', 'sonnet5', 'training cutoff', '2026-02-01', ('sonnet_protocol', 'sonnet_posted')),
          ('Claude Sonnet 5', 'sonnet5_reliable', 'reliable-knowledge cutoff', '2026-02-01', ('sonnet_protocol', 'sonnet_posted'))]


def lm_is(D, msk):
    d = D[msk]
    if len(d) == 0:
        return (np.nan, np.nan, np.nan)
    return skill_ci(d.truth, d.A, d.B, d.nct_id, 2000)


def rows(version='v2.1'):
    C = load(version); FL = pd.read_csv(FLAGS).set_index('nct_id'); R = []
    for fam in FAMS:
        tr, te, P = registry_cached(C, fam, version)
        y, g = te.delta.to_numpy(), te.nct_id.to_numpy(); assert te.nct_id.isin(FL.index).all()
        arms = {a: lm_arm(a, version, C=C) for a in ARMS if ARMS[a]['fam'] == fam}
        for model, tag, kind, b, (pa, qa) in BOUNDS:
            u = FL.loc[te.nct_id, f'uncont_{tag}'].to_numpy(bool)
            for st, msk in (('uncontaminated', u), ('complement', ~u)):
                r = dict(version=version, family=fam, model=model, flag=f'uncont_{tag}', cutoff_kind=kind, boundary=b, stratum=st,
                         n=int(msk.sum()), trials=int(len(np.unique(g[msk]))))
                if msk.sum() >= 2 and len(np.unique(g[msk])) >= 2:
                    s = shared_boot_mean(y[msk], [P[('full', s)][msk] for s in SEEDS], [P[('blind_ctrl', s)][msk] for s in SEEDS], g[msk])
                    r.update(gbt_IS=s[0], gbt_IS_lo=s[1], gbt_IS_hi=s[2])
                for lab, a in (('protocol', pa), ('posted', qa)):
                    key = a if fam == 'binary' else ('sonnet_cont_posted' if (lab == 'posted' and model == 'Claude Sonnet 5') else None)
                    if key is None:
                        continue
                    s = lm_is(arms[key], msk); r.update({f'{lab}_IS': s[0], f'{lab}_IS_lo': s[1], f'{lab}_IS_hi': s[2]})
                r['note'] = 'not interpretable (n < 30)' if r['n'] < 30 else ''
                R.append(r)
    return pd.DataFrame(R)


def tex_of(R):
    b = R[(R.family == 'binary') & (R.flag != 'uncont_sonnet5_reliable')].reset_index(drop=True); L = []
    def cell(r, k):
        if pd.isna(r.get(k, np.nan)):
            return '--'
        if r.n < 30:
            return m(r[k])[:-1] + '^{\\dagger}$'
        return m(r[k]) + (f" {ci(r[k + '_lo'], r[k + '_hi'])}" if pd.notna(r[k + '_lo']) else '')
    for i, (_, r) in enumerate(b.iterrows()):
        kind = 'training / reliable-knowledge cutoff' if r.flag == 'uncont_sonnet5' else r.cutoff_kind
        head = f"{r.model.replace('Claude ', '')} & {kind} ({r.boundary})" if r.stratum == 'uncontaminated' else ' & '
        L.append(f"{head} & {r.stratum} & {r.n} & {r.trials} & {cell(r, 'protocol_IS')} & {cell(r, 'posted_IS')} & {cell(r, 'gbt_IS')} \\\\")
        if r.stratum == 'complement' and i < len(b) - 1:
            L.append('\\addlinespace')
    nb = int(b.n.iloc[0] + b.n.iloc[1]); tb = int(R[(R.family == 'binary')].trials.iloc[0] + R[(R.family == 'binary')].trials.iloc[1])
    cap = ("Interventional skill inside and outside each model's uncontaminated set, v2.1 binary temporal test split "
           f"({comma(nb)} contrasts). A trial is uncontaminated for a model when its primary completion and first results posting both "
           "fall on or after the model's boundary date and no linked results publication predates it. Language-model columns use "
           "protocol-only and posted-record prompts (2{,}000 trial resamples); the registry \\gbt\\ column (five-seed mean, one shared "
           "set of 1{,}000 trial resamples) has no pretraining exposure and shows the effect of the date stratum alone. Strata with "
           "fewer than 30 contrasts ($^{\\dagger}$) are not interpretable and are shown without intervals. For Sonnet~5 the "
           "training and reliable-knowledge boundaries coincide. Continuous-family "
           "strata are in \\texttt{results/cr/contamination.csv}.")
    return ("\\begin{table}[h]\\centering\\small\n\\caption{" + cap + "}\n\\label{tab:contamination}\n"
            "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lllrrccc}\\toprule model & boundary & stratum & contrasts & trials & "
            "$\\IS$ protocol-only & $\\IS$ posted-record & $\\IS$ \\gbt \\\\ \\midrule\n" + '\n'.join(L) +
            "\n\\bottomrule\\end{tabular}}\\end{table}\n")


if __name__ == '__main__':
    with Timer():
        R = rows(); write('contamination', R, tex_of(R))
        print(R.drop(columns=['flag', 'note']).round(3).to_string())
