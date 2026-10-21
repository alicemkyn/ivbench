"""Table 1 (tab:main): skill against the global training mean and interventional skill on the temporal test split.

  python scripts/cr/tables/tab_main.py

Primary: benchmark v2.1 (data/v2.1/benchmark_contrasts_v2.1_final.csv.gz). Writes results/cr/main.csv, tables/main.tex and
the same rows on the v2 test set to results/cr/main_v2_sensitivity.csv.
Registry rows: GBT (ivbench.specs.run_models), five-seed mean (seeds 0-4) with one shared set of 1,000 trial resamples.
LM rows: archived per-call outputs (no LM calls); single runs, 2,000 trial resamples (seed 0). On v2.1, 'kept' test rows use
the v2 prediction by cid and new/re-paired rows the v2.1 call (results/llm/cr/v21new_*). Protocol-only prompts are the
headline LM condition; posted-record (original) prompts are the second row; the continuous LM uses the posted-record prompt.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *  # noqa: E402,F403

LM_ROWS = [('binary', 'sonnet_protocol', '\\lm\\ large, protocol-only'), ('binary', 'sonnet_posted', '\\lm\\ large, posted-record'),
           ('binary', 'haiku_protocol', '\\lm\\ small, protocol-only'), ('binary', 'haiku_posted', '\\lm\\ small, posted-record'),
           ('binary', 'llama_protocol', 'Llama 3.1 70B, protocol-only'), ('binary', 'llama_posted', 'Llama 3.1 70B, posted-record'),
           ('continuous', 'sonnet_cont_posted', '\\lm\\ large, posted-record')]


def rows_for(version):
    C = load(version); rows = []
    for fam in FAMS:
        tr, te, P = registry_cached(C, fam, version)
        y, g = te.delta.to_numpy(), te.nct_id.to_numpy(); ref = np.full(len(y), tr.delta.mean()); bt = shared_resamples(g)
        Fp, Bp = [P[('full', s)] for s in SEEDS], [P[('blind_ctrl', s)] for s in SEEDS]
        sb = shared_boot_mean(y, Bp, [ref] * 5, g, boots=bt); sf = shared_boot_mean(y, Fp, [ref] * 5, g, boots=bt)
        iv = shared_boot_mean(y, Fp, Bp, g, boots=bt)
        rows.append(dict(version=version, family=fam, system='registry GBT (5-seed mean)', arm='gbt', prompt='', n=len(y),
                         trials=len(np.unique(g)), skill_blind=sb[0], skill_blind_lo=sb[1], skill_blind_hi=sb[2],
                         skill_full=sf[0], skill_full_lo=sf[1], skill_full_hi=sf[2], IS=iv[0], IS_lo=iv[1], IS_hi=iv[2],
                         IS_seed0=me.skill(y, Fp[0], Bp[0]), n_boot=1000))
        for fam2, arm, _ in LM_ROWS:
            if fam2 != fam:
                continue
            D = lm_arm(arm, version, C=C); r = lm_metrics(D, tr.delta.mean())
            rows.append(dict(version=version, family=fam, system=ARMS[arm]['model'], arm=arm, prompt=ARMS[arm]['prompt'],
                             n=r['n'], trials=r['trials'], skill_blind=r['skill_B'], skill_blind_lo=r['skill_B_lo'],
                             skill_blind_hi=r['skill_B_hi'], skill_full=r['skill_A'], skill_full_lo=r['skill_A_lo'],
                             skill_full_hi=r['skill_A_hi'], IS=r['IS'], IS_lo=r['IS_lo'], IS_hi=r['IS_hi'], n_boot=2000))
    return pd.DataFrame(rows)


def tex_of(R):
    b, c = R[R.family == 'binary'], R[R.family == 'continuous']
    cell = lambda r, k: f"{f(r[k])} {ci(r[k + '_lo'], r[k + '_hi'])}"
    isc = lambda r, bold: (f"\\textbf{{{f(r.IS)}}}" if bold else m(r.IS)) + f" {ci(r.IS_lo, r.IS_hi)}"
    lab = {a: l for _, a, l in LM_ROWS}
    L = []
    L.append(f"\\multirow{{{2 + len(b) - 1}}}{{*}}{{Risk difference}}")
    L.append(" & global mean & -- & 0 & -- \\\\")
    for _, r in b.iterrows():
        name = '\\gbt\\ (5 seeds)' if r.arm == 'gbt' else lab[r.arm]
        L.append(f" & {name} & {cell(r, 'skill_blind')} & {cell(r, 'skill_full')} & {isc(r, r.arm != 'gbt')} \\\\")
    L.append("\\midrule")
    for i, (_, r) in enumerate(c.iterrows()):
        name = '\\gbt\\ (5 seeds)' if r.arm == 'gbt' else lab[r.arm]
        head = f"\\multirow{{{len(c)}}}{{*}}{{Standardised difference}}" if i == 0 else ''
        L.append(f"{head} & {name} & {cell(r, 'skill_blind')} & {cell(r, 'skill_full')} & {isc(r, r.arm != 'gbt')} \\\\")
    nb, tb, nc, tc = int(b.n.iloc[0]), int(b.trials.iloc[0]), int(c.n.iloc[0]), int(c.trials.iloc[0])
    cap = ("\\textbf{Skill against the global training mean and interventional skill, v2.1 temporal test split.} "
           f"Binary: {comma(nb)} contrasts / {tb} trials; continuous: {comma(nc)} / {tc}. \\gbt: five-seed mean, 95\\% interval from one "
           "shared set of 1{,}000 trial resamples; language models (large = Claude Sonnet~5, small = Claude Haiku~4.5; Llama~3.1 70B, "
           "open weights): single runs, 2{,}000 trial resamples. Protocol-only prompts hold registered fields only; posted-record prompts "
           "(the submitted version's) add the posted results-group labels and descriptions; the continuous \\lm\\ row is posted-record. "
           "$\\IS = 1 - \\mathrm{MSE}(\\full)/\\mathrm{MSE}(\\blind)$; v2 values in \\texttt{results/cr/main\\_v2\\_sensitivity.csv}.")
    return ("\\begin{table}[t]\n\\centering\n\\small\n\\caption{" + cap + "}\n\\label{tab:main}\n\\footnotesize\n"
            "\\setlength{\\tabcolsep}{4pt}\\resizebox{\\linewidth}{!}{\\begin{tabular}{llccc}\n\\toprule\n"
            "Task & System & skill(\\blind) & skill(\\full) & interventional skill \\\\\n\\midrule\n"
            + '\n'.join(L) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


if __name__ == '__main__':
    with Timer():
        R21, R2 = rows_for('v2.1'), rows_for('v2')
        write('main', R21, tex_of(R21), extra={'main_v2_sensitivity.csv': R2})
        print(R21[['family', 'system', 'prompt', 'n', 'skill_blind', 'skill_full', 'IS', 'IS_lo', 'IS_hi']].round(3).to_string())
        print(R2[['family', 'system', 'prompt', 'n', 'IS', 'IS_lo', 'IS_hi']].round(3).to_string())
