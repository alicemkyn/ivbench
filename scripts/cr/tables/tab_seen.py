"""Table tab:seen (paper Table 10): seen vs unseen interventions for the registry model on benchmark v2.1 (binary).
Intervention keys = ivbench.specs.norm_name of the treatment intervention names (fallback: treatment group title). A test
contrast is 'seen' when (exact) one of its keys equals a key of any training contrast, or (token-Jaccard >= 0.5) one of its keys
has token Jaccard >= 0.5 with a training key. Headline blind (with comparator type) and full gradient-boosting models, seeds 0-4,
evaluated on each stratum; per-seed trial bootstrap (1,000 resamples, seed 0); the five-seed mean IS interval applies one shared
set of 1,000 trial resamples to every seed (the v2 paper table averaged the per-seed bounds instead; both are in the CSV
for the mean rows). v2 rows (recomputed; exact-name rows reproduce results/positive_control/
v2_seen_unseen.csv) go to the CSV only.
  python scripts/cr/tables/tab_seen.py"""
import os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common_b import ROOT, load, split, shared_boot_mean, f, ci, n, write, me
from ivbench.specs import run_models, eval_row, norm_name
import numpy as np, pandas as pd

LABEL = 'seen'
SEEDS = [0, 1, 2, 3, 4]


def keys(D):
    return [set(norm_name(a) or norm_name(b)) for a, b in zip(D.treat_intervention, D.treat_group)]


def strata(tr, te):
    ktr, kte = keys(tr), keys(te)
    alltr = set().union(*ktr)
    exact = np.array([bool(s & alltr) for s in kte])
    trtok = [set(k.split()) for k in sorted(alltr)]; inv = defaultdict(set)
    for i, ts in enumerate(trtok):
        for t in ts: inv[t].add(i)

    def jseen(s):
        for key in s:
            a = set(key.split())
            for i in set().union(*[inv[t] for t in a]) if a else ():
                if len(a & trtok[i]) / len(a | trtok[i]) >= 0.5:
                    return True
        return False
    return {'exact normalized name': exact, 'token-Jaccard>=0.5': np.array([jseen(s) for s in kte])}


def compute(C):
    tr, te = split(C, 'binary')
    y, g, mean = te.delta.to_numpy(), te.nct_id.to_numpy(), np.full(len(te), tr.delta.mean())
    P = {s: run_models(tr, te, ['blind_ctrl', 'full'], seed=s) for s in SEEDS}
    rows = []
    for variant, seen in strata(tr, te).items():
        for stratum, m in [('seen', seen), ('unseen', ~seen)]:
            ym, gm, rm = y[m], g[m], mean[m]
            for s in SEEDS:
                sk, lo, hi = eval_row(ym, P[s]['full'][m], P[s]['blind_ctrl'][m], gm, n_boot=1000, seed=0)
                rows.append(dict(variant=variant, stratum=stratum, seed=s, n=int(m.sum()), n_trials=len(np.unique(gm)),
                                 blind_ctrl_skill=me.skill(ym, P[s]['blind_ctrl'][m], rm), full_skill=me.skill(ym, P[s]['full'][m], rm),
                                 interventional_skill=sk, ci_low=lo, ci_high=hi))
            R = pd.DataFrame(rows[-len(SEEDS):])
            lo, hi = shared_boot_mean(ym, [P[s]['full'][m] for s in SEEDS], [P[s]['blind_ctrl'][m] for s in SEEDS], gm, n=1000, seed=0)
            rows.append(dict(variant=variant, stratum=stratum, seed='mean', n=int(m.sum()), n_trials=len(np.unique(gm)),
                             blind_ctrl_skill=R.blind_ctrl_skill.mean(), full_skill=R.full_skill.mean(),
                             interventional_skill=R.interventional_skill.mean(), ci_low=lo, ci_high=hi,
                             ci_low_mean_of_seed_bounds=R.ci_low.mean(), ci_high_mean_of_seed_bounds=R.ci_high.mean()))
    return pd.DataFrame(rows)


def main():
    out = []
    for version in ['v2.1', 'v2']:
        R = compute(load(version)); R.insert(0, 'version', version); out.append(R)
    D = pd.concat(out, ignore_index=True); D['seed'] = D.seed.astype(str)
    ref = pd.read_csv(os.path.join(ROOT, 'results', 'positive_control', 'v2_seen_unseen.csv')); ref['seed'] = ref.seed.astype(str)
    M = D[D.version == 'v2'].merge(ref, on=['variant', 'stratum', 'seed'], how='left', suffixes=('', '_rel'))
    ok = np.isclose(M.interventional_skill, M.interventional_skill_rel, rtol=0, atol=1e-12) & (M.n == M.n_rel)
    D.loc[D.version == 'v2', 'v2_released_IS'] = M.interventional_skill_rel.to_numpy()
    D.loc[D.version == 'v2', 'v2_reproduces_released'] = np.where(M.n_rel.isna(), np.nan, ok)
    T = D[(D.version == 'v2.1') & (D.seed == 'mean')]
    rows = '\n'.join(f"{r.variant.replace('>=', '$' + chr(92) + 'geq$')} & {r.stratum} & {r.n} & {r.n_trials} & {f(r.blind_ctrl_skill)} & "
                     f"{f(r.full_skill)} & {f(r.interventional_skill)} {ci(r.ci_low, r.ci_high)} \\\\" for _, r in T.iterrows())
    te = T.n.iat[0] + T.n.iat[1]
    tex = ("\\begin{table}[h]\\centering\\small\n"
           f"\\caption{{Seen vs unseen interventions on v2.1 ({n(te)} binary test contrasts; five-seed means; headline models "
           "evaluated on the stratum). ``Seen'' = the test contrast's intervention key appears among training contrasts. "
           "Intervals: one shared set of 1{,}000 trial resamples applied to all seeds. Seen molecules do not help: $\\IS$ is more negative where the molecule was in training. v2 values in the released CSV.}\n"
           "\\label{tab:seen}\n\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrccc}\\toprule matching & stratum & contrasts & trials & skill(\\blind) & "
           "skill(\\full) & $\\IS$ \\\\ \\midrule\n" + rows + "\n\\bottomrule\\end{tabular}}\\end{table}")
    write(LABEL, D, tex)
    print(D[D.version == 'v2'][['variant', 'stratum', 'seed', 'n', 'interventional_skill', 'v2_released_IS', 'v2_reproduces_released']].round(4).to_string())


if __name__ == '__main__':
    main()
