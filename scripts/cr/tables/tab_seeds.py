"""Table 6 (tab:seeds): gradient boosting, binary risk difference, v2.1 temporal test split, by seed.

  python scripts/cr/tables/tab_seeds.py

Columns 2-5: skill against the global training mean for specs 'blind' (\\blind$^{-}$), 'blind_ctrl' (\\blind), 'full' and
'interv_only'; IS = skill(full | blind_ctrl) with a per-seed 1,000-resample trial bootstrap (seed 0); the mean row applies
one shared set of 1,000 trial resamples to all seeds. Also writes the same table on v2 to results/cr/seeds_v2.csv.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *  # noqa: E402,F403


def rows_for(version):
    C = load(version); tr, te, P = registry_cached(C, 'binary', version)
    y, g = te.delta.to_numpy(), te.nct_id.to_numpy(); ref = np.full(len(y), tr.delta.mean()); rows = []
    for s in SEEDS:
        r = dict(version=version, seed=str(s), n=len(y), trials=len(np.unique(g)))
        for n in ALL_SPECS:
            r[f'skill_{n}'] = me.skill(y, P[(n, s)], ref)
        r['IS'], r['IS_lo'], r['IS_hi'] = skill_ci(y, P[('full', s)], P[('blind_ctrl', s)], g, 1000)
        rows.append(r)
    R = pd.DataFrame(rows)
    mean = dict(version=version, seed='mean', n=len(y), trials=len(np.unique(g)),
                **{f'skill_{n}': R[f'skill_{n}'].mean() for n in ALL_SPECS})
    mean['IS'], mean['IS_lo'], mean['IS_hi'] = shared_boot_mean(y, [P[('full', s)] for s in SEEDS], [P[('blind_ctrl', s)] for s in SEEDS], g)
    mean['IS_seed_sd'] = float(R.IS.std(ddof=1))
    return pd.concat([R, pd.DataFrame([mean])], ignore_index=True)


def tex_of(R):
    n, t = int(R.n.iloc[0]), int(R.trials.iloc[0]); L = []
    for _, r in R.iterrows():
        pre = '\\midrule mean' if r.seed == 'mean' else r.seed
        L.append(f"{pre} & {f(r.skill_blind)} & {f(r.skill_blind_ctrl)} & {f(r.skill_full)} & {f(r.skill_interv_only)} & "
                 f"{m(r.IS)} {ci(r.IS_lo, r.IS_hi)} \\\\")
    return ("\\begin{table}[h]\\centering\\small\n\\caption{Gradient boosting, binary risk difference, v2.1 temporal test split "
            f"({n} contrasts / {t} trials), by seed. Columns 2--5: skill against the global training mean. \\blind$^{{-}}$ omits "
            "comparator type; \\blind\\ includes it and is the reference for $\\IS$. The mean-row interval applies one shared set of "
            "1{,}000 trial resamples to all seeds. v2 values in the released CSV.}\n\\label{tab:seeds}\n"
            "\\begin{tabular}{lccccc}\\toprule seed & \\blind$^{-}$ & \\blind & \\full & intervention only & $\\IS$ (\\full\\ vs \\blind) \\\\ \\midrule\n"
            + '\n'.join(L) + "\n\\bottomrule\\end{tabular}\\end{table}\n")


if __name__ == '__main__':
    with Timer():
        R21, R2 = rows_for('v2.1'), rows_for('v2')
        write('seeds', R21, tex_of(R21), extra={'seeds_v2.csv': R2})
        print(R21.round(3).to_string()); print(R2.round(3).to_string())
