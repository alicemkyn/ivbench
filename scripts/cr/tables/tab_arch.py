"""Table 8 (tab:arch): registry-model architectures on the v2.1 binary temporal test split.

  python scripts/cr/tables/tab_arch.py

Runs scripts/cr/run_architectures.py (alternative architectures; writes results/cr/arch_raw.csv) on
data/v2.1/benchmark_contrasts_v2.1_final.csv.gz with PubMedBERT embeddings from results/cr/embeddings_v21_binary.npz.
The embeddings are the output of scripts/cr/embed_v21.py (pritamdeka/S-PubMedBert-MS-MARCO via sentence-transformers;
needs network to download the model) and are shipped as a data file so this script runs offline.
The two headline rows come from the registry GBT (ivbench.specs.run_models, seeds 0-4; tab_seeds.py) and are shown at the
precision of Table 6 rounded to two decimals (the approved draft rounds the Table 6 value -0.205 to -0.20).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *  # noqa: E402,F403
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'cr'))
import run_architectures as ra  # noqa: E402

EMB = os.path.join(ROOT, 'results', 'cr', 'embeddings_v21_binary.npz')
RAW = os.path.join(ROOT, 'results', 'cr', 'arch_raw.csv')


def headline():
    C = load('v2.1'); tr, te, P = registry_cached(C, 'binary', 'v2.1')
    y, g = te.delta.to_numpy(), te.nct_id.to_numpy(); ref = np.full(len(y), tr.delta.mean())
    iv = shared_boot_mean(y, [P[('full', s)] for s in SEEDS], [P[('blind_ctrl', s)] for s in SEEDS], g)
    sb = np.mean([me.skill(y, P[('blind_ctrl', s)], ref) for s in SEEDS]); sf = np.mean([me.skill(y, P[('full', s)], ref) for s in SEEDS])
    si = np.mean([me.skill(y, P[('interv_only', s)], ref) for s in SEEDS])
    return [dict(arch='Headline GBT, shared SVD (120)', seed='mean', seeds=5, skill_blind=sb, skill_full=sf, IS=iv[0], IS_lo=iv[1], IS_hi=iv[2], n=len(y), trials=len(np.unique(g))),
            dict(arch='Headline GBT, intervention text only', seed='mean', seeds=5, skill_blind=np.nan, skill_full=si, IS=np.nan, IS_lo=np.nan, IS_hi=np.nan, n=len(y), trials=len(np.unique(g)))]


def tex_label(a):
    a = a.replace('alpha ', '$\\alpha$ ')
    return a


def tex_of(T):
    def two(x, head):
        return f(round(float(x), 3), 2) if head else f(x, 2)   # headline rows: Table 6 precision, then two decimals
    L = []
    for _, r in T.iterrows():
        h = r.arch.startswith('Headline')
        sb = '--' if pd.isna(r.skill_blind) else two(r.skill_blind, h)
        isc = '--' if pd.isna(r.IS) else f"${two(r.IS, h)}$ \\ci{{{two(r.IS_lo, h)}}}{{{two(r.IS_hi, h)}}}"
        L.append(f"{tex_label(r.arch)} & {int(r.seeds)} & {sb} & {two(r.skill_full, h)} & {isc} \\\\")
    n, t = int(T.n.iloc[0]), int(T.trials.iloc[0])
    cap = (f"Registry-model architectures on the v2.1 binary temporal test split ({n} contrasts, {t} trials). Skill columns are against "
           "the global training mean; \\blind\\ includes comparator type, and $\\IS$ is \\full\\ vs \\blind\\ of the same architecture. "
           "The headline GBT embeds endpoint and intervention text in one shared 120-dimensional SVD space; the other rows remove that "
           "sharing. Five-seed rows report the seed mean with an interval from one shared set of 1{,}000 trial resamples; ridge rows are "
           "deterministic, with $\\alpha$ chosen by trial-grouped 5-fold cross-validation. The two-stage model adds a shrunken "
           "correction to the \\blind\\ prediction and so stays near zero by construction. No row has a positive interval. v2 values in "
           "the released CSV.")
    return ("\\begin{table}[h]\\centering\\footnotesize\n\\caption{" + cap + "}\n\\label{tab:arch}\n"
            "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lcccc}\\toprule architecture & seeds & skill(\\blind) & skill(\\full) & $\\IS$ \\\\ \\midrule\n"
            + '\n'.join(L) + "\n\\bottomrule\\end{tabular}}\\end{table}\n")


if __name__ == '__main__':
    with Timer():
        os.makedirs(OUT_CSV, exist_ok=True)
        sys.argv = ['run_architectures.py', '--contrasts', V21, '--embeddings', EMB, '--out', RAW]
        ra.main()
        A = pd.read_csv(RAW)
        A['seeds'] = A.groupby('arch').seed.transform(lambda s: 5 if (s.astype(str) == 'mean').any() else 1)
        A = A[(A.seeds == 1) | (A.seed.astype(str) == 'mean')]
        T = pd.concat([pd.DataFrame(headline()), A], ignore_index=True)
        order = ['Headline GBT, shared SVD (120)', 'Headline GBT, intervention text only', 'GBT, separate SVD blocks (120+120)',
                 'Ridge on sparse TF-IDF', 'Ridge, intervention text only', 'Two-stage', 'GBT on PubMedBERT', 'Ridge on PubMedBERT',
                 'Ridge, PubMedBERT intervention']
        T['o'] = [next(i for i, o in enumerate(order) if a.startswith(o)) for a in T.arch]
        T = T.sort_values('o', kind='stable').drop(columns='o').reset_index(drop=True)
        write('arch', T, tex_of(T))
        print(T[['arch', 'seeds', 'skill_blind', 'skill_full', 'IS', 'IS_lo', 'IS_hi']].round(4).to_string())
