"""New table (tab:design): interventional skill on design and valid-conversion subsets, v2.1.

  python scripts/cr/tables/tab_design.py

Ported to v2.1 from scripts/cr/stats/design_subset.py and valid_conversion.py.
  randomized parallel-group: data/cr/design_trial_fields.csv (Oct 2026 CT.gov re-fetch, treated as data):
                             allocation_now == RANDOMIZED and intervention_model == PARALLEL
  randomized:                benchmark field allocation == RANDOMIZED
  binary participant-level unit: the outcome value is a count or percentage of participants. For v2.1 'kept' rows the flag is
                             read from data/cr/valid_conversion_flags.csv by cid; the 62 new/re-paired rows take the flag of the
                             same (param_type, unit) pair among the v2 test rows (in v2 the flag is a function of that pair;
                             asserted below). The original rule (fetch_audit_records.binary_participant) is not in the repository.
  continuous strict:         param_type == MEAN; sensitivity: MEAN or LEAST_SQUARES_MEAN
Registry GBT: five-seed mean with one shared set of 1,000 trial resamples (and seed 0 with 1,000 resamples, CSV only).
LM arms: 2,000 trial resamples. The sampling-noise ceiling on each subset (valid_conversion.py formulas) is written to
results/cr/design_noise_ceiling.csv.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *  # noqa: E402,F403

DESIGN = os.path.join(ROOT, 'data', 'cr', 'design_trial_fields.csv')
VALID = os.path.join(ROOT, 'data', 'cr', 'valid_conversion_flags.csv')
PU = 'participant-level unit'


def participant_unit(te21, C2):
    VF = pd.read_csv(VALID); te2 = test(C2, 'binary', 'v2').merge(VF[['cid', PU]], on='cid')
    key = te2.param_type.astype(str) + '|' + te2.unit.astype(str)
    g = te2.groupby(key)[PU].agg(['min', 'max']); assert (g['min'] == g['max']).all(), 'v2 flag is not a function of (param_type, unit)'
    lut = g['min'].to_dict(); byc = VF.set_index('cid')[PU]
    k21 = (te21.param_type.astype(str) + '|' + te21.unit.astype(str))
    kept = (te21.v21_status == 'kept').to_numpy()
    out = np.where(kept, te21.cid.map(byc), k21.map(lut))
    assert pd.notna(out).all(), 'unmapped participant-unit flag'
    return out.astype(bool)


def sampling_var(te, family):
    n1, n2 = te.n_treat.to_numpy(float), te.n_ctrl.to_numpy(float)
    if family == 'binary':
        p1 = np.clip(te.y_treat.to_numpy(float), 0.5 / n1, 1 - 0.5 / n1); p2 = np.clip(te.y_ctrl.to_numpy(float), 0.5 / n2, 1 - 0.5 / n2)
        return p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
    d = te.delta.to_numpy(float); return (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))


def subsets(fam, te, C2):
    M = pd.read_csv(DESIGN).set_index('nct_id')
    rp = ((te.nct_id.map(M.allocation_now) == 'RANDOMIZED') & (te.nct_id.map(M.intervention_model) == 'PARALLEL')).to_numpy()
    rand = (te.allocation == 'RANDOMIZED').to_numpy(); al = np.ones(len(te), bool)
    S = {'all test': al, 'randomized parallel-group': rp, 'not randomized parallel-group': ~rp}
    if fam == 'binary':
        v = participant_unit(te, C2)
        S.update({PU: v, f'{PU} & randomized': v & rand, f'{PU} & randomized parallel': v & rp})
    else:
        mean = (te.param_type == 'MEAN').to_numpy(); ls = te.param_type.isin(['MEAN', 'LEAST_SQUARES_MEAN']).to_numpy()
        S.update({'strict: arithmetic MEAN': mean, 'strict & randomized': mean & rand, 'strict & randomized parallel': mean & rp,
                  'sensitivity: MEAN or LS MEAN & randomized': ls & rand, 'sensitivity: MEAN or LS MEAN & randomized parallel': ls & rp})
    return S


def rows():
    C, C2 = load('v2.1'), load('v2'); R, N = [], []
    for fam in FAMS:
        tr, te, P = registry_cached(C, fam, 'v2.1'); y, g = te.delta.to_numpy(), te.nct_id.to_numpy()
        arms = {a: lm_arm(a, 'v2.1', C=C) for a in ARMS if ARMS[a]['fam'] == fam}; sv_all = sampling_var(te, fam)
        for name, msk in subsets(fam, te, C2).items():
            yy, gg = y[msk], g[msk]
            base = dict(version='v2.1', family=fam, subset=name, n=int(msk.sum()), trials=int(len(np.unique(gg))), frac_of_test=float(msk.mean()))
            s5 = shared_boot_mean(yy, [P[('full', s)][msk] for s in SEEDS], [P[('blind_ctrl', s)][msk] for s in SEEDS], gg)
            per = [me.skill(yy, P[('full', s)][msk], P[('blind_ctrl', s)][msk]) for s in SEEDS]
            R.append(dict(base, arm='gbt_5seed', IS=s5[0], IS_lo=s5[1], IS_hi=s5[2], seed_sd=float(np.std(per, ddof=1))))
            s0 = skill_ci(yy, P[('full', 0)][msk], P[('blind_ctrl', 0)][msk], gg, 1000)
            R.append(dict(base, arm='gbt_seed0', IS=s0[0], IS_lo=s0[1], IS_hi=s0[2]))
            for a, D in arms.items():
                d = D[msk]; s = skill_ci(d.truth, d.A, d.B, d.nct_id, 2000); R.append(dict(base, arm=a, IS=s[0], IS_lo=s[1], IS_hi=s[2]))
            sv = sv_all[msk]; tot = float(np.var(yy)); mse_mean = float(np.mean((yy - tr.delta.mean()) ** 2))
            rng = np.random.default_rng(0); u = np.unique(gg); idx = {t: np.flatnonzero(gg == t) for t in u}; bo = []
            for _ in range(1000):
                ii = np.concatenate([idx[t] for t in rng.choice(u, len(u), replace=True)]); bo.append(1 - sv[ii].mean() / np.var(yy[ii]))
            nb = dict(base, mean_sampling_var=float(sv.mean()), total_var_delta=tot, reliability=1 - sv.mean() / tot,
                      reliability_lo=float(np.percentile(bo, 2.5)), reliability_hi=float(np.percentile(bo, 97.5)),
                      ceiling_skill_vs_mean=1 - sv.mean() / mse_mean)
            refs = [('registry blind_ctrl seed 0', P[('blind_ctrl', 0)][msk])] + [(f'{a} B', D.B.to_numpy()[msk]) for a, D in arms.items()]
            for rn, rp_ in refs:
                N.append(dict(nb, reference=rn, max_IS_given_reference=1 - sv.mean() / float(np.mean((yy - rp_) ** 2))))
    return pd.DataFrame(R), pd.DataFrame(N)


COLS = [('gbt_5seed', '\\gbt'), ('sonnet_protocol', 'Sonnet 5, protocol-only'), ('sonnet_posted', 'Sonnet 5, posted-record'),
        ('haiku_protocol', 'Haiku 4.5, protocol-only'), ('llama_protocol', 'Llama 3.1, protocol-only')]


def tex_of(R):
    """Two tables: binary (tab:design) and continuous (tab:designcont), one row per subset x model, no resizebox."""
    def block(fam, cols, label, cap):
        Rf = R[R.family == fam]; L = []
        for sub, G in Rf.groupby('subset', sort=False):
            L.append(f"\\multicolumn{{3}}{{l}}{{\\emph{{{sub.replace('&', chr(92) + '&')}}}: {comma(int(G.n.iloc[0]))} contrasts, {comma(int(G.trials.iloc[0]))} trials}} \\\\")
            for a, lab in cols:
                r = G[G.arm == a]
                if r.empty:
                    continue
                L.append(f"\\quad {lab} & {m(r.IS.iloc[0])} {ci(r.IS_lo.iloc[0], r.IS_hi.iloc[0])} & {f(r.seed_sd.iloc[0]) if pd.notna(r.seed_sd.iloc[0]) else '--'} \\\\")
            L.append('\\midrule')
        L = L[:-1]
        return ("\\begin{table}[h]\\centering\\footnotesize\n\\caption{" + cap + "}\n\\label{" + label + "}\n"
                "\\begin{tabular}{lcc}\\toprule system & $\\IS$ & seed SD \\\\ \\midrule\n" + '\n'.join(L) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    common = ("\\gbt: five-seed mean with one shared set of 1{,}000 trial resamples, seed SD across the five seeds; language models: single runs, "
              "2{,}000 trial resamples. Haiku and Llama posted-record rows, seed-0 registry values and the noise ceiling per subset are in "
              "\\texttt{results/cr/design.csv} and \\texttt{results/cr/design\\_noise\\_ceiling.csv}.")
    cap_b = ("Interventional skill on design subsets of the v2.1 binary temporal test split. Randomized parallel-group: allocation and "
             "intervention model from an October 2026 re-fetch of the registry records. Participant-level unit: the binary outcome is a "
             "count or percentage of participants. " + common)
    cap_c = ("Interventional skill on design and valid-conversion subsets of the v2.1 continuous temporal test split (posted-record prompt "
             "for the language model). Strict: arithmetic means only; sensitivity rows also admit least-squares means. " + common)
    cols_b = [('gbt_5seed', '\\gbt'), ('sonnet_protocol', 'Sonnet 5, protocol-only'), ('sonnet_posted', 'Sonnet 5, posted-record'),
              ('haiku_protocol', 'Haiku 4.5, protocol-only'), ('llama_protocol', 'Llama 3.1 70B, protocol-only')]
    cols_c = [('gbt_5seed', '\\gbt'), ('sonnet_cont_posted', 'Sonnet 5, posted-record')]
    return block('binary', cols_b, 'tab:design', cap_b), block('continuous', cols_c, 'tab:designcont', cap_c)


if __name__ == '__main__':
    with Timer():
        R, N = rows(); tb, tc = tex_of(R); write('design', R, tb, extra={'design_noise_ceiling.csv': N})
        with open(os.path.join(OUT_TEX, 'designcont.tex'), 'w') as fh:
            fh.write(tc.rstrip() + '\n')
        print(R[R.arm.isin(['gbt_5seed', 'sonnet_protocol', 'sonnet_posted', 'sonnet_cont_posted'])][['family', 'subset', 'n', 'trials', 'arm', 'IS', 'IS_lo', 'IS_hi']].round(3).to_string())
