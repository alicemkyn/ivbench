"""New table (tab:lmrevision): language-model revision checks.

  python scripts/cr/tables/tab_lmrevision.py

(a) Prompt condition: IS with protocol-only vs posted-record prompts and their paired difference (one shared set of 2,000
    trial resamples, seed 0), Sonnet 5 / Haiku 4.5 / Llama 3.1 70B, v2.1 and v2.
(b) Arm swap (Sonnet 5, original prompts): condition A with arms exchanged. Sign reversal among contrasts with
    |original A| > 0.01, correlation of the swapped prediction with the negated original, mean(swapped + original).
(c) Blind repeat (Sonnet 5, original prompts): a second run of condition B; IS of the original A against it.
(d) Pre-registered negative control (results/README.md, section "Negative-control pre-registration", copied to
    results/llm/cr/negative_control_preregistration.md): drug replaced by loratadine (cardiometabolic) or ezetimibe (all other
    areas) in the original condition-A prompt; A = original condition A, NC = substituted, B = original condition B.
    H1: mean(|A| - |NC|) > 0 (2,000-resample trial bootstrap, seed 0) and |NC| < |A| for a majority of contrasts.
    H2 (inert control): mean |NC| < mean |A| and OLS slope of NC on A through the origin < 0.5.
    H3 (active comparator, contrasts with A > 0): mean(NC - A) < 0.
    3 of 1,020 negative-control calls were refused (no prediction) and are excluded.
Perturbation runs exist for the 1,020 v2 binary test contrasts and the 62 new/re-paired v2.1 test contrasts; rows are given on v2, on the 816 v2.1 'kept' contrasts and on the full v2.1 test set (displayed).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *  # noqa: E402,F403

B = 2000


def row(panel, ts, stat, val, lo=np.nan, hi=np.nan, n=np.nan, trials=np.nan, criterion='', verdict=''):
    return dict(panel=panel, test_set=ts, statistic=stat, value=float(val), lo=lo, hi=hi, n=n, trials=trials,
                criterion=criterion, verdict=verdict)


def prompt_rows(C, v):
    out = []
    for name, p, q in (('Claude Sonnet 5', 'sonnet_protocol', 'sonnet_posted'), ('Claude Haiku 4.5', 'haiku_protocol', 'haiku_posted'),
                       ('Llama 3.1 70B', 'llama_protocol', 'llama_posted')):
        P, Q = lm_arm(p, v, C=C), lm_arm(q, v, C=C); assert (P.cid == Q.cid).all()
        y, g = P.truth.to_numpy(), P.nct_id.to_numpy()
        a1, b1, a2, b2 = P.A.to_numpy(), P.B.to_numpy(), Q.A.to_numpy(), Q.B.to_numpy()
        sp, sq = skill_ci(y, a1, b1, g, B), skill_ci(y, a2, b2, g, B)
        d = boot_stat(lambda i: me.skill(y[i], a2[i], b2[i]) - me.skill(y[i], a1[i], b1[i]), g, B)
        k = dict(n=len(y), trials=len(np.unique(g)))
        out += [row('a', v, f'{name}: IS, protocol-only', *sp, **k), row('a', v, f'{name}: IS, posted-record', *sq, **k),
                row('a', v, f'{name}: IS difference, posted-record minus protocol-only', sq[0] - sp[0], *d, **k)]
    return out


def perturbation_rows(D, ts):
    """D: v2-ordered frame with truth, nct_id, ctrl_kind, A (orig), B (orig), SW (swap), RB (repeat B), NC (negative control)."""
    out = []
    y, g = D.truth.to_numpy(), D.nct_id.to_numpy(); A, Bo, SW, RB = (D[c].to_numpy() for c in ('A', 'B', 'SW', 'RB'))
    # (b) swap
    msk = np.abs(A) > 0.01; gm = g[msk]; As, Ss = A[msk], SW[msk]
    sr = lambda i: np.mean(np.sign(Ss[i]) == -np.sign(As[i]))
    out.append(row('b', ts, 'sign reversal, |A| > 0.01', sr(np.arange(len(As))), *boot_stat(sr, gm, B), n=int(msk.sum()), trials=len(np.unique(gm)),
                   criterion='a model that reads the arms returns about -A'))
    cr_ = lambda i: np.corrcoef(SW[i], -A[i])[0, 1]
    out.append(row('b', ts, 'corr(swapped, -A)', cr_(np.arange(len(A))), *boot_stat(cr_, g, B), n=len(A), trials=len(np.unique(g))))
    ms = lambda i: np.mean(SW[i] + A[i])
    out.append(row('b', ts, 'mean(swapped + A) (ideal 0)', ms(np.arange(len(A))), *boot_stat(ms, g, B), n=len(A), trials=len(np.unique(g))))
    out.append(row('b', ts, 'mean |A| (scale reference)', np.mean(np.abs(A)), n=len(A), trials=len(np.unique(g))))
    # (c) blind repeat
    k = dict(n=len(A), trials=len(np.unique(g)))
    out.append(row('c', ts, 'IS, original A vs original B', *skill_ci(y, A, Bo, g, B), **k))
    out.append(row('c', ts, 'IS, original A vs repeat B', *skill_ci(y, A, RB, g, B), **k))
    dd = lambda i: me.skill(y[i], A[i], RB[i]) - me.skill(y[i], A[i], Bo[i])
    out.append(row('c', ts, 'IS difference, repeat B minus original B', dd(np.arange(len(A))), *boot_stat(dd, g, B), **k))
    out.append(row('c', ts, 'corr(repeat B, original B)', np.corrcoef(RB, Bo)[0, 1], **k))
    out.append(row('c', ts, 'fraction of identical B points', np.mean(np.isclose(RB, Bo)), **k))
    # (d) negative control
    E = D.dropna(subset=['NC']).reset_index(drop=True); y, g = E.truth.to_numpy(), E.nct_id.to_numpy()
    A, Bo, NC = E.A.to_numpy(), E.B.to_numpy(), E.NC.to_numpy(); k = dict(n=len(E), trials=len(np.unique(g)))
    h1 = lambda i: np.mean(np.abs(A[i]) - np.abs(NC[i])); v1 = h1(np.arange(len(A))); c1 = boot_stat(h1, g, B)
    fr = lambda i: np.mean(np.abs(NC[i]) < np.abs(A[i])); v1b = fr(np.arange(len(A))); c1b = boot_stat(fr, g, B)
    ok1 = (c1[0] > 0) and (v1b > 0.5)
    out.append(row('d', ts, 'H1: mean(|A| - |NC|)', v1, *c1, **k, criterion='> 0, CI excludes 0', verdict='supported' if c1[0] > 0 else 'not supported'))
    out.append(row('d', ts, 'H1: fraction |NC| < |A|', v1b, *c1b, **k, criterion='> 0.5', verdict='supported' if v1b > 0.5 else 'not supported'))
    out.append(row('d', ts, 'H1: fraction |NC| = |A| (ties)', np.mean(np.isclose(np.abs(NC), np.abs(A))), **k))
    out.append(row('d', ts, 'H1 overall', float(ok1), **k, criterion='both H1 conditions', verdict='supported' if ok1 else 'not supported'))
    inert = (E.ctrl_kind == 'inert_control').to_numpy(); gi, Ai, Ni = g[inert], A[inert], NC[inert]; ki = dict(n=int(inert.sum()), trials=len(np.unique(gi)))
    out.append(row('d', ts, 'H2 (inert control): mean |A|', np.mean(np.abs(Ai)), **ki))
    out.append(row('d', ts, 'H2 (inert control): mean |NC|', np.mean(np.abs(Ni)), **ki))
    h2 = lambda i: np.mean(np.abs(Ai[i])) - np.mean(np.abs(Ni[i])); c2 = boot_stat(h2, gi, B)
    out.append(row('d', ts, 'H2 (inert control): mean |A| - mean |NC|', h2(np.arange(len(Ai))), *c2, **ki, criterion='> 0',
                   verdict='supported' if c2[0] > 0 else 'not supported'))
    sl = lambda i: np.sum(Ni[i] * Ai[i]) / np.sum(Ai[i] ** 2); cs = boot_stat(sl, gi, B)
    out.append(row('d', ts, 'H2 (inert control): slope of NC on A through origin', sl(np.arange(len(Ai))), *cs, **ki, criterion='< 0.5',
                   verdict='supported' if cs[1] < 0.5 else ('not supported' if cs[0] >= 0.5 else 'inconclusive')))
    ok2 = (c2[0] > 0) and (cs[1] < 0.5)
    out.append(row('d', ts, 'H2 overall', float(ok2), **ki, criterion='both H2 conditions', verdict='supported' if ok2 else 'not supported'))
    act = ((E.ctrl_kind == 'active_comparator') & (E.A > 0)).to_numpy(); ga, Aa, Na = g[act], A[act], NC[act]
    h3 = lambda i: np.mean(Na[i] - Aa[i]); c3 = boot_stat(h3, ga, B)
    out.append(row('d', ts, 'H3 (active comparator, A > 0): mean(NC - A)', h3(np.arange(len(Aa))), *c3, n=int(act.sum()), trials=len(np.unique(ga)),
                   criterion='< 0', verdict='supported' if c3[1] < 0 else 'not supported'))
    out.append(row('d', ts, 'H3: active-comparator contrasts (all A)', int((E.ctrl_kind == 'active_comparator').sum())))
    mu = E.train_mean.iloc[0]
    out.append(row('d', ts, 'secondary: skill(NC) vs training mean', *skill_ci(y, NC, np.full(len(y), mu), g, B), **k))
    out.append(row('d', ts, 'secondary: skill(A | NC)', *skill_ci(y, A, NC, g, B), **k))
    out.append(row('d', ts, 'secondary: skill(NC | B)', *skill_ci(y, NC, Bo, g, B), **k))
    out.append(row('d', ts, 'secondary: corr(NC, A)', np.corrcoef(NC, A)[0, 1], **k))
    out.append(row('d', ts, 'secondary: corr(NC, B)', np.corrcoef(NC, Bo)[0, 1], **k))
    return out


def rows():
    C21, C2 = load('v2.1'), load('v2'); R = prompt_rows(C21, 'v2.1') + prompt_rows(C2, 'v2')
    te2 = test(C2, 'binary', 'v2'); te21 = test(C21, 'binary', 'v2.1'); kept = te21[te21.v21_status == 'kept']
    D = te2[['cid', 'nct_id', 'delta', 'ctrl_kind']].rename(columns={'delta': 'truth'}).reset_index(drop=True)
    for col, src in (('A', ARMS['sonnet_posted']['v2']['A']()), ('B', ARMS['sonnet_posted']['v2']['B']()), ('SW', cr_frame('swap_A')),
                     ('RB', cr_frame('repeat_B')), ('NC', cr_frame('negctl_A'))):
        D[col] = src.point.reindex(D.cid).to_numpy(float)
    assert D[['A', 'B', 'SW', 'RB']].notna().all().all() and D.NC.isna().sum() == 3
    D['train_mean'] = train(C2, 'binary').delta.mean(); R += perturbation_rows(D, 'v2')
    Dk = D[D.cid.isin(kept.cid)].reset_index(drop=True)
    k21 = kept.set_index('cid'); assert np.allclose(Dk.truth, k21.delta.loc[Dk.cid]); Dk['ctrl_kind'] = k21.ctrl_kind.loc[Dk.cid].to_numpy()
    Dk['train_mean'] = train(C21, 'binary').delta.mean(); R += perturbation_rows(Dk, 'v2.1 kept')
    # full v2.1 test set: kept rows take the v2 run by cid, new/re-paired rows take the v21new_* runs by their v2.1 cid
    D21 = te21[['cid', 'nct_id', 'delta', 'ctrl_kind', 'v21_status']].rename(columns={'delta': 'truth'}).reset_index(drop=True)
    newA, newB = frame(long_read('v21new_sonnet_binary', 'posted_A')), frame(long_read('v21new_sonnet_binary', 'posted_B'))
    for col, old_, new_ in (('A', ARMS['sonnet_posted']['v2']['A'](), newA), ('B', ARMS['sonnet_posted']['v2']['B'](), newB),
                            ('SW', cr_frame('swap_A'), cr_frame('v21new_swap_A')), ('RB', cr_frame('repeat_B'), cr_frame('v21new_repeat_B')),
                            ('NC', cr_frame('negctl_A'), cr_frame('v21new_negctl_A'))):
        D21[col] = assemble(te21, 'v2.1', old_, new_).point.to_numpy(float)
    assert D21[['A', 'B', 'SW', 'RB']].notna().all().all() and D21.NC.isna().sum() <= 3, D21.isna().sum().to_dict()
    D21['train_mean'] = train(C21, 'binary').delta.mean(); R += perturbation_rows(D21, 'v2.1')
    return pd.DataFrame(R)


def texify(s):
    s = s.replace('|A|', '@A@').replace('|NC|', '@N@').replace(' | ', ' $\\mid$ ').replace('>', '$>$').replace('<', '$<$')
    return s.replace('@A@', '$|A|$').replace('@N@', '$|\\mathrm{NC}|$').replace('-A', '$-A$')


def tex_of(R):
    lab = {'a': '(a) prompt condition', 'b': '(b) arm swap', 'c': '(c) blind repeat', 'd': '(d) pre-registered negative control'}
    show_a = R[(R.panel == 'a') & (R.test_set == 'v2.1')]
    rest = R[(R.panel != 'a') & (R.test_set == 'v2.1') & ~R.statistic.str.startswith(('H3: active-comparator contrasts (all',))]
    L = []
    for p, G in [('a', show_a)] + [(p, rest[rest.panel == p]) for p in 'bcd']:
        L.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{lab[p]}, v2.1}}}} \\\\")
        for _, r in G.iterrows():
            val = '--' if r.statistic.endswith('overall') else m(r.value) + (f" {ci(r.lo, r.hi)}" if pd.notna(r.lo) else '')
            crit = r.criterion if isinstance(r.criterion, str) else ''
            ver = r.verdict if isinstance(r.verdict, str) else ''
            L.append(f"\\quad {texify(r.statistic)} & {val} & {'' if pd.isna(r.n) else int(r.n)} & "
                     f"{texify((crit + ': ' + ver) if ver else crit)} \\\\")
        L.append('\\midrule')
    L = L[:-1]
    cap = ("Language-model revision checks. (a) Interventional skill with protocol-only and posted-record prompts on the v2.1 binary "
           "test split, with the paired difference on one shared set of 2{,}000 trial resamples. (b)--(d) use Claude Sonnet~5 with the "
           "original (posted-record) prompts on the 878 v2.1 binary test contrasts (v2 and 816-contrast values in "
           "\\texttt{results/cr/lmrevision.csv}). (b) Condition A with the "
           "arms exchanged; A is the original condition-A prediction. (c) A second run of condition B. (d) Hypotheses H1--H3 exactly as "
           "pre-registered before any negative-control call (\\texttt{results/llm/cr/negative\\_control\\_preregistration.md}): the "
           "treatment is replaced by loratadine (cardiometabolic trials) or ezetimibe (all other areas); NC is the substituted prediction; "
           "refused calls (3 on v2) are excluded. Intervals are 95\\% trial-level bootstrap (2{,}000 resamples).")
    return ("\\begin{table}[h]\\centering\\footnotesize\n\\caption{" + cap + "}\n\\label{tab:lmrevision}\n"
            "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lcrl}\\toprule statistic & value & $n$ & pre-registered criterion / outcome \\\\ \\midrule\n"
            + '\n'.join(L) + "\n\\bottomrule\\end{tabular}}\\end{table}\n")


if __name__ == '__main__':
    with Timer():
        R = rows(); write('lmrevision', R, tex_of(R))
        print(R[['panel', 'test_set', 'statistic', 'value', 'lo', 'hi', 'n', 'verdict']].round(3).to_string())
