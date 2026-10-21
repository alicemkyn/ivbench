"""Table 4 (tab:lmruns): all language-model runs.

  python scripts/cr/tables/tab_lmruns.py

Upper block: the seven benchmark arms on v2.1 (protocol-only and posted-record prompts; continuous uses the posted-record
prompt). Lower block: perturbation runs with the original (posted-record) prompts. Arm swap, repeat run of B, negative control and
class guess were run on the 1,020 v2 binary test contrasts and on the 62 v2.1 new/re-paired contrasts that survive in the
final v2.1 file (v21new_* files; 66 were targeted, 4 were removed by the post-audit exclusions), so
they are reported on the full 878-contrast v2.1 test set; the repeat run of A exists for v2 only and is reported on the
816 v2.1 'kept' contrasts. The CSV also gives every row on v2 and on the kept subset.
Skill against the family's training mean; IS = skill(A | B), 2,000 trial resamples (seed 0); MAE, coverage of the
self-stated 90% interval and mean width as A / B.
  repeat run of A   results/llm/llm_repeat_A_predictions_full.csv, scored against the original B
  repeat run of B   results/llm/cr/repeat_B_predictions.csv, original A scored against it
  negative control  results/llm/cr/negctl_A_predictions.csv (pre-registered drug substitution) as A vs the original B;
                    3 refusals (no prediction) are excluded
  arm swap          results/llm/cr/swap_A_predictions.csv: condition A with arms exchanged, scored against the negated truth
                    (the swapped contrast's true value) and the negated training mean
  class guess       results/llm/cr/classguess_B + classjudge(_perm): class match against the true vs a permuted treatment
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *  # noqa: E402,F403

ARM_ROWS = ['sonnet_protocol', 'sonnet_posted', 'haiku_protocol', 'haiku_posted', 'llama_protocol', 'llama_posted', 'sonnet_cont_posted']


def perturbation_frames(te, version='v2'):
    """Binary test frames for the perturbation runs. version='v2': the 1,020 v2 contrasts by v2 cid. version='v2.1': the full
    878-contrast v2.1 test set, kept rows by v2 cid and new/re-paired rows from the v21new_* runs; the repeat run of A was not
    extended to v2.1 and is omitted there."""
    base = te[['cid', 'nct_id', 'delta']].rename(columns={'delta': 'truth'}).reset_index(drop=True)
    if version == 'v2':
        oA, oB = assemble(te, 'v2', ARMS['sonnet_posted']['v2']['A']()), assemble(te, 'v2', ARMS['sonnet_posted']['v2']['B']())
        rA = assemble(te, 'v2', pd.read_csv(os.path.join(LLM, 'llm_repeat_A_predictions_full.csv')).set_index('cid')[['point', 'low', 'high']])
        rB, nc, sw = (assemble(te, 'v2', cr_frame(v)) for v in ('repeat_B', 'negctl_A', 'swap_A'))
    else:
        newA, newB = frame(long_read('v21new_sonnet_binary', 'posted_A')), frame(long_read('v21new_sonnet_binary', 'posted_B'))
        oA, oB = assemble(te, 'v2.1', ARMS['sonnet_posted']['v2']['A'](), newA), assemble(te, 'v2.1', ARMS['sonnet_posted']['v2']['B'](), newB)
        rA = None
        rB, nc, sw = (assemble(te, 'v2.1', cr_frame(v), cr_frame('v21new_' + v)) for v in ('repeat_B', 'negctl_A', 'swap_A'))

    def mk(a, b, neg=False):
        D = base.copy()
        if neg:
            D['truth'] = -D['truth']
        for side, src in (('A', a), ('B', b)):
            if src is None:
                continue
            D[side] = src.point.to_numpy(float); D[side + '_low'] = src.low.to_numpy(float); D[side + '_high'] = src.high.to_numpy(float)
        return D.dropna(subset=[c for c in ('A', 'B') if c in D]).reset_index(drop=True)
    out = {'Claude Sonnet 5, repeat run A vs original B': mk(rA, oB)} if rA is not None else {}
    out.update({'Claude Sonnet 5, original A vs repeat run B': mk(oA, rB),
            'Claude Sonnet 5, negative control (drug substituted) vs original B': mk(nc, oB),
            'Claude Sonnet 5, arms swapped (vs negated truth)': mk(sw, None, neg=True)})
    return out


def class_guess(te2, cids=None):
    J, P = cr_read('classjudge').set_index('cid'), cr_read('classjudge_perm').set_index('cid')
    Jn, Pn = cr_read('v21new_classjudge').set_index('cid'), cr_read('v21new_classjudge_perm').set_index('cid')
    cols = ['class_match', 'mechanism_match', 'exact_agent', 'parse_ok']
    J, P = pd.concat([J[cols], Jn[cols]]), pd.concat([P[cols], Pn[cols]])
    ids = te2.cid if cids is None else pd.Index(cids)
    j, p = J.reindex(ids), P.reindex(ids); assert j.parse_ok.notna().all(), 'class-guess row missing for a cid'
    j = j[j.parse_ok.astype(bool)]; p = p[p.parse_ok.astype(bool)]
    return dict(n=int(len(j)), n_perm=int(len(p)), class_match=float(j.class_match.astype(bool).mean()),
                mechanism_match=float(j.mechanism_match.astype(bool).mean()), exact_agent=float(j.exact_agent.astype(bool).mean()),
                class_match_perm=float(p.class_match.astype(bool).mean()), mechanism_match_perm=float(p.mechanism_match.astype(bool).mean()),
                exact_agent_perm=float(p.exact_agent.astype(bool).mean()))


def rows():
    C21, C2 = load('v2.1'), load('v2'); R = []
    mu = {(v, fam): train(C, fam).delta.mean() for v, C in (('v2.1', C21), ('v2', C2)) for fam in FAMS}
    for v, C in (('v2.1', C21), ('v2', C2)):
        for arm in ARM_ROWS:
            a = ARMS[arm]; r = lm_metrics(lm_arm(arm, v, C=C), mu[(v, a['fam'])])
            R.append(dict(block='arm', test_set=v, run=f"{a['model']}, {a['prompt']}", model=a['model'], prompt=a['prompt'], family=a['fam'], **r))
    te2 = test(C2, 'binary', 'v2'); te21 = test(C21, 'binary', 'v2.1'); kept = te21[te21.v21_status == 'kept']
    t21 = kept.set_index('cid').delta
    for name, D in perturbation_frames(te2).items():
        neg = 'swapped' in name
        for ts, d in (('v2', D), ('v2.1 kept', D[D.cid.isin(kept.cid)].reset_index(drop=True))):
            if ts != 'v2':
                assert np.allclose(np.abs(d.truth), np.abs(t21.loc[d.cid].to_numpy()))
            ref = (-1 if neg else 1) * mu[('v2' if ts == 'v2' else 'v2.1', 'binary')]
            if neg:
                y, g = d.truth.to_numpy(), d.nct_id.to_numpy()
                r = dict(n=len(d), trials=int(d.nct_id.nunique()), skill_A=me.skill(y, d.A.to_numpy(), np.full(len(y), ref)),
                         mae_A=float(np.mean(np.abs(y - d.A))), cov_A=float(np.mean((y >= d.A_low) & (y <= d.A_high))),
                         width_A=float(np.mean(d.A_high - d.A_low)))
            else:
                r = lm_metrics(d, ref)
            R.append(dict(block='perturbation', test_set=ts, run=name, model='Claude Sonnet 5', prompt='posted-record', family='binary', **r))
    mu21 = mu[('v2.1', 'binary')]
    for name, d in perturbation_frames(te21, 'v2.1').items():
        neg = 'swapped' in name; ref = (-1 if neg else 1) * mu21
        if neg:
            y, g = d.truth.to_numpy(), d.nct_id.to_numpy()
            r = dict(n=len(d), trials=int(d.nct_id.nunique()), skill_A=me.skill(y, d.A.to_numpy(), np.full(len(y), ref)),
                     mae_A=float(np.mean(np.abs(y - d.A))), cov_A=float(np.mean((y >= d.A_low) & (y <= d.A_high))),
                     width_A=float(np.mean(d.A_high - d.A_low)))
        else:
            r = lm_metrics(d, ref)
        R.append(dict(block='perturbation', test_set='v2.1', run=name, model='Claude Sonnet 5', prompt='posted-record', family='binary', **r))
    for ts, cids in (('v2', None), ('v2.1 kept', kept.cid), ('v2.1', te21.cid)):
        R.append(dict(block='class guess', test_set=ts, run='Claude Haiku 4.5, class guess from condition-B prompt (judge: Haiku 4.5)',
                      model='Claude Haiku 4.5', prompt='posted-record (B)', family='binary', **class_guess(te2, cids)))
    return pd.DataFrame(R)


def tex_of(R):
    SHORT = [('Claude Sonnet 5', 'Sonnet 5'), ('Claude Haiku 4.5', 'Haiku 4.5'), (' Instruct', ''), ('protocol-only', 'protocol'),
             ('posted-record', 'posted'), ('continuous SMD', 'cont.'), ('negative control', 'neg. control'), ('intervention', 'interv.')]
    def line(r, label, tsl):
        for a_, b_ in SHORT: label = label.replace(a_, b_)
        g = lambda k, d=2: f(r.get(k), d) if k in r and pd.notna(r.get(k)) else '--'
        isc = f"{m(r.IS)} {ci(r.IS_lo, r.IS_hi)}" if pd.notna(r.get('IS', np.nan)) else '--'
        pair = lambda k: f"{g(k + '_A', 3 if k == 'mae' else 2)} / {g(k + '_B', 3 if k == 'mae' else 2)}"
        return (f"{label} & {r.family} & {tsl} & {comma(int(r.n))} & {comma(int(r.trials))} & {g('skill_A', 3)} & {g('skill_B', 3)} & {isc} & "
                f"{pair('cov')} \\\\")
    A = R[(R.block == 'arm') & (R.test_set == 'v2.1')]
    PA = R[(R.block == 'perturbation') & (R.test_set == 'v2.1 kept') & R.run.str.contains('repeat run A')]
    P = R[(R.block == 'perturbation') & (R.test_set == 'v2.1')]
    cg = R[(R.block == 'class guess') & (R.test_set == 'v2.1')].iloc[0]
    L = ([line(r, r.run, 'v2.1') for _, r in A.iterrows()] + ['\\midrule'] + [line(r, r.run, 'v2.1 kept') for _, r in PA.iterrows()]
         + [line(r, r.run, 'v2.1') for _, r in P.iterrows()])
    cgtxt = (f"Class guess (Claude Haiku~4.5 from the condition-B prompt, judged by Haiku~4.5; v2.1, {int(cg.n)} contrasts): the drug class "
             f"matches the true treatment in {f(cg.class_match)} vs {f(cg.class_match_perm)} for a permuted treatment ({int(cg.n_perm)} judged), "
             f"mechanism {f(cg.mechanism_match)} vs {f(cg.mechanism_match_perm)}, exact agent {f(cg.exact_agent)} vs {f(cg.exact_agent_perm)}.")
    cap = ("All language-model runs. Upper block: benchmark arms on the v2.1 temporal test split; protocol-only prompts contain "
           "registered protocol fields only, posted-record prompts are the original prompts (continuous uses the posted-record prompt). "
           "Lower block: perturbation runs of Claude Sonnet~5 with the original (posted-record) prompts on the 878 v2.1 binary test "
           "contrasts (the 62 new or re-paired contrasts were run separately with identical prompts); the repeat run of condition~A "
           "exists for v2 only and is reported on the 816 v2.1 contrasts it covers (`v2.1 kept'). v2 values in \\texttt{results/cr/lmruns.csv}. Skill against "
           "the family's training mean; $\\IS$ = \\full\\ (A) vs \\blind\\ (B) with 95\\% trial-level bootstrap intervals "
           "(2{,}000 resamples); coverage of the self-stated 90\\% interval as A / B (MAE and mean width in the CSV). The "
           "negative-control row excludes 3 refusals; the arm-swap row is scored against the negated truth. Run names are abbreviated (full names in the CSV). " + cgtxt)
    return ("\\begin{table}[h]\\centering\\footnotesize\n\\caption{" + cap + "}\n\\label{tab:lmruns}\n"
            "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lllrrcccc}\\toprule run & family & test set & $n$ & trials & skill(A) & "
            "skill(B) & $\\IS$ & coverage A / B \\\\ \\midrule\n" + '\n'.join(L) + "\n\\bottomrule\\end{tabular}}\\end{table}\n")


if __name__ == '__main__':
    with Timer():
        R = rows(); write('lmruns', R, tex_of(R))
        print(R[['block', 'test_set', 'run', 'n', 'skill_A', 'skill_B', 'IS', 'IS_lo', 'IS_hi']].round(3).to_string())
