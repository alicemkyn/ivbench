"""Table tab:mol (paper Table 12): same-molecule retrieval positive control on benchmark v2.1 (binary), all variants.
A test contrast is matched to TRAIN contrasts (other trials) sharing an intervention key, optionally with the same comparator
kind. residual: blind prediction + n/(n+k) * mean out-of-fold blind residual of the n matches (5-fold GroupKFold by trial);
level: n/(n+k) * mean matched delta + k/(n+k) * training mean. Reported as interventional skill against the blind (with
comparator type) prediction, trial bootstrap 1,000 resamples seed 0; last column: gradient-boosting full model IS on the same
subset. Keys: name = ivbench.specs.norm_name of treatment intervention names (fallback group title); mesh = trial-level
intervention MeSH terms (data/cr/intervention_mesh_terms.csv, ClinicalTrials.gov derivedSection.interventionBrowseModule,
fetched 2026-10-17) whose term string occurs in the treatment intervention name or group title; union = name | mesh.
Headline (main text) variants: name keys (reproducible from the released data). MeSH-keyed variants go to tables/molmesh.tex, flagged provisional.
Reproduction status on v2 (CSV v2 rows carry the released values): name / same comparator kind rows reproduce
results/positive_control/v2_same_molecule.csv exactly; name / any comparator rows match the subset but not the retrieval
estimate; mesh and union rows cannot be reproduced because the original harvest-time MeSH term snapshot and term-to-arm rule are
not in the repository (re-fetched terms are used instead).
  python scripts/cr/tables/tab_mol.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common_b import ROOT, load, split, f, ci, n, write, me
from ivbench.specs import run_models, eval_row, norm_name
from sklearn.model_selection import GroupKFold
import numpy as np, pandas as pd

LABEL = 'mol'
MESH = pd.read_csv(os.path.join(ROOT, 'data', 'cr', 'intervention_mesh_terms.csv')).fillna('')
MESH_OF = dict(zip(MESH.nct_id, MESH.mesh_terms))


def keys(D, norm):
    name = [set(norm_name(a) or norm_name(b)) for a, b in zip(D.treat_intervention, D.treat_group)]
    if norm == 'name':
        return name
    mesh = []
    for nct, ti, tg in zip(D.nct_id, D.treat_intervention.fillna(''), D.treat_group.fillna('')):
        txt = (ti + ' ' + tg).lower()
        mesh.append({t.lower() for t in MESH_OF.get(nct, '').split('|') if t and t.lower() in txt})
    return mesh if norm == 'mesh' else [a | b for a, b in zip(name, mesh)]


def base_models(C, seed=0):
    tr, te = split(C, 'binary')
    P = run_models(tr, te, ['blind_ctrl', 'full'], seed=seed)
    oof = np.zeros(len(tr))
    for a, b in GroupKFold(5).split(tr, groups=tr.nct_id):
        oof[b] = run_models(tr.iloc[a].reset_index(drop=True), tr.iloc[b].reset_index(drop=True), ['blind_ctrl'], seed=seed)['blind_ctrl']
    return tr, te, P, tr.delta.to_numpy() - oof


def variant(tr, te, P, res, norm, same, mode, k):
    dtr = tr.delta.to_numpy(); ktr, kte = keys(tr, norm), keys(te, norm); index = {}
    for i, ks in enumerate(ktr):
        for kk in ks: index.setdefault((kk, tr.ctrl_kind.iat[i] if same else None), []).append(i)
    pred, mask, nm = P['blind_ctrl'].copy(), np.zeros(len(te), bool), []
    for j, ks in enumerate(kte):
        m = sorted({i for kk in ks for i in index.get((kk, te.ctrl_kind.iat[j] if same else None), []) if tr.nct_id.iat[i] != te.nct_id.iat[j]})
        if m:
            mask[j] = True; nm.append(len(m)); w = len(m) / (len(m) + k)
            pred[j] = pred[j] + w * res[m].mean() if mode == 'residual' else w * dtr[m].mean() + (1 - w) * dtr.mean()
    y, g = te.delta.to_numpy()[mask], te.nct_id.to_numpy()[mask]; ref = np.full(mask.sum(), dtr.mean())
    sk, lo, hi = eval_row(y, pred[mask], P['blind_ctrl'][mask], g, n_boot=1000, seed=0)
    gk, glo, ghi = eval_row(y, P['full'][mask], P['blind_ctrl'][mask], g, n_boot=1000, seed=0)
    return dict(normalization=norm, same_ctrl_kind=same, mode=mode, k=k, n_subset=int(mask.sum()), n_trials_subset=int(len(np.unique(g))),
                median_n_matches=float(np.median(nm)), retrieval_interventional_skill=sk, ci_low=lo, ci_high=hi,
                gbt_interventional_skill_same_subset=gk, gbt_ci_low=glo, gbt_ci_high=ghi, retrieval_skill_vs_mean=me.skill(y, pred[mask], ref),
                blind_skill_vs_mean_subset=me.skill(y, P['blind_ctrl'][mask], ref), full_skill_vs_mean_subset=me.skill(y, P['full'][mask], ref))


def compute(C):
    tr, te, P, res = base_models(C)
    return pd.DataFrame([variant(tr, te, P, res, nrm, s, md, k) for nrm in ['union', 'name', 'mesh'] for s in [True, False]
                         for md in ['residual', 'level'] for k in [3, 1]])


def vname(r):
    return f"{'primary_' if r.normalization == 'union' and r.same_ctrl_kind else ''}{r.normalization}_{'sameCtrl' if r.same_ctrl_kind else 'anyCtrl'}_k{r.k}_{r['mode']}"


def main():
    out = []
    for version in ['v2.1', 'v2']:
        R = compute(load(version)); R.insert(0, 'version', version); out.append(R)
    D = pd.concat(out, ignore_index=True); D.insert(1, 'variant', [vname(r) for _, r in D.iterrows()])
    ref = pd.read_csv(os.path.join(ROOT, 'results', 'positive_control', 'v2_same_molecule.csv'))
    m = D.version == 'v2'
    M = D[m].merge(ref, on=['normalization', 'same_ctrl_kind', 'mode', 'k'], how='left', suffixes=('', '_rel'))
    D.loc[m, 'v2_released_n_subset'] = M.n_subset_rel.to_numpy()
    D.loc[m, 'v2_released_retrieval_IS'] = M.retrieval_interventional_skill_rel.to_numpy()
    D.loc[m, 'v2_reproduces_released'] = (np.isclose(M.retrieval_interventional_skill, M.retrieval_interventional_skill_rel, rtol=0, atol=1e-12)
                                          & (M.n_subset == M.n_subset_rel)).to_numpy()
    def row(r):
        return (f"{r.variant.replace('_', chr(92) + '_')} & {r.n_subset} & {r.n_trials_subset} & {f(r.retrieval_interventional_skill)} "
                f"{ci(r.ci_low, r.ci_high)} & {f(r.gbt_interventional_skill_same_subset)} {ci(r.gbt_ci_low, r.gbt_ci_high)} \\\\")
    V = D[(D.version == 'v2.1') & (D.k == 3)]
    head = ("\\resizebox{\\linewidth}{!}{\\begin{tabular}{lrrcc}\\toprule variant & contrasts & trials & retrieval $\\IS$ & \\gbt\\ $\\IS$, same subset "
            "\\\\ \\midrule\n")
    # headline (main text): name-keyed variants only; keys are reproducible from the released data alone
    T = V[(V.normalization == 'name')]
    tex = ("\\begin{table}[h]\\centering\\footnotesize\n"
           "\\caption{Same-molecule retrieval positive control on v2.1 (binary test split, $k=3$ matched training contrasts of the same "
           "normalised treatment name). ``residual'' adds the shrunk mean out-of-fold \\blind\\ residual of the matched contrasts to the "
           "\\blind\\ prediction; ``level'' replaces the prediction by their mean delta shrunk towards the training mean. sameCtrl restricts "
           "matches to the same comparator kind. 95\\% trial-level bootstrap intervals (1{,}000 resamples). Last column: the gradient-boosting "
           "\\full\\ model's $\\IS$ on the same subset. Variants keyed on intervention MeSH terms are in Table~\\ref{tab:molmesh}.}\n"
           "\\label{tab:mol}\n" + head + '\n'.join(row(r) for _, r in T.iterrows()) + "\n\\bottomrule\\end{tabular}}\\end{table}")
    write(LABEL, D, tex)
    # appendix: MeSH-keyed variants, flagged as provisional (terms re-fetched in October 2026, not the harvest-time terms)
    T2 = V[V.normalization.isin(['union', 'mesh'])]
    tex2 = ("\\begin{table}[h]\\centering\\footnotesize\n"
            "\\caption{Same-molecule retrieval, MeSH-keyed variants (provisional). union keys = normalised treatment names $\\cup$ intervention "
            "MeSH terms; mesh keys = MeSH terms only. The harvest-time MeSH terms were not preserved; the terms used here were re-fetched from "
            "ClinicalTrials.gov in October 2026, so the matched subsets differ from those of the submitted version (on v2: 342 vs 358 contrasts "
            "for the union variant) and these rows should be read as indicative. Same conventions as Table~\\ref{tab:mol}.}\n"
            "\\label{tab:molmesh}\n" + head + '\n'.join(row(r) for _, r in T2.iterrows()) + "\n\\bottomrule\\end{tabular}}\\end{table}")
    with open(os.path.join(ROOT, 'tables', 'molmesh.tex'), 'w') as fh:
        fh.write(tex2 + '\n')

    print(D[m][['variant', 'n_subset', 'v2_released_n_subset', 'retrieval_interventional_skill', 'v2_released_retrieval_IS', 'v2_reproduces_released']].round(4).to_string())


if __name__ == '__main__':
    main()
