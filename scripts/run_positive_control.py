"""Positive-control and robustness experiments for the registry-trained simulator (paper Sec. 5.3; Tables 1, 6, 7,
11, 12 and the noise ceiling in Sec. 3).

  python scripts/run_positive_control.py --out results/positive_control [--seeds 0 1 2 3 4] [--n-boot 1000]

Writes, in the same schema as the released tables:
  v2_ablation_seeds.csv   blind / blind_ctrl / full / interv_only skill vs the training mean, and full vs blind_ctrl
                          (interventional skill), binary risk difference for every seed and continuous SMD for seed 0
  v2_noise_ceiling.csv    sampling-noise ceiling per family and the seed-0 models' position relative to it
  v2_robustness.csv       alternative temporal cuts (retrained) and comparator / phase / text-richness strata
                          (headline models evaluated on the stratum, and retrained within it)
  v2_level_task.csv       arm-level task: predicting each test arm's own outcome
  v2_same_molecule.csv    same-molecule retrieval positive control, `name` variant (keys = normalised treatment
                          intervention names, same comparator kind, residual mode, k = 3), reproducing the
                          normalization=name rows of the released table (n = 276, IS = -0.039). The paper's headline
                          `union` variant (0.006, n = 358) additionally keys on trial-level MeSH intervention terms
                          from the raw harvest (interventionBrowseModule), which are not carried into the released
                          contrast table; regenerate data/trials_raw.jsonl.gz with scripts/run_pipeline.sh to rebuild it.
With the defaults (seeds 0-4, 1000 bootstrap resamples) the ablation, noise-ceiling, robustness and level-task files
reproduce the released tables bit-for-bit, point estimates and intervals alike (verified).
"""
import argparse, json, os, sys, re
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ivbench import models as me
from ivbench.specs import SPECS, ENDPOINT_TXT, INTERV_TXT, POP_CAT, POP_NUM, run_models, eval_row, norm_name

ROOT = os.path.join(os.path.dirname(__file__), '..')
MODELS = ['blind', 'blind_ctrl', 'full', 'interv_only']


def split(C, family, cut=2023):
    D = C[C.family == family]
    return D[D.year_posted < cut].reset_index(drop=True), D[D.year_posted >= cut].reset_index(drop=True)


def ablation(C, seeds, nb):
    rows = []
    for family, fseeds in [('binary', seeds), ('continuous', [seeds[0]])]:
        tr, te = split(C, family)
        y, g, ref = te.delta.to_numpy(), te.nct_id.to_numpy(), np.full(len(te), tr.delta.mean())
        for s in fseeds:
            P = run_models(tr, te, MODELS, seed=s)
            for m in MODELS:
                sk, lo, hi = eval_row(y, P[m], ref, g, n_boot=nb, seed=0)
                rows.append(dict(family=family, seed=s, model=m, reference='train_mean', skill=sk, ci_low=lo, ci_high=hi,
                                 n_test=len(te), n_trials=te.nct_id.nunique()))
            sk, lo, hi = eval_row(y, P['full'], P['blind_ctrl'], g, n_boot=nb, seed=0)
            rows.append(dict(family=family, seed=s, model='full', reference='blind_ctrl', skill=sk, ci_low=lo, ci_high=hi,
                             n_test=len(te), n_trials=te.nct_id.nunique()))
            print(f'  ablation {family} seed {s}: IS = {sk:+.4f} [{lo:+.3f}, {hi:+.3f}]', flush=True)
    return pd.DataFrame(rows)


def sampling_var(te, family):
    n1, n2 = te.n_treat.to_numpy(float), te.n_ctrl.to_numpy(float)
    if family == 'binary':
        p1 = np.clip(te.y_treat.to_numpy(float), 0.5 / n1, 1 - 0.5 / n1)
        p2 = np.clip(te.y_ctrl.to_numpy(float), 0.5 / n2, 1 - 0.5 / n2)
        return p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
    d = te.delta.to_numpy(float)
    return (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))


def noise_ceiling(C, AB, nb):
    rows = []
    for family in ['binary', 'continuous']:
        tr, te = split(C, family)
        sv = sampling_var(te, family); y = te.delta.to_numpy(); g = te.nct_id.to_numpy()
        mse_mean = float(np.mean((y - tr.delta.mean()) ** 2)); tot = float(np.var(y))
        rel = 1 - sv.mean() / tot
        rng = np.random.default_rng(0); trials = np.unique(g); boots = []
        for _ in range(nb):
            pick = rng.choice(trials, len(trials), replace=True)
            idx = np.concatenate([np.flatnonzero(g == t) for t in pick])
            boots.append(1 - sv[idx].mean() / np.var(y[idx]))
        base = dict(family=family, n_test=len(te), n_trials=te.nct_id.nunique(), mean_sampling_var=float(sv.mean()),
                    median_sampling_var=float(np.median(sv)), total_var_delta=tot, mse_train_mean=mse_mean, reliability=float(rel),
                    reliability_ci_low=float(np.percentile(boots, 2.5)), reliability_ci_high=float(np.percentile(boots, 97.5)),
                    ceiling_skill_vs_mean=float(1 - sv.mean() / mse_mean), frac_noise_dominated=float(np.mean(sv > tot)))
        for _, r in AB[(AB.family == family) & (AB.seed == AB.seed.min()) & (AB.reference == 'train_mean') & (AB.model != 'interv_only')].iterrows():
            mse_model = mse_mean * (1 - r.skill)
            rows.append(dict(base, model=r.model, seed=int(r.seed), mse_model=mse_model, skill_vs_mean=r.skill,
                             frac_of_ceiling=r.skill / base['ceiling_skill_vs_mean'], attainable_remaining=base['ceiling_skill_vs_mean'] - r.skill,
                             max_interventional_skill_given_ref=1 - sv.mean() / mse_model))
        print(f'  noise ceiling {family}: reliability {rel:.3f}, ceiling skill {base["ceiling_skill_vs_mean"]:.3f}', flush=True)
    return pd.DataFrame(rows)


def robustness(C, nb, seed=0):
    rows = []
    tr, te = split(C, 'binary')
    P = run_models(tr, te, ['blind_ctrl', 'full'], seed=seed)

    def row(setting, trn, tst, pb, pf, cut, mode):
        y, g = tst.delta.to_numpy(), tst.nct_id.to_numpy()
        sk, lo, hi = eval_row(y, pf, pb, g, n_boot=nb, seed=0); ref = np.full(len(tst), trn.delta.mean())
        rows.append(dict(setting=setting, n_train=len(trn), n_test=len(tst), n_trials=tst.nct_id.nunique(),
                         blind_ctrl_skill=me.skill(y, pb, ref), full_skill=me.skill(y, pf, ref), interventional_skill=sk,
                         ci_low=lo, ci_high=hi, cut_year=cut, mode=mode))
        print(f'  robustness [{mode}] {setting}: IS = {sk:+.4f} [{lo:+.3f}, {hi:+.3f}]', flush=True)

    row(f'headline seed={seed} cut=2023', tr, te, P['blind_ctrl'], P['full'], 2023, 'headline')
    for cut in [2021, 2022, 2024]:
        trc, tec = split(C, 'binary', cut); Pc = run_models(trc, tec, ['blind_ctrl', 'full'], seed=seed)
        row(f'cut={cut}', trc, tec, Pc['blind_ctrl'], Pc['full'], cut, 'retrained')
    rich = lambda D: (D.treat_arm_matched.astype(str).str.lower().isin(['true', '1'])) & D.treat_intervention_desc.fillna('').str.strip().ne('')
    strata = [('ctrl_kind=inert_control', lambda D: D.ctrl_kind == 'inert_control'),
              ('ctrl_kind=active_comparator', lambda D: D.ctrl_kind == 'active_comparator'),
              ('phase contains PHASE3', lambda D: D.phase.astype(str).str.contains('PHASE3')),
              ('phase PHASE2 only', lambda D: D.phase.astype(str) == 'PHASE2'),
              ('rich intervention text (arm matched & desc non-empty)', rich)]
    for name, f in strata:
        mte = f(te).to_numpy()
        row(name, tr, te[mte].reset_index(drop=True), P['blind_ctrl'][mte], P['full'][mte], 2023, 'headline models evaluated on stratum')
        trs, tes = tr[f(tr)].reset_index(drop=True), te[mte].reset_index(drop=True)
        Ps = run_models(trs, tes, ['blind_ctrl', 'full'], seed=seed)
        row(name, trs, tes, Ps['blind_ctrl'], Ps['full'], 2023, 'retrained within stratum')
    return pd.DataFrame(rows)


ARM_ENDP = ['outcome_title', 'outcome_desc', 'outcome_timeframe', 'conditions']
ARM_INTV = ['intervention_names', 'intervention_desc', 'arm_desc', 'group_title', 'group_desc']
ARM_CAT = ['phase', 'sponsor_class', 'allocation', 'masking', 'purpose', 'area', 'sex']
ARM_NUM = ['n_analyzed', 'age_mean', 'frac_female', 'year_posted']
ARM_SPECS = {'blind': dict(text=ARM_ENDP, cat=ARM_CAT + ['role'], num=ARM_NUM),
             'blind_norole': dict(text=ARM_ENDP, cat=ARM_CAT, num=ARM_NUM),
             'full': dict(text=ARM_ENDP + ARM_INTV, cat=ARM_CAT + ['role', 'intervention_types'], num=ARM_NUM)}


def level_task(A, nb, seed=0):
    D = A[A.family == 'binary']
    tr, te = D[D.year_posted < 2023].reset_index(drop=True), D[D.year_posted >= 2023].reset_index(drop=True)
    P = run_models(tr, te, list(ARM_SPECS), seed=seed, target='y', specs=ARM_SPECS)
    y, g, ref = te.y.to_numpy(), te.nct_id.to_numpy(), np.full(len(te), tr.y.mean())
    rows = []
    for m in ARM_SPECS:
        sk, lo, hi = eval_row(y, P[m], ref, g, n_boot=nb, seed=0)
        rows.append(dict(model=m, reference='train_mean', skill=sk, ci_low=lo, ci_high=hi, n_test=len(te), n_trials=te.nct_id.nunique()))
    for r in ['blind', 'blind_norole']:
        sk, lo, hi = eval_row(y, P['full'], P[r], g, n_boot=nb, seed=0)
        rows.append(dict(model='full', reference=r, skill=sk, ci_low=lo, ci_high=hi, n_test=len(te), n_trials=te.nct_id.nunique()))
    print(f"  level task: blind {rows[0]['skill']:.3f}, full {rows[2]['skill']:.3f}", flush=True)
    return pd.DataFrame(rows)


def same_molecule(C, nb, seed=0, k=3):
    """`name` variant: keys = normalised treatment-intervention names (fallback group label); a test contrast is matched
    to TRAIN contrasts sharing a key AND the same comparator kind. Prediction = blind_ctrl prediction + n/(n+k) * mean
    out-of-fold blind_ctrl residual of the matches. Reported against the blind_ctrl prediction (interventional skill)."""
    tr, te = split(C, 'binary')
    P = run_models(tr, te, ['blind_ctrl', 'full'], seed=seed)
    oof = np.zeros(len(tr))
    for f, (a, b) in enumerate(GroupKFold(5).split(tr, groups=tr.nct_id)):
        oof[b] = run_models(tr.iloc[a].reset_index(drop=True), tr.iloc[b].reset_index(drop=True), ['blind_ctrl'], seed=seed)['blind_ctrl']
    res = tr.delta.to_numpy() - oof
    keys = lambda D: [set(norm_name(a) or norm_name(b)) for a, b in zip(D.treat_intervention, D.treat_group)]
    ktr, kte = keys(tr), keys(te)
    index = {}
    for i, ks in enumerate(ktr):
        for kk in ks: index.setdefault((kk, tr.ctrl_kind.iat[i]), []).append(i)
    pred, mask, nmatch = P['blind_ctrl'].copy(), np.zeros(len(te), bool), []
    for j, ks in enumerate(kte):
        m = sorted({i for kk in ks for i in index.get((kk, te.ctrl_kind.iat[j]), []) if tr.nct_id.iat[i] != te.nct_id.iat[j]})
        if m:
            mask[j] = True; nmatch.append(len(m)); pred[j] += len(m) / (len(m) + k) * res[m].mean()
    y, g = te.delta.to_numpy()[mask], te.nct_id.to_numpy()[mask]
    sk, lo, hi = eval_row(y, pred[mask], P['blind_ctrl'][mask], g, n_boot=nb, seed=0)
    gk, glo, ghi = eval_row(y, P['full'][mask], P['blind_ctrl'][mask], g, n_boot=nb, seed=0)
    ref = np.full(mask.sum(), tr.delta.mean())
    print(f'  same-molecule retrieval: n={mask.sum()}, IS = {sk:+.4f} [{lo:+.3f}, {hi:+.3f}] (GBT on subset {gk:+.4f})', flush=True)
    return pd.DataFrame([dict(normalization='name', same_ctrl_kind=True, mode='residual', k=k, n_subset=int(mask.sum()),
                              n_trials_subset=int(pd.Series(g).nunique()), median_n_matches=float(np.median(nmatch)),
                              retrieval_interventional_skill=sk, ci_low=lo, ci_high=hi, gbt_interventional_skill_same_subset=gk,
                              gbt_ci_low=glo, gbt_ci_high=ghi, retrieval_skill_vs_mean=me.skill(y, pred[mask], ref),
                              blind_skill_vs_mean_subset=me.skill(y, P['blind_ctrl'][mask], ref), full_skill_vs_mean_subset=me.skill(y, P['full'][mask], ref))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contrasts', default=os.path.join(ROOT, 'data', 'benchmark_contrasts.csv.gz'))
    ap.add_argument('--arms', default=os.path.join(ROOT, 'data', 'benchmark_arms.csv.gz'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'positive_control'))
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--only', nargs='+', default=['ablation', 'ceiling', 'robustness', 'level', 'molecule'])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    C = pd.read_csv(a.contrasts, low_memory=False)
    if 'ablation' in a.only or 'ceiling' in a.only:
        AB = ablation(C, a.seeds, a.n_boot); AB.to_csv(os.path.join(a.out, 'v2_ablation_seeds.csv'), index=False)
    if 'ceiling' in a.only:
        noise_ceiling(C, AB, a.n_boot).to_csv(os.path.join(a.out, 'v2_noise_ceiling.csv'), index=False)
    if 'robustness' in a.only:
        robustness(C, a.n_boot, a.seeds[0]).to_csv(os.path.join(a.out, 'v2_robustness.csv'), index=False)
    if 'level' in a.only:
        level_task(pd.read_csv(a.arms, low_memory=False), a.n_boot, a.seeds[0]).to_csv(os.path.join(a.out, 'v2_level_task.csv'), index=False)
    if 'molecule' in a.only:
        same_molecule(C, a.n_boot, a.seeds[0]).to_csv(os.path.join(a.out, 'v2_same_molecule.csv'), index=False)


if __name__ == '__main__':
    main()
