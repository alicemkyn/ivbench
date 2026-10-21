"""Alternative registry-model architectures (paper Table 8): is the null/negative interventional skill a featurisation
artefact of the headline model, which embeds endpoint and intervention text in one shared 120-d SVD space?

  python scripts/cr/run_architectures.py --contrasts data/v2.1/benchmark_contrasts_v2.1_final.csv.gz \
      [--embeddings results/cr/embeddings_v21_binary.npz] --out results/cr/arch_raw.csv
  (camera-ready table wrapper: scripts/cr/tables/tab_arch.py)

Binary family, temporal split (split_time). Skill columns are against the global training mean; IS is FULL vs BLIND of the
same architecture (BLIND includes control type, as blind_ctrl). Seeds: GBT variants 0-4, ridge variants deterministic.
CIs: 1,000 trial-level resamples (ivbench.models.boot_skill, seed 0); 5-seed means use one shared set of resamples.
Architectures:
  GBT, separate SVD blocks   structure one-hot + endpoint SVD(120) | + intervention SVD(120) + intervention types
  Ridge, sparse TF-IDF        same blocks without SVD, alpha by trial-grouped 5-fold CV
  Ridge, intervention only    intervention TF-IDF block alone (skill only)
  Two-stage ridge             blind ridge + ridge on intervention text fitted to the blind model's out-of-fold residuals
  GBT, PubMedBERT             structure + 768-d endpoint embedding | + 768-d intervention embedding (needs --embeddings)
  Ridge, PubMedBERT           same features, ridge
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from ivbench import models as me
from ivbench.specs import POP_CAT, POP_NUM, ENDPOINT_TXT as ENDP, INTERV_TXT as IVT


def text_col(df, cols): return df[cols].fillna('').astype(str).agg(' '.join, axis=1)


def struct(tr, te, cat, num):
    enc = OneHotEncoder(handle_unknown='ignore', min_frequency=5)
    Xc_tr = enc.fit_transform(tr[cat].fillna('NA').astype(str)); Xc_te = enc.transform(te[cat].fillna('NA').astype(str))
    Xn_tr = tr[num].astype(float); mu, sd = Xn_tr.mean(), Xn_tr.std().replace(0, 1)
    Xn_tr = ((Xn_tr - mu) / sd).fillna(0).to_numpy(); Xn_te = ((te[num].astype(float) - mu) / sd).fillna(0).to_numpy()
    return sp.hstack([Xc_tr, sp.csr_matrix(Xn_tr)]).tocsr(), sp.hstack([Xc_te, sp.csr_matrix(Xn_te)]).tocsr()


def blocks(tr, te, groups, n_svd=None):
    out_tr, out_te = [], []
    for cols in groups:
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words='english', max_features=60000)
        A = v.fit_transform(text_col(tr, cols)); B = v.transform(text_col(te, cols))
        if n_svd:
            s = TruncatedSVD(min(n_svd, A.shape[1] - 1), random_state=0); A, B = s.fit_transform(A), s.transform(B)
        out_tr.append(A); out_te.append(B)
    return out_tr, out_te


def gbt(seed):
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, min_samples_leaf=20, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.15, random_state=seed)


def ridge_cv(Xtr, ytr, Xte, groups, alphas=(0.3, 1, 3, 10, 30, 100)):
    best = None
    for a in alphas:
        oof = np.zeros(len(ytr))
        for f_tr, f_va in GroupKFold(5).split(Xtr, ytr, groups=groups):
            oof[f_va] = Ridge(alpha=a).fit(Xtr[f_tr], ytr[f_tr]).predict(Xtr[f_va])
        mse = np.mean((oof - ytr) ** 2)
        if best is None or mse < best[0]: best = (mse, a)
    return Ridge(alpha=best[1]).fit(Xtr, ytr).predict(Xte), best[1], oof if False else None


def key_of(D):
    return (D.nct_id.astype(str) + '|' + D.outcome_title.fillna('').astype(str) + '|' + D.treat_group.fillna('').astype(str) + '|'
            + D.ctrl_group.fillna('').astype(str) + '|' + D.split_time.astype(str)).to_numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--contrasts', required=True); ap.add_argument('--embeddings', default=None)
    ap.add_argument('--out', required=True); ap.add_argument('--seeds', type=int, nargs='*', default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    C = pd.read_csv(a.contrasts, low_memory=False); C = C[C.family == 'binary']
    tr = C[C.split_time == 'train'].reset_index(drop=True); te = C[C.split_time == 'test'].reset_index(drop=True)
    ytr, yte = tr.delta.to_numpy(float), te.delta.to_numpy(float); g = te.nct_id.to_numpy(); ref = np.full(len(yte), ytr.mean())
    tri = np.unique(g); gi = {t: np.flatnonzero(g == t) for t in tri}; rng = np.random.default_rng(0)
    boots = [np.concatenate([gi[t] for t in rng.choice(tri, len(tri))]) for _ in range(1000)]
    rows = []

    def report(arch, preds_b, preds_f):
        """preds_*: list over seeds (or one element). Reports per-seed rows and, for >1 seed, the shared-resample mean."""
        for i, (pb, pf) in enumerate(zip(preds_b, preds_f)):
            iv = np.nan if pb is None else me.skill(yte, pf, pb)
            ivb = [] if pb is None else [me.skill(yte[ix], pf[ix], pb[ix]) for ix in boots]
            rows.append(dict(arch=arch, seed=i if len(preds_f) > 1 else 0, skill_blind=np.nan if pb is None else me.skill(yte, pb, ref),
                             skill_full=me.skill(yte, pf, ref), IS=iv, IS_lo=np.percentile(ivb, 2.5) if ivb else np.nan,
                             IS_hi=np.percentile(ivb, 97.5) if ivb else np.nan, n=len(yte), trials=len(tri)))
        if len(preds_f) > 1 and preds_b[0] is not None:
            m = [np.mean([me.skill(yte[ix], pf[ix], pb[ix]) for pb, pf in zip(preds_b, preds_f)]) for ix in boots]
            rows.append(dict(arch=arch, seed='mean', skill_blind=np.mean([me.skill(yte, pb, ref) for pb in preds_b]),
                             skill_full=np.mean([me.skill(yte, pf, ref) for pf in preds_f]),
                             IS=np.mean([me.skill(yte, pf, pb) for pb, pf in zip(preds_b, preds_f)]),
                             IS_lo=np.percentile(m, 2.5), IS_hi=np.percentile(m, 97.5), n=len(yte), trials=len(tri)))

    Sb_tr, Sb_te = struct(tr, te, POP_CAT + ['ctrl_kind'], POP_NUM)
    Sf_tr, Sf_te = struct(tr, te, POP_CAT + ['ctrl_kind', 'treat_intervention_types'], POP_NUM)
    (E_tr, I_tr), (E_te, I_te) = blocks(tr, te, [ENDP, IVT], n_svd=120)
    Xb_tr, Xb_te = np.hstack([Sb_tr.toarray(), E_tr]), np.hstack([Sb_te.toarray(), E_te])
    Xf_tr, Xf_te = np.hstack([Sf_tr.toarray(), E_tr, I_tr]), np.hstack([Sf_te.toarray(), E_te, I_te])
    pb = [gbt(s).fit(Xb_tr, ytr).predict(Xb_te) for s in a.seeds]; pf = [gbt(s).fit(Xf_tr, ytr).predict(Xf_te) for s in a.seeds]
    report('GBT, separate SVD blocks (120+120)', pb, pf)

    (Es_tr, Is_tr), (Es_te, Is_te) = blocks(tr, te, [ENDP, IVT], n_svd=None)
    Rb_tr, Rb_te = sp.hstack([Sb_tr, Es_tr]).tocsr(), sp.hstack([Sb_te, Es_te]).tocsr()
    Rf_tr, Rf_te = sp.hstack([Sf_tr, Es_tr, Is_tr]).tocsr(), sp.hstack([Sf_te, Es_te, Is_te]).tocsr()
    grp = tr.nct_id.to_numpy()
    pb3, ab, _ = ridge_cv(Rb_tr, ytr, Rb_te, grp); pf3, af, _ = ridge_cv(Rf_tr, ytr, Rf_te, grp)
    report(f'Ridge on sparse TF-IDF, no SVD (alpha {ab}/{af})', [pb3], [pf3])
    pi3, ai, _ = ridge_cv(Is_tr.tocsr(), ytr, Is_te.tocsr(), grp)
    report(f'Ridge, intervention text only (alpha {ai})', [None], [pi3])
    oof_b = np.zeros(len(ytr))
    for f_tr, f_va in GroupKFold(5).split(Rb_tr, ytr, groups=grp):
        oof_b[f_va] = Ridge(alpha=ab).fit(Rb_tr[f_tr], ytr[f_tr]).predict(Rb_tr[f_va])
    res = ytr - oof_b; best = None
    for al in (1, 3, 10, 30, 100):
        oof = np.zeros(len(ytr))
        for f_tr, f_va in GroupKFold(5).split(Is_tr, res, groups=grp):
            oof[f_va] = Ridge(alpha=al).fit(Is_tr[f_tr], res[f_tr]).predict(Is_tr[f_va])
        mse = np.mean((oof - res) ** 2); best = (mse, al) if best is None or mse < best[0] else best
    p2 = pb3 + Ridge(alpha=best[1]).fit(Is_tr, res).predict(Is_te)
    report(f'Two-stage: ridge blind + ridge(intervention) on OOF residuals (alpha {best[1]})', [pb3], [p2])

    if a.embeddings:
        z = np.load(a.embeddings, allow_pickle=True); pos = {k: i for i, k in enumerate(z['key'])}
        it, ie = np.array([pos[k] for k in key_of(tr)]), np.array([pos[k] for k in key_of(te)])
        Etr, Ete, Itr, Ite = z['E'][it], z['E'][ie], z['I'][it], z['I'][ie]
        Xb_tr, Xb_te = np.hstack([Sb_tr.toarray(), Etr]), np.hstack([Sb_te.toarray(), Ete])
        Xf_tr, Xf_te = np.hstack([Sf_tr.toarray(), Etr, Itr]), np.hstack([Sf_te.toarray(), Ete, Ite])
        pb = [gbt(s).fit(Xb_tr, ytr).predict(Xb_te) for s in a.seeds]; pf = [gbt(s).fit(Xf_tr, ytr).predict(Xf_te) for s in a.seeds]
        report('GBT on PubMedBERT embeddings (768+768)', pb, pf)
        pbE, aB, _ = ridge_cv(sp.csr_matrix(Xb_tr), ytr, sp.csr_matrix(Xb_te), grp); pfE, aF, _ = ridge_cv(sp.csr_matrix(Xf_tr), ytr, sp.csr_matrix(Xf_te), grp)
        report(f'Ridge on PubMedBERT embeddings (alpha {aB}/{aF})', [pbE], [pfE])
        piE, aI, _ = ridge_cv(sp.csr_matrix(Itr), ytr, sp.csr_matrix(Ite), grp)
        report(f'Ridge, PubMedBERT intervention embedding only (alpha {aI})', [None], [piE])
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
