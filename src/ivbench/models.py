"""Phase 1: level vs contrast predictors, skill metrics, split conformal, selective prediction."""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder

RNG = np.random.default_rng(0)

STRUCT_CAT = ['phase', 'sponsor_class', 'allocation', 'masking', 'purpose', 'area', 'ctrl_kind',
              'treat_intervention_types', 'sex']
STRUCT_NUM = ['n_treat', 'n_ctrl', 'age_mean', 'frac_female', 'year_posted']


def text_of(df, cols):
    return df[cols].fillna('').agg(' \n '.join, axis=1)


def featurize(train, test, text_cols, n_svd=120, cat_cols=None, num_cols=None):
    """Shared feature matrix: ordinal-encoded categoricals + numerics + SVD of TF-IDF text."""
    global STRUCT_CAT, STRUCT_NUM
    _sc, _sn = STRUCT_CAT, STRUCT_NUM
    if cat_cols is not None:
        STRUCT_CAT = cat_cols
    if num_cols is not None:
        STRUCT_NUM = num_cols
    try:
        return _featurize(train, test, text_cols, n_svd)
    finally:
        STRUCT_CAT, STRUCT_NUM = _sc, _sn


def _featurize(train, test, text_cols, n_svd=120):
    tv = TfidfVectorizer(max_features=60000, ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                         strip_accents='unicode', stop_words='english')
    Xt_tr = tv.fit_transform(text_of(train, text_cols))
    Xt_te = tv.transform(text_of(test, text_cols))
    svd = TruncatedSVD(n_components=min(n_svd, Xt_tr.shape[1] - 1), random_state=0)
    Ztr, Zte = svd.fit_transform(Xt_tr), svd.transform(Xt_te)

    cats = [c for c in STRUCT_CAT if c in train.columns]
    nums = [c for c in STRUCT_NUM if c in train.columns]
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    Ctr = enc.fit_transform(train[cats].astype(str))
    Cte = enc.transform(test[cats].astype(str))
    Ntr = train[nums].astype(float).to_numpy()
    Nte = test[nums].astype(float).to_numpy()
    Xtr = np.hstack([Ctr, Ntr, Ztr])
    Xte = np.hstack([Cte, Nte, Zte])
    return Xtr, Xte, (tv, svd, enc, cats, nums), (Xt_tr, Xt_te)


def fit_gbt(Xtr, ytr, Xte, seed=0):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=None,
                                      min_samples_leaf=20, l2_regularization=1.0,
                                      early_stopping=True, validation_fraction=0.15, random_state=seed)
    m.fit(Xtr, ytr)
    return m, m.predict(Xte)


def retrieval_pred(Xt_tr, Xt_te, ytr, k=25):
    nn = NearestNeighbors(n_neighbors=min(k, Xt_tr.shape[0]), metric='cosine').fit(Xt_tr)
    d, idx = nn.kneighbors(Xt_te)
    w = np.clip(1 - d, 1e-6, None)
    return (ytr[idx] * w).sum(1) / w.sum(1)


def prior_pred(train, test, ytr, key_cols):
    """Group mean over a protocol key that carries no intervention information."""
    tr = train.copy(); tr['_y'] = ytr
    g = tr.groupby(key_cols)['_y'].agg(['mean', 'size'])
    g = g[g['size'] >= 5]['mean']
    keys = list(test[key_cols].itertuples(index=False, name=None))
    if len(key_cols) == 1:
        keys = [k[0] for k in keys]
    gm = ytr.mean()
    return np.array([g.get(k, gm) for k in keys])


# ---------------- metrics ----------------
def skill(y, pred, ref):
    """1 - MSE(pred)/MSE(ref): fraction of reference error removed. 0 = no better than reference."""
    return 1 - np.mean((y - pred) ** 2) / np.mean((y - ref) ** 2)


def boot_skill(y, pred, ref, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        out.append(skill(y[idx], pred[idx], ref[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ---------------- conformal + selective ----------------
def split_conformal(res_cal, alpha=0.1):
    """Constant-width interval: (1-alpha) quantile of calibration absolute residuals."""
    n = len(res_cal)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(np.abs(res_cal), min(q, 1.0)))


def normalized_conformal(res_cal, sigma_cal, alpha=0.1):
    n = len(res_cal)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(np.abs(res_cal) / np.clip(sigma_cal, 1e-6, None), min(q, 1.0)))


def selective_curve(y, pred, conf_rank, grid=None):
    """conf_rank: higher = model expects to be worse. Returns (coverage, MAE) as we drop the worst."""
    order = np.argsort(conf_rank)          # keep most confident first
    grid = grid if grid is not None else np.linspace(0.1, 1.0, 19)
    out = []
    for c in grid:
        k = max(1, int(round(c * len(y))))
        sel = order[:k]
        out.append((c, float(np.mean(np.abs(y[sel] - pred[sel]))), k))
    return pd.DataFrame(out, columns=['coverage', 'mae', 'n'])
