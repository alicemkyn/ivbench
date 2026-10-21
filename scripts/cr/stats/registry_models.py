"""Registry-model wrapper that reproduces ivbench.specs.fit_spec exactly but keeps the fitted transforms, so the same
fitted model can score new frames (holdout, swapped contrasts). Predictions on the test split are identical to run_models."""
from common import *
from ivbench.specs import SPECS

def fit_predictor(train, spec, seed=0, target='delta'):
    Xtr, _, meta, _ = me.featurize(train, train.iloc[:2], spec['text'], cat_cols=spec['cat'], num_cols=spec['num'])
    tv, svd, enc, cats, nums = meta
    m, _ = me.fit_gbt(Xtr, train[target].to_numpy(), Xtr[:2], seed=seed)
    def transform(df):
        Z = svd.transform(tv.transform(me.text_of(df, spec['text'])))
        return np.hstack([enc.transform(df[cats].astype(str)), df[nums].astype(float).to_numpy(), Z])
    return lambda df: m.predict(transform(df))

def fit_family(C, fam, seeds=(0, 1, 2, 3, 4), names=('blind_ctrl', 'full'), specs=None):
    specs = specs or SPECS; tr = train_set(C, fam)
    return {(n, s): fit_predictor(tr, specs[n], seed=s) for s in seeds for n in names}
