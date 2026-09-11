"""v2 benchmark analyses: feature sets + model wrappers built on models_eval (featurize/fit_gbt/skill/boot_skill)."""
import numpy as np, pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from ivbench import models as me

ENDPOINT_TXT = ['outcome_title', 'outcome_desc', 'outcome_timeframe', 'conditions']
INTERV_TXT = ['treat_intervention', 'treat_intervention_desc', 'treat_arm_desc', 'treat_group', 'treat_group_desc',
              'ctrl_intervention', 'ctrl_arm_desc', 'ctrl_group', 'ctrl_group_desc']
POP_CAT = ['phase', 'sponsor_class', 'allocation', 'masking', 'purpose', 'area', 'sex']
POP_NUM = ['n_treat', 'n_ctrl', 'age_mean', 'frac_female', 'year_posted']

SPECS = {
    'blind':       dict(text=ENDPOINT_TXT, cat=POP_CAT, num=POP_NUM),
    'blind_ctrl':  dict(text=ENDPOINT_TXT, cat=POP_CAT + ['ctrl_kind'], num=POP_NUM),
    'full':        dict(text=ENDPOINT_TXT + INTERV_TXT, cat=POP_CAT + ['ctrl_kind', 'treat_intervention_types'], num=POP_NUM),
    'interv_only': dict(text=INTERV_TXT, cat=['treat_intervention_types'], num=[]),
}


def featurize_spec(train, test, spec):
    """Wrapper around me.featurize; handles the no-text case with the same ordinal/numeric encoding."""
    if spec['text']:
        Xtr, Xte, meta, _ = me.featurize(train, test, spec['text'], cat_cols=spec['cat'], num_cols=spec['num'])
        return Xtr, Xte
    cats = [c for c in spec['cat'] if c in train.columns]
    nums = [c for c in spec['num'] if c in train.columns]
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    Ctr = enc.fit_transform(train[cats].astype(str)) if cats else np.zeros((len(train), 0))
    Cte = enc.transform(test[cats].astype(str)) if cats else np.zeros((len(test), 0))
    Ntr = train[nums].astype(float).to_numpy(); Nte = test[nums].astype(float).to_numpy()
    return np.hstack([Ctr, Ntr]), np.hstack([Cte, Nte])


def fit_spec(train, test, spec, seed=0, target='delta'):
    Xtr, Xte = featurize_spec(train, test, spec)
    m, pred = me.fit_gbt(Xtr, train[target].to_numpy(), Xte, seed=seed)
    return pred


def run_models(train, test, names, seed=0, target='delta', specs=None):
    specs = specs or SPECS
    return {n: fit_spec(train, test, specs[n], seed=seed, target=target) for n in names}


def eval_row(y, pred, ref, groups, n_boot=1000, seed=0):
    s = me.skill(y, pred, ref)
    lo, hi = me.boot_skill(y, pred, ref, groups, n=n_boot, seed=seed)
    return float(s), lo, hi


def norm_name(s):
    """Normalize an intervention name string: strip type prefix, dose/route/salt tokens."""
    import re
    if not isinstance(s, str):
        return []
    out = []
    for part in s.split('|'):
        p = part.split(':', 1)[-1].lower()
        p = re.sub(r'\(.*?\)', ' ', p)
        p = re.sub(r'\d+(\.\d+)?\s*(mg|mcg|µg|ug|g|ml|iu|units?|%|mg/kg|mg/m2|mg/m\^2|mcg/kg|kg)\b', ' ', p)
        p = re.sub(r'\d+(\.\d+)?', ' ', p)
        toks = re.findall(r'[a-z][a-z0-9\-]+', p)
        toks = [t for t in toks if t not in STOP]
        if toks:
            out.append(' '.join(toks))
    return out


STOP = set('''mg mcg ug g ml iu unit units dose doses tablet tablets capsule capsules oral iv intravenous subcutaneous sc
im intramuscular topical inhaled inhalation injection injections infusion infusions solution suspension gel cream ointment
patch spray drops daily twice once weekly monthly bid tid qd qw q2w q4w q8w q12w per day week weeks days high low medium
hydrochloride hcl sodium potassium calcium sulfate sulphate acetate citrate mesylate maleate tartrate succinate fumarate
phosphate besylate bromide chloride dihydrate monohydrate hydrate hemihydrate tosylate malate the and or with plus of in
for to a an placebo matching arm group treatment drug biological device procedure other dietary supplement behavioral
radiation combination product extended release er xr sr la immediate delayed film coated fdc'''.split())


def jaccard(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if a | b else 0.0
