"""Shared helpers for the camera-ready table scripts (scripts/cr/tables/tab_*.py).

Run every table script from the repository root: `python scripts/cr/tables/tab_<label>.py`. Scripts read only repository
files, make no network or language-model calls, and are deterministic (fixed seeds throughout).

Conventions (paper Sec. 3): skill = 1 - MSE(pred)/MSE(ref); interventional skill IS = skill(FULL | BLIND) with BLIND =
spec 'blind_ctrl' (the paper's \\blind, comparator type included). Trial-level bootstrap over nct_id with
ivbench.models.boot_skill: registry 1,000 resamples (seed 0), language models 2,000 resamples (seed 0). Five-seed registry
means use one shared set of 1,000 trial resamples applied to every seed (shared_boot_mean).
"""
import os, sys, time
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from ivbench import models as me                      # noqa: E402
from ivbench.specs import run_models                   # noqa: E402

V21 = os.path.join(ROOT, 'data', 'v2.1', 'benchmark_contrasts_v2.1_final.csv.gz')
V2 = os.path.join(ROOT, 'data', 'benchmark_contrasts.csv.gz')
LLM = os.path.join(ROOT, 'results', 'llm')
OUT_CSV = os.path.join(ROOT, 'results', 'cr')
OUT_TEX = os.path.join(ROOT, 'tables')
SEEDS = [0, 1, 2, 3, 4]
FAMS = ['binary', 'continuous']


# ---------------------------------------------------------------- data
def load(version):
    return pd.read_csv(V21 if version == 'v2.1' else V2, low_memory=False)


def train(C, fam):
    return C[(C.family == fam) & (C.split_time == 'train')].reset_index(drop=True)


def test(C, fam, version):
    """Test contrasts with cids. v2: llm_prompts.load_rows order (c0000.., k0000..); v2.1: file order, cid column."""
    d = C[(C.family == fam) & (C.split_time == 'test')]
    if version == 'v2':
        d = d.sort_values(['nct_id', 'outcome_title', 'treat_group']).reset_index(drop=True)
        d['cid'] = [f"{'c' if fam == 'binary' else 'k'}{i:04d}" for i in range(len(d))]
        return d
    return d.reset_index(drop=True)


# ---------------------------------------------------------------- registry models
def registry_preds(C, fam, version, names=('blind_ctrl', 'full'), seeds=SEEDS):
    """{(spec, seed): test predictions} from ivbench.specs.run_models (GBT), trained on the same version's train split."""
    tr, te = train(C, fam), test(C, fam, version)
    P = {}
    for s in seeds:
        for n, p in run_models(tr, te, list(names), seed=s).items():
            P[(n, s)] = p
    return tr, te, P


ALL_SPECS = ('blind', 'blind_ctrl', 'full', 'interv_only')


def registry_cached(C, fam, version):
    """registry_preds for all four specs and seeds 0-4, cached in results/cr/registry_preds_<version>_<fam>.csv.gz.
    The cache is a deterministic function of the repository data (first script to run writes it; delete it to refit)."""
    tr, te = train(C, fam), test(C, fam, version)
    path = os.path.join(OUT_CSV, f'registry_preds_{version}_{fam}.csv.gz')
    if os.path.exists(path):
        D = pd.read_csv(path, float_precision='round_trip')
        if len(D) == len(te) and (D.cid.to_numpy() == te.cid.to_numpy()).all() and np.allclose(D.truth, te.delta):
            return tr, te, {(n, s): D[f'{n}_s{s}'].to_numpy() for n in ALL_SPECS for s in SEEDS}
    _, _, P = registry_preds(C, fam, version, names=ALL_SPECS)
    D = pd.DataFrame({'cid': te.cid, 'nct_id': te.nct_id, 'truth': te.delta})
    for (n, s), p in P.items():
        D[f'{n}_s{s}'] = p
    os.makedirs(OUT_CSV, exist_ok=True); tmp = path + f'.tmp{os.getpid()}.csv.gz'
    D.to_csv(tmp, index=False, float_format='%.17g'); os.replace(tmp, path)
    return tr, te, {k: D[f'{k[0]}_s{k[1]}'].to_numpy() for k in P}


def shared_resamples(g, n=1000, seed=0):
    rng = np.random.default_rng(seed); u = np.unique(g); idx = {t: np.flatnonzero(g == t) for t in u}
    return [np.concatenate([idx[t] for t in rng.choice(u, size=len(u), replace=True)]) for _ in range(n)]


def shared_boot_mean(y, preds, refs, g, n=1000, seed=0, boots=None):
    """Mean over seeds of skill(pred_s | ref_s); 95% interval from ONE shared set of trial resamples."""
    boots = boots if boots is not None else shared_resamples(g, n, seed)
    out = [np.mean([me.skill(y[ii], a[ii], b[ii]) for a, b in zip(preds, refs)]) for ii in boots]
    point = float(np.mean([me.skill(y, a, b) for a, b in zip(preds, refs)]))
    return point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def boot_stat(stat, g, n=2000, seed=0):
    """95% trial-level bootstrap interval of stat(idx) over nct_id (same resampling scheme as ivbench.models.boot_skill)."""
    g = np.asarray(g); rng = np.random.default_rng(seed); u = np.unique(g); idx = {t: np.where(g == t)[0] for t in u}
    out = [stat(np.concatenate([idx[t] for t in rng.choice(u, size=len(u), replace=True)])) for _ in range(n)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def skill_ci(y, a, b, g, n, seed=0):
    y, a, b, g = map(np.asarray, (y, a, b, g))
    s = float(me.skill(y, a, b))
    lo, hi = me.boot_skill(y, a, b, g, n=n, seed=seed) if len(np.unique(g)) > 1 else (np.nan, np.nan)
    return s, lo, hi


# ---------------------------------------------------------------- language-model arms
def cr_read(variant):
    return pd.read_csv(os.path.join(LLM, 'cr', f'{variant}_predictions.csv'), low_memory=False)


def long_read(fn, cond):
    d = cr_read(fn); return d[d.condition == cond].set_index('cid')


def frame(df, p='point', lo='low', hi='high'):
    return pd.DataFrame({'point': df[p], 'low': df[lo], 'high': df[hi]}, index=df.index)


def paired_read(fn, side):
    d = pd.read_csv(os.path.join(LLM, fn)).set_index('cid')
    return frame(d, f'point_{side}', f'low_{side}', f'high_{side}')


def cr_frame(variant):
    return frame(cr_read(variant).set_index('cid'))


# v2 sources (keyed by v2 cid) and v2.1 new/re-paired sources (keyed by c21_/k21_ cid), per arm and condition.
ARMS = {
    'sonnet_protocol': dict(fam='binary', model='Claude Sonnet 5', prompt='protocol-only',
                            v2=dict(A=lambda: cr_frame('protocol_A'), B=lambda: cr_frame('protocol_B')),
                            new=dict(A=('v21new_sonnet_binary', 'protocol_A'), B=('v21new_sonnet_binary', 'protocol_B'))),
    'sonnet_posted': dict(fam='binary', model='Claude Sonnet 5', prompt='posted-record',
                          v2=dict(A=lambda: paired_read('llm_v2_paired.csv', 'A'), B=lambda: paired_read('llm_v2_paired.csv', 'B')),
                          new=dict(A=('v21new_sonnet_binary', 'posted_A'), B=('v21new_sonnet_binary', 'posted_B'))),
    'haiku_protocol': dict(fam='binary', model='Claude Haiku 4.5', prompt='protocol-only',
                           v2=dict(A=lambda: cr_frame('haiku_protocol_A'), B=lambda: cr_frame('haiku_protocol_B')),
                           new=dict(A=('v21new_haiku_binary', 'protocol_A'), B=('v21new_haiku_binary', 'protocol_B'))),
    'haiku_posted': dict(fam='binary', model='Claude Haiku 4.5', prompt='posted-record',
                         v2=dict(A=lambda: paired_read('llm_haiku_paired.csv', 'A'), B=lambda: paired_read('llm_haiku_paired.csv', 'B')),
                         new=dict(A=('v21new_haiku_binary', 'posted_A'), B=('v21new_haiku_binary', 'posted_B'))),
    'llama_protocol': dict(fam='binary', model='Llama 3.1 70B Instruct', prompt='protocol-only',
                           v2=dict(A=lambda: frame(long_read('llama70b_protocol_posted', 'protocol_A')),
                                   B=lambda: frame(long_read('llama70b_protocol_posted', 'protocol_B'))),
                           new=dict(A=('llama70b_protocol_posted', 'protocol_A'), B=('llama70b_protocol_posted', 'protocol_B'))),
    'llama_posted': dict(fam='binary', model='Llama 3.1 70B Instruct', prompt='posted-record',
                         v2=dict(A=lambda: frame(long_read('llama70b_protocol_posted', 'posted_A')),
                                 B=lambda: frame(long_read('llama70b_protocol_posted', 'posted_B'))),
                         new=dict(A=('llama70b_protocol_posted', 'posted_A'), B=('llama70b_protocol_posted', 'posted_B'))),
    'sonnet_cont_posted': dict(fam='continuous', model='Claude Sonnet 5', prompt='posted-record',
                               v2=dict(A=lambda: paired_read('llm_cont_paired.csv', 'A'), B=lambda: paired_read('llm_cont_paired.csv', 'B')),
                               new=dict(A=('v21new_sonnet_continuous', 'posted_A'), B=('v21new_sonnet_continuous', 'posted_B'))),
}


def assemble(te, version, old, new=None):
    """Per-test-row (point, low, high): v2 rows by v2 cid; v2.1 'kept' rows by v2 cid, other rows from `new` by cid."""
    if version == 'v2' or new is None:
        return old.reindex(te.cid).reset_index(drop=True)
    kept = (te.v21_status == 'kept').to_numpy()
    return pd.DataFrame(np.where(kept[:, None], old.reindex(te.cid).to_numpy(), new.reindex(te.cid).to_numpy()),
                        columns=old.columns)


def lm_arm(arm, version, C=None, te=None):
    """Test frame for one LM arm with columns A, B (points), A_low.., B_high.., truth, nct_id, cid (+ v21_status).
    v2.1: 'kept' rows take the v2 prediction by cid, other rows the v2.1 new/re-paired prediction by cid (the 8 excluded
    cids are absent from the final file and so drop out)."""
    spec = ARMS[arm]; fam = spec['fam']
    if te is None:
        te = test(C if C is not None else load(version), fam, version)
    keep = ['cid', 'nct_id', 'delta'] + (['v21_status'] if 'v21_status' in te else [])
    out = te[keep].rename(columns={'delta': 'truth'}).reset_index(drop=True)
    for side in 'AB':
        fn, cond = spec['new'][side]
        src = assemble(te, version, spec['v2'][side](), frame(long_read(fn, cond)) if version == 'v2.1' else None)
        out[side] = src['point'].to_numpy(float); out[side + '_low'] = src['low'].to_numpy(float)
        out[side + '_high'] = src['high'].to_numpy(float)
    assert out[['A', 'B']].notna().all().all(), (arm, version, int(out[['A', 'B']].isna().sum().sum()))
    return out


def lm_metrics(D, ref_mean, a='A', b='B'):
    """Skill vs training mean, IS (2,000 trial resamples, seed 0), MAE, 90%-interval coverage and width for a and b."""
    y, g = D.truth.to_numpy(), D.nct_id.to_numpy(); r = np.full(len(y), ref_mean)
    s, lo, hi = skill_ci(y, D[a], D[b], g, 2000)
    out = dict(n=len(D), trials=int(len(np.unique(g))), IS=s, IS_lo=lo, IS_hi=hi)
    for tag, side in (('A', a), ('B', b)):
        out[f'skill_{tag}'] = float(me.skill(y, D[side].to_numpy(), r))
        out[f'skill_{tag}_lo'], out[f'skill_{tag}_hi'] = me.boot_skill(y, D[side].to_numpy(), r, g, n=2000, seed=0)
        out[f'mae_{tag}'] = float(np.mean(np.abs(y - D[side])))
        if side + '_low' in D:
            out[f'cov_{tag}'] = float(np.mean((y >= D[side + '_low']) & (y <= D[side + '_high'])))
            out[f'width_{tag}'] = float(np.mean(D[side + '_high'] - D[side + '_low']))
    return out


# ---------------------------------------------------------------- formatting / output
def f(x, d=3):
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return '--'
    v = round(float(x), d)
    out = f'{0.0 if v == 0 else v:.{d}f}'
    return f'${out}$' if out.startswith('-') and not _in_math else out


_in_math = False  # negative plain values are wrapped in $...$ so the minus sign is typeset as a minus; inside \ci and m() the macro already supplies math mode


def ci(lo, hi, d=3):
    global _in_math
    _in_math = True
    try:
        return f'\\ci{{{f(lo, d)}}}{{{f(hi, d)}}}'
    finally:
        _in_math = False


def m(x, d=3):
    """Value in math mode, as the paper writes signed IS values ($-0.123$)."""
    global _in_math
    _in_math = True
    try:
        return f'${f(x, d)}$'
    finally:
        _in_math = False


def comma(n):
    return f'{int(n):,}'.replace(',', '{,}')


def write(label, df, tex, extra=None):
    os.makedirs(OUT_CSV, exist_ok=True); os.makedirs(OUT_TEX, exist_ok=True)
    csv = os.path.join(OUT_CSV, f'{label}.csv'); df.to_csv(csv, index=False)
    texf = os.path.join(OUT_TEX, f'{label}.tex')
    assert '\u2014' not in tex, 'em dash in table'
    with open(texf, 'w') as fh:
        fh.write(tex.rstrip() + '\n')
    for p, d in (extra or {}).items():
        d.to_csv(os.path.join(OUT_CSV, p), index=False)
    print('wrote', os.path.relpath(csv, ROOT), os.path.relpath(texf, ROOT))


class Timer:
    def __enter__(self):
        self.t = time.time(); return self

    def __exit__(self, *a):
        print(f'runtime_s {time.time() - self.t:.1f}')
