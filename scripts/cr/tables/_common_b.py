"""Shared helpers for the camera-ready table scripts (scripts/cr/tables/tab_<label>.py). Run scripts from the repo root."""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from ivbench import models as me                                   # noqa: E402
from ivbench.specs import run_models, eval_row                      # noqa: E402,F401

V2 = os.path.join(ROOT, 'data', 'benchmark_contrasts.csv.gz')
V21 = os.path.join(ROOT, 'data', 'v2.1', 'benchmark_contrasts_v2.1_final.csv.gz')
LLM = os.path.join(ROOT, 'results', 'llm')


def load(version='v2.1'):
    return pd.read_csv(V21 if version == 'v2.1' else V2, low_memory=False)


def split(C, family, cut=2023):
    """Temporal split by year results were posted (train < cut). For cut=2023 this equals the split_time column."""
    D = C[C.family == family]
    return D[D.year_posted < cut].reset_index(drop=True), D[D.year_posted >= cut].reset_index(drop=True)


def shared_boot_mean(y, preds, refs, g, n=1000, seed=0):
    """Percentile CI of the seed-mean skill, one shared set of trial resamples applied to every seed."""
    rng = np.random.default_rng(seed); u = np.unique(g); idx = {t: np.where(g == t)[0] for t in u}; out = []
    for _ in range(n):
        ii = np.concatenate([idx[t] for t in rng.choice(u, len(u), replace=True)])
        out.append(np.mean([me.skill(y[ii], a[ii], b[ii]) for a, b in zip(preds, refs)]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ---------------- language-model arms on the v2.1 binary test set ----------------
COLS = ['point_A', 'low_A', 'high_A', 'point_B', 'low_B', 'high_B']


def lm_arm_v21(arm, te):
    """FULL (A) and BLIND (B) prompt predictions (point/low/high) for the v2.1 binary test rows `te`, Sonnet 5.
    arm: 'protocol' (protocol-only prompts, headline) or 'posted' (original posted-record prompts).
    'kept' rows take the archived v2 prediction by cid; repaired / added rows take the v2.1 new-call prediction by cid."""
    if arm == 'protocol':
        A = pd.read_csv(os.path.join(LLM, 'cr', 'protocol_A_predictions.csv')).set_index('cid')
        B = pd.read_csv(os.path.join(LLM, 'cr', 'protocol_B_predictions.csv')).set_index('cid')
        old = pd.DataFrame({'point_A': A.point, 'low_A': A.low, 'high_A': A.high, 'point_B': B.point, 'low_B': B.low,
                            'high_B': B.high, 'truth': A.truth})
    elif arm == 'posted':
        old = pd.read_csv(os.path.join(LLM, 'llm_v2_paired.csv')).set_index('cid')
    else:
        raise ValueError(arm)
    N = pd.read_csv(os.path.join(LLM, 'cr', 'v21new_sonnet_binary_predictions.csv'))
    nA, nB = N[N.condition == f'{arm}_A'].set_index('cid'), N[N.condition == f'{arm}_B'].set_index('cid')
    new = pd.DataFrame({'point_A': nA.point, 'low_A': nA.low, 'high_A': nA.high, 'point_B': nB.point, 'low_B': nB.low,
                        'high_B': nB.high, 'truth': nA.truth})
    kept = (te.v21_status == 'kept').to_numpy()
    out = pd.DataFrame(index=te.index, columns=COLS + ['truth'], dtype=float)
    out.loc[kept] = old.loc[te.cid[kept], COLS + ['truth']].to_numpy()
    out.loc[~kept] = new.loc[te.cid[~kept], COLS + ['truth']].to_numpy()
    assert out.notna().all().all(), 'missing LM predictions'
    assert np.allclose(out.truth.to_numpy(), te.delta.to_numpy()), 'archived truth differs from v2.1 delta'
    return out


# ---------------- formatting ----------------
def f(x, d=3):
    s = f'{x:.{d}f}'
    s = s[1:] if s == '-0.' + '0' * d else s
    return f'${s}$' if s.startswith('-') and not _in_math else s


_in_math = False  # negative plain values are wrapped in $...$ so the minus sign is typeset as a minus; inside \ci the macro supplies math mode


def ci(lo, hi, d=3):
    global _in_math
    _in_math = True
    try:
        return f'\\ci{{{f(lo, d)}}}{{{f(hi, d)}}}'
    finally:
        _in_math = False


def write(label, df, tex):
    os.makedirs(os.path.join(ROOT, 'results', 'cr'), exist_ok=True); os.makedirs(os.path.join(ROOT, 'tables'), exist_ok=True)
    assert '\u2014' not in tex, 'em dash in table'
    df.to_csv(os.path.join(ROOT, 'results', 'cr', f'{label}.csv'), index=False)
    with open(os.path.join(ROOT, 'tables', f'{label}.tex'), 'w') as fh:
        fh.write(tex.rstrip() + '\n')
    print(f'wrote results/cr/{label}.csv ({len(df)} rows) and tables/{label}.tex')


def n(x):
    """Integer with LaTeX thousands separator, e.g. 4{,}275."""
    return f'{int(x):,}'.replace(',', '{,}')
