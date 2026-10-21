"""Shared helpers for the camera-ready statistics track (ivbench). Run from work/stats with REPO pointing at ivbench-main."""
import os, sys, re, json, gzip
import numpy as np, pandas as pd
REPO = os.environ.get('IVBENCH_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ivbench-main'))
sys.path.insert(0, os.path.join(REPO, 'src'))
from ivbench import models as me

BOUND = {'llama3.1': pd.Timestamp('2024-01-01'), 'haiku4.5': pd.Timestamp('2025-08-01'), 'sonnet5': pd.Timestamp('2026-02-01')}
RELI = {'haiku4.5': pd.Timestamp('2025-03-01'), 'sonnet5': pd.Timestamp('2026-02-01')}   # reliable-knowledge cutoffs Feb 2025 / Jan 2026
MON = {m: i + 1 for i, m in enumerate('Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split())}

def load_contrasts():
    return pd.read_csv(os.path.join(REPO, 'data', 'benchmark_contrasts.csv.gz'), low_memory=False)

def test_sets(C):
    """Test contrasts in the language-model order (cid c0000.. binary, k0000.. continuous), as llm_prompts.load_rows."""
    TE = {}
    for fam, pre in [('binary', 'c'), ('continuous', 'k')]:
        d = C[(C.family == fam) & (C.split_time == 'test')].sort_values(['nct_id', 'outcome_title', 'treat_group']).reset_index(drop=True)
        d['cid'] = [f'{pre}{i:04d}' for i in range(len(d))]; TE[fam] = d
    return TE

def train_set(C, fam):
    return C[(C.family == fam) & (C.split_time == 'train')].reset_index(drop=True)

def load_lm(TE):
    f = {'sonnet_bin': ('llm_v2_paired.csv', 'binary'), 'haiku_bin': ('llm_haiku_paired.csv', 'binary'), 'sonnet_cont': ('llm_cont_paired.csv', 'continuous')}
    out = {}
    for k, (fn, fam) in f.items():
        D = pd.read_csv(os.path.join(REPO, 'results', 'llm', fn)).set_index('cid').loc[TE[fam].cid].reset_index()
        assert np.allclose(D.truth.to_numpy(), TE[fam].delta.to_numpy())
        out[k] = D
    return out

def todate(s):
    if not isinstance(s, str) or not s: return pd.NaT
    s = s.strip()
    if re.fullmatch(r'\d{4}-\d{2}', s): s += '-01'
    if re.fullmatch(r'\d{4}', s): s += '-01-01'
    return pd.to_datetime(s, errors='coerce')

def pm_date(s):
    """Earliest calendar day consistent with a PubMed date string (partial dates map to period start: conservative)."""
    if not isinstance(s, str) or not s.strip(): return pd.NaT
    m = re.match(r'(\d{4})(?:[ /](\w{3}))?(?:[ /](\d{1,2}))?', s.strip())
    if not m: return pd.NaT
    y = int(m.group(1)); g2 = m.group(2)
    mo = (int(g2) if g2.isdigit() else MON.get(g2[:3].title(), 1)) if g2 else 1
    dd = int(m.group(3)) if m.group(3) else 1
    try: return pd.Timestamp(y, mo, dd)
    except Exception: return pd.Timestamp(y, mo, 1)

def load_meta(path='ctgov_meta_oct2026.jsonl.gz'):
    rows = []
    for line in gzip.open(path, 'rt'):
        s = json.loads(line); ps = s['protocolSection']; st = ps.get('statusModule', {}); d = ps.get('designModule', {}); di = d.get('designInfo', {})
        rows.append(dict(nct_id=ps['identificationModule']['nctId'], pcd=st.get('primaryCompletionDateStruct', {}).get('date'),
                         pcd_type=st.get('primaryCompletionDateStruct', {}).get('type'), rfp=st.get('resultsFirstPostDateStruct', {}).get('date'),
                         allocation_now=di.get('allocation'), intervention_model=di.get('interventionModel'),
                         masking_now=di.get('maskingInfo', {}).get('masking'), phases=','.join(d.get('phases', []) or [])))
    M = pd.DataFrame(rows); M['pcd_dt'] = M.pcd.map(todate); M['rfp_dt'] = M.rfp.map(todate)
    return M

def load_result_refs(path='ctgov_references_pubmed.csv'):
    R = pd.read_csv(path, dtype={'pmid': str})
    def hist_min(h):
        if not isinstance(h, str): return pd.NaT
        ds = [pd.to_datetime(x.split('=')[1][:10], errors='coerce') for x in h.split(';') if '=' in x]
        ds = [d for d in ds if pd.notna(d)]; return min(ds) if ds else pd.NaT
    R['pubdate_dt'] = R.pubdate.map(pm_date); R['epub_dt'] = R.epubdate.map(pm_date); R['entrez_dt'] = R.pmc_hist.map(hist_min)
    R['cit_year'] = R.citation.str.extract(r'\b(19[5-9]\d|20[0-2]\d)\b')[0].astype(float)
    R['pub_earliest'] = R[['pubdate_dt', 'epub_dt', 'entrez_dt']].min(axis=1)
    RR = R[R.ref_type == 'RESULT'].copy()
    m = RR.pub_earliest.isna() & RR.cit_year.notna()
    RR.loc[m, 'pub_earliest'] = pd.to_datetime(RR.loc[m, 'cit_year'].astype(int).astype(str) + '-01-01')
    return RR

def contamination_flags(M, RR, b):
    """Per-trial flags vs boundary b. Undated RESULT references count as 'before boundary' (conservative)."""
    pre = RR.groupby('nct_id').pub_earliest.apply(lambda s: bool(((s < b) | s.isna()).any())).astype(bool)
    f = M[['nct_id', 'pcd_dt', 'rfp_dt']].copy()
    f['c_pcd'] = (f.pcd_dt >= b).to_numpy(); f['c_rfp'] = (f.rfp_dt >= b).to_numpy()
    f['c_nopub'] = ~f.nct_id.map(pre.to_dict()).fillna(False).astype(bool).to_numpy()
    f['uncont'] = f.c_pcd & f.c_rfp & f.c_nopub
    return f.set_index('nct_id')

def skill_ci(y, a, b, g, n, seed=0):
    y, a, b, g = map(np.asarray, (y, a, b, g))
    s = me.skill(y, a, b)
    lo, hi = me.boot_skill(y, a, b, g, n=n, seed=seed) if len(np.unique(g)) > 1 else (np.nan, np.nan)
    return float(s), lo, hi


def shared_boot_mean(y, preds_full, preds_ref, g, n=1000, seed=0):
    """Mean over seeds of skill, with one shared set of trial resamples (same resample applied to every seed)."""
    rng = np.random.default_rng(seed); u = np.unique(g); idx = {t: np.where(g == t)[0] for t in u}; out = []
    for _ in range(n):
        ii = np.concatenate([idx[t] for t in rng.choice(u, len(u), replace=True)])
        out.append(np.mean([me.skill(y[ii], a[ii], b[ii]) for a, b in zip(preds_full, preds_ref)]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

