# Redaction audit + metrics (Phase-1 definitions). Expects in namespace: df, samp, prompts, parsed(dict key->(point,low,high)), SEED
import numpy as np, pandas as pd, re

REF_MEAN = df[(df.split_time=='train') & (df.family=='binary')]['delta'].mean()   # global TRAINING mean of delta (binary)

# ---------- redaction audit ----------
def training_intervention_vocab(df):
    """Name tokens from TRAINING-set intervention name fields, classified with the same dictionary+suffix classifier."""
    tr = df[df.split_time=='train']
    vocab=set()
    for _,r in tr.iterrows():
        toks,_,_ = name_sets(r, cols=['treat_intervention','ctrl_intervention'])
        vocab |= toks
    return vocab

def audit_prompt(text, vocab, protect):
    words = set(w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-']{3,}", text))
    leaks = sorted(w for w in words if w in vocab and w not in protect and w not in COMMON and w not in TEMPLATE_WORDS)
    suffix_hits = sorted(w for w in words if w not in ENG and w not in protect and PHARMA.match(w))
    return leaks, suffix_hits

def run_audit(samp, prompts, vocab):
    rows=[]
    for _,r in samp.iterrows():
        toks, phrases, protect = name_sets(r)
        pb = prompts[(r['cid'],'B')]; pa = prompts[(r['cid'],'A')]
        own = sorted(t for t in toks if re.search(r'\b'+re.escape(t)+r'\b', pb, flags=re.I))
        leaks, suf = audit_prompt(pb, vocab, protect)
        _, sufA = audit_prompt(pa, vocab, protect)
        rows.append(dict(cid=r['cid'], nct_id=r['nct_id'], own_tokens_surviving=";".join(own),
                         train_vocab_tokens_surviving=";".join(leaks), n_train_vocab_leaks=len(leaks),
                         pharma_suffix_words_in_B=";".join(suf), pharma_suffix_words_in_A=";".join(sufA),
                         n_redacted_spans=pb.count("[REDACTED]"), len_A=len(pa), len_B=len(pb)))
    return pd.DataFrame(rows)

# ---------- metrics ----------
def skill_of(truth, pred, ref=None):
    ref = REF_MEAN if ref is None else ref
    return 1 - np.mean((truth-pred)**2)/np.mean((truth-ref)**2)

def boot_ci(dfp, fn, n_boot=2000, seed=SEED):
    """Trial-level bootstrap: resample nct_id clusters with replacement. fn(sub_df)->float."""
    rng = np.random.default_rng(seed)
    trials = dfp['nct_id'].unique()
    groups = {t: g for t,g in dfp.groupby('nct_id')}
    vals=[]
    for _ in range(n_boot):
        pick = rng.choice(trials, size=len(trials), replace=True)
        sub = pd.concat([groups[t] for t in pick], ignore_index=True)
        vals.append(fn(sub))
    vals=np.array(vals)
    return float(np.nanpercentile(vals,2.5)), float(np.nanpercentile(vals,97.5))

def cond_metrics(wide, cond):
    t=wide['truth'].values; p=wide[f'point_{cond}'].values; lo=wide[f'lo_{cond}'].values; hi=wide[f'hi_{cond}'].values
    sk = skill_of(t,p)
    ci = boot_ci(wide, lambda s: skill_of(s['truth'].values, s[f'point_{cond}'].values))
    return dict(condition=cond, n=int(len(wide)), skill=float(sk), ci_low=ci[0], ci_high=ci[1],
                mae=float(np.mean(np.abs(t-p))), rmse=float(np.sqrt(np.mean((t-p)**2))),
                interval_coverage=float(np.mean((t>=lo)&(t<=hi))), mean_interval_width=float(np.mean(hi-lo)))

def diff_skill(wide, c1, c2):
    f = lambda s: skill_of(s['truth'].values, s[f'point_{c1}'].values) - skill_of(s['truth'].values, s[f'point_{c2}'].values)
    v = f(wide); ci = boot_ci(wide, f)
    return dict(comparison=f"{c1}_minus_{c2}", n=int(len(wide)), value=float(v), ci_low=ci[0], ci_high=ci[1])

def selective(wide, cond='A', covs=(0.25,0.5,0.75,1.0)):
    w = wide.copy(); w['width'] = w[f'hi_{cond}']-w[f'lo_{cond}']
    w = w.sort_values('width', kind='mergesort')
    out=[]
    for c in covs:
        k = max(1, int(round(c*len(w)))); s=w.iloc[:k]
        out.append(dict(coverage=float(c), n=int(k), mae=float(np.mean(np.abs(s['truth']-s[f'point_{cond}']))),
                        skill=float(skill_of(s['truth'].values, s[f'point_{cond}'].values)),
                        max_width=float(s['width'].max())))
    return out
