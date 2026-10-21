"""Step 5 (language-model part): trial-bootstrap intervals (2,000 resamples over nct_id, seed 0) for the coverage of the
models' self-stated 90% intervals, and honest abstention: thresholds on stated interval width chosen on one random half of
test trials and evaluated on the other half (both directions), repeated 500 times.
Outputs: lm_interval_coverage.csv, lm_abstention_honest.csv, lm_selective_insample.csv"""
from common import *
C = load_contrasts(); TE = test_sets(C); L = load_lm(TE)
RUNS = [('claude-sonnet-5', 'binary', 'sonnet_bin'), ('claude-haiku-4-5-20251001', 'binary', 'haiku_bin'), ('claude-sonnet-5', 'continuous', 'sonnet_cont')]

def boot_mean(x, g, n=2000, seed=0):
    rng = np.random.default_rng(seed); u = np.unique(g); idx = {t: np.where(g == t)[0] for t in u}; out = []
    for _ in range(n):
        ii = np.concatenate([idx[t] for t in rng.choice(u, len(u), replace=True)]); out.append(x[ii].mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def choose_cut(w, ae, rule, val):
    o = np.argsort(w, kind='stable')
    if rule == 'mae_target':
        cum = np.cumsum(ae[o]) / np.arange(1, len(o) + 1); ok = np.where(cum <= val)[0]
        return w[o][ok.max()] if len(ok) else -np.inf
    return np.quantile(w, val)

if __name__ == '__main__':
    cov, hon, ins = [], [], []
    for mid, fam, key in RUNS:
        D = L[key]; y = D.truth.to_numpy(); g = D.nct_id.to_numpy()
        for cond in ['A', 'B']:
            c = ((D[f'low_{cond}'] <= y) & (y <= D[f'high_{cond}'])).to_numpy().astype(float)
            lo, hi = boot_mean(c, g)
            cov.append(dict(model=mid, family=fam, condition={'A': 'FULL (intervention shown)', 'B': 'BLIND (withheld)'}[cond], coverage=c.mean(), ci_low=lo, ci_high=hi,
                            ci_method='trial bootstrap 2000', mean_width=float((D[f'high_{cond}'] - D[f'low_{cond}']).mean()), n=len(y), n_trials=len(np.unique(g))))
        w = (D.high_A - D.low_A).to_numpy(); ae = np.abs(y - D.point_A.to_numpy())
        sc = me.selective_curve(y, D.point_A.to_numpy(), w, grid=np.array([0.10, 0.25, 0.50, 0.75, 1.0])); sc.insert(0, 'family', fam); sc.insert(0, 'model', mid); ins.append(sc)
        rules = [('mae_target', t) for t in ([0.10, 0.08, 0.05, 0.02] if fam == 'binary' else [0.30, 0.25, 0.20])] + [('quantile', f) for f in (0.10, 0.25, 0.50)]
        u = np.unique(g); rng = np.random.default_rng(0)
        for rule, val in rules:
            fr, mae, frt = [], [], []
            for r in range(500):
                h = set(rng.choice(u, len(u) // 2, replace=False)); m1 = np.array([t in h for t in g])
                for a, b in [(m1, ~m1), (~m1, m1)]:
                    cut = choose_cut(w[a], ae[a], rule, val); ans = b & (w <= cut)
                    fr.append(ans.sum() / b.sum()); mae.append(ae[ans].mean() if ans.any() else np.nan)
                    frt.append(len(np.unique(g[ans])) / len(np.unique(g[b])))
            fr, mae = np.array(fr), np.array(mae)
            hon.append(dict(model=mid, family=fam, condition='FULL (A)', rule=rule, value=val, answered_frac_mean=fr.mean(), answered_frac_p2_5=np.percentile(fr, 2.5),
                            answered_frac_p97_5=np.percentile(fr, 97.5), answered_trials_frac_mean=float(np.mean(frt)), mae_answered_mean=np.nanmean(mae),
                            mae_answered_p2_5=np.nanpercentile(mae, 2.5), mae_answered_p97_5=np.nanpercentile(mae, 97.5),
                            frac_target_met=float(np.nanmean(mae <= val)) if rule == 'mae_target' else np.nan, mae_all=ae.mean(), n_repeats=500, n_evaluations=len(fr)))
    pd.DataFrame(cov).to_csv('lm_interval_coverage.csv', index=False); pd.DataFrame(hon).to_csv('lm_abstention_honest.csv', index=False)
    pd.concat(ins).to_csv('lm_selective_insample.csv', index=False)
    pd.set_option('display.width', 250)
    print(pd.DataFrame(cov).round(3).to_string()); print(pd.DataFrame(hon).round(3).to_string()); print(pd.concat(ins).round(4).to_string())
