"""Step 8: how blind is BLIND? For test contrasts whose normalized treatment-intervention name (ivbench.specs.norm_name on
treat_intervention) occurs among training contrasts, retrieve the top-k (k=1,5,10) training contrasts by cosine similarity of the
BLIND text (endpoint title/description/timeframe + conditions; TF-IDF fit on training with the featurizer's settings, and its
120-d SVD used by the model) and score a hit if any retrieved contrast shares a normalized name. Chance = expected hit rate of k
uniformly drawn training contrasts (hypergeometric), averaged over the same test contrasts.
Output: blind_identifiability.csv"""
from common import *
from ivbench.specs import norm_name, ENDPOINT_TXT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.special import gammaln
def p_hit(N, m, k):
    if m == 0: return 0.0
    if N - m < k: return 1.0
    return 1 - np.exp(gammaln(N - m + 1) - gammaln(N - m - k + 1) - gammaln(N + 1) + gammaln(N - k + 1))
if __name__ == '__main__':
    C = load_contrasts(); TE = test_sets(C); rows = []
    for fam in ['binary', 'continuous']:
        tr, te = train_set(C, fam), TE[fam]
        ktr = [set(norm_name(x)) for x in tr.treat_intervention]; kte = [set(norm_name(x)) for x in te.treat_intervention]
        inv = {}
        for i, ks in enumerate(ktr):
            for k in ks: inv.setdefault(k, set()).add(i)
        elig = [j for j, ks in enumerate(kte) if any(k in inv for k in ks)]
        tv = TfidfVectorizer(max_features=60000, ngram_range=(1, 2), min_df=3, sublinear_tf=True, strip_accents='unicode', stop_words='english')
        Xtr = tv.fit_transform(me.text_of(tr, ENDPOINT_TXT)); Xte = tv.transform(me.text_of(te.iloc[elig], ENDPOINT_TXT))
        svd = TruncatedSVD(n_components=120, random_state=0); Ztr = svd.fit_transform(Xtr); Zte = svd.transform(Xte)
        nz = lambda Z: Z / np.clip(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12, None)
        sims = {'tfidf': (Xte @ Xtr.T).toarray(), 'svd120 (model input)': nz(Zte) @ nz(Ztr).T}
        match = [set().union(*[inv.get(k, set()) for k in kte[j]]) for j in elig]
        for rep, S in sims.items():
            order = np.argsort(-S, axis=1)[:, :10]
            for k in (1, 5, 10):
                hits = np.array([len(set(order[i, :k]) & match[i]) > 0 for i in range(len(elig))])
                chance = np.mean([p_hit(len(tr), len(match[i]), k) for i in range(len(elig))])
                g = te.nct_id.to_numpy()[elig]
                rng = np.random.default_rng(0); u = np.unique(g); idx = {t: np.where(g == t)[0] for t in u}
                bs = [hits[np.concatenate([idx[t] for t in rng.choice(u, len(u))])].mean() for _ in range(1000)]
                rows.append(dict(family=fam, representation=rep, k=k, n_eligible=len(elig), n_eligible_trials=len(u), n_test=len(te),
                                 hit_rate=hits.mean(), ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5), chance=chance, lift=hits.mean() / chance,
                                 median_matches_in_train=float(np.median([len(m) for m in match]))))
    R = pd.DataFrame(rows); R.to_csv('blind_identifiability.csv', index=False); pd.set_option('display.width', 250); print(R.round(3).to_string())
