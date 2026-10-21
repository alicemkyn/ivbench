"""New table (tab:blindness): how blind is BLIND?

  python scripts/cr/tables/tab_blindness.py

(a) Language model. Claude Haiku 4.5 is shown the condition-B (blind, original) prompt and asked for the withheld treatment's
    drug class, mechanism and an example agent (results/llm/cr/classguess_B_predictions.csv). A Haiku 4.5 judge scores each guess
    against the true treatment (classjudge) and, as a chance reference, against the treatment of a randomly permuted other
    contrast (classjudge_perm; 2 unparsed judge calls excluded). Run on the 1,020 v2 binary test contrasts; also reported on the
    816 v2.1 'kept' contrasts. Intervals: 95% trial-level bootstrap (2,000 resamples, seed 0).
(b) Registry BLIND text. Port of scripts/cr/stats/blind_identifiability.py: for test contrasts whose normalised treatment name
    (ivbench.specs.norm_name of treat_intervention) occurs among training contrasts, retrieve the top-k training contrasts by
    cosine similarity of the BLIND text (endpoint title/description/timeframe + conditions; TF-IDF with the featurizer's
    settings, and its 120-d SVD used by the model) and score a hit if any shares a normalised name. Chance = expected hit rate of
    k uniformly drawn training contrasts (hypergeometric). Training and test from the same benchmark version (v2.1; v2 in CSV).
    Intervals: 1,000 trial resamples (seed 0).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *  # noqa: E402,F403
from ivbench.specs import norm_name, ENDPOINT_TXT  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from scipy.special import gammaln  # noqa: E402


def p_hit(N, m_, k):
    if m_ == 0:
        return 0.0
    if N - m_ < k:
        return 1.0
    return 1 - np.exp(gammaln(N - m_ + 1) - gammaln(N - m_ - k + 1) - gammaln(N + 1) + gammaln(N - k + 1))


def class_rows(te2, kept_cids, te21_cids=None):
    J, P = cr_read('classjudge').set_index('cid'), cr_read('classjudge_perm').set_index('cid')
    Jn, Pn = cr_read('v21new_classjudge').set_index('cid'), cr_read('v21new_classjudge_perm').set_index('cid')
    cols = ['nct_id', 'class_match', 'mechanism_match', 'exact_agent', 'parse_ok']
    J = pd.concat([J[cols], Jn[cols]]); P = pd.concat([P[cols], Pn[cols]]); R = []
    sets = [('v2', te2.cid), ('v2.1 kept', te2.cid[te2.cid.isin(kept_cids)])] + ([('v2.1', te21_cids)] if te21_cids is not None else [])
    for ts, ids in sets:
        j = J.reindex(ids).dropna(subset=['nct_id']); j = j[j.parse_ok.astype(bool)]; p = P.reindex(ids).dropna(subset=['nct_id']); p = p[p.parse_ok.astype(bool)]
        for col, lab in (('class_match', 'drug class'), ('mechanism_match', 'mechanism'), ('exact_agent', 'exact agent')):
            for src, d, ref in (('true treatment', j, 'classjudge'), ('permuted treatment (chance)', p, 'classjudge_perm')):
                x = d[col].astype(bool).to_numpy(); g = d.nct_id.to_numpy()
                lo, hi = boot_stat(lambda i: x[i].mean(), g, 2000)
                R.append(dict(panel='a', test_set=ts, statistic=f'{lab} match, {src}', value=float(x.mean()), lo=lo, hi=hi, n=len(x),
                              trials=int(len(np.unique(g))), source=ref))
    return R


def retrieval_rows(C, version):
    rows = []
    for fam in FAMS:
        tr, te = train(C, fam), test(C, fam, version)
        ktr = [set(norm_name(x)) for x in tr.treat_intervention]; kte = [set(norm_name(x)) for x in te.treat_intervention]
        inv = {}
        for i, ks in enumerate(ktr):
            for k in ks:
                inv.setdefault(k, set()).add(i)
        elig = [j for j, ks in enumerate(kte) if any(k in inv for k in ks)]
        tv = TfidfVectorizer(max_features=60000, ngram_range=(1, 2), min_df=3, sublinear_tf=True, strip_accents='unicode', stop_words='english')
        Xtr = tv.fit_transform(me.text_of(tr, ENDPOINT_TXT)); Xte = tv.transform(me.text_of(te.iloc[elig], ENDPOINT_TXT))
        svd = TruncatedSVD(n_components=120, random_state=0); Ztr = svd.fit_transform(Xtr); Zte = svd.transform(Xte)
        nz = lambda Z: Z / np.clip(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12, None)
        sims = {'tfidf': (Xte @ Xtr.T).toarray(), 'svd120 (model input)': nz(Zte) @ nz(Ztr).T}
        match = [set().union(*[inv.get(k, set()) for k in kte[j]]) for j in elig]
        g = te.nct_id.to_numpy()[elig]
        for rep, S in sims.items():
            order = np.argsort(-S, axis=1)[:, :10]
            for k in (1, 5, 10):
                hits = np.array([len(set(order[i, :k]) & match[i]) > 0 for i in range(len(elig))])
                chance = float(np.mean([p_hit(len(tr), len(match[i]), k) for i in range(len(elig))]))
                rng = np.random.default_rng(0); u = np.unique(g); idx = {t: np.where(g == t)[0] for t in u}
                bs = [hits[np.concatenate([idx[t] for t in rng.choice(u, len(u))])].mean() for _ in range(1000)]
                rows.append(dict(panel='b', test_set=version, family=fam, representation=rep, k=k, statistic=f'same-molecule hit rate, {rep}, top-{k}',
                                 value=float(hits.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), chance=chance,
                                 lift=float(hits.mean() / chance), n=len(elig), trials=len(u), n_test=len(te),
                                 median_matches_in_train=float(np.median([len(x) for x in match]))))
    return rows


def tex_of(R):
    A = R[(R.panel == 'a') & (R.test_set == 'v2.1')]
    B = R[(R.panel == 'b') & (R.test_set == 'v2.1')]
    L = ["\\multicolumn{5}{l}{\\emph{(a) language model: treatment class guessed from the \\blind\\ prompt (Claude Haiku~4.5, v2.1 binary)}} \\\\"]
    for lab in ('drug class', 'mechanism', 'exact agent'):
        t = A[A.statistic == f'{lab} match, true treatment'].iloc[0]; p = A[A.statistic == f'{lab} match, permuted treatment (chance)'].iloc[0]
        L.append(f"\\quad {lab} & {f(t.value)} {ci(t.lo, t.hi)} & {f(p.value)} {ci(p.lo, p.hi)} & {f(t.value / p.value, 1)} & {t.n} ({t.trials}) \\\\")
    L.append('\\midrule')
    L.append("\\multicolumn{5}{l}{\\emph{(b) registry \\blind\\ text: same-molecule training contrast retrieved (v2.1)}} \\\\")
    for _, r in B.iterrows():
        L.append(f"\\quad {r.family}, {r.representation}, top-{int(r.k)} & {f(r.value)} {ci(r.lo, r.hi)} & {f(r.chance)} & {f(r.lift, 1)} & {r.n} ({r.trials}) \\\\")
    cap = ("How blind is \\blind? (a) Claude Haiku~4.5 reads the original condition-B prompt and guesses the withheld treatment's drug "
           "class, mechanism and an example agent; a Haiku~4.5 judge scores the guess against the true treatment and, as a chance "
           "reference, against the treatment of a randomly permuted contrast (judge calls that failed to parse are excluded). On the 878 v2.1 "
           "binary test contrasts (v2 values in \\texttt{results/cr/blindness.csv}); 95\\% trial-level bootstrap intervals (2{,}000 resamples). "
           "(b) For v2.1 test contrasts whose normalised treatment name occurs in the training split, the share for which a training "
           "contrast with the same molecule is among the $k$ nearest by \\blind\\ text (endpoint and condition fields), as TF-IDF and as "
           "the 120-dimensional SVD the registry model uses; chance is the hypergeometric rate for $k$ random training contrasts; "
           "1{,}000 trial resamples.")
    return ("\\begin{table}[h]\\centering\\footnotesize\n\\caption{" + cap + "}\n\\label{tab:blindness}\n"
            "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lcccr}\\toprule statistic & observed & chance & ratio & $n$ (trials) \\\\ \\midrule\n"
            + '\n'.join(L) + "\n\\bottomrule\\end{tabular}}\\end{table}\n")


if __name__ == '__main__':
    with Timer():
        C21, C2 = load('v2.1'), load('v2'); te2 = test(C2, 'binary', 'v2'); te21 = test(C21, 'binary', 'v2.1')
        R = pd.DataFrame(class_rows(te2, te21.cid[te21.v21_status == 'kept'], te21.cid) + retrieval_rows(C21, 'v2.1') + retrieval_rows(C2, 'v2'))
        write('blindness', R, tex_of(R))
        print(R[['panel', 'test_set', 'statistic', 'value', 'lo', 'hi', 'n']].round(3).to_string())
