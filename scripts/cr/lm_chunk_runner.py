"""Chunk runner for the camera-ready language-model runs (protocol-only, swap, blind repeat, negative control,
blind drug-class guess). Runs inside a Claude Science python kernel and calls the models through host.llm
(list form, max_concurrency=8), with the same parse rule and one-retry rule as the original runs.

Usage (python kernel; every path is a local file path, e.g. from host.artifact_path(<version_id>)):

    exec(open(RUNNER_PATH).read())
    df = run_chunk(run_name='protocol_A', variant='protocol_A', cids=[...], chunk=0,
                   prompts_path=PROMPT_MODULE_PATH, repo_zip_path=REPO_ZIP_PATH)
    # writes ./<run_name>_<chunk:02d>.csv and ./<run_name>_<chunk:02d>_run.json and returns the DataFrame

Variants (model is fixed per variant; prompts come from the repository's src/ivbench/llm_prompts.py):
    protocol_A    claude-sonnet-5            build_prompt_protocol(r, 'A')
    protocol_B    claude-sonnet-5            build_prompt_protocol(r, 'B')
    swap_A        claude-sonnet-5            build_prompt_swap(r)              (arms exchanged)
    repeat_B      claude-sonnet-5            build_prompt(r, 'B')              (original blind prompt, repeat run)
    negctl_A      claude-sonnet-5            build_prompt_negctrl(r)           (pre-registered substitution)
    classguess_B  claude-haiku-4-5-20251001  build_prompt_classguess(r)        (drug-class question on the B prompt)
    classjudge    claude-haiku-4-5-20251001  build_judge_prompt(r, guess)      (scores a guess vs the true treatment;
                                                                                needs guesses_path = merged classguess CSV)
    classjudge_perm  same as classjudge, but each guess is scored against the true treatment of a different contrast
                     (fixed derangement: the guess of cid i is paired with the row of PERM_MAP[i]); chance level.

Per-row output: cid, nct_id, run_name, variant, condition, model, model_returned, point, low, high, parse_ok, n_calls,
raw_response (first 300 chars), prompt_sha1, error, truth, year_posted, area, ctrl_kind
(classguess_B adds drug_class, mechanism, example_agent; classjudge* add class_match, mechanism_match, exact_agent,
judged_cid). Default temperature, max_tokens = MAX_TOKENS (1500), no system prompt, no thinking parameter: identical
to the original runs. A provider error (no text returned) is retried up to 3 times after a pause and is not counted
in n_calls (n_calls counts accepted attempts: 1, or 2 when the parse retry was used); if it still fails the row is
written with parse_ok=False and the error text so that the cid can be re-dispatched.
"""
import hashlib, json, os, sys, time, zipfile, importlib.util
import numpy as np, pandas as pd

SONNET = 'claude-sonnet-5'
HAIKU = 'claude-haiku-4-5-20251001'
VARIANTS = {
    'protocol_A': (SONNET, 'A'), 'protocol_B': (SONNET, 'B'), 'swap_A': (SONNET, 'A'), 'repeat_B': (SONNET, 'B'),
    'negctl_A': (SONNET, 'A'), 'classguess_B': (HAIKU, 'B'), 'classjudge': (HAIKU, 'J'), 'classjudge_perm': (HAIKU, 'J'),
    'haiku_protocol_A': (HAIKU, 'A'), 'haiku_protocol_B': (HAIKU, 'B'),
}
BATCH = 50          # requests per host.llm list call
CONCURRENCY = 8


def load_prompt_module(prompts_path):
    spec = importlib.util.spec_from_file_location('ivbench_llm_prompts', prompts_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_binary_rows(L, repo_zip_path, workdir='.'):
    """Extract data/benchmark_contrasts.csv.gz from the repository zip and build the 1,020 binary test rows."""
    dst = os.path.join(workdir, '_benchmark_contrasts.csv.gz')
    if not os.path.exists(dst):
        with zipfile.ZipFile(repo_zip_path) as z:
            name = [n for n in z.namelist() if n.endswith('data/benchmark_contrasts.csv.gz')][0]
            open(dst, 'wb').write(z.read(name))
    rows = L.load_rows(dst, 'binary')
    assert len(rows) == 1020 and rows[0]['cid'] == 'c0000' and rows[-1]['cid'] == 'c1019', len(rows)
    return {r['cid']: r for r in rows}


def perm_map(all_cids, seed=0):
    """Fixed derangement used for the chance level of the drug-class judge."""
    rng = np.random.default_rng(seed)
    cids = list(all_cids)
    while True:
        p = rng.permutation(len(cids))
        if not np.any(p == np.arange(len(cids))):
            return {cids[i]: cids[p[i]] for i in range(len(cids))}


def build(L, variant, r, guess=None):
    if variant in ('protocol_A', 'haiku_protocol_A'):
        return L.build_prompt_protocol(r, 'A')
    if variant in ('protocol_B', 'haiku_protocol_B'):
        return L.build_prompt_protocol(r, 'B')
    if variant == 'swap_A':
        return L.build_prompt_swap(r)
    if variant == 'repeat_B':
        return L.build_prompt(r, 'B')
    if variant == 'negctl_A':
        return L.build_prompt_negctrl(r)
    if variant == 'classguess_B':
        return L.build_prompt_classguess(r)
    if variant in ('classjudge', 'classjudge_perm'):
        return L.build_judge_prompt(r, guess)
    raise ValueError(variant)


def _parser(L, variant):
    if variant == 'classguess_B':
        return L.parse_classguess, L.CLASSGUESS_RETRY
    if variant.startswith('classjudge'):
        return L.parse_judge, ('IMPORTANT: output must be exactly one JSON object with keys class_match, mechanism_match, '
                               'exact_agent and nothing else.')
    return (lambda t: L.parse_response(t, 'binary')), L.RETRY_SUFFIX


def _call_batch(host, prompts, model, max_tokens):
    """One host.llm list call per BATCH; provider errors retried (up to 3 more times, with pauses)."""
    out = [None] * len(prompts)
    todo = list(range(len(prompts)))
    for attempt in range(4):
        if not todo:
            break
        if attempt:
            time.sleep(20 * attempt)
        res = host.llm([{'prompt': prompts[i], 'model': model, 'max_tokens': max_tokens} for i in todo],
                       max_concurrency=CONCURRENCY)
        nxt = []
        for i, rr in zip(todo, res):
            if isinstance(rr, dict) and 'error' not in rr and isinstance(rr.get('text'), str):
                out[i] = rr
            else:
                out[i] = {'error': str(rr.get('error') if isinstance(rr, dict) else rr)[:300]}
                nxt.append(i)
        todo = nxt
    return out


def run_chunk(run_name, variant, cids, chunk, prompts_path, repo_zip_path, out_dir='.', guesses_path=None, host_obj=None):
    host_ = host_obj if host_obj is not None else globals().get('host')
    assert host_ is not None, 'host object not found (run inside a Claude Science python kernel)'
    assert variant in VARIANTS, variant
    model, cond = VARIANTS[variant]
    L = load_prompt_module(prompts_path)
    rows = load_binary_rows(L, repo_zip_path, out_dir)
    cids = list(cids)
    assert len(set(cids)) == len(cids) and all(c in rows for c in cids)
    guesses, pmap = {}, {}
    if variant.startswith('classjudge'):
        G = pd.read_csv(guesses_path)
        G = G[G['parse_ok'].astype(bool)].drop_duplicates('cid', keep='last')
        guesses = {c: dict(drug_class=g.drug_class, mechanism=g.mechanism, example_agent='' if pd.isna(g.example_agent) else g.example_agent)
                   for c, g in zip(G['cid'], G.itertuples())}
        pmap = perm_map(sorted(rows)) if variant == 'classjudge_perm' else {c: c for c in rows}
        cids = [c for c in cids if c in guesses]
    parse, retry_suffix = _parser(L, variant)
    prompts = {c: build(L, variant, rows[pmap.get(c, c)] if variant.startswith('classjudge') else rows[c], guesses.get(c))
               for c in cids}
    t0 = time.time()
    records = []
    for b in range(0, len(cids), BATCH):
        bc = cids[b:b + BATCH]
        first = _call_batch(host_, [prompts[c] for c in bc], model, L.MAX_TOKENS)
        parsed = {c: (parse(r.get('text')) if 'error' not in r else None) for c, r in zip(bc, first)}
        need = [c for c, r in zip(bc, first) if 'error' not in r and parsed[c] is None]
        second = dict(zip(need, _call_batch(host_, [prompts[c] + '\n' + retry_suffix for c in need], model, L.MAX_TOKENS))) if need else {}
        for c, r1 in zip(bc, first):
            r = rows[c]
            rec = dict(cid=c, nct_id=r['nct_id'], run_name=run_name, variant=variant, condition=cond, model=model,
                       prompt_sha1=hashlib.sha1(prompts[c].encode()).hexdigest(), truth=r['delta'], year_posted=r['year_posted'],
                       area=r['area'], ctrl_kind=r['ctrl_kind'])
            if 'error' in r1:
                rec.update(parse_ok=False, n_calls=0, raw_response='', model_returned='', error=r1['error'])
                p = None
            elif c in second:
                r2 = second[c]
                p = parse(r2.get('text')) if 'error' not in r2 else None
                rec.update(parse_ok=p is not None, n_calls=2, raw_response=(r2.get('text') or '')[:300],
                           model_returned=r2.get('model', ''), error=r2.get('error', ''))
            else:
                p = parsed[c]
                rec.update(parse_ok=True, n_calls=1, raw_response=r1['text'][:300], model_returned=r1.get('model', ''), error='')
            if variant.startswith('classjudge'):
                rec['judged_cid'] = pmap[c]
                rec.update(dict(guesses[c], **(p or {})))
            else:
                rec.update(p or {})
            records.append(rec)
        print(f'{run_name} chunk {chunk}: {len(records)}/{len(cids)}', flush=True)
    D = pd.DataFrame(records)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f'{run_name}_{int(chunk):02d}')
    D.to_csv(base + '.csv', index=False)
    meta = dict(run_name=run_name, variant=variant, chunk=int(chunk), model_requested=model,
                models_returned=sorted(set(D['model_returned'].dropna().astype(str)) - {''}), n_cids=len(cids),
                n_parsed=int(D['parse_ok'].sum()), n_retries=int((D['n_calls'] == 2).sum()),
                n_provider_errors=int((D['n_calls'] == 0).sum()), max_tokens=L.MAX_TOKENS, max_concurrency=CONCURRENCY,
                wall_seconds=round(time.time() - t0, 1), utc_finished=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    json.dump(meta, open(base + '_run.json', 'w'), indent=1)
    return D


def merge_and_verify(variant, csv_paths, plan_path):
    """Merge chunk CSVs for one variant; returns (merged DataFrame with one accepted row per cid, list of missing cids,
    list of rows whose prompt_sha1 does not match the chunk plan). Missing = no parse_ok row."""
    P = json.load(open(plan_path))
    want = P['variants'][variant]['cids']
    sha = P['variants'][variant].get('prompt_sha1', {})
    D = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    D = D[D['variant'] == variant]
    ok = D[D['parse_ok'].astype(bool)].drop_duplicates('cid', keep='last')
    missing = [c for c in want if c not in set(ok['cid'])]
    bad_sha = [c for c, s in zip(ok['cid'], ok['prompt_sha1']) if sha and sha.get(c) not in (None, s)]
    ok = ok.set_index('cid').loc[[c for c in want if c in set(ok['cid'])]].reset_index()
    return ok, missing, bad_sha
