"""Language-model calls for the v2.1 test contrasts that are new or re-paired relative to v2 (benchmark repair).
Same prompt builders (llm_prompts.py), parser and one-retry rule as all earlier runs; models called through host.llm
with the provider default temperature, exactly as the original Claude runs.

    exec(open(RUNNER).read())
    out = run_v21(host, model, family, conds, prompts_path, v21_contrasts_path, need_path, out_csv)

conds: subset of protocol_A, protocol_B, posted_A, posted_B. Rows are taken from the v2.1 contrast file by their v2.1 cid
(the cid column; no positional renumbering), restricted to the cids listed in need_path for the given family.
"""
import importlib.util, os, time
import pandas as pd

SONNET = 'claude-sonnet-5'
HAIKU = 'claude-haiku-4-5-20251001'


def _load(prompts_path):
    spec = importlib.util.spec_from_file_location('ivbench_llm_prompts_v21', prompts_path)
    L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L); return L


def _rows(v21_path, need_path, family):
    C = pd.read_csv(v21_path, low_memory=False)
    need = pd.read_csv(need_path)
    ids = set(need[need.family == family].cid)
    C = C[(C.family == family) & (C.split_time == 'test') & C.cid.isin(ids)].copy()
    assert len(C) == len(ids), (len(C), len(ids))
    return [dict(r, family=family) for r in C.to_dict('records')]


def _build(L, cond, r):
    if cond == 'protocol_A': return L.build_prompt_protocol(r, 'A')
    if cond == 'protocol_B': return L.build_prompt_protocol(r, 'B')
    if cond == 'posted_A': return L.build_prompt(r, 'A')
    if cond == 'posted_B': return L.build_prompt(r, 'B')
    raise ValueError(cond)


def _batch(host, prompts, model, max_tokens):
    out = [None] * len(prompts); todo = list(range(len(prompts)))
    for attempt in range(4):
        if not todo: break
        if attempt: time.sleep(20 * attempt)
        res = host.llm([{'prompt': prompts[i], 'model': model, 'max_tokens': max_tokens} for i in todo], max_concurrency=8)
        nxt = []
        for i, r in zip(todo, res):
            if 'error' in r: nxt.append(i); out[i] = r
            else: out[i] = r
        todo = nxt
    return out


def run_v21(host, model, family, conds, prompts_path, v21_path, need_path, out_csv):
    L = _load(prompts_path); rows = _rows(v21_path, need_path, family)
    recs = []
    for cond in conds:
        ps = [_build(L, cond, r) for r in rows]
        first = _batch(host, ps, model, L.MAX_TOKENS)
        parsed = [L.parse_response(f.get('text'), family) if 'error' not in f else None for f in first]
        need = [i for i, p in enumerate(parsed) if p is None and 'error' not in first[i]]
        second = dict(zip(need, _batch(host, [ps[i] + '\n' + L.RETRY_SUFFIX for i in need], model, L.MAX_TOKENS))) if need else {}
        for i, r in enumerate(rows):
            p, n, f = parsed[i], 1, first[i]
            if p is None and i in second:
                f = second[i]; n = 2; p = L.parse_response(f.get('text'), family) if 'error' not in f else None
            rec = dict(cid=r['cid'], nct_id=r['nct_id'], family=family, condition=cond, truth=r['delta'], model=model,
                       model_returned=f.get('model', ''), n_calls=n, parse_ok=p is not None, error=f.get('error', '') if p is None else '',
                       point=p['point'] if p else None, low=p['low'] if p else None, high=p['high'] if p else None,
                       raw=(f.get('text') or '')[:300])
            recs.append(rec)
    D = pd.DataFrame(recs); D.to_csv(out_csv, index=False)
    return D
