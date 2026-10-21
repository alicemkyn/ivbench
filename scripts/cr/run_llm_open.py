"""Older-cutoff open-weight model arm (camera-ready): Llama 3.1 70B Instruct (bf16) through a hosted
OpenAI-compatible chat-completions endpoint (Together or Fireworks), JSON mode on, temperature 0.

Same prompts (ivbench.llm_prompts.build_prompt, conditions A / B), same parser (parse_response) and the same one-retry
rule (RETRY_SUFFIX appended on parse failure) as the Claude runs. Resumable: rows already present in --out with an
accepted prediction are skipped, and every finished row is appended immediately.

Environment:
    LLM_API_KEY     provider API key (Together: TOGETHER_API_KEY is also accepted; Fireworks: FIREWORKS_API_KEY)
Provider presets (--provider):
    together   https://api.together.xyz/v1/chat/completions        model meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
               (405B: meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo; 8B: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo)
    fireworks  https://api.fireworks.ai/inference/v1/chat/completions model accounts/fireworks/models/llama-v3p1-70b-instruct
               (405B: accounts/fireworks/models/llama-v3p1-405b-instruct; 8B: accounts/fireworks/models/llama-v3p1-8b-instruct)
    Check the provider's model page that the served precision is bf16 (non-quantised) and record it in the run JSON
    (--precision-note). Override with --endpoint / --model.

Day-1 quality gate (20 prompts, fixed random sample of binary test cids with numpy default_rng(0), condition A):
    python scripts/run_llm_open.py --provider together --condition A --gate --out results/llm/open_gate_70b.csv
    -> prints parse rate, number of unique point predictions, share of the modal point value; the model is judged
       inadequate if parse rate < 0.90 or modal share > 0.50 (then rerun the gate with the 405B model).
Full runs:
    python scripts/run_llm_open.py --provider together --condition A --out results/llm/llm_llama70b_A_predictions.csv
    python scripts/run_llm_open.py --provider together --condition B --out results/llm/llm_llama70b_B_predictions.csv
"""
import argparse, json, os, sys, time, urllib.request, urllib.error
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ivbench.llm_prompts import load_rows, build_prompt, parse_response, RETRY_SUFFIX, MAX_TOKENS

PRESETS = {
    'together': dict(endpoint='https://api.together.xyz/v1/chat/completions',
                     models={'70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
                             '405b': 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
                             '8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'}, env='TOGETHER_API_KEY'),
    'fireworks': dict(endpoint='https://api.fireworks.ai/inference/v1/chat/completions',
                      models={'70b': 'accounts/fireworks/models/llama-v3p1-70b-instruct',
                              '405b': 'accounts/fireworks/models/llama-v3p1-405b-instruct',
                              '8b': 'accounts/fireworks/models/llama-v3p1-8b-instruct'}, env='FIREWORKS_API_KEY'),
}
OUT_COLS = ['cid', 'nct_id', 'condition', 'family', 'model', 'model_returned', 'provider', 'truth', 'year_posted', 'area',
            'ctrl_kind', 'n_treat', 'n_ctrl', 'parse_ok', 'n_calls', 'point', 'low', 'high', 'raw_response', 'error']


def post(url, body, key, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
    for attempt in range(7):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (408, 409, 429, 500, 502, 503, 504, 529) and attempt < 6:
                time.sleep(min(120, 5 * 2 ** attempt)); continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < 6:
                time.sleep(min(120, 5 * 2 ** attempt)); continue
            raise


def call(prompt, a, key):
    body = {'model': a.model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
            'max_tokens': MAX_TOKENS, 'response_format': {'type': 'json_object'}}
    j = post(a.endpoint, body, key)
    return j['choices'][0]['message']['content'] or '', j.get('model', a.model)


def gate_cids(rows, n=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(rows), size=n, replace=False).tolist())
    return [rows[i]['cid'] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contrasts', default=os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmark_contrasts.csv.gz'))
    ap.add_argument('--condition', choices=['A', 'B'], required=True)
    ap.add_argument('--provider', choices=list(PRESETS), default='together')
    ap.add_argument('--size', choices=['70b', '405b', '8b'], default='70b')
    ap.add_argument('--endpoint', default=None); ap.add_argument('--model', default=None)
    ap.add_argument('--precision-note', default='bf16 per provider model page (verify)')
    ap.add_argument('--gate', action='store_true', help='run only the 20-prompt day-1 quality gate')
    ap.add_argument('--cids', nargs='*', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    pre = PRESETS[a.provider]
    a.endpoint = a.endpoint or pre['endpoint']; a.model = a.model or pre['models'][a.size]
    key = os.environ.get('LLM_API_KEY') or os.environ.get(pre['env'])
    if not key:
        sys.exit(f"no API key: set LLM_API_KEY or {pre['env']}")
    rows = load_rows(a.contrasts, 'binary')
    if a.gate:
        want = set(gate_cids(rows))
        rows = [r for r in rows if r['cid'] in want]
    if a.cids:
        rows = [r for r in rows if r['cid'] in set(a.cids)]
    done = set()
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        done = set(prev.loc[prev['parse_ok'].astype(bool) & (prev['model'] == a.model), 'cid'])
    else:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
        pd.DataFrame(columns=OUT_COLS).to_csv(a.out, index=False)
    todo = [r for r in rows if r['cid'] not in done]
    print(f'{len(rows)} rows, {len(done)} already done, {len(todo)} to run', flush=True)
    model_seen, t0 = None, time.time()
    for i, r in enumerate(todo):
        p = build_prompt(r, a.condition); err = ''
        try:
            txt, model_seen = call(p, a, key); parsed = parse_response(txt, 'binary'); n_calls = 1
            if parsed is None:
                txt, model_seen = call(p + '\n' + RETRY_SUFFIX, a, key); parsed = parse_response(txt, 'binary'); n_calls = 2
        except Exception as e:  # recorded, not accepted; rerun the script to resume
            txt, parsed, n_calls, err = '', None, 0, repr(e)[:300]
        rec = dict(cid=r['cid'], nct_id=r['nct_id'], condition=a.condition, family='binary', model=a.model,
                   model_returned=model_seen or '', provider=a.provider, truth=r['delta'], year_posted=r['year_posted'],
                   area=r['area'], ctrl_kind=r['ctrl_kind'], n_treat=r['n_treat'], n_ctrl=r['n_ctrl'],
                   parse_ok=parsed is not None, n_calls=n_calls, point=None, low=None, high=None,
                   raw_response=(txt or '')[:300], error=err)
        rec.update(parsed or {})
        pd.DataFrame([rec], columns=OUT_COLS).to_csv(a.out, mode='a', header=False, index=False)
        if (i + 1) % 50 == 0:
            print(f'{i + 1}/{len(todo)}', flush=True)
    D = pd.read_csv(a.out)
    D = D[D['model'] == a.model]
    acc = D[D['parse_ok'].astype(bool)].drop_duplicates('cid', keep='last')
    meta = dict(model_requested=a.model, model_returned=model_seen, provider=a.provider, endpoint=a.endpoint,
                precision_note=a.precision_note, condition=a.condition, temperature=0, json_mode=True,
                max_tokens=MAX_TOKENS, n_rows_requested=len(rows), n_accepted=int(acc['cid'].nunique()),
                n_retries=int((D['n_calls'] == 2).sum()), wall_seconds=round(time.time() - t0, 1), gate=a.gate)
    if a.gate:
        sub = D.drop_duplicates('cid', keep='last')
        pts = acc['point'].round(6)
        meta.update(gate_cids=sorted(sub['cid']), parse_rate=float(sub['parse_ok'].astype(bool).mean()),
                    n_unique_points=int(pts.nunique()),
                    modal_value=(float(pts.mode().iloc[0]) if len(pts) else None),
                    modal_share=(float(pts.value_counts().iloc[0] / len(pts)) if len(pts) else None))
        meta['adequate'] = bool(meta['parse_rate'] >= 0.90 and (meta['modal_share'] or 1.0) <= 0.50)
    json.dump(meta, open(os.path.splitext(a.out)[0] + '_run.json', 'w'), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
