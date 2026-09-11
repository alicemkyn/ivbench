"""Protocol-reader arm (paper Sec. 5.4, Tables 1, 4, 5, 15, 16). Builds the FULL (A) and BLIND (B) prompts with
ivbench.llm_prompts and sends them to a chat endpoint; reproduces the prompt construction byte-for-byte and the
parse/retry policy (one retry with RETRY_SUFFIX appended on parse failure).

Endpoints (choose with --api):
  anthropic  Anthropic Messages API (what the paper used).  env: ANTHROPIC_API_KEY
             models used in the paper: claude-sonnet-5 (large), claude-haiku-4-5-20251001 (small)
  openai     any OpenAI-compatible /v1/chat/completions endpoint.  env: LLM_ENDPOINT, LLM_API_KEY

  python scripts/run_llm_arm.py --api anthropic --model claude-sonnet-5 --family binary --condition A \\
      --contrasts data/benchmark_contrasts.csv.gz --out results/llm/rerun_sonnet_binary_A.csv

The paper's runs were executed through a batched interface with max_concurrency 8; the prompts, model ids,
max_tokens and retry rule are identical. Provider usage-limit errors are retried after a pause and are not counted
as parse retries (n_calls counts accepted attempts only, as in the released prediction files).
"""
import argparse, json, os, sys, time, urllib.request, urllib.error
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ivbench.llm_prompts import load_rows, build_prompt, parse_response, RETRY_SUFFIX, MAX_TOKENS

ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages'


def _post(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', **headers})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < 5:
                time.sleep(2 ** attempt * 5); continue
            raise


def call(prompt, model, api):
    if api == 'anthropic':
        j = _post(ANTHROPIC_URL, {'model': model, 'max_tokens': MAX_TOKENS, 'messages': [{'role': 'user', 'content': prompt}]},
                  {'x-api-key': os.environ['ANTHROPIC_API_KEY'], 'anthropic-version': '2023-06-01'})
        return ''.join(b.get('text', '') for b in j['content']), j.get('model', model)
    j = _post(os.environ['LLM_ENDPOINT'], {'model': model, 'max_tokens': MAX_TOKENS, 'messages': [{'role': 'user', 'content': prompt}]},
              {'Authorization': f"Bearer {os.environ.get('LLM_API_KEY', '')}"})
    return j['choices'][0]['message']['content'], j.get('model', model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contrasts', default=os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmark_contrasts.csv.gz'))
    ap.add_argument('--condition', choices=['A', 'B'], required=True, help='A = FULL (intervention shown), B = BLIND (withheld)')
    ap.add_argument('--family', choices=['binary', 'continuous'], default='binary')
    ap.add_argument('--api', choices=['anthropic', 'openai'], default='anthropic')
    ap.add_argument('--model', default='claude-sonnet-5')
    ap.add_argument('--out', required=True); ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--cids', nargs='*', default=None, help='optional subset of contrast ids (e.g. to complete a partial run)')
    a = ap.parse_args()
    rows = load_rows(a.contrasts, a.family)
    if a.cids: rows = [r for r in rows if r['cid'] in set(a.cids)]
    rows = rows[: a.limit]
    out, model_seen = [], None
    for i, r in enumerate(rows):
        p = build_prompt(r, a.condition)
        txt, model_seen = call(p, a.model, a.api); parsed = parse_response(txt, a.family); n_calls = 1
        if parsed is None:
            txt, model_seen = call(p + '\n' + RETRY_SUFFIX, a.model, a.api); parsed = parse_response(txt, a.family); n_calls = 2
        out.append(dict(cid=r['cid'], nct_id=r['nct_id'], condition=a.condition, family=a.family, truth=r['delta'],
                        year_posted=r['year_posted'], area=r['area'], ctrl_kind=r['ctrl_kind'], n_treat=r['n_treat'], n_ctrl=r['n_ctrl'],
                        parse_ok=parsed is not None, n_calls=n_calls, **(parsed or {}), raw_response=(txt or '')[:300]))
        if (i + 1) % 50 == 0: print(f'{i + 1}/{len(rows)}', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.DataFrame(out).to_csv(a.out, index=False)
    json.dump(dict(model_requested=a.model, model_returned=model_seen, api=a.api, family=a.family, condition=a.condition,
                   n_rows=len(out), n_parsed=int(sum(o['parse_ok'] for o in out)), n_retries=int(sum(o['n_calls'] - 1 for o in out)),
                   max_tokens=MAX_TOKENS), open(os.path.splitext(a.out)[0] + '_run.json', 'w'), indent=1)
    print(f"{len(out)} rows, parsed {sum(o['parse_ok'] for o in out)}, retries {sum(o['n_calls'] - 1 for o in out)}")


if __name__ == '__main__':
    main()
