"""Llama 3.1 70B Instruct (BF16, dedicated deployment) runs for the camera-ready memorisation test.
Same prompt builders, parser and one-retry rule as the Claude runs; JSON mode on, temperature 0.
Conditions: protocol_A / protocol_B (protocol-time fields; primary) and posted_A / posted_B (original prompts).
Rows: the 1,020 v2 binary test contrasts plus the 66 binary test contrasts that are new or re-paired in v2.1.
"""
import json, os, time, urllib.request, urllib.error, threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

ENDPOINT = 'https://api.fireworks.ai/inference/v1/chat/completions'


def post(body, key, timeout=300):
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
    last = None
    for attempt in range(7):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = f'HTTP {e.code}: {e.read().decode()[:200]}'
            if e.code in (408, 409, 429, 500, 502, 503, 504, 529):
                time.sleep(min(60, 3 * 2 ** attempt)); continue
            break
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(e)[:200]; time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError(last)


def build(L, cond, r):
    if cond == 'protocol_A': return L.build_prompt_protocol(r, 'A')
    if cond == 'protocol_B': return L.build_prompt_protocol(r, 'B')
    if cond == 'posted_A': return L.build_prompt(r, 'A')
    if cond == 'posted_B': return L.build_prompt(r, 'B')
    raise ValueError(cond)


def run(L, rows, conds, model, key, out_csv, workers=16):
    done = set()
    if os.path.exists(out_csv):
        D = pd.read_csv(out_csv)
        done = set(zip(D.condition, D.cid))
    lock = threading.Lock()
    tasks = [(c, r) for c in conds for r in rows if (c, r['cid']) not in done]

    def one(t):
        cond, r = t
        p = build(L, cond, r)
        rec = dict(cid=r['cid'], nct_id=r['nct_id'], condition=cond, truth=r.get('delta'), model=model)
        n = 0
        try:
            for i, prompt in enumerate([p, p + '\n' + L.RETRY_SUFFIX]):
                body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
                        'max_tokens': L.MAX_TOKENS, 'response_format': {'type': 'json_object'}}
                j = post(body, key); n += 1
                txt = j['choices'][0]['message']['content'] or ''
                parsed = L.parse_response(txt, 'binary')
                if parsed is not None:
                    pt, lo, hi = (parsed['point'], parsed.get('low'), parsed.get('high')) if isinstance(parsed, dict) else parsed[:3]
                    rec.update(point=pt, low=lo, high=hi, parse_ok=True, n_calls=n, raw=txt[:300], error='')
                    break
            else:
                rec.update(parse_ok=False, n_calls=n, raw=txt[:300], error='parse')
        except Exception as e:
            rec.update(parse_ok=False, n_calls=n, raw='', error=str(e)[:200])
        with lock:
            pd.DataFrame([rec]).to_csv(out_csv, mode='a', header=not os.path.exists(out_csv), index=False)
        return rec

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, tasks))
