"""v2.1 re-audit: same procedure as the v2 audit (audit_checks.check deterministic re-derivation + the same model-assisted
pairing/conversion review prompt, model host.reasoning_model(), max_tokens 600, one retry at 4000 for unparsable replies).
Run inside the Claude Science python kernel (needs `host.llm`):  exec(open('audit_v21.py').read()); AU = run_v21(host)
Inputs: audit_samples_v21.csv, audit_records_v21.jsonl.gz. Output: audit_results_v21.csv. Judgements are model-assisted."""
import json, re, datetime
from audit_checks import *
from audit_llm_review import CONV_Q, pj, cat

def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def run_v21(host, samples='audit_samples_v21.csv', records='audit_records_v21.jsonl.gz', out='audit_results_v21.csv'):
    REC = load_records(records); S = pd.read_csv(samples, low_memory=False)
    chk = pd.DataFrame([check(r, REC[r.nct_id]) for r in S.itertuples()])
    chk.insert(0, 'sample', S['sample'].values); chk.insert(1, 'row_id', S.row_id.values); chk['cid'] = S['cid'].values
    chk['v21_status'] = S['v21_status'].values
    prompts = [pairing_prompt(r, REC[r.nct_id]) + CONV_Q + ' Output only the JSON object.' for r in S.itertuples()]
    model = host.reasoning_model()
    out_ = host.llm([dict(prompt=p, max_tokens=600, model=model) for p in prompts], max_concurrency=8)
    J = [pj(o.get('text')) for o in out_]; bad = [i for i, j in enumerate(J) if j is None]
    if bad:
        for i, o in zip(bad, host.llm([dict(prompt=prompts[i], max_tokens=4000, model=model) for i in bad], max_concurrency=8)):
            out_[i] = o; J[i] = pj(o.get('text'))
    JD = pd.DataFrame([j or {} for j in J]); JD.columns = ['llm_' + c for c in JD.columns]
    AU = pd.concat([chk.reset_index(drop=True), JD], axis=1)
    AU['llm_raw'] = [o.get('text') for o in out_]; AU['llm_model'] = model
    AU['llm_call_date'] = datetime.date.today().isoformat()
    AU['llm_max_tokens'] = [4000 if i in bad else 600 for i in range(len(out_))]
    AU['llm_parsed'] = [j is not None for j in J]
    AU['audit_category'] = AU.apply(cat, axis=1)
    AU['rand_parallel'] = (AU.allocation == 'RANDOMIZED') & (AU.intervention_model == 'PARALLEL')
    AU.to_csv(out, index=False); return AU
