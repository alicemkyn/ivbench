"""Model-assisted pairing/conversion review of the 150 audit contrasts (run inside the Claude Science python kernel, which
provides `host.llm`). Model: host.reasoning_model() (claude-sonnet-5 on 2026-10-03), max_tokens 600, retried once with
max_tokens 4000 when the response was empty (34 of 150). Deterministic checks come from audit_checks.check.
Output: audit_results.csv (one row per sampled contrast; raw model text in llm_raw). Judgements are model-assisted, not human-verified."""
import json, re
from audit_checks import *
CONV_Q = ('\nAlso judge the outcome conversion. Add keys: "binary_participant_proportion": true/false/null (for a count/percentage outcome: the value is '
          'the number or percentage of participants with the endpoint, so dividing by the analysed participants gives a proportion; null if not a count/percentage outcome), '
          '"continuous_arithmetic_mean_with_valid_dispersion": true/false/null (for a mean outcome: an arithmetic mean of a participant-level measure whose '
          'dispersion is an SD, an SE of that arm mean, or a 95% CI of that arm mean; false for LS means from models, geometric means, medians, or CIs of differences; null if not a mean outcome), '
          '"conversion_issue": short string.')

def pj(t):
    m = re.search(r'\{.*\}', t or '', re.S)
    try: return json.loads(m.group(0))
    except Exception: return None

def cat(r):
    if r.llm_pairing_valid == 'yes': return 'valid'
    if r.llm_concurrent_distinct_arms == False: return 'not concurrent distinct arms (crossover/sequential/pooled/subgroup)'
    if r.llm_treat_is_intervention_arm == False: return 'treatment group is not the investigational arm (reversed or comparator chosen)'
    if r.llm_ctrl_is_comparator_arm == False: return 'comparator group is another active/investigational dose'
    if r.llm_ctrl_kind_correct == False: return 'comparator kind label wrong only'
    return 'other'

def run(host):
    REC = load_records(); S = pd.read_csv('audit_samples.csv', low_memory=False)
    chk = pd.DataFrame([check(r, REC[r.nct_id]) for r in S.itertuples()])
    chk.insert(0, 'sample', S['sample'].values); chk.insert(1, 'row_id', S.row_id.values)
    prompts = [pairing_prompt(r, REC[r.nct_id]) + CONV_Q + ' Output only the JSON object.' for r in S.itertuples()]
    model = host.reasoning_model()
    out = host.llm([dict(prompt=p, max_tokens=600, model=model) for p in prompts], max_concurrency=8)
    J = [pj(o.get('text')) for o in out]; bad = [i for i, j in enumerate(J) if j is None]
    for i, o in zip(bad, host.llm([dict(prompt=prompts[i], max_tokens=4000, model=model) for i in bad], max_concurrency=8)):
        out[i] = o; J[i] = pj(o.get('text'))
    JD = pd.DataFrame([j or {} for j in J]); JD.columns = ['llm_' + c for c in JD.columns]
    AU = pd.concat([chk.reset_index(drop=True), JD], axis=1)
    AU['llm_raw'] = [o.get('text') for o in out]; AU['llm_model'] = model
    AU['llm_max_tokens'] = [4000 if i in bad else 600 for i in range(len(out))]
    AU['audit_category'] = AU.apply(cat, axis=1)
    AU['rand_parallel'] = (AU.allocation == 'RANDOMIZED') & (AU.intervention_model == 'PARALLEL')
    AU.to_csv('audit_results.csv', index=False); return AU
