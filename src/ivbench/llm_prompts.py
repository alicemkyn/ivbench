"""Shared prompt construction for the protocol-reader arm (benchmark v2).

Two conditions on the binary test contrasts:
  A  FULL  — protocol + treatment/control intervention names, intervention descriptions, arm descriptions,
             results-group titles and descriptions.
  B  BLIND — identical protocol fields; every intervention-identifying field replaced by a fixed withheld
             string, and intervention-derived name tokens removed from the retained free text.

Usage (in a python kernel):
    exec(open('llm_prompts_v2.py').read())
    rows = load_rows(path_to_contrasts_csv_gz)            # sorted, with contrast ids c0000..
    prompts_A = [build_prompt(r, 'A') for r in rows]
    prompts_B = [build_prompt(r, 'B') for r in rows]
    parse_response(text) -> dict(point, low, high) or None
"""
import json, re
import numpy as np, pandas as pd

SEED = 20260915
MAX_TOKENS = 1500
RETRY_SUFFIX = 'IMPORTANT: output must be exactly one JSON object with keys point, low, high and nothing else.'

HEADER = (
    "You are simulating the results of a randomized clinical trial. You are given protocol information for one "
    "trial and one binary primary endpoint (an event or responder proportion). Predict the observed risk difference "
    "between the two arms.\n\n"
    "Definition: risk difference = (proportion of participants in the TREATMENT arm with the endpoint) minus "
    "(proportion in the CONTROL arm with the endpoint). Both proportions are on a 0-1 fraction scale, so the risk "
    "difference lies in [-1, 1]. A higher endpoint value is not necessarily clinically better - predict the sign as "
    "measured, not as desirable.\n\n"
    "Respond with STRICT JSON only, no prose, no markdown fences:\n"
    "{\"point\": <number>, \"low\": <number>, \"high\": <number>}\n"
    "where point is your best point estimate and [low, high] is a 90% central interval for the observed risk difference."
)

HEADER_CONT = (
    "You are simulating the results of a randomized clinical trial. You are given protocol information for one "
    "trial and one continuous primary endpoint (a measured quantity such as a score, level or duration). Predict the "
    "observed standardized mean difference between the two arms.\n\n"
    "Definition: standardized mean difference (SMD, Cohen's d) = (mean in the TREATMENT arm) minus (mean in the "
    "CONTROL arm), divided by the pooled standard deviation of the endpoint at the analysis timepoint. Typical trial "
    "SMDs lie in [-2, 2]; |d| = 0.2 is small, 0.5 medium, 0.8 large. A higher endpoint value is not necessarily "
    "clinically better - predict the sign as measured (treatment minus control), not as desirable.\n\n"
    "Respond with STRICT JSON only, no prose, no markdown fences:\n"
    "{\"point\": <number>, \"low\": <number>, \"high\": <number>}\n"
    "where point is your best point estimate and [low, high] is a 90% central interval for the observed SMD."
)

WITHHELD = '[withheld]'
SUFFIX_RE = re.compile(r'(?i)\b\w{3,}(mab|nib|ciclib|parib|tinib|prazole|sartan|floxacin|cycline|mycin|micin|cillin|'
                       r'glutide|gliptin|glitazone|setron|statin|afil|vir|cept|zumab|ximab|lumab|umab|pril|olol|dipine|'
                       r'azepam|oxetine|triptan|conazole|lukast|tidine|profen|caine|barbital|cort|sone|olone|relin|'
                       r'tropin|kinra|ase|zole|tide)\b')


def _s(x, n=None):
    x = '' if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)
    return x[:n] if n else x


def load_rows(path, family='binary'):
    C = pd.read_csv(path)
    C = C[(C['family'] == family) & (C['split_time'] == 'test')].copy()
    C = C.sort_values(['nct_id', 'outcome_title', 'treat_group']).reset_index(drop=True)
    prefix = 'c' if family == 'binary' else 'k'
    C['cid'] = [f'{prefix}{i:04d}' for i in range(len(C))]
    C['family'] = family
    return C.to_dict('records')


def intervention_tokens(r):
    """Name-like tokens derived from the intervention/arm fields of one contrast (for redacting retained text)."""
    txt = ' '.join(_s(r.get(k)) for k in ('treat_intervention', 'treat_intervention_desc', 'treat_arm_desc',
                                           'treat_group', 'treat_group_desc', 'ctrl_intervention',
                                           'ctrl_intervention_desc', 'ctrl_arm_desc', 'ctrl_group', 'ctrl_group_desc'))
    toks = set()
    for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", txt):
        tl = t.lower()
        if SUFFIX_RE.search(tl) or (t[0].isupper() and tl not in COMMON) or re.search(r'\d', t):
            toks.add(tl)
    return toks


COMMON = set("""placebo control treatment arm group dose doses daily weekly once twice oral intravenous subcutaneous
tablet tablets capsule capsules injection infusion mg ml kg standard care usual therapy patients participants
subjects received receive week weeks day days month months year years plus versus with without and the for per
open label blinded double single randomized randomised active comparator sham vehicle matching identical
combination monotherapy background maintenance induction cycle cycles phase study trial arm cohort high low
medium total""".split())


def redact(text, toks):
    if not text:
        return text
    out = text
    for t in sorted(toks, key=len, reverse=True):
        out = re.sub(r'(?i)\b' + re.escape(t) + r'\b', WITHHELD, out)
    out = SUFFIX_RE.sub(WITHHELD, out)
    return re.sub(r'(\[withheld\]\s*){2,}', WITHHELD + ' ', out)


def ctrl_type_text(r):
    return ('inert control (placebo / sham / no-treatment)' if r.get('ctrl_kind') == 'inert_control'
            else 'active comparator (another active treatment)')


def build_prompt(r, cond):
    toks = intervention_tokens(r) if cond == 'B' else set()
    R = (lambda s, n=None: redact(_s(s, n), toks)) if cond == 'B' else (lambda s, n=None: _s(s, n))
    age = r.get('age_mean'); ff = r.get('frac_female')
    hdr = HEADER_CONT if r.get('family') == 'continuous' else HEADER
    lines = [hdr, '', '--- TRIAL ---',
             f"Phase: {_s(r.get('phase'))}",
             f"Condition(s): {R(r.get('conditions'))}",
             f"Eligibility criteria (truncated): {R(r.get('eligibility_text'), 1500)}", '',
             f"Endpoint title: {R(r.get('outcome_title'))}",
             f"Endpoint description: {R(r.get('outcome_desc'), 900)}",
             f"Endpoint timeframe: {R(r.get('outcome_timeframe'))}",
             f"Endpoint unit as reported: {R(r.get('unit'))}", '',
             f"Treatment arm size (n): {int(r['n_treat'])}",
             f"Control arm size (n): {int(r['n_ctrl'])}",
             f"Control type: {ctrl_type_text(r)}",
             f"Population mean age (years): {'not reported' if age is None or (isinstance(age, float) and np.isnan(age)) else f'{age:.1f}'}",
             f"Population fraction female: {'not reported' if ff is None or (isinstance(ff, float) and np.isnan(ff)) else f'{ff:.3f}'}", '']
    if cond == 'A':
        lines += [f"TREATMENT arm label: {_s(r.get('treat_group'))}",
                  f"TREATMENT arm intervention name(s): {_s(r.get('treat_intervention'))}",
                  f"TREATMENT arm intervention description: {_s(r.get('treat_intervention_desc'), 600)}",
                  f"TREATMENT arm description: {_s(r.get('treat_arm_desc'), 600)} {_s(r.get('treat_group_desc'), 600)}".strip(),
                  f"CONTROL arm label: {_s(r.get('ctrl_group'))}",
                  f"CONTROL arm intervention name(s): {_s(r.get('ctrl_intervention'))}",
                  f"CONTROL arm intervention description: {_s(r.get('ctrl_intervention_desc'), 600)}",
                  f"CONTROL arm description: {_s(r.get('ctrl_arm_desc'), 600)} {_s(r.get('ctrl_group_desc'), 600)}".strip()]
    else:
        lines += ["TREATMENT arm label: the experimental intervention [withheld]",
                  "TREATMENT arm intervention name(s): [withheld]",
                  "TREATMENT arm intervention description: [withheld]",
                  "TREATMENT arm description: [withheld]",
                  f"CONTROL arm label: the control condition ({ctrl_type_text(r)})",
                  "CONTROL arm intervention name(s): [withheld]",
                  "CONTROL arm intervention description: [withheld]",
                  "CONTROL arm description: [withheld]"]
    lines += ['', '--- END TRIAL ---', 'Return only the JSON object.']
    return '\n'.join(lines)


def parse_response(text, family='binary'):
    lim = 1.0 if family == 'binary' else 5.0
    if not isinstance(text, str):
        return None
    m = re.search(r'\{[^{}]*\}', text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        p, lo, hi = float(d['point']), float(d['low']), float(d['high'])
    except Exception:
        return None
    if not all(np.isfinite([p, lo, hi])) or lo > hi:
        return None
    return dict(point=max(-lim, min(lim, p)), low=max(-lim, min(lim, lo)), high=max(-lim, min(lim, hi)))
