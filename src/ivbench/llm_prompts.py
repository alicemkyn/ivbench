"""Shared prompt construction for the protocol-reader arm (benchmark v2).

Two conditions on the binary test contrasts:
  A  FULL: protocol + treatment/control intervention names, intervention descriptions, arm descriptions,
             results-group titles and descriptions.
  B  BLIND: identical protocol fields; every intervention-identifying field replaced by a fixed withheld
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


# ---------------------------------------------------------------------------------------------------------------
# Camera-ready additions (Oct 2026). Everything above this line is unchanged, so build_prompt() renders the
# original condition-A / condition-B prompts byte-for-byte. The functions below add four further renderings.
#
#   build_prompt_protocol(r, cond)  protocol-only variant of A / B: drops every results-derived field
#                                   (results-group labels and descriptions, analysed n per arm, baseline mean age and
#                                   fraction female; participant flow is never rendered by any template).
#   build_prompt_swap(r)            condition A with the TREATMENT and CONTROL arms exchanged (labels, intervention
#                                   names, intervention descriptions, arm/group descriptions, arm sizes, control type).
#   build_prompt_negctrl(r)         condition A with the treatment intervention replaced by the pre-registered unrelated
#                                   drug (NEGCTRL_RULE, see results/README.md); original treatment tokens redacted.
#   build_prompt_classguess(r)      condition-B trial block with a question asking for the withheld treatment's class.
#   audit_blind(prompt, r)          redaction auditor used for the protocol-only blind prompts.
# ---------------------------------------------------------------------------------------------------------------

def _num_or_nr(x, fmt):
    return 'not reported' if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, fmt)


def _arm_block_A(r, swap=False):
    t, c = ('ctrl', 'treat') if swap else ('treat', 'ctrl')
    return [f"TREATMENT arm label: {_s(r.get(t + '_group'))}",
            f"TREATMENT arm intervention name(s): {_s(r.get(t + '_intervention'))}",
            f"TREATMENT arm intervention description: {_s(r.get(t + '_intervention_desc'), 600)}",
            f"TREATMENT arm description: {_s(r.get(t + '_arm_desc'), 600)} {_s(r.get(t + '_group_desc'), 600)}".strip(),
            f"CONTROL arm label: {_s(r.get(c + '_group'))}",
            f"CONTROL arm intervention name(s): {_s(r.get(c + '_intervention'))}",
            f"CONTROL arm intervention description: {_s(r.get(c + '_intervention_desc'), 600)}",
            f"CONTROL arm description: {_s(r.get(c + '_arm_desc'), 600)} {_s(r.get(c + '_group_desc'), 600)}".strip()]


def _trial_block(r, R, protocol_only=False, swap=False, ctrl_line=None):
    lines = ['--- TRIAL ---',
             f"Phase: {_s(r.get('phase'))}",
             f"Condition(s): {R(r.get('conditions'))}",
             f"Eligibility criteria (truncated): {R(r.get('eligibility_text'), 1500)}", '',
             f"Endpoint title: {R(r.get('outcome_title'))}",
             f"Endpoint description: {R(r.get('outcome_desc'), 900)}",
             f"Endpoint timeframe: {R(r.get('outcome_timeframe'))}",
             f"Endpoint unit as reported: {R(r.get('unit'))}", '']
    if not protocol_only:
        nt, nc = (r['n_ctrl'], r['n_treat']) if swap else (r['n_treat'], r['n_ctrl'])
        lines += [f"Treatment arm size (n): {int(nt)}", f"Control arm size (n): {int(nc)}"]
    lines += [f"Control type: {ctrl_line or ctrl_type_text(r)}"]
    if not protocol_only:
        lines += [f"Population mean age (years): {_num_or_nr(r.get('age_mean'), '.1f')}",
                  f"Population fraction female: {_num_or_nr(r.get('frac_female'), '.3f')}"]
    return lines + ['']


def build_prompt_protocol(r, cond):
    """Protocol-only rendering. A: registered arm/intervention fields only (no results-group label or description).
    B: identical to the original blind arm block (already free of group fields), same redaction, minus results fields."""
    toks = intervention_tokens(r) if cond == 'B' else set()
    R = (lambda s, n=None: redact(_s(s, n), toks)) if cond == 'B' else (lambda s, n=None: _s(s, n))
    hdr = HEADER_CONT if r.get('family') == 'continuous' else HEADER
    lines = [hdr, ''] + _trial_block(r, R, protocol_only=True)
    if cond == 'A':
        lines += [f"TREATMENT arm intervention name(s): {_s(r.get('treat_intervention'))}",
                  f"TREATMENT arm intervention description: {_s(r.get('treat_intervention_desc'), 600)}",
                  f"TREATMENT arm description: {_s(r.get('treat_arm_desc'), 600)}",
                  f"CONTROL arm intervention name(s): {_s(r.get('ctrl_intervention'))}",
                  f"CONTROL arm intervention description: {_s(r.get('ctrl_intervention_desc'), 600)}",
                  f"CONTROL arm description: {_s(r.get('ctrl_arm_desc'), 600)}"]
    else:
        lines += ["TREATMENT arm intervention name(s): [withheld]",
                  "TREATMENT arm intervention description: [withheld]",
                  "TREATMENT arm description: [withheld]",
                  f"CONTROL arm: the control condition ({ctrl_type_text(r)})",
                  "CONTROL arm intervention name(s): [withheld]",
                  "CONTROL arm intervention description: [withheld]",
                  "CONTROL arm description: [withheld]"]
    lines += ['', '--- END TRIAL ---', 'Return only the JSON object.']
    return '\n'.join(lines)


SWAP_CTRL_TEXT = {'inert_control': 'active comparator (another active treatment)',
                  'active_comparator': 'active comparator (another active treatment)'}


def build_prompt_swap(r):
    """Condition A with arms exchanged. The control-type line describes the arm now labelled CONTROL (for an
    inert-control trial that arm is the active drug, so it is rendered as an active comparator). A model that reads the
    arms correctly should return approximately the negative of its condition-A prediction."""
    hdr = HEADER_CONT if r.get('family') == 'continuous' else HEADER
    R = lambda s, n=None: _s(s, n)
    lines = [hdr, ''] + _trial_block(r, R, swap=True, ctrl_line=SWAP_CTRL_TEXT[r.get('ctrl_kind')])
    lines += _arm_block_A(r, swap=True)
    lines += ['', '--- END TRIAL ---', 'Return only the JSON object.']
    return '\n'.join(lines)


# Pre-registered negative-control substitution (results/README.md, section "Negative-control pre-registration").
NEGCTRL_RULE = {
    'cardiometabolic': ('Loratadine', 'Loratadine 10 mg tablet taken by mouth once daily (a second-generation H1 antihistamine '
                                      'used for seasonal allergic rhinitis).'),
    'default': ('Ezetimibe', 'Ezetimibe 10 mg tablet taken by mouth once daily (a cholesterol-absorption inhibitor used to lower '
                             'LDL cholesterol).'),
}


def negctrl_drug(r):
    return NEGCTRL_RULE['cardiometabolic' if r.get('area') == 'cardiometabolic' else 'default']


def treat_tokens(r):
    """Name-like tokens from the TREATMENT arm fields only (to remove the original drug from retained text)."""
    rr = {k: (r.get(k) if k.startswith('treat') else None) for k in r}
    return intervention_tokens(rr)


def build_prompt_negctrl(r):
    name, desc = negctrl_drug(r)
    toks = treat_tokens(r) - {w.lower() for w in re.findall(r"[A-Za-z]+", _s(r.get('ctrl_intervention')))}
    R = lambda s, n=None: redact(_s(s, n), toks)
    hdr = HEADER_CONT if r.get('family') == 'continuous' else HEADER
    lines = [hdr, ''] + _trial_block(r, R)
    lines += [f"TREATMENT arm label: {name}",
              f"TREATMENT arm intervention name(s): {name}",
              f"TREATMENT arm intervention description: {desc}",
              f"TREATMENT arm description: Participants receive {name.lower()} 10 mg by mouth once daily.",
              f"CONTROL arm label: {R(r.get('ctrl_group'))}",
              f"CONTROL arm intervention name(s): {R(r.get('ctrl_intervention'))}",
              f"CONTROL arm intervention description: {R(r.get('ctrl_intervention_desc'), 600)}",
              f"CONTROL arm description: {R(r.get('ctrl_arm_desc'), 600)} {R(r.get('ctrl_group_desc'), 600)}".strip()]
    lines += ['', '--- END TRIAL ---', 'Return only the JSON object.']
    return '\n'.join(lines)


CLASSGUESS_HEADER = (
    "You are given protocol information for one randomized clinical trial in which the identity of the experimental "
    "treatment has been withheld. Based only on the information given, state your best guess of the withheld "
    "experimental treatment's drug class (or intervention class, if it is not a drug) and its mechanism of action.\n\n"
    "Respond with STRICT JSON only, no prose, no markdown fences:\n"
    "{\"drug_class\": <string>, \"mechanism\": <string>, \"example_agent\": <string>}"
)
CLASSGUESS_RETRY = 'IMPORTANT: output must be exactly one JSON object with keys drug_class, mechanism, example_agent and nothing else.'


def build_prompt_classguess(r):
    """Condition-B trial block (byte-identical to build_prompt(r, 'B') between the TRIAL markers) with the class question."""
    b = build_prompt(r, 'B')
    body = b[b.index('--- TRIAL ---'): b.index('--- END TRIAL ---') + len('--- END TRIAL ---')]
    return CLASSGUESS_HEADER + '\n\n' + body + '\nReturn only the JSON object.'


def parse_classguess(text):
    if not isinstance(text, str):
        return None
    m = re.search(r'\{[^{}]*\}', text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return dict(drug_class=str(d['drug_class']), mechanism=str(d['mechanism']), example_agent=str(d.get('example_agent', '')))
    except Exception:
        return None


JUDGE_PROMPT = (
    "You are scoring a guess about the drug class of a clinical-trial intervention.\n\n"
    "TRUE intervention name(s): {true_name}\nTRUE intervention description: {true_desc}\n\n"
    "GUESSED drug class: {g_class}\nGUESSED mechanism: {g_mech}\nGUESSED example agent: {g_ex}\n\n"
    "Rubric. class_match = true if the guessed drug class names the same pharmacological or therapeutic class as the true "
    "intervention at the level of a standard class (for example 'SGLT2 inhibitor', 'anti-PD-1 antibody', 'proton pump "
    "inhibitor', 'mRNA vaccine', 'corticosteroid', 'probiotic'); a broader umbrella that would also cover most other "
    "drugs used in the same disease (for example 'immunotherapy', 'antibiotic' when the true agent is a specific "
    "antibiotic class, 'anticancer drug', 'vaccine' for a vaccine of a different platform) is NOT a match, except "
    "that 'vaccine' matches a vaccine against the same pathogen. mechanism_match = true if the guessed mechanism "
    "names the same molecular target or mechanism. exact_agent = true if the guessed example agent is the true "
    "agent (same active ingredient).\n\n"
    "Respond with STRICT JSON only: {{\"class_match\": <true|false>, \"mechanism_match\": <true|false>, \"exact_agent\": <true|false>}}"
)


def build_judge_prompt(r, guess):
    return JUDGE_PROMPT.format(true_name=_s(r.get('treat_intervention')) or _s(r.get('treat_group')),
                               true_desc=(_s(r.get('treat_intervention_desc'), 400) or _s(r.get('treat_arm_desc'), 400)),
                               g_class=guess.get('drug_class', ''), g_mech=guess.get('mechanism', ''),
                               g_ex=guess.get('example_agent', ''))


def parse_judge(text):
    if not isinstance(text, str):
        return None
    m = re.search(r'\{[^{}]*\}', text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return {k: bool(d[k]) for k in ('class_match', 'mechanism_match', 'exact_agent')}
    except Exception:
        return None


RETAINED_FIELDS = ('conditions', 'eligibility_text', 'outcome_title', 'outcome_desc', 'outcome_timeframe', 'unit')


def audit_blind(prompt, r, render=None):
    """Redaction audit of one blind prompt. Candidate tokens are the contrast's own intervention-derived name tokens,
    intervention_tokens(r) (length >= 4; capitalised, containing a digit or carrying a pharmacological suffix; from all
    ten intervention/arm/group fields; minus COMMON). A token "survives" if it occurs as a whole word (case-insensitive)
    anywhere in the blind prompt. Survivors are then split into template words (the token also occurs in the same
    prompt re-rendered with every retained free-text field emptied, i.e. it belongs to the fixed header / field labels)
    and leaks (it occurs only through the retained trial text). `render` re-renders a row (default: protocol-only B).
    Returns dict(n_candidate, surviving, template_words, leaks, suffix_words) with sorted token lists."""
    render = render or (lambda rr: build_prompt_protocol(rr, 'B'))
    toks = intervention_tokens(r)
    hit = lambda t, s: re.search(r'(?i)\b' + re.escape(t) + r'\b', s) is not None
    surv = sorted(t for t in toks if hit(t, prompt))
    empty = dict(r); empty.update({k: '' for k in RETAINED_FIELDS})
    skeleton = render(empty)
    templ = sorted(t for t in surv if hit(t, skeleton))
    body = prompt[prompt.index('--- TRIAL ---'):]
    return dict(n_candidate=len(toks), surviving=surv, template_words=templ, leaks=sorted(set(surv) - set(templ)),
                suffix_words=sorted({m.group(0).lower() for m in SUFFIX_RE.finditer(body)}))
