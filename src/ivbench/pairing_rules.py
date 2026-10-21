"""Benchmark v2.1 arm-pairing rules (camera-ready repair). build.py is left untouched; this module re-implements the
contrast loop of build.build with three additional rules, and reproduces v2 exactly when no rule is enabled.

Rules
  (a) same-molecule comparator: a comparator whose normalised molecule key equals the treatment's (different dose or
      schedule of the same drug) is not an admissible comparator for that treatment arm. The comparator is then chosen
      among the remaining candidates by the v2 preference (inert before active, largest n); if none remain the contrast
      is dropped.
  (b) non-concurrent arms:
      b1  trials whose designInfo.interventionModel is CROSSOVER or SEQUENTIAL are dropped;
      b2  results groups whose title matches POOLED_TITLE_PATTERNS or EXTENSION_TITLE_PATTERNS, or whose description
          matches POOLED_DESC_PATTERNS, are removed from the candidate pools;
      b3  a treatment and comparator must carry the same study-stage label (part / period / phase / stage / cohort /
          segment / step / panel, or a double-blind / open-label phase prefix); groups from different stages, or one
          staged and one unstaged group, are not paired.
  (c) investigational arm: the registered armGroupType is recovered for every results group (released v2 match, else
      re-matched against armGroups re-fetched from ClinicalTrials.gov API v2). When the trial has at least one
      EXPERIMENTAL results group whose own title is not a placebo/sham/vehicle label, the treatment pool is those
      groups and the comparator pool is the remaining groups with control roles; if that leaves no comparator, the v2
      rule (text/type role) is used for the trial. Roles of re-matched groups are recomputed with parse.arm_role.
"""
import re
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

from ivbench import parse as pt
from ivbench.specs import norm_name

NONCONCURRENT_MODELS = ('CROSSOVER', 'SEQUENTIAL')

# --- rule (b2): pooled / summary groups (title) ------------------------------------------------------------------
POOLED_TITLE_PATTERNS = [
    r'^\s*(all|total|overall|pooled|combined)\b',
    r'\b(all|total|overall|pooled|combined)\s*\W*$',
    r'\ball (participants|patients|subjects|doses|dose groups|arms|groups|treated|randomi[sz]ed|active|treatments|cohorts)\b',
    r'\bpooled\b',
    r'\b(average|avg)\b',
    r'\b(any|all) (active )?doses?\b',
    r'\btotal (active|drug|treated|population)\b',
    r'\bcombined (dose|doses|arms|groups|active|treatment)\b',
]
# --- rule (b2): pooled groups (results-group description; narrower, descriptions are free text) ----------------------
POOLED_DESC_PATTERNS = [
    r'\bpooled\b',
    r'\b(data|results) (from|of) (all|both) (arms|groups|doses|dose groups|cohorts)\b',
    r'\baverage (dose )?effect\b',
    r'\bcombined (data|analysis|groups|arms|doses|dose groups)\b',
    r'\b(includes|including|consists of|comprises) (all )?(participants|patients|subjects) (from|in) (all|both)\b',
]
# --- rule (b2): extension / rollover / prior-assignment / pre-treatment groups (title) -------------------------------
EXTENSION_TITLE_PATTERNS = [
    r'\bopen[- ]label extension\b',
    r'\bOLE\b',
    r'\bextension (study|phase|period|part|treatment)\b',
    r'\broll[- ]?over\b',
    r'\b(early|delayed)[- ](start|treated|treatment)\b',
    r'\b(from|in) (the )?(parent|lead[- ]in|previous|prior|preceding|core|feeder|antecedent|double[- ]blind) (study|studies|trial)\b',
    r'\b(from|in) study\s+[a-z0-9][a-z0-9-]{2,}',
    r'\blead[- ]in\b',
    r'\bparent stud(y|ies)\b',
    r'\bpre-?treatment\b',
    r'^\s*baseline\b',
    r'\bwashout\b',
    r'\bfollow[- ]?up (phase|period)\b',
]
# --- rule (b3): study-stage labels (title) ---------------------------------------------------------------------------
STAGE_PATTERNS = [
    r'\b(part|period|phase|stage|cohort|segment|step|panel|epoch)\s*[:#-]?\s*(\d+[a-z]?|[ivx]{1,4}|[a-h])\b',
    r'\b(double[- ]blind|open[- ]label|db|ol)\s*(phase|period|part|stage|treatment period|:)',
]
_POOLED_T = [re.compile(p, re.I) for p in POOLED_TITLE_PATTERNS]
_POOLED_D = [re.compile(p, re.I) for p in POOLED_DESC_PATTERNS]
_EXT_T = [re.compile(p, re.I) for p in EXTENSION_TITLE_PATTERNS[:1]] + [re.compile(EXTENSION_TITLE_PATTERNS[1])] + \
         [re.compile(p, re.I) for p in EXTENSION_TITLE_PATTERNS[2:]]          # 'OLE' is case-sensitive
_STAGE = [re.compile(p, re.I) for p in STAGE_PATTERNS]
_ROMAN = {'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5', 'vi': '6', 'vii': '7', 'viii': '8', 'ix': '9', 'x': '10'}
INERT_TITLE = re.compile(r'(?i)\b(?:placebo|sham|vehicle|dummy|saline|no treatment|no intervention|untreated)\b')


def _txt(x):
    return '' if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)


def pooled_reason(title, desc=''):
    t, d = _txt(title), _txt(desc)
    for p in _POOLED_T:
        if p.search(t):
            return f'pooled_title:{p.pattern}'
    for p in _EXT_T:
        if p.search(t):
            return f'extension_title:{p.pattern}'
    for p in _POOLED_D:
        if p.search(d):
            return f'pooled_desc:{p.pattern}'
    return ''


def stage_key(title):
    t = _txt(title)
    out = set()
    m = _STAGE[0].finditer(t)
    for x in m:
        kind, lab = x.group(1).lower(), x.group(2).lower()
        lab = _ROMAN.get(lab, lab)
        out.add(('stage', lab) if kind != 'cohort' else ('cohort', lab))
    for x in _STAGE[1].finditer(t):
        k = x.group(1).lower().replace(' ', '-')
        out.add(('blind', 'db' if k in ('double-blind', 'db') else 'ol'))
    return frozenset(out)


# --- rule (a): molecule key -------------------------------------------------------------------------------------------
PLACEBO_PART = re.compile(r'(?i)\b(placebo|sham|vehicle|saline|dummy|matching|no (treatment|intervention)|standard (of )?care|'
                          r'usual care|best supportive care|observation|control)\b')
_TITLE_SPLIT = re.compile(r'(?i)\s*(?:\+|/|,|;|&|\bplus\b|\band\b|\bwith\b|\bfollowed by\b|\bthen\b|\s-\s|->|\u2192)\s*')
INERT_KEY = frozenset(['__inert__'])


def mol_key_names(names):
    parts = [p for p in _txt(names).split('|') if p.strip()]
    keep = [p for p in parts if not PLACEBO_PART.search(p)]
    if parts and not keep:
        return INERT_KEY
    return frozenset(norm_name('|'.join(keep)))


_DOSE = re.compile(r'(?i)\d+(?:[.,]\d+)?\s*(?:x\s*)?(?:mg/kg|mg/m2|mg/m\^2|mcg/kg|\u00b5g/kg|mg|mcg|\u00b5g|\u03bcg|ug|ng|g|ml|iu|units?|u|%|vp|tcid50|tcid|pfu|e\d+)\b')
_GENERIC = set('''dose doses dosing level levels regimen regimens therapy therapies active arm arms group groups cohort cohorts part
phase stage period periods low high lower higher middle mid standard fixed flexible i ii iii iv v vi vii viii ix x a b c d e f
q2w q3w q4w q6w q8w q12w qd bid tid qid qw qm od hs pm am every alternating subjects participants patients population study
treated treatment treatments arm-a arm-b only alone'''.split())


def _title_tokens(part):
    p = part.split(':', 1)[-1] if re.match(r'(?i)^\s*(arm|group|gp|cohort|part|period|phase|stage)\b[^:]{0,12}:', part) else part
    p = _DOSE.sub(' ', p.lower().replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' '))
    toks = re.findall(r'[a-z\u00e0-\u00ff][a-z0-9\u00e0-\u00ff.\-]*[a-z0-9]|[a-z]{2,}', p)
    from ivbench.specs import STOP
    return [t.strip('.-') for t in toks if t.strip('.-') and t.strip('.-') not in STOP and t.strip('.-') not in _GENERIC]


def mol_key_title(title):
    """Molecule key from a results-group title: split into components (+, /, and, plus, with, then, ->, ' - '), drop
    placebo/vehicle/standard-care components, strip doses/units/schedule words, keep alphanumeric product codes."""
    t = _txt(title)
    parts = [p for p in _TITLE_SPLIT.split(t) if p and p.strip()]
    keep = [p for p in parts if not PLACEBO_PART.search(p)]
    if parts and not keep:
        return INERT_KEY
    out = set()
    for p in keep:
        tk = _title_tokens(p)
        if tk:
            out.add(' '.join(tk))
    return frozenset(out)


TRUE_PLACEBO_TITLE = re.compile(r'(?i)(\b(placebo|vehicle|sham|dummy)\s+(matched|matching|for|of|to|with|control|comparator)\b|'
                                r'\b(matched|matching)\s+(placebo|vehicle)\b|\bvehicle\b)')


def same_molecule(t_title, t_names, c_title, c_names):
    """True when the comparator is a different dose/schedule/route/product of the treatment's molecule(s).
    Molecule keys come from the results-group titles when both are informative, else from the intervention names.
    Comparators whose title is a placebo/vehicle/sham label (alone, or 'placebo matching X', 'X vehicle') never match."""
    if TRUE_PLACEBO_TITLE.search(_txt(c_title)):
        return False
    kc_t, kt_t = mol_key_title(c_title), mol_key_title(t_title)
    if kc_t == INERT_KEY:
        return False
    if kt_t and kc_t and kt_t != INERT_KEY:
        return kt_t == kc_t
    kt_n, kc_n = mol_key_names(t_names), mol_key_names(c_names)
    if kt_n and kc_n and INERT_KEY not in (kt_n, kc_n):
        return kt_n == kc_n
    return False


# --- rule (c): arm-type recovery --------------------------------------------------------------------------------------
_TOKSTOP = set('the and of with a an in to for on at by or arm group groups participants patients subjects who will be were '
               'was received receive receiving given administered daily once twice per day days week weeks study'.split())


def _toks(s):
    s = re.sub(r'(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)', ' ', _txt(s).lower())
    return {t for t in re.findall(r'[a-z0-9]+', s) if t not in _TOKSTOP}


def _jac(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def match_score(g_title, g_desc, a_label, a_desc, a_ivnames=''):
    ts = pt._sim(pt.norm_title(g_title), pt.norm_title(a_label))
    tt, ta = _toks(g_title), _toks(f'{a_label} {a_desc} {a_ivnames}')
    cont = (len(tt & ta) / len(tt)) if tt else 0.0
    dj = _jac(_toks(g_desc), _toks(a_desc)) if (len(_toks(g_desc)) >= 3 and len(_toks(a_desc)) >= 3) else 0.0
    return max(ts, 0.9 * cont * min(1.0, len(tt) / 2), dj)


def rematch_types(A, arm_groups, threshold=0.5, margin=0.05):
    """A: released arm table. arm_groups: {nct_id: [armGroup dicts]} re-fetched. Returns A with columns
    arm_type_v21, arm_label_v21, type_source ('released' | 'rematched' | 'unmatched' | 'no_registry_arms'), role_v21."""
    A = A.copy()
    A['arm_type_v21'] = A['arm_type']
    A['arm_label_v21'] = A['arm_label']
    A['type_source'] = np.where(A['arm_type'].notna(), 'released', 'unmatched')
    A['role_v21'] = A['role']
    for nct, g in A[A['arm_type'].isna()].groupby('nct_id'):
        arms = arm_groups.get(nct) or []
        if not arms:
            A.loc[g.index, 'type_source'] = 'no_registry_arms'
            continue
        used = set(pt.norm_title(x) for x in A.loc[(A.nct_id == nct) & A.arm_label.notna(), 'arm_label'])
        cand = [a for a in arms if pt.norm_title(a.get('label')) not in used]
        trip = []
        for i, r in g.iterrows():
            sc = sorted(((match_score(r.group_title, r.group_desc, a.get('label'), a.get('description'),
                                      ' '.join(a.get('interventionNames') or [])), j) for j, a in enumerate(cand)), reverse=True)
            if not sc or sc[0][0] < threshold:
                continue
            if len(sc) > 1 and sc[0][0] - sc[1][0] < margin:
                continue
            for s, j in sc:
                trip.append((s, i, j))
        trip.sort(reverse=True)
        ui, uj = set(), set()
        for s, i, j in trip:
            if s < threshold or i in ui or j in uj:
                continue
            ui.add(i); uj.add(j)
            a = cand[j]
            A.at[i, 'arm_type_v21'] = a.get('type')
            A.at[i, 'arm_label_v21'] = a.get('label')
            A.at[i, 'type_source'] = 'rematched'
            r = A.loc[i]
            A.at[i, 'role_v21'] = pt.arm_role(r.group_title, _txt(a.get('description')) + ' ' + _txt(r.group_desc), a.get('type'))
    return A


# --- contrast construction --------------------------------------------------------------------------------------------
def _row(tr, c0, fam, delta, scale):
    """Field-for-field copy of the row dict in build.build (v2)."""
    return dict(
        nct_id=tr['nct_id'], family=fam, area=tr['area'], phase=tr['phase'], sponsor_class=tr['sponsor_class'],
        sponsor=tr['sponsor'], allocation=tr['allocation'], masking=tr['masking'], purpose=tr['purpose'],
        year_posted=tr['year_posted'], split_time=tr['split_time'],
        conditions=tr['conditions'], mesh_conditions=tr['mesh_conditions'],
        outcome_title=tr['outcome_title'], outcome_desc=tr['outcome_desc'],
        outcome_timeframe=tr['outcome_timeframe'], unit=tr['unit'], param_type=tr['param_type'],
        eligibility_text=tr['eligibility_text'], brief_summary=tr['brief_summary'],
        min_age=tr['min_age'], max_age=tr['max_age'], sex=tr['sex'],
        treat_group=tr['group_title'], ctrl_group=c0['group_title'],
        treat_intervention=tr['intervention_names'], ctrl_intervention=c0['intervention_names'],
        treat_intervention_desc=tr['intervention_desc'], ctrl_intervention_desc=c0['intervention_desc'],
        treat_arm_desc=tr['arm_desc'], ctrl_arm_desc=c0['arm_desc'],
        treat_group_desc=tr['group_desc'], ctrl_group_desc=c0['group_desc'],
        treat_arm_matched=tr['arm_matched'], ctrl_arm_matched=c0['arm_matched'],
        treat_intervention_types=tr['intervention_types'], ctrl_role=c0['_role'],
        n_treat=tr['n_analyzed'], n_ctrl=c0['n_analyzed'],
        y_treat=tr['y'], y_ctrl=c0['y'], delta=delta, scale=scale,
        age_mean=tr['age_mean'], frac_female=tr['frac_female'],
        completion_rate_treat=tr['completion_rate'], completion_rate_ctrl=c0['completion_rate'],
        treat_arm_type_v21=tr.get('arm_type_v21'), ctrl_arm_type_v21=c0.get('arm_type_v21'),
        treat_type_source=tr.get('type_source'), ctrl_type_source=c0.get('type_source'),
    )


def _effect(tr, c0, fam):
    if fam == 'binary':
        return tr['y'] - c0['y'], 1.0
    n1, n2, s1, s2 = tr['n_analyzed'], c0['n_analyzed'], tr['sd'], c0['sd']
    if not (pd.notna(s1) and pd.notna(s2)) or (n1 + n2 - 2) <= 0:
        return None
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if not np.isfinite(sp) or sp <= 0:
        return None
    return (tr['y'] - c0['y']) / sp, sp


def build_contrasts(A, rules=(), interv_model=None, log=None):
    """A: filtered arm table (benchmark_arms.csv.gz; with rematch_types columns when rule c is used).
    rules: subset of {'a', 'b', 'c'}. interv_model: {nct_id: interventionModel}. log: list collecting rule events."""
    rules = set(rules)
    interv_model = interv_model or {}
    log = log if log is not None else []
    rows = []
    for nct, g in A.groupby('nct_id'):
        g = g.copy()
        g['_role'] = g['role_v21'] if ('c' in rules and 'role_v21' in g) else g['role']
        if 'b' in rules:
            im = interv_model.get(nct)
            if im in NONCONCURRENT_MODELS:
                log.append(dict(nct_id=nct, rule='b1_model', detail=im)); continue
            pr = [pooled_reason(t, d) for t, d in zip(g['group_title'], g['group_desc'])]
            for (i, r), why in zip(g.iterrows(), pr):
                if why:
                    log.append(dict(nct_id=nct, rule='b2_group', group=r['group_title'], detail=why))
            g = g[[not w for w in pr]]
        t = g[g['_role'] == 'treatment']
        inert = g[g['_role'] == 'inert_control']
        active = g[g['_role'] == 'active_comparator']
        if 'c' in rules:
            exp = g[(g['arm_type_v21'] == 'EXPERIMENTAL') & ~g['group_title'].fillna('').str.contains(INERT_TITLE)]
            if len(exp):
                rest = g.drop(exp.index)
                inert_r = rest[rest['_role'] == 'inert_control']; active_r = rest[rest['_role'] == 'active_comparator']
                if len(inert_r) + len(active_r):
                    t, inert, active = exp, inert_r, active_r
                    log.append(dict(nct_id=nct, rule='c_exp_pool', detail=f'{len(exp)} experimental groups'))
                else:
                    log.append(dict(nct_id=nct, rule='c_fallback_v2', detail='no comparator outside experimental groups'))
        if len(t) == 0 or (len(inert) + len(active)) == 0:
            continue
        fam = g['family'].iloc[0]
        if not rules:                                            # v2 path, verbatim
            c = inert if len(inert) else active
            c0 = c.sort_values('n_analyzed', ascending=False).iloc[0]
        ordered = t.sort_values('n_analyzed', ascending=False)
        if not rules:
            ordered = ordered.head(3)                            # v2: three largest treatment arms
        n_made = 0
        for _, tr in ordered.iterrows():
            if rules and n_made >= 3:                            # v2.1: three largest treatment arms with an admissible comparator
                break
            if rules:
                ci, ca = inert, active
                if 'a' in rules:
                    keep_i = [not same_molecule(tr['group_title'], tr['intervention_names'], r['group_title'], r['intervention_names'])
                              for _, r in ci.iterrows()]
                    keep_a = [not same_molecule(tr['group_title'], tr['intervention_names'], r['group_title'], r['intervention_names'])
                              for _, r in ca.iterrows()]
                    for (_, r), k in list(zip(ci.iterrows(), keep_i)) + list(zip(ca.iterrows(), keep_a)):
                        if not k:
                            log.append(dict(nct_id=nct, rule='a_same_molecule', group=tr['group_title'], detail=r['group_title']))
                    ci, ca = ci[keep_i] if len(ci) else ci, ca[keep_a] if len(ca) else ca
                if 'b' in rules:
                    sk = stage_key(tr['group_title'])
                    ki = [stage_key(x) == sk for x in ci['group_title']]
                    ka = [stage_key(x) == sk for x in ca['group_title']]
                    for x, k in list(zip(ci['group_title'], ki)) + list(zip(ca['group_title'], ka)):
                        if not k:
                            log.append(dict(nct_id=nct, rule='b3_stage', group=tr['group_title'], detail=x))
                    ci, ca = ci[ki] if len(ci) else ci, ca[ka] if len(ca) else ca
                c = ci if len(ci) else ca
                if len(c) == 0:
                    log.append(dict(nct_id=nct, rule='no_admissible_comparator', group=tr['group_title'], detail=''))
                    continue
                c0 = c.sort_values('n_analyzed', ascending=False).iloc[0]
                n_made += 1                                      # counts admissible arms, as v2 counts arms before SD checks
            eff = _effect(tr, c0, fam)
            if eff is None:
                continue
            rows.append(_row(tr, c0, fam, *eff))
    C = pd.DataFrame(rows)
    C = C[np.isfinite(C['delta'])]
    C = C.drop_duplicates(['nct_id', 'treat_group', 'ctrl_group'])
    C = C[~((C['family'] == 'continuous') & (C['delta'].abs() > 5))]
    C['ctrl_kind'] = C['ctrl_role']
    return C.reset_index(drop=True), log
