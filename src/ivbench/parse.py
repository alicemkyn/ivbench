"""Parse ClinicalTrials.gov study records into arm-level and contrast-level benchmark rows."""
import re

CONTINUOUS = {'MEAN', 'LEAST_SQUARES_MEAN', 'MEDIAN', 'GEOMETRIC_MEAN', 'GEOMETRIC_LEAST_SQUARES_MEAN'}
COUNTLIKE = {'NUMBER', 'COUNT_OF_PARTICIPANTS', 'COUNT_OF_UNITS'}
PCT_UNIT = re.compile(r'(?i)percent|%|proportion')
CTRL_PAT = re.compile(r'(?i)\b(placebo|control|sham|vehicle|standard of care|standard care|usual care|'
                      r'no treatment|observation|comparator|reference)\b')
ACTIVE_CTRL = re.compile(r'(?i)active comparator')


def _f(x):
    try:
        v = float(str(x).replace(',', ''))
        return v
    except (TypeError, ValueError):
        return None


def _date(d):
    if not d:
        return None
    return (d.get('date') if isinstance(d, dict) else d)


def pick_primary(rs):
    """First PRIMARY outcome that is posted, single-class, single-category."""
    for o in rs.get('outcomeMeasuresModule', {}).get('outcomeMeasures', []):
        if o.get('type') != 'PRIMARY' or o.get('reportingStatus') != 'POSTED':
            continue
        classes = o.get('classes', [])
        if len(classes) != 1:
            continue
        cats = classes[0].get('categories', [])
        if len(cats) != 1:
            continue
        if not cats[0].get('measurements'):
            continue
        return o, cats[0]['measurements']
    return None, None


def baseline_by_group(rs):
    """Per-baseline-group N, age (numeric if available), fraction female."""
    bm = rs.get('baselineCharacteristicsModule', {})
    out = {}
    for d in bm.get('denoms', []):
        if d.get('units', '').lower().startswith('particip'):
            for c in d.get('counts', []):
                out.setdefault(c['groupId'], {})['n_baseline'] = _f(c.get('value'))
    titles = {g['id']: g.get('title', '') for g in bm.get('groups', [])}
    for meas in bm.get('measures', []):
        t = (meas.get('title') or '').lower()
        ptype = meas.get('paramType', '')
        for cl in meas.get('classes', []):
            for cat in cl.get('categories', []):
                ctitle = (cat.get('title') or '').lower()
                for m in cat.get('measurements', []):
                    gid, v = m.get('groupId'), _f(m.get('value'))
                    if gid is None or v is None:
                        continue
                    rec = out.setdefault(gid, {})
                    if 'age' in t and ptype in CONTINUOUS and 'continuous' in (meas.get('unitOfMeasure') or 'years').lower() + 'continuous':
                        rec.setdefault('age_mean', v)
                    if t.startswith('sex') or t.startswith('gender'):
                        if 'female' in ctitle:
                            rec['n_female'] = v
                        elif 'male' in ctitle:
                            rec['n_male'] = v
    for gid, rec in out.items():
        rec['group_title'] = titles.get(gid, '')
        nf, nm = rec.get('n_female'), rec.get('n_male')
        if nf is not None and nm is not None and (nf + nm) > 0:
            rec['frac_female'] = nf / (nf + nm)
    return out


def flow_by_group(rs):
    """Keyed by normalized flow-group title (flow group ids differ from outcome group ids)."""
    pf = rs.get('participantFlowModule', {})
    titles = {g['id']: g.get('title', '') for g in pf.get('groups', [])}
    acc = {}
    for per in pf.get('periods', []):
        for milestone in per.get('milestones', []):
            key = (milestone.get('type') or '').upper()
            if key not in ('STARTED', 'COMPLETED'):
                continue
            for a in milestone.get('achievements', []):
                gid = a.get('groupId')
                v = _f(a.get('numSubjects') if a.get('numSubjects') is not None else a.get('value'))
                if gid is not None and v is not None:
                    acc.setdefault(gid, {}).setdefault(key.lower(), v)
        break  # first period only
    out = {}
    for gid, rec in acc.items():
        if rec.get('started'):
            rec['completion_rate'] = (rec.get('completed') or 0) / rec['started']
        out[norm_title(titles.get(gid, ''))] = rec
    return out


def norm_title(s):
    return re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).strip()


_STOP = {'arm', 'group', 'cohort', 'the', 'and', 'with', 'plus', 'of', 'a', 'in', 'to', 'db', 'total'}


def _sim(a, b):
    """Blend of character ratio and token Jaccard (stopwords removed); 0..1."""
    from difflib import SequenceMatcher
    ta = set(a.split()) - _STOP
    tb = set(b.split()) - _STOP
    jac = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    return 0.5 * SequenceMatcher(None, a, b).ratio() + 0.5 * jac


def fuzzy_assign(queries, candidates, threshold=0.55):
    """One-to-one greedy assignment of normalized titles to candidate titles above a similarity threshold."""
    if not queries or not candidates:
        return {}
    pairs = sorted(((_sim(q, c), q, c) for q in queries for c in candidates if q and c), reverse=True)
    out, used_q, used_c = {}, set(), set()
    for s, q, c in pairs:
        if s < threshold:
            break
        if q in used_q or c in used_c:
            continue
        out[q] = c
        used_q.add(q); used_c.add(c)
    return out


INERT_TYPE = {'PLACEBO_COMPARATOR', 'SHAM_COMPARATOR', 'NO_INTERVENTION'}
INERT_TEXT = re.compile(r'(?i)\b(placebo|sham|vehicle|no treatment|no intervention|observation only|'
                        r'untreated|matching placebo|saline control)\b')
ACTIVE_TEXT = re.compile(r'(?i)\b(active comparator|active control|standard of care|standard care|'
                         r'usual care|reference (drug|treatment|therapy)|comparator (drug|arm|therapy))\b')


def arm_role(title, description, arm_type):
    """3 classes: inert_control (placebo/sham/none), active_comparator, treatment."""
    at = (arm_type or '').upper().replace(' ', '_')
    txt = f"{title} {description or ''}"
    if at in INERT_TYPE:
        return 'inert_control'
    if INERT_TEXT.search(txt):
        return 'inert_control'
    if at == 'ACTIVE_COMPARATOR' or ACTIVE_TEXT.search(txt):
        return 'active_comparator'
    return 'treatment'


def parse_study(st):
    ps, rs = st.get('protocolSection', {}), st.get('resultsSection', {})
    if not rs:
        return None
    ident = ps.get('identificationModule', {})
    status = ps.get('statusModule', {})
    design = ps.get('designModule', {})
    elig = ps.get('eligibilityModule', {})
    arms_mod = ps.get('armsInterventionsModule', {})

    o, meas = pick_primary(rs)
    if o is None:
        return None
    ptype = o.get('paramType', '')
    unit = o.get('unitOfMeasure', '') or ''
    if ptype in CONTINUOUS:
        family = 'continuous'
    elif ptype in COUNTLIKE:
        family = 'binary' if (PCT_UNIT.search(unit) or 'particip' in unit.lower()) else 'count'
    else:
        return None

    denoms = {}
    for d in o.get('denoms', []):
        for c in d.get('counts', []):
            denoms[c['groupId']] = _f(c.get('value'))
    ogroups = {g['id']: g for g in o.get('groups', [])}
    base = baseline_by_group(rs)
    flow = flow_by_group(rs)
    base_by_title = {norm_title(v.get('group_title')): v for v in base.values()}
    arm_list = arms_mod.get('armGroups', []) or []
    arm_by_title = {norm_title(a.get('label')): a for a in arm_list}
    # CT.gov v2 lists armGroups.interventionNames as "Drug: Name"; interventions[].name is bare "Name".
    iv_list = arms_mod.get('interventions', []) or []
    iv_by_name = {}
    for i in iv_list:
        nm = (i.get('name') or '').strip()
        iv_by_name[nm] = i
        iv_by_name[norm_title(nm)] = i
        iv_by_name[norm_title(f"{i.get('type', '')}: {nm}")] = i
    # reverse map arm label -> interventions via interventions[].armGroupLabels (more reliable than names)
    ivs_by_arm = {}
    for i in iv_list:
        for lab in i.get('armGroupLabels', []) or []:
            ivs_by_arm.setdefault(norm_title(lab), []).append(i)
    group_titles = [norm_title(g.get('title', '')) for g in o.get('groups', [])]
    fuzzy_arm = fuzzy_assign(group_titles, list(arm_by_title.keys()))
    fuzzy_base = fuzzy_assign(group_titles, list(base_by_title.keys()))

    meta = dict(
        nct_id=ident.get('nctId'),
        title=ident.get('briefTitle'),
        sponsor_class=(ps.get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}) or {}).get('class'),
        sponsor=(ps.get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}) or {}).get('name'),
        phase='|'.join(design.get('phases', []) or []),
        allocation=(design.get('designInfo', {}) or {}).get('allocation'),
        masking=((design.get('designInfo', {}) or {}).get('maskingInfo', {}) or {}).get('masking'),
        purpose=(design.get('designInfo', {}) or {}).get('primaryPurpose'),
        enrollment=(design.get('enrollmentInfo', {}) or {}).get('count'),
        conditions='|'.join(ps.get('conditionsModule', {}).get('conditions', []) or []),
        mesh_conditions='|'.join(m.get('term', '') for m in
                                 (st.get('derivedSection', {}).get('conditionBrowseModule', {}) or {}).get('meshes', [])),
        start_date=_date(status.get('startDateStruct')),
        primary_completion_date=_date(status.get('primaryCompletionDateStruct')),
        completion_date=_date(status.get('completionDateStruct')),
        results_posted_date=_date(status.get('resultsFirstPostDateStruct')),
        min_age=elig.get('minimumAge'), max_age=elig.get('maximumAge'), sex=elig.get('sex'),
        healthy_volunteers=elig.get('healthyVolunteers'),
        eligibility_text=(elig.get('eligibilityCriteria') or '')[:6000],
        brief_summary=(ps.get('descriptionModule', {}).get('briefSummary') or '')[:4000],
        outcome_title=o.get('title'), outcome_desc=(o.get('description') or '')[:2000],
        outcome_timeframe=o.get('timeFrame'), param_type=ptype, dispersion_type=o.get('dispersionType'),
        unit=unit, family=family, n_arms_outcome=len(meas),
    )

    rows = []
    for m in meas:
        gid = m.get('groupId')
        v = _f(m.get('value'))
        if v is None:
            continue
        g = ogroups.get(gid, {})
        gt = g.get('title', '')
        key = norm_title(gt)
        arm = arm_by_title.get(key) or arm_by_title.get(fuzzy_arm.get(key, ''), {})
        b = base_by_title.get(key) or base_by_title.get(fuzzy_base.get(key, ''), {})
        n = denoms.get(gid) or b.get('n_baseline')
        arm_key = norm_title(arm.get('label', '')) if arm else ''
        ivs = ivs_by_arm.get(arm_key, [])
        if not ivs:
            ivs = [iv_by_name.get(norm_title(re.sub(r'^[A-Za-z_ ]+:\s*', '', nm))) or iv_by_name.get(nm)
                   for nm in (arm.get('interventionNames', []) or [])]
            ivs = [i for i in ivs if i]
        iv_names = [i.get('name', '') for i in ivs] or \
                   [re.sub(r'^[A-Za-z_ ]+:\s*', '', nm) for nm in (arm.get('interventionNames', []) or [])]
        iv_text = ' ; '.join(filter(None, [(i.get('description') or '') for i in ivs]))
        iv_types = '|'.join(sorted({(i.get('type') or '') for i in ivs} - {''}))
        arm_matched = bool(arm)
        # normalize outcome to a comparable scale
        if family == 'binary':
            if PCT_UNIT.search(unit):
                y = v / 100.0
            elif n:
                y = v / n
            else:
                y = None
        else:
            y = v
        rows.append(dict(
            **meta, group_id=gid, group_title=gt, group_desc=(g.get('description') or '')[:1200],
            arm_type=arm.get('type'), arm_label=arm.get('label'),
            arm_desc=(arm.get('description') or '')[:1200],
            intervention_names='|'.join(iv_names), intervention_types=iv_types,
            intervention_desc=iv_text[:2000], arm_matched=arm_matched,
            role=arm_role(gt, (arm.get('description') or '') + ' ' + (g.get('description') or ''), arm.get('type')),
            n_analyzed=n, value_raw=v, spread=_f(m.get('spread')),
            low=_f(m.get('lowerLimit')), high=_f(m.get('upperLimit')), y=y,
            age_mean=b.get('age_mean'), frac_female=b.get('frac_female'),
            completion_rate=(flow.get(norm_title(gt)) or {}).get('completion_rate'),
        ))
    return rows if rows else None


def make_contrasts(arm_rows):
    """Within-trial contrasts: every treatment arm against every control arm (fallback: all pairs)."""
    by_trial = {}
    for r in arm_rows:
        by_trial.setdefault(r['nct_id'], []).append(r)
    out = []
    for nct, rows in by_trial.items():
        rows = [r for r in rows if r.get('y') is not None]
        if len(rows) < 2:
            continue
        ctrls = [r for r in rows if r['role'] in ('control', 'active_comparator')]
        treats = [r for r in rows if r['role'] == 'treatment']
        pairs = [(t, c) for t in treats for c in ctrls] or \
                [(rows[i], rows[j]) for i in range(len(rows)) for j in range(i + 1, len(rows))]
        for t, c in pairs:
            base_keys = ['nct_id', 'phase', 'conditions', 'mesh_conditions', 'sponsor', 'sponsor_class',
                         'allocation', 'masking', 'purpose', 'enrollment', 'start_date', 'completion_date',
                         'results_posted_date', 'outcome_title', 'outcome_desc', 'outcome_timeframe',
                         'param_type', 'unit', 'family', 'eligibility_text', 'brief_summary',
                         'min_age', 'max_age', 'sex', 'title']
            out.append({**{k: t[k] for k in base_keys},
                        'treat_group': t['group_title'], 'ctrl_group': c['group_title'],
                        'treat_role': t['role'], 'ctrl_role': c['role'],
                        'treat_intervention': t['intervention_names'], 'ctrl_intervention': c['intervention_names'],
                        'treat_intervention_desc': t['intervention_desc'], 'ctrl_intervention_desc': c['intervention_desc'],
                        'treat_arm_desc': t['arm_desc'], 'ctrl_arm_desc': c['arm_desc'],
                        'treat_intervention_types': t['intervention_types'],
                        'n_treat': t['n_analyzed'], 'n_ctrl': c['n_analyzed'],
                        'y_treat': t['y'], 'y_ctrl': c['y'], 'delta': t['y'] - c['y'],
                        'spread_treat': t['spread'], 'spread_ctrl': c['spread'],
                        'age_mean': t['age_mean'], 'frac_female': t['frac_female'],
                        'completion_rate_treat': t['completion_rate'], 'completion_rate_ctrl': c['completion_rate']})
    return out
