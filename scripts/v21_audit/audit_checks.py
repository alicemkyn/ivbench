"""Deterministic re-derivation of sampled contrasts from the Oct 2026 source records (audit_records.jsonl.gz):
outcome identity, arm values, denominators, SD conversion, delta; arm types of the paired groups; design fields.
Also builds the pairing-review prompt used for the model-assisted pairing judgement."""
import json, gzip
from common import *
from ivbench import parse as pt
from ivbench.build import sd_from_dispersion

def load_records(path='audit_records.jsonl.gz'):
    return {json.loads(l)['protocolSection']['identificationModule']['nctId']: json.loads(l) for l in gzip.open(path, 'rt')}

def check(row, st):
    ps, rs = st['protocolSection'], st.get('resultsSection', {})
    di = ps.get('designModule', {}).get('designInfo', {})
    out = dict(nct_id=row.nct_id, family=row.family, allocation=di.get('allocation'), intervention_model=di.get('interventionModel'))
    o, meas = pt.pick_primary(rs)
    out['outcome_found'] = o is not None
    if o is None: return out
    out['outcome_title_match'] = (o.get('title') or '').strip() == str(row.outcome_title).strip()
    out['param_type_src'] = o.get('paramType'); out['unit_src'] = o.get('unitOfMeasure'); out['dispersion_src'] = o.get('dispersionType')
    groups = {g['id']: g for g in o.get('groups', [])}
    title2id = {}
    for gid, g in groups.items(): title2id.setdefault(g.get('title', ''), []).append(gid)
    denoms = {c['groupId']: pt._f(c.get('value')) for d in o.get('denoms', []) for c in d.get('counts', [])}
    mv = {m['groupId']: m for m in meas}
    out['n_outcome_groups'] = len(groups)
    arms = ps.get('armsInterventionsModule', {}).get('armGroups', []) or []
    arm_by = {pt.norm_title(a.get('label', '')): a for a in arms}
    fz = pt.fuzzy_assign([pt.norm_title(g.get('title', '')) for g in groups.values()], list(arm_by))
    out['n_arm_groups'] = len(arms)
    vals = {}
    for side, title, n_b, y_b in [('treat', row.treat_group, row.n_treat, row.y_treat), ('ctrl', row.ctrl_group, row.n_ctrl, row.y_ctrl)]:
        ids = title2id.get(title, [])
        out[f'{side}_group_found'] = len(ids) == 1
        if len(ids) != 1: continue
        gid = ids[0]; m = mv.get(gid, {}); v = pt._f(m.get('value')); n = denoms.get(gid)
        unit = o.get('unitOfMeasure') or ''
        if row.family == 'binary':
            y = v / 100.0 if pt.PCT_UNIT.search(unit) else (v / n if (n and v is not None) else None)
        else:
            y = v
        out[f'{side}_value_src'] = v; out[f'{side}_n_src'] = n; out[f'{side}_y_rederived'] = y
        out[f'{side}_y_match'] = (y is not None and abs(y - y_b) < 1e-6)
        out[f'{side}_n_match'] = (n is not None and abs(n - n_b) < 1e-6) if n is not None else None
        a = arm_by.get(pt.norm_title(title)) or arm_by.get(fz.get(pt.norm_title(title), ''), {})
        out[f'{side}_arm_type'] = a.get('type') if a else None
        out[f'{side}_arm_label'] = a.get('label') if a else None
        if row.family == 'continuous' and v is not None:
            vals[side] = sd_from_dispersion(dict(dispersion_type=o.get('dispersionType'), spread=pt._f(m.get('spread')), n_analyzed=n,
                                                 low=pt._f(m.get('lowerLimit')), high=pt._f(m.get('upperLimit'))))
    if row.family == 'continuous' and len(vals) == 2 and out.get('treat_y_match') is not None:
        try:
            n1, n2, s1, s2 = out['treat_n_src'], out['ctrl_n_src'], vals['treat'], vals['ctrl']
            sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
            out['delta_rederived'] = (out['treat_y_rederived'] - out['ctrl_y_rederived']) / sp
        except Exception:
            out['delta_rederived'] = np.nan
    elif row.family == 'binary' and out.get('treat_y_rederived') is not None and out.get('ctrl_y_rederived') is not None:
        out['delta_rederived'] = out['treat_y_rederived'] - out['ctrl_y_rederived']
    out['delta_match'] = bool(abs(out.get('delta_rederived', np.nan) - row.delta) < 1e-6) if pd.notna(out.get('delta_rederived', np.nan)) else False
    out['treat_type_ok'] = out.get('treat_arm_type') in ('EXPERIMENTAL', 'OTHER', None)
    out['ctrl_type_ok'] = out.get('ctrl_arm_type') in ('PLACEBO_COMPARATOR', 'SHAM_COMPARATOR', 'NO_INTERVENTION', 'ACTIVE_COMPARATOR', 'OTHER', None)
    return out

def pairing_prompt(row, st):
    ps, rs = st['protocolSection'], st.get('resultsSection', {})
    o, _ = pt.pick_primary(rs)
    di = ps.get('designModule', {}).get('designInfo', {})
    arms = ps.get('armsInterventionsModule', {}).get('armGroups', []) or []
    L = [f"Trial {row.nct_id}. Design: allocation={di.get('allocation')}, intervention model={di.get('interventionModel')}.",
         "Registered arm groups:"]
    L += [f"- [{a.get('type')}] {a.get('label')}: {(a.get('description') or '')[:300]}" for a in arms]
    if o:
        L += [f"Primary outcome analysed: {o.get('title')} (param type {o.get('paramType')}, unit '{o.get('unitOfMeasure')}', dispersion {o.get('dispersionType')}).",
              "Outcome result groups:"]
        L += [f"- {g.get('title')}: {(g.get('description') or '')[:250]}" for g in o.get('groups', [])]
    L += ["", f"A benchmark built from this record treats result group '{row.treat_group}' as the TREATMENT arm and result group "
          f"'{row.ctrl_group}' as the COMPARATOR arm (comparator kind: {row.ctrl_kind}), and computes treatment minus comparator on the outcome.",
          "Judge the pairing. Answer with one JSON object only, keys:",
          '"treat_is_intervention_arm": true/false (the treatment group received the investigational intervention, not a placebo/standard-care/comparator),',
          '"ctrl_is_comparator_arm": true/false (the comparator group is a placebo, sham, no-treatment, standard-care or active-comparator arm),',
          '"concurrent_distinct_arms": true/false (the two groups are distinct concurrently assigned arms, not periods of a crossover, sequential cohorts, subgroups or pooled groups),',
          '"ctrl_kind_correct": true/false (the comparator kind label matches the comparator group),',
          '"pairing_valid": "yes"/"no"/"unclear", "issue": short string ("" if none).']
    return '\n'.join(L)
