"""Build benchmark v2.1 (arm-pairing repair) from the released v2 arm table plus re-fetched registry arm groups.

Usage:  python scripts/build_v21.py [--out data/v2.1]
Inputs: data/benchmark_arms.csv.gz, data/benchmark_contrasts.csv.gz (v2, untouched),
        data/v2.1/ctgov_arms_design_oct2026.jsonl.gz (armsInterventionsModule + designModule for the 18,077 arm-table
        trials, re-fetched from ClinicalTrials.gov API v2 on 2026-10-04 by fetch_arm_types.py).
Outputs (in --out): benchmark_contrasts_v2.1.csv.gz, benchmark_arms_v2.1.csv.gz, datacard_v2.1.json,
        v2_to_v21_reconciliation.csv, v21_new_contrasts_log.csv, rule_counts_v2.1.csv, rule_b_subcounts_v2.1.csv,
        rule_events_v2.1.csv.gz, candidate_new_trial_contrasts_v2.1.csv.gz (contrasts from trials absent from v2;
        NOT part of v2.1).
v2.1 universe: the trials that contribute contrasts to v2 (8,011). Rules (a)-(c) in src/ivbench/pairing_rules.py.
"""
import argparse, gzip, json, os, sys
from collections import defaultdict
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from ivbench import pairing_rules as prl

KEY = ['nct_id', 'treat_group', 'ctrl_group']


def v2_test_cids(C):
    """cid convention of the language-model arm (common.test_sets / llm_prompts.load_rows)."""
    out = {}
    for fam, pre in [('binary', 'c'), ('continuous', 'k')]:
        d = C[(C.family == fam) & (C.split_time == 'test')].sort_values(['nct_id', 'outcome_title', 'treat_group']).reset_index(drop=True)
        for i, k in enumerate(map(tuple, d[KEY].values)):
            out[k] = f'{pre}{i:04d}'
    return out


def content_changed(Cn, Co, cols):
    m = Cn.merge(Co, on=KEY, how='inner', suffixes=('_n', ''))
    neq = np.zeros(len(m), bool)
    for c in cols:
        if c in KEY:
            continue
        a, b = m[c + '_n'], m[c]
        if c in ('treat_arm_matched', 'ctrl_arm_matched'):
            a, b = a.astype(bool), b.astype(bool)
        eq = ((a == b) | (a.isna() & b.isna())).to_numpy()
        try:
            eq = eq | np.isclose(a.astype(float), b.astype(float), equal_nan=True)
        except (TypeError, ValueError):
            pass
        neq |= ~eq
    return set(map(tuple, m.loc[neq, KEY].values))


def reconcile(Cn, Co, cols):
    """Per-trial one-to-one reconciliation. v2 rows: kept | relabelled | repaired | dropped.
    new rows: kept | relabelled | repaired | added_in_v2_trial | added (trial absent from v2)."""
    kn, ko = set(map(tuple, Cn[KEY].values)), set(map(tuple, Co[KEY].values))
    chg = content_changed(Cn, Co, cols)
    R, N = defaultdict(list), defaultdict(list)
    for k in map(tuple, Co[KEY].values):
        if k not in kn: R[k[0]].append(k)
    for k in map(tuple, Cn[KEY].values):
        if k not in ko: N[k[0]].append(k)
    st, nst, how = {}, {}, {}
    v2t = set(Co.nct_id)
    for nct in set(R) | set(N):
        r, n = list(R[nct]), list(N[nct])
        for stage in ('same_treatment_new_comparator', 'same_comparator_new_treatment', 'new_treatment_and_comparator'):
            for k in list(r):
                cand = [x for x in n if stage == 'new_treatment_and_comparator' or
                        (stage == 'same_treatment_new_comparator' and x[1] == k[1]) or
                        (stage == 'same_comparator_new_treatment' and x[2] == k[2])]
                if cand:
                    x = cand[0]; st[k] = ('repaired', x, stage); nst[x] = ('repaired', k, stage); r.remove(k); n.remove(x)
        for k in r: st[k] = ('dropped', None, '')
        for x in n: nst[x] = ('added' if nct not in v2t else 'added_in_v2_trial', None, '')
    def lab(k, d):
        return d.get(k, ('relabelled' if k in chg else 'kept', None, ''))
    O = Co[KEY + ['family', 'split_time']].copy()
    L = [lab(k, st) for k in map(tuple, O[KEY].values)]
    O['status'] = [x[0] for x in L]; O['repair_type'] = [x[2] for x in L]
    O['new_treat_group'] = [x[1][1] if x[1] else None for x in L]; O['new_ctrl_group'] = [x[1][2] if x[1] else None for x in L]
    Nn = Cn[KEY + ['family', 'split_time']].copy()
    L = [lab(k, nst) for k in map(tuple, Nn[KEY].values)]
    Nn['status'] = [x[0] for x in L]; Nn['repair_type'] = [x[2] for x in L]
    Nn['v2_treat_group'] = [x[1][1] if x[1] else None for x in L]; Nn['v2_ctrl_group'] = [x[1][2] if x[1] else None for x in L]
    return O, Nn


def seg(df):
    return np.where(df.split_time == 'train', 'train_' + df.family, 'test_' + df.family)


def count_table(O, Nn, label):
    O = O.assign(seg=seg(O)); Nn = Nn.assign(seg=seg(Nn))
    t1 = O.groupby(['seg', 'status']).size().unstack(fill_value=0)
    t2 = Nn[Nn.status.isin(['added', 'added_in_v2_trial'])].groupby(['seg', 'status']).size().unstack(fill_value=0)
    T = t1.join(t2, how='outer').fillna(0).astype(int)
    for c in ['kept', 'relabelled', 'repaired', 'dropped', 'added_in_v2_trial', 'added']:
        if c not in T: T[c] = 0
    T = T[['kept', 'relabelled', 'repaired', 'dropped', 'added_in_v2_trial', 'added']]
    T['train_all'] = 0
    T.loc['train_all'] = T.loc[[i for i in T.index if i.startswith('train')]].sum()
    T.loc['total'] = T.loc[[i for i in T.index if i.startswith('t') and i != 'train_all' and i != 'total']].sum()
    T = T.drop(columns='train_all')
    T.insert(0, 'rules', label)
    return T.reset_index().rename(columns={'index': 'segment', 'seg': 'segment'})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'v2.1'))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ARM = pd.read_csv(os.path.join(ROOT, 'data', 'benchmark_arms.csv.gz'), low_memory=False)
    CON = pd.read_csv(os.path.join(ROOT, 'data', 'benchmark_contrasts.csv.gz'), low_memory=False)
    V2COLS = list(CON.columns)
    AG, IM = {}, {}
    for line in gzip.open(os.path.join(a.out, 'ctgov_arms_design_oct2026.jsonl.gz'), 'rt'):
        s = json.loads(line); ps = s['protocolSection']; n = ps['identificationModule']['nctId']
        AG[n] = ps.get('armsInterventionsModule', {}).get('armGroups', []) or []
        IM[n] = (ps.get('designModule', {}).get('designInfo', {}) or {}).get('interventionModel')
    R = prl.rematch_types(ARM, AG)
    R['intervention_model'] = R.nct_id.map(IM)
    R['pooled_reason'] = [prl.pooled_reason(t, d) for t, d in zip(R.group_title, R.group_desc)]
    R['stage_key'] = [';'.join(sorted(f'{k}:{v}' for k, v in prl.stage_key(t))) for t in R.group_title]

    # sanity: no rules == v2 exactly
    C0, _ = prl.build_contrasts(R, rules=())
    assert len(C0) == len(CON) and not content_changed(C0, CON, V2COLS) and \
        set(map(tuple, C0[KEY].values)) == set(map(tuple, CON[KEY].values)), 'no-rule build does not reproduce v2'

    builds, tabs = {}, []
    for rules, label in [(('a',), 'a only'), (('b',), 'b only'), (('c',), 'c only'), (('b', 'a'), 'b+a'), (('a', 'b', 'c'), 'a+b+c')]:
        log = []; C, _ = prl.build_contrasts(R, rules=rules, interv_model=IM, log=log)
        builds[rules] = (C, pd.DataFrame(log))
        O, Nn = reconcile(C, CON, V2COLS)
        tabs.append(count_table(O, Nn, label))
    T = pd.concat(tabs, ignore_index=True)
    T.to_csv(os.path.join(a.out, 'rule_counts_v2.1.csv'), index=False)

    # rule (b) sub-attribution (b alone vs v2)
    Cb, Lb = builds[('b',)]
    Ob, _ = reconcile(Cb, CON, V2COLS)
    b2g = set(zip(Lb.loc[Lb.rule == 'b2_group', 'nct_id'], Lb.loc[Lb.rule == 'b2_group', 'group']))
    def battr(r):
        if IM.get(r.nct_id) in prl.NONCONCURRENT_MODELS: return 'b1_model_' + IM.get(r.nct_id)
        if (r.nct_id, r.treat_group) in b2g or (r.nct_id, r.ctrl_group) in b2g: return 'b2_pooled_or_extension_group'
        return 'b3_stage_mismatch'
    X = Ob[Ob.status != 'kept'].copy(); X['sub_rule'] = X.apply(battr, axis=1); X['segment'] = seg(X)
    BS = X.groupby(['sub_rule', 'status', 'segment']).size().unstack(fill_value=0)
    BS['train_all'] = BS[[c for c in BS.columns if c.startswith('train')]].sum(1); BS['total'] = BS[[c for c in BS.columns if c.startswith('t') and c not in ('train_all',)]].sum(1)
    BS.reset_index().to_csv(os.path.join(a.out, 'rule_b_subcounts_v2.1.csv'), index=False)

    # final v2.1: all rules, restricted to the v2 trial universe
    CF, LF = builds[('a', 'b', 'c')]
    v2t = set(CON.nct_id)
    CAND = CF[~CF.nct_id.isin(v2t)].copy()
    CF = CF[CF.nct_id.isin(v2t)].copy()
    O, Nn = reconcile(CF, CON, V2COLS)
    cid2 = v2_test_cids(CON)
    O['v2_cid'] = [cid2.get(k) for k in map(tuple, O[KEY].values)]
    CF = CF.merge(Nn[KEY + ['status', 'repair_type', 'v2_treat_group', 'v2_ctrl_group']], on=KEY, how='left')
    CF = CF.rename(columns={'status': 'v21_status'})
    # cids: unchanged contrasts keep their v2 cid; every other test contrast gets a new c21_/k21_ id
    CF['v2_cid'] = [cid2.get(k) if s == 'kept' else None for k, s in zip(map(tuple, CF[KEY].values), CF.v21_status)]
    CF['cid'] = CF['v2_cid']
    for fam, pre in [('binary', 'c21_'), ('continuous', 'k21_')]:
        m = (CF.family == fam) & (CF.split_time == 'test') & CF.cid.isna()
        idx = CF[m].sort_values(['nct_id', 'outcome_title', 'treat_group']).index
        CF.loc[idx, 'cid'] = [f'{pre}{i:04d}' for i in range(len(idx))]
    # repaired v2 rows -> the cid of their replacement
    rep = CF.set_index(KEY)['cid']
    O['v21_cid_of_replacement'] = [rep.get((r.nct_id, r.new_treat_group, r.new_ctrl_group)) if r.status == 'repaired' else
                                   (rep.get((r.nct_id, r.treat_group, r.ctrl_group)) if r.status in ('kept', 'relabelled') else None)
                                   for r in O.itertuples()]
    CF['intervention_model'] = CF.nct_id.map(IM)
    CF = CF.sort_values(['family', 'split_time', 'nct_id', 'outcome_title', 'treat_group']).reset_index(drop=True)
    CF.to_csv(os.path.join(a.out, 'benchmark_contrasts_v2.1.csv.gz'), index=False)
    CAND.to_csv(os.path.join(a.out, 'candidate_new_trial_contrasts_v2.1.csv.gz'), index=False)
    O.to_csv(os.path.join(a.out, 'v2_to_v21_reconciliation.csv'), index=False)
    Nn.to_csv(os.path.join(a.out, 'v21_new_contrasts_log.csv'), index=False)
    LF.to_csv(os.path.join(a.out, 'rule_events_v2.1.csv.gz'), index=False)
    R.to_csv(os.path.join(a.out, 'benchmark_arms_v2.1.csv.gz'), index=False)
    # first-rule attribution of every v2 contrast that is not kept (b before a before c, from the single-rule builds)
    st = {r: reconcile(builds[r][0], CON, V2COLS)[0].set_index(KEY).status for r in [('b',), ('a',), ('c',)]}
    def first_rule(r):
        k = (r.nct_id, r.treat_group, r.ctrl_group)
        if st[('b',)].get(k) != 'kept':
            return 'b1_crossover_sequential' if IM.get(r.nct_id) in prl.NONCONCURRENT_MODELS else 'b2/b3_pooled_extension_or_stage'
        if st[('a',)].get(k) != 'kept': return 'a_same_molecule'
        if st[('c',)].get(k) != 'kept': return 'c_investigational_arm'
        return 'interaction'
    X = O[O.status != 'kept'].copy(); X['reason'] = X.apply(first_rule, axis=1)
    X.to_csv(os.path.join(a.out, 'v2_removed_by_reason_v2.1.csv'), index=False)
    T21 = count_table(O, Nn, 'v2.1 (a+b+c, v2 trial universe)')
    pd.concat([T, T21], ignore_index=True).to_csv(os.path.join(a.out, 'rule_counts_v2.1.csv'), index=False)

    card = json.load(open(os.path.join(ROOT, 'data', 'datacard.json')))
    card = {k: card[k] for k in ('trials_harvested', 'arm_rows_parsed', 'trials_parsed', 'min_n_per_arm', 'test_from_year', 'source')}
    card.update(version='2.1: arm-pairing repair (same-molecule comparator, non-concurrent arms, investigational-arm preference); '
                        'contrast universe restricted to the 8,011 v2 trials; v2 files unchanged',
                rules_module='src/ivbench/pairing_rules.py', build_script='scripts/build_v21.py',
                registry_refetch='armsInterventionsModule + designModule, ClinicalTrials.gov API v2, 2026-10-04, 18,077 trials',
                arm_type_source=R.type_source.value_counts().to_dict(),
                contrast_rows_final=int(len(CF)), contrast_trials_final=int(CF.nct_id.nunique()),
                contrast_by_family=CF.family.value_counts().to_dict(), contrast_by_split=CF.split_time.value_counts().to_dict(),
                contrast_by_family_split={f'{f}_{s}': int(n) for (f, s), n in CF.groupby(['family', 'split_time']).size().items()},
                ctrl_kind=CF.ctrl_kind.value_counts().to_dict(),
                v2_reconciliation=O.status.value_counts().to_dict(), v21_rows_by_status=CF.v21_status.value_counts().to_dict(),
                excluded_candidate_contrasts_new_trials=int(len(CAND)), excluded_candidate_trials=int(CAND.nct_id.nunique()),
                regex=dict(POOLED_TITLE_PATTERNS=prl.POOLED_TITLE_PATTERNS, POOLED_DESC_PATTERNS=prl.POOLED_DESC_PATTERNS,
                           EXTENSION_TITLE_PATTERNS=prl.EXTENSION_TITLE_PATTERNS, STAGE_PATTERNS=prl.STAGE_PATTERNS,
                           NONCONCURRENT_MODELS=list(prl.NONCONCURRENT_MODELS)))
    json.dump(card, open(os.path.join(a.out, 'datacard_v2.1.json'), 'w'), indent=1, default=str)
    print(json.dumps({k: card[k] for k in ('contrast_rows_final', 'contrast_trials_final', 'contrast_by_family_split', 'v2_reconciliation',
                                           'v21_rows_by_status', 'excluded_candidate_contrasts_new_trials')}, default=str))


if __name__ == '__main__':
    main()
