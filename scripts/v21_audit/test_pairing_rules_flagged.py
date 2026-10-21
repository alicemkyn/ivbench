"""Apply each v2.1 pairing rule to the 24 contrasts flagged by the v2 audit (audit_flagged_review.csv) and record which
rule fires. Output: pairing_rules_flagged_tests.csv"""
import os, gzip, json
from common import *
from ivbench import pairing_rules as prl
FR = pd.read_csv('audit_flagged_review.csv')
ARM = pd.read_csv(os.path.join(REPO, 'data', 'v2.1', 'benchmark_arms_v2.1.csv.gz'), low_memory=False)
rows = []
for r in FR.itertuples():
    g = ARM[ARM.nct_id == r.nct_id].set_index('group_title')
    t, c = g.loc[r.treatment_group], g.loc[r.comparator_group]
    t = t.iloc[0] if isinstance(t, pd.DataFrame) else t; c = c.iloc[0] if isinstance(c, pd.DataFrame) else c
    rows.append(dict(nct_id=r.nct_id, cid_v2=r.cid_v2, judgement=r.judgement, intervention_model=t.intervention_model,
                     b1_model_rule=t.intervention_model in prl.NONCONCURRENT_MODELS,
                     b2_treat=prl.pooled_reason(r.treatment_group, t.group_desc), b2_ctrl=prl.pooled_reason(r.comparator_group, c.group_desc),
                     b3_stage_treat=t.stage_key if isinstance(t.stage_key, str) else '', b3_stage_ctrl=c.stage_key if isinstance(c.stage_key, str) else '',
                     b3_mismatch=prl.stage_key(r.treatment_group) != prl.stage_key(r.comparator_group),
                     a_same_molecule=prl.same_molecule(r.treatment_group, t.intervention_names, r.comparator_group, c.intervention_names),
                     a_key_treat='|'.join(sorted(prl.mol_key_title(r.treatment_group))), a_key_ctrl='|'.join(sorted(prl.mol_key_title(r.comparator_group))),
                     c_treat_type_v21=t.arm_type_v21, c_treat_type_source=t.type_source, c_treat_role_v21=t.role_v21,
                     c_ctrl_type_v21=c.arm_type_v21, c_ctrl_type_source=c.type_source, c_ctrl_role_v21=c.role_v21,
                     v21_status=r.v21_status, v21_new_pair=r.v21_new_pair))
D = pd.DataFrame(rows); D.to_csv('pairing_rules_flagged_tests.csv', index=False)
print(D[['nct_id', 'judgement', 'b1_model_rule', 'b2_treat', 'b2_ctrl', 'b3_mismatch', 'a_same_molecule', 'c_treat_type_v21', 'v21_status']].to_string())
