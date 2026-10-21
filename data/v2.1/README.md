# Benchmark v2.1 (arm-pairing repair)

v2 files in data/ are unchanged. v2.1 is rebuilt by `python scripts/build_v21.py` from data/benchmark_arms.csv.gz plus
ctgov_arms_design_oct2026.jsonl.gz (armsInterventionsModule + designModule re-fetched from ClinicalTrials.gov API v2 on
2026-10-04 by scripts/fetch_arm_types.py). Rules: src/ivbench/pairing_rules.py.

* benchmark_contrasts_v2.1.csv.gz: 10,004 contrasts (v2 columns unchanged in meaning) plus v21_status (kept / repaired /
  relabelled / added_in_v2_trial), repair_type, v2_treat_group / v2_ctrl_group (pair replaced), cid (test only: kept rows keep
  their v2 cid; all others get c21_ / k21_ ids), treat/ctrl_arm_type_v21 and type_source, intervention_model.
* benchmark_arms_v2.1.csv.gz: v2 arm table plus arm_type_v21, arm_label_v21, type_source, role_v21, intervention_model,
  pooled_reason, stage_key.
* v2_to_v21_reconciliation.csv: every v2 contrast with its v2.1 status and replacement; v2_removed_by_reason_v2.1.csv: first rule
  responsible for each change; v21_new_contrasts_log.csv; rule_events_v2.1.csv.gz; rule_counts_v2.1.csv; rule_b_subcounts_v2.1.csv.
* candidate_new_trial_contrasts_v2.1.csv.gz: contrasts rule (c) would add from trials absent from v2. Not part of v2.1.
* datacard_v2.1.json: counts and the explicit regex lists.
Audit, re-audit and skill re-computation: scripts/v21_audit/, results/v2.1/ (see v21_repair_report.md).
