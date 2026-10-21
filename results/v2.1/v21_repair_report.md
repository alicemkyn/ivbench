# ivbench v2.1: arm-pairing repair and re-audit (camera-ready statistics track)

All numbers below are read from the CSV files listed at the end. Pairing judgements in the flagged review are proposals
from reading the source records and must be confirmed by the human author. The re-audit pairing labels are
model-assisted (same prompt and model family as the v2 audit), not human-verified.

## 1. Root cause in v2 (src/ivbench/build.py, parse.py)

* parse.arm_role assigns roles from text before arm type: any group whose title or description mentions placebo, vehicle
  or sham becomes `inert_control`, including EXPERIMENTAL double-dummy arms (for example "Ertapenem ... matching placebo to
  metronidazole"), and an unmatched group with no control text becomes `treatment` (for example "10 mg Haloperidol").
* 28% of result groups were not matched to a registered arm group (title fuzzy match only), so their armGroupType was unknown.
* build.build takes the three largest `treatment` groups and pairs each with the single largest-n control of the
  preferred kind, with no check of molecule identity, design (crossover/sequential), pooled groups, or study stage.

## 2. Flagged-contrast review (audit_flagged_review.csv)

Of the 24 contrasts flagged in the v2 audit (100-contrast sample, 76 valid), my reading of the source records judges
19 invalid and 5 valid (NCT00807027, NCT01023581, NCT01652703, NCT01715896, NCT02271230; two of these have a wrong
comparator-kind label only). This implies 81/100 valid pairings for v2 under my reading (provisional; author to confirm).
All 19 invalid pairs are removed in v2.1 (14 dropped, 5 replaced by a valid pair from the same trial); the 5 valid ones are kept.
NCT01463033 (levetiracetam registered as ACTIVE_COMPARATOR vs a no-intervention control) is dropped rather than re-paired.

## 3. Rules (src/ivbench/pairing_rules.py; build: scripts/build_v21.py)

With no rule enabled the new contrast loop reproduces v2 exactly (11,274 contrasts, identical on every v2 column).

* (a) same-molecule comparator: molecule key from result-group titles (components split on +, /, and, plus, with, then,
  ' - '; placebo/vehicle/standard-care components removed; doses, units, schedules and generic words stripped; product
  codes kept), falling back to intervention names when a title key is empty. Comparators labelled as placebo/vehicle
  ("placebo matching X", "X vehicle gel") never match. A same-molecule comparator is excluded for that treatment arm and the
  next admissible comparator is used (inert before active, largest n); otherwise the contrast is dropped.
* (b) non-concurrent arms: b1 interventionModel CROSSOVER or SEQUENTIAL (re-fetched designModule, 2026-10-04; agrees with
  design_trial_fields.csv on all 8,011 v2 trials); b2 pooled / extension group regexes (explicit lists in the module and
  in datacard_v2.1.json); b3 treatment and comparator must carry the same stage label (part/period/phase/stage/cohort/
  segment/step/panel/epoch number or letter, or a double-blind / open-label phase prefix). Design choice: staged groups are
  paired within stage instead of being dropped outright; the literal variant (drop every staged group) would remove 502 v2
  contrasts, 337 of which are same-stage pairs that b3 keeps.
* (c) investigational arm: armGroupType recovered for every result group (released match: 28,590 arm rows; re-matched
  against re-fetched armGroups by title, title-token containment and description similarity, one-to-one, threshold 0.5,
  margin 0.05: 4,170; still unknown: 7,690). If a trial has EXPERIMENTAL groups whose own title is not a placebo label, they
  form the treatment pool and the remaining control-role groups the comparator pool; if no comparator remains the v2 rule
  is used for that trial. Roles of re-matched groups are recomputed with parse.arm_role. A spot check of 40 re-matched groups found most plausible and at least 2 clearly wrong (e.g. 'Stage 2 - CONTROL' matched to arm 'Stage 1'), so re-matched types are provisional. Treatment cap stays at three arms
  per trial, now counted over arms with an admissible comparator.
* v2.1 universe: the 8,011 trials already in v2. Rule (c) would also admit 1,516 contrasts from 1,099 trials absent from v2
  (trials in which v2 found no usable treatment/control pair under its text roles; not audited); these are saved separately as
  candidate_new_trial_contrasts_v2.1.csv.gz and are NOT part of v2.1 (scope decision for the author).

### Rule counts (v2 contrasts; data/v2.1/rule_counts_v2.1.csv)

'a only', 'b only', 'c only' are marginal (each rule alone against v2); 'b+a' and 'a+b+c' are cumulative. Re-paired = the
v2 slot is replaced by a different pair from the same trial; relabelled = same pair, comparator kind changed.

| rules | segment | kept | relabelled | re-paired | dropped | added (v2 trial) | added (new trial, excluded) |
|---|---|---|---|---|---|---|---|
| a only | test_binary | 974 | 0 | 18 | 28 | 0 | 0 |
| a only | test_continuous | 1428 | 0 | 21 | 21 | 0 | 0 |
| a only | train_all | 8457 | 0 | 141 | 186 | 0 | 0 |
| a only | total | 10859 | 0 | 180 | 235 | 0 | 0 |
| b only | test_binary | 898 | 0 | 20 | 102 | 0 | 0 |
| b only | test_continuous | 1349 | 0 | 12 | 109 | 0 | 0 |
| b only | train_all | 8014 | 0 | 58 | 712 | 0 | 1 |
| b only | total | 10261 | 0 | 90 | 923 | 0 | 1 |
| c only | test_binary | 946 | 0 | 37 | 37 | 17 | 147 |
| c only | test_continuous | 1416 | 1 | 20 | 33 | 23 | 166 |
| c only | train_all | 8102 | 5 | 256 | 421 | 222 | 1389 |
| c only | total | 10464 | 6 | 313 | 491 | 262 | 1702 |
| b+a | test_binary | 860 | 0 | 35 | 125 | 0 | 0 |
| b+a | test_continuous | 1311 | 0 | 31 | 128 | 0 | 0 |
| b+a | train_all | 7724 | 0 | 183 | 877 | 0 | 1 |
| b+a | total | 9895 | 0 | 249 | 1130 | 0 | 1 |
| a+b+c | test_binary | 816 | 0 | 50 | 154 | 16 | 130 |
| a+b+c | test_continuous | 1274 | 1 | 39 | 156 | 19 | 142 |
| a+b+c | train_all | 7293 | 5 | 286 | 1200 | 205 | 1244 |
| a+b+c | total | 9383 | 6 | 375 | 1510 | 240 | 1516 |
| v2.1 (a+b+c, v2 trial universe) | test_binary | 816 | 0 | 50 | 154 | 16 | 0 |
| v2.1 (a+b+c, v2 trial universe) | test_continuous | 1274 | 1 | 39 | 156 | 19 | 0 |
| v2.1 (a+b+c, v2 trial universe) | train_all | 7293 | 5 | 286 | 1200 | 205 | 0 |
| v2.1 (a+b+c, v2 trial universe) | total | 9383 | 6 | 375 | 1510 | 240 | 0 |

### Rule (b) sub-counts (rule b alone; data/v2.1/rule_b_subcounts_v2.1.csv)

| sub-rule | status | train | test binary | test continuous | total |
|---|---|---|---|---|---|
| b1_model_CROSSOVER | dropped | 485 | 17 | 46 | 548 |
| b1_model_SEQUENTIAL | dropped | 34 | 38 | 10 | 82 |
| b2_pooled_or_extension_group | dropped | 158 | 22 | 37 | 217 |
| b2_pooled_or_extension_group | repaired | 27 | 1 | 8 | 36 |
| b3_stage_mismatch | dropped | 35 | 25 | 16 | 76 |
| b3_stage_mismatch | repaired | 31 | 19 | 4 | 54 |

v2.1 totals: 10,004 contrasts from 7,210 trials (binary train 3,136 / test 882; continuous train 4,653 / test 1,333);
9,383 identical to v2, 375 re-paired, 6 relabelled, 240 new pairs in v2 trials; 1,510 v2 contrasts dropped.

## 4. Re-audit of v2.1 (audit_results_v21.csv)

100 contrasts drawn uniformly from the v2.1 file (numpy default_rng(1), sorted), 98 trials, records re-fetched
2026-10-04. Deterministic re-derivation (outcome identity, group values, denominators, SD conversion, delta): 100/100 match.
Model-assisted pairing review (claude-sonnet-5, same prompt): valid pairing 96/100, Wilson 95% CI
[0.902, 0.984] (trial-cluster bootstrap [0.920, 0.990]); v2 audit 76/100, Wilson [0.668, 0.833].
One reply was empty after the scripted retry and was obtained with one extra call (recorded in llm_attempts).
Residual failure modes (4): comparator from a different disease cohort or randomization stratum (NCT02059291 crFMF vs
HIDS/MKD placebo; NCT05412004 GPI1 vs GPI2 strata), and result groups contaminated by mid-trial re-randomization
(NCT05335356 Stelara arm switched at week 16; NCT02712008 pooled re-randomized arms). None are caught by rules (a)-(c).
Separately (not a pairing issue) the conversion review flags 42 continuous contrasts whose value is not an arithmetic mean
with a valid arm-level dispersion (LS means, medians, etc.), as in the v2 audit.

## 5. Interventional skill on v2.1 test sets

Restricted set = test contrasts whose cid and content are unchanged from v2 (binary 816 of 1,020; continuous 1,274 of
1,470). Registry models trained on the v2.1 train split; v2 rows reproduce the released numbers (binary seed 0 -0.0799).

| family | train | test set | n (trials) | IS seed 0 [95% CI] | IS 5-seed mean [95% CI] |
|---|---|---|---|---|---|
| binary | v2 | v2_test_full | 1020 (740) | -0.080 [-0.163, -0.012] | -0.047 [-0.118, +0.015] |
| binary | v2 | v2_test_restricted_to_v21_kept | 816 (632) | -0.097 [-0.200, -0.013] | -0.081 [-0.164, +0.001] |
| binary | v2.1 | v21_test_kept | 816 (632) | -0.078 [-0.157, -0.002] | -0.091 [-0.171, -0.017] |
| binary | v2.1 | v21_test_full | 882 (657) | -0.120 [-0.206, -0.040] | -0.130 [-0.217, -0.049] |
| binary | v2.1 | v21_test_repaired_or_new | 66 (41) | -0.526 [-0.761, -0.322] | -0.552 [-0.768, -0.306] |
| continuous | v2 | v2_test_full | 1470 (1069) | +0.010 [-0.059, +0.086] | -0.026 [-0.094, +0.037] |
| continuous | v2 | v2_test_restricted_to_v21_kept | 1274 (953) | +0.046 [-0.025, +0.113] | -0.003 [-0.080, +0.059] |
| continuous | v2.1 | v21_test_kept | 1274 (953) | -0.007 [-0.087, +0.068] | +0.004 [-0.068, +0.074] |
| continuous | v2.1 | v21_test_full | 1333 (979) | -0.004 [-0.089, +0.065] | -0.002 [-0.076, +0.061] |
| continuous | v2.1 | v21_test_repaired_or_new | 59 (38) | +0.032 [-0.321, +0.209] | -0.114 [-0.476, +0.034] |

Language-model IS (existing per-call outputs, 2,000 trial resamples, seed 0):

| arm | test set | n (trials) | IS [95% CI] |
|---|---|---|---|
| sonnet_binary | v2_test_full | 1020 (740) | +0.385 [+0.267, +0.488] |
| sonnet_binary | v21_test_kept | 816 (632) | +0.253 [+0.135, +0.360] |
| haiku_binary | v2_test_full | 1020 (740) | +0.176 [+0.060, +0.276] |
| haiku_binary | v21_test_kept | 816 (632) | +0.099 [-0.026, +0.218] |
| sonnet_continuous | v2_test_full | 1470 (1069) | +0.275 [+0.095, +0.430] |
| sonnet_continuous | v21_test_kept | 1274 (953) | +0.243 [+0.062, +0.412] |
| sonnet_binary_protocol_only | v2_test_full | 1020 (740) | +0.261 [+0.146, +0.375] |
| sonnet_binary_protocol_only | v21_test_kept | 816 (632) | +0.190 [+0.068, +0.306] |

The removed contrasts carried much of the language-model IS (IS computed on the removed v2 test contrasts only):

| arm | removal reason (first rule) | n | IS on removed v2 contrasts |
|---|---|---|---|
| sonnet_binary | a_same_molecule | 38 | +0.886 |
| sonnet_binary | b1_crossover_sequential | 55 | +0.344 |
| sonnet_binary | b2/b3_pooled_extension_or_stage | 67 | +0.748 |
| sonnet_binary | c_investigational_arm | 44 | +0.386 |
| haiku_binary | a_same_molecule | 38 | +0.613 |
| haiku_binary | b1_crossover_sequential | 55 | +0.092 |
| haiku_binary | b2/b3_pooled_extension_or_stage | 67 | +0.563 |
| haiku_binary | c_investigational_arm | 44 | +0.156 |
| sonnet_continuous | a_same_molecule | 38 | +0.243 |
| sonnet_continuous | b1_crossover_sequential | 56 | +0.321 |
| sonnet_continuous | b2/b3_pooled_extension_or_stage | 65 | +0.727 |
| sonnet_continuous | c_investigational_arm | 37 | -1.443 |

Test contrasts that need new language-model calls: 125 (binary 50 re-paired + 16 new pairs in v2 trials; continuous
39 re-paired + 1 relabelled + 19 new pairs), listed with their new pairing in v21_test_contrasts_needing_new_lm_calls.csv.

Supplementary check before new language-model calls (audit_results_repaired_test_v21.csv; same deterministic and
model-assisted procedure applied to all 125 such test contrasts, not a random sample): model-assisted valid pairing 115/125
(re-paired 79/89, new pairs in v2 trials 35/35, relabelled 1/1). The 10 failures: subgroups by indication or prior-study
cohort (c21_0000, c21_0002, c21_0024), pooled groups across parts or a contrast row used as a group (k21_0005, k21_0006,
k21_0021), same molecule on a different schedule not caught by rule (a) (k21_0007), and comparator-kind label only
(c21_0022, c21_0061, k21_0014). Value re-derivation matched for 124/125; c21_0051 (NCT05625477) does not re-derive the
comparator value from the current record and should be checked. The per-contrast verdicts are columns of
v21_test_contrasts_needing_new_lm_calls.csv.
