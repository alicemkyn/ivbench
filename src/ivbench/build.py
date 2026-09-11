"""Stage 3: inclusion criteria, contrast construction, standardization, splits, data card."""
import numpy as np
import pandas as pd
import re, json

MIN_N = 20                 # analysed participants per arm
TEST_FROM = 2023           # temporal split on results-posting year

AREAS = [
    ('oncology', r'(?i)\b(cancer|carcinom|tumou?r|neoplas|leukemi|lymphom|myelom|melanom|sarcom|glioma)'),
    ('cardiometabolic', r'(?i)\b(diabet|cardiac|cardiovascular|heart|hypertens|cholesterol|lipid|obesity|coronary|stroke|atrial)'),
    ('infectious', r'(?i)\b(infect|hiv|hepatitis|influenza|covid|sars|tuberculosis|malaria|vaccin|sepsis|pneumoni)'),
    ('neuro_psych', r'(?i)\b(depress|schizophren|anxiety|alzheim|parkinson|epilep|migraine|multiple sclerosis|bipolar|adhd|insomnia|pain)'),
    ('immune_inflam', r'(?i)\b(arthritis|psoria|crohn|colitis|lupus|asthma|copd|dermatitis|allerg|immune)'),
]


def area_of(text):
    for name, pat in AREAS:
        if re.search(pat, text or ''):
            return name
    return 'other'


def sd_from_dispersion(row):
    d, s, n = row['dispersion_type'], row['spread'], row['n_analyzed']
    lo, hi = row.get('low'), row.get('high')
    if not n or n <= 1:
        return np.nan
    if d == 'Standard Deviation' and pd.notna(s):
        return s
    if d == 'Standard Error' and pd.notna(s):
        return s * np.sqrt(n)
    if d == '95% Confidence Interval' and pd.notna(lo) and pd.notna(hi):
        return (hi - lo) / 3.92 * np.sqrt(n)
    return np.nan


def build(A):
    card = {}
    card['arm_rows_parsed'] = len(A)
    card['trials_parsed'] = A['nct_id'].nunique()

    A = A[A['family'].isin(['binary', 'continuous'])].copy()
    card['after_family_filter'] = len(A)

    A = A[A['y'].notna() & A['n_analyzed'].notna()]
    A = A[A['n_analyzed'] >= MIN_N]
    card[f'after_n>={MIN_N}'] = len(A)

    bad_binary = (A['family'] == 'binary') & ((A['y'] < 0) | (A['y'] > 1))
    A = A[~bad_binary]
    card['after_binary_range'] = len(A)

    A['sd'] = A.apply(sd_from_dispersion, axis=1)
    A['results_posted_date'] = pd.to_datetime(A['results_posted_date'], errors='coerce')
    A['year_posted'] = A['results_posted_date'].dt.year
    A = A[A['year_posted'].notna()]
    card['after_date_present'] = len(A)

    A['area'] = (A['conditions'].fillna('') + ' ' + A['mesh_conditions'].fillna('')).map(area_of)
    A['split_time'] = np.where(A['year_posted'] >= TEST_FROM, 'test', 'train')

    # ---- contrasts: treatment vs control within trial, capped per trial ----
    rows = []
    for nct, g in A.groupby('nct_id'):
        t = g[g['role'] == 'treatment']
        inert = g[g['role'] == 'inert_control']
        active = g[g['role'] == 'active_comparator']
        c = inert if len(inert) else active          # prefer placebo/sham over active comparator
        if len(t) == 0 or len(c) == 0:
            continue
        fam = g['family'].iloc[0]
        c0 = c.sort_values('n_analyzed', ascending=False).iloc[0]
        for _, tr in t.sort_values('n_analyzed', ascending=False).head(3).iterrows():
            if fam == 'binary':
                delta = tr['y'] - c0['y']
                scale = 1.0
            else:
                n1, n2, s1, s2 = tr['n_analyzed'], c0['n_analyzed'], tr['sd'], c0['sd']
                if not (pd.notna(s1) and pd.notna(s2)) or (n1 + n2 - 2) <= 0:
                    continue
                sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
                if not np.isfinite(sp) or sp <= 0:
                    continue
                delta = (tr['y'] - c0['y']) / sp
                scale = sp
            rows.append(dict(
                nct_id=nct, family=fam, area=tr['area'], phase=tr['phase'], sponsor_class=tr['sponsor_class'],
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
                treat_intervention_types=tr['intervention_types'], ctrl_role=c0['role'],
                n_treat=tr['n_analyzed'], n_ctrl=c0['n_analyzed'],
                y_treat=tr['y'], y_ctrl=c0['y'], delta=delta, scale=scale,
                age_mean=tr['age_mean'], frac_female=tr['frac_female'],
                completion_rate_treat=tr['completion_rate'], completion_rate_ctrl=c0['completion_rate'],
            ))
    C = pd.DataFrame(rows)
    card['contrast_rows_raw'] = len(C)
    card['contrast_trials_raw'] = C['nct_id'].nunique() if len(C) else 0
    if len(C):
        C = C[np.isfinite(C['delta'])]
        card['contrast_rows_finite'] = len(C)
        C = C.drop_duplicates(['nct_id', 'treat_group', 'ctrl_group'])
        card['after_dedup'] = len(C)
        # implausible standardized effects indicate a mis-parsed dispersion field
        implausible = (C['family'] == 'continuous') & (C['delta'].abs() > 5)
        card['dropped_implausible_smd'] = int(implausible.sum())
        C = C[~implausible]
        card['after_smd_filter'] = len(C)
        card['contrast_rows_final'] = len(C)
        card['contrast_trials_final'] = C['nct_id'].nunique()
        card['contrast_by_family'] = C['family'].value_counts().to_dict()
        card['contrast_by_split'] = C['split_time'].value_counts().to_dict()
        card['contrast_by_area'] = C['area'].value_counts().to_dict()
    card['arm_by_family'] = A['family'].value_counts().to_dict()
    card['arm_by_split'] = A['split_time'].value_counts().to_dict()
    card['arm_trials'] = A['nct_id'].nunique()
    return A, C, card
