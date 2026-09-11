"""Assemble every number quoted in the paper from results/ into results/paper_numbers.json."""
import json, os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ivbench import models as me

R = os.path.join(os.path.dirname(__file__), '..', 'results')
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
NB_LM = 2000  # language-model intervals use 2,000 trial-level resamples, seed 0 (llm/llm_canonical_numbers.json)
out = {}
ab = pd.read_csv(f'{R}/positive_control/v2_ablation_seeds.csv')
iv = ab[(ab.family == 'binary') & (ab.model == 'full') & (ab.reference == 'blind_ctrl')]
out['gbt_interventional'] = dict(mean=iv.skill.mean(), min=iv.skill.min(), max=iv.skill.max(),
                                 ci_low_mean=iv.ci_low.mean(), ci_high_mean=iv.ci_high.mean())
out['gbt_interventional_continuous'] = ab[(ab.family == 'continuous') & (ab.model == 'full') &
                                          (ab.reference == 'blind_ctrl')].iloc[0][['skill', 'ci_low', 'ci_high']].to_dict()
lm = pd.read_csv(f'{R}/llm/llm_v2_paired.csv')
y, g = lm.truth.to_numpy(), lm.nct_id.to_numpy()
tr = pd.read_csv('data/benchmark_contrasts.csv.gz'); tr = tr[(tr.family == 'binary') & (tr.split_time == 'train')]
ref = np.full(len(y), tr.delta.mean())
for name, a, b in [('lm_full_vs_mean', lm.point_A, ref), ('lm_blind_vs_mean', lm.point_B, ref),
                   ('lm_interventional', lm.point_A, lm.point_B)]:
    s = me.skill(y, a.to_numpy() if hasattr(a, 'to_numpy') else a, b.to_numpy() if hasattr(b, 'to_numpy') else b)
    lo, hi = me.boot_skill(y, np.asarray(a), np.asarray(b), g, n=NB_LM, seed=0)
    out[name] = dict(skill=s, ci_low=lo, ci_high=hi)
canon = json.load(open(f'{R}/llm/llm_canonical_numbers.json'))['sonnet']
for name, key in [('lm_full_vs_mean', 'A'), ('lm_blind_vs_mean', 'B'), ('lm_interventional', 'IS')]:
    assert abs(out[name]['skill'] - canon[key][0]) < 1e-9 and abs(out[name]['ci_low'] - canon[key][1]) < 1e-9, (name, out[name], canon[key])
out['check'] = 'language-model numbers match llm/llm_canonical_numbers.json to 1e-9'
json.dump(out, open(f'{R}/paper_numbers.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
