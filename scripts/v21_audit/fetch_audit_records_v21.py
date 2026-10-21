"""Draw the v2.1 re-audit sample and fetch full CT.gov records for it.
Sample: 100 contrasts uniformly from all v2.1 contrasts (data/v2.1/benchmark_contrasts_v2.1.csv.gz, file row order),
numpy default_rng(1).choice(n, 100, replace=False), sorted; same design as sample A of the v2 audit (rng 2026 on v2).
Records: same modules as the v2 audit fetch (holdout_harvest.MODULES), ClinicalTrials.gov API v2."""
import json, gzip, os, sys
import numpy as np, pandas as pd
from fetch_ctgov_meta import fetch_ids
MODULES = ['protocolSection.identificationModule', 'protocolSection.statusModule', 'protocolSection.sponsorCollaboratorsModule',
           'protocolSection.descriptionModule', 'protocolSection.conditionsModule', 'protocolSection.designModule',
           'protocolSection.armsInterventionsModule', 'protocolSection.outcomesModule', 'protocolSection.eligibilityModule',
           'resultsSection.participantFlowModule', 'resultsSection.baselineCharacteristicsModule', 'resultsSection.outcomeMeasuresModule',
           'derivedSection.conditionBrowseModule', 'derivedSection.interventionBrowseModule', 'protocolSection.referencesModule']
if __name__ == '__main__':
    C = pd.read_csv(sys.argv[1], low_memory=False); C['row_id'] = np.arange(len(C))
    S = C.iloc[np.sort(np.random.default_rng(1).choice(len(C), 100, replace=False))].copy(); S['sample'] = 'v21_arm_match_audit'
    S.to_csv('audit_samples_v21.csv', index=False)
    recs = fetch_ids(sorted(S.nct_id.unique()), fields=','.join(MODULES), batch=50)
    with gzip.open('audit_records_v21.jsonl.gz', 'wt') as f:
        for s in recs: f.write(json.dumps(s, separators=(',', ':')) + '\n')
    print(len(S), S.nct_id.nunique(), len(recs))
