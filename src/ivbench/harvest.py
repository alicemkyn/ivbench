import gzip, json, time, urllib.parse, urllib.request

MODULES = ['protocolSection.identificationModule', 'protocolSection.statusModule',
           'protocolSection.sponsorCollaboratorsModule', 'protocolSection.descriptionModule',
           'protocolSection.conditionsModule', 'protocolSection.designModule',
           'protocolSection.armsInterventionsModule', 'protocolSection.outcomesModule',
           'protocolSection.eligibilityModule', 'resultsSection.participantFlowModule',
           'resultsSection.baselineCharacteristicsModule', 'resultsSection.outcomeMeasuresModule',
           'derivedSection.conditionBrowseModule', 'derivedSection.interventionBrowseModule']

BASE = "https://clinicaltrials.gov/api/v2/studies"

def page(token=None, size=500):
    q = {'filter.overallStatus': 'COMPLETED', 'aggFilters': 'results:with,studyType:int,phase:2 3',
         'pageSize': str(size), 'fields': ','.join(MODULES)}
    if token:
        q['pageToken'] = token
    req = urllib.request.Request(BASE + "?" + urllib.parse.urlencode(q), headers={'User-Agent': 'python-urllib'})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)

t0 = time.time()
n, token = 0, None
with gzip.open('trials_raw.jsonl.gz', 'wt') as f:
    while True:
        for attempt in range(4):
            try:
                r = page(token)
                break
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:80]}"
                if attempt == 3:
                    raise
                time.sleep(4 + 6 * attempt)
        for s in r['studies']:
            f.write(json.dumps(s, separators=(',', ':')) + "\n")
            n += 1
        token = r.get('nextPageToken')
        if n % 5000 < 500:
            print(f"{n} trials  {time.time()-t0:.0f}s", flush=True)
        if not token:
            break
print(f"DONE {n} trials in {time.time()-t0:.0f}s")
