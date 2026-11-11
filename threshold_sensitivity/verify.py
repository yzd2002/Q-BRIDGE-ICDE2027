"""Read out thresholds from saved exact capacities; no fitting or LP solves."""
from pathlib import Path
from fractions import Fraction
import argparse,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,help='Optional original decomposition_occurrences.jsonl from the full archive');a=p.parse_args()
d=json.loads((Path(__file__).parent/'capacity_records.json').read_text());rows=d['rows']
assert len(rows)==240 and len({x['occurrence_id'] for x in rows})==240
if a.source:
    assert hashlib.sha256(a.source.read_bytes()).hexdigest()==d['source_sha256']
    raw=[json.loads(x) for x in a.source.open()]
    raw={x['occurrence_id']:x for x in raw if x['group']==d['primary_group']}
    assert len(raw)==240
    for row in rows:
        assert all(raw[row['occurrence_id']][k]==v for k,v in row.items())
elig=[x for x in rows if x['primary_107_eligible']];assert len(elig)==107
counts=[]
for q in d['query_thresholds_probability']:
    t=Fraction(q)
    counts.append({'threshold_pp':float(100*t),'all_240_family_above':sum(Fraction(x['R'])>t for x in rows),'original_107_family_above':sum(Fraction(x['R'])>t for x in elig),'original_107_candidates_above':sum(Fraction(x['L3_plus'])>t for x in elig)})
assert counts==d['counts']
assert [x['original_107_family_above'] for x in counts]==[86,71,42,12]
assert all(Fraction(x['L3_plus'])==0 for x in elig)
print(json.dumps({'status':'PASS','source_rows_checked':bool(a.source),'exact_rational_comparison':True,'counts':counts},indent=2))
