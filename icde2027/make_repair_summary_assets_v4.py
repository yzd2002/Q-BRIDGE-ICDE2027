"""Summarize the complete, independently audited represented-law repair.

Does not run inference, optimize, alter prior assets, or infer unknown Coat truth.
"""
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

from make_coat_block_assets import reconcile as full_block_reconcile

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs/icde2027'
PAPER = ROOT / 'icde2027/paper'
AUDIT = RUNS / 'audit/represented_law_repair_v4_result_audit.json'
RECORDS = RUNS / 'audit/represented_law_repair_v4_result_records.json'
RUN = RUNS / 'analysis/represented_law_repair_v4_run'
AUDIT_SHA = 'd40f6ecfe7bef2359843396db3c6949c084f7206f0d5466fb44b64bb918c6a97'
RECORDS_SHA = '7fbe38ab474a17afc60def16736069a3ecb9a66334c4cd8194790487e937ac07'
RETAINED_PINS = {
    'icde2027/make_coat_block_assets.py': '4ec049199286455b2ae7953f5f7278eb52f3d1f17d328ed33442dd77958af448',
    'runs/icde2027/coat_case_v1/summary.json': '8b39e0a5ab1cb019debeed96dfc6f278b32fc141d8333489901b8cd567a84e2a',
    'runs/icde2027/coat_case_v1/full_cell_control.json': '17344cd5ed43e1625b1a0ed81fa13cc7f19d7a7d8b7f059e89636fdc38f0296f',
}
METHODS = [('observation_consistency','OBS'), ('randomized_projected','R projected'),
           ('fusion_projected','Fused projected'), ('randomized_group_score','R group score'),
           ('fusion_group_score','Fused group score')]


def require(ok, reason):
    if not ok: raise ValueError(reason)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def fraction(pair):
    return F(int(pair[0]), int(pair[1]))


def main():
    require(sha(AUDIT) == AUDIT_SHA and sha(RECORDS) == RECORDS_SHA, 'Independent repair audit identity changed')
    a = read(AUDIT); record_file = read(RECORDS)
    require(all(a.get(k) is True for k in ('complete','passed','independent_review')) and
            a['production_entry_constructor_checker_imported_or_used'] is False and record_file['complete'] is True,
            'Complete independent audit required')
    pins = dict(a['sources_sha256'])
    pins.update({str(AUDIT.relative_to(ROOT)): AUDIT_SHA, str(RECORDS.relative_to(ROOT)): RECORDS_SHA})
    for rel,value in RETAINED_PINS.items():
        require(rel not in pins or pins[rel]==value,'Conflicting retained-control source')
        pins[rel]=value
    for rel, expected in pins.items():
        p = Path(rel)
        require(not p.is_absolute() and '..' not in p.parts and (ROOT/p).resolve().is_relative_to(ROOT), 'Unsafe source')
        require(sha(ROOT/p) == expected, 'Audited source changed: '+rel)
    records = record_file['records']
    require(len(records) == a['views_checked'] == 6060 and a['responses_checked'] == 18900, 'Incomplete audit census')
    counters = defaultdict(Counter); statuses = Counter(); deltas = Counter(); changes = []
    seen = set()
    for r in records:
        key = (r['source_file'],r['record'],r['method']); require(key not in seen,'Duplicate audited row'); seen.add(key)
        is_world = r['source_file'].startswith('runs/icde2027/fresh_worlds_v1/worlds/world_')
        counts = counters[r['method']] if is_world else Counter()
        if is_world:
            counts['views'] += 1; counts['misses'] += r['coverage'] is False
            counts['refused'] += r['view_status'] != 'ready'; counts['coverage_unavailable'] += r['coverage'] is None
        for family,q in r['queries'].items():
            counts['queries'] += 1; counts['false_certificates'] += q['false_assertion']
            statuses[q['status']] += 1; deltas[q['lower_difference_direction']] += 1
            if q['status_changed']: changes.append({'key':key,'family':family,'old':q['old_status'],'new':q['status']})
            if is_world:
                require(q['true_contrast_exact'] is not None, 'Synthetic truth unavailable')
                counts['query_violations'] += (q['new_lower_bound'] is not None and
                                               F(q['new_lower_bound']) > fraction(q['true_contrast_exact']))
    expected = a['independently_recomputed_counts']
    require(sum(statuses.values()) == 18900 and dict(statuses) == expected['query_status'] and
            dict(deltas) == expected['lower_directions'] and len(changes) == expected['status_changes'],
            'Audit summaries disagree')
    require(set(counters) == {m for m,_ in METHODS}, 'Incomplete synthetic method grid')
    coverage = []
    for method,label in METHODS:
        counts = dict(counters[method]); require(counts['views'] == 1200 and counts['queries'] == 3600, 'Method denominator differs')
        coverage.append({'method':method,'label':label,**counts})
    require(sum(r['misses'] for r in coverage) == expected['coverage']['misses'], 'Coverage summary differs')

    coat = read(RUN/'results/coat_thinned.json')
    protocol = read(RUNS/'coat_case_v1/protocol.json')
    grid = coat['records']; require(len(grid)==12, 'Incomplete Coat grid')
    primary_rows = [r for r in grid if r['original_configuration']=={'radius':protocol['primary_radius'],
                                                                   'rho':protocol['primary_transport_tolerance']}]
    require(len(primary_rows)==1, 'Coat primary declaration missing or duplicated'); primary=primary_rows[0]
    q_index = next(i for i,q in enumerate(primary['original_queries'])
                   if all(q[k]==v for k,v in protocol['queries']['primary'].items()))
    primary_record = grid.index(primary)
    by_key = {(r['record'],r['method']):r for r in records if r['source_file']=='runs/icde2027/coat_case_v1/results.json'}
    full = full_block_reconcile(RUNS/'coat_full_block_v1', RUNS/'audit/coat_full_block_result_audit.json')
    require(full['primary_query_index']==q_index, 'Coat query identity differs')
    retained=read(RUNS/'coat_case_v1/full_cell_control.json')
    old_summary=read(RUNS/'coat_case_v1/summary.json')
    require(len(retained['answers'])==15 and old_summary['primary_query_index']==q_index,
            'Retained full-cell control census differs')
    matrix=retained['document']['view']['matrix']; dimension=len(retained['document']['view']['lower'])
    require(dimension==300 and matrix=={'indptr':[0],'indices':[],'data':[]} and
            retained['document']['view']['metadata']['assumption']=='static full-cell transport',
            'Retained full-cell control is not the original transported randomized box')
    for q,answer in zip(primary['original_queries'],retained['answers'],strict=True):
        certificate=answer['certificate']
        require(certificate is not None and certificate['objective']==q['objective'] and
                certificate['view_digest']==retained['document']['digest'] and
                certificate['decision_threshold']==protocol['threshold'] and
                certificate['claimed_lower']==answer['lower_bound'] and answer['status']=='unresolved',
                'Retained full-cell answer/query binding differs')
    for row in full['primary_comparisons']:
        if row['method']=='full_cell_control':
            require(all(row['thinned'][k]==retained['answers'][q_index][k]==old_summary['primary_full_cell_control'][k]
                        for k in ('status','lower_bound')),'Retained control summary differs from saved answer')
            row['thinned_provenance']='Original full-cell-transport randomized box, retained separately; not part of the 900 repaired queries.'
            continue
        answer = primary['methods'][row['method']]['answers'][str(q_index)]
        independent = by_key[primary_record,row['method']]['queries'][str(q_index)]
        require(answer['status']==independent['status'] and answer['lower_bound']==independent['new_lower_bound'],
                'Coat repaired primary disagrees with independent audit')
        row['thinned']={k:answer.get(k) for k in ('status','lower_bound','route','reason')}
        row['thinned_provenance']='Complete represented-law repair with normalized exact q and kappa endpoint mapping.'
    require(all(set(m['answers']) == {str(i) for i in range(15)} for r in grid for m in r['methods'].values()),
            'Coat query keys differ')
    coat_statuses = Counter(q['status'] for r in grid for m in r['methods'].values() for q in m['answers'].values())
    require(sum(coat_statuses.values())==900,'Coat response census differs')
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Construction & Set misses & Query violations & Refusals\\',r'\midrule']
    for r in coverage:
        lines.append(f"{r['label']} & {r['misses']}/{r['views']} & {r['query_violations']}/{r['queries']} & {r['refused']}/{r['views']}"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (PAPER/'coverage_table_repaired_v4.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{lrr}',r'\toprule',r'Construction & \shortstack{One item\\per source} & \shortstack{Full user\\blocks}\\',r'\midrule']
    for row in full['primary_comparisons']:
        lines.append(f"{row['label']} & ${row['thinned']['lower_bound']:.5f}$ & ${row['full_panel']['lower_bound']:.5f}$"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (PAPER/'coat_table_repaired_v4.tex').write_text('\n'.join(lines)+'\n')
    facts={'status':'complete','generator_sha256':sha(Path(__file__)),'sources_sha256':pins,'coverage':coverage,
           'all_query_statuses':dict(statuses),'lower_bound_change_directions':dict(deltas),'status_changes':changes,
           'coat_repaired_query_statuses':dict(coat_statuses),'coat_comparison':full,
           'scope':'Current repaired inference only. Original design views/confirmation and full-cell control remain historical. Bound changes are new minus old, not signs of the new bounds.'}
    for rel in ('audit/coat_full_block_result_audit.json','coat_full_block_v1/manifest.json',
                'coat_full_block_v1/protocol.json','coat_full_block_v1/results.json',
                'coat_full_block_v1/full_cell_control.json','coat_full_block_v1/retained_direct_controls.json'):
        facts['sources_sha256']['runs/icde2027/'+rel]=sha(RUNS/rel)
    (PAPER/'repair_summary_facts_v4.json').write_text(json.dumps(facts,indent=2)+'\n')
    print(json.dumps({'status':'complete','coverage':coverage,'coat_statuses':dict(coat_statuses),
                      'changes':len(changes),'bound_change_directions':dict(deltas)}))

if __name__=='__main__': main()
