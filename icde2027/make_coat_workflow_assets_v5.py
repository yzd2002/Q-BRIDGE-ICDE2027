"""Generate manuscript assets from the two complete independent workflow audits.

This is deterministic analysis of saved records. It executes no query or timer.
The caller supplies the SHA of the actually completed timing audit; no future
audit identity or favourable-result condition is embedded in this program.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import math
import statistics


ROOT = Path(__file__).resolve().parents[1]
QUALITY = 'runs/icde2027/audit/coat_policy_workflow_v5_quality_audit.json'
QUALITY_SHA = '56af5756f67b632e50b986dc9555beace2c729c02d79315b14ac05cee95ab54d'
TIMING = 'runs/icde2027/audit/coat_policy_workflow_v5_result_audit.json'
VIEWS = ('observation', 'randomized_group_score', 'randomized_cell_projection',
         'fusion_group_score', 'fusion_cell_projection')
METHODS = ('current_prepared', 'optimized_last', 'native_full_hot')
LABELS = ('OBS', 'R group', 'R projected', 'Fusion group', 'Fusion projected')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(timing_sha):
    require(len(timing_sha) == 64 and all(c in '0123456789abcdef' for c in timing_sha),
            'Supply the actual independently completed timing-audit SHA256')
    sources = {}

    def pin(name, expected):
        path = (ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and str(path.relative_to(ROOT)) == name,
                'Canonical project-relative source required')
        require(sha(path) == expected, 'Changed source: '+name)
        require(name not in sources or sources[name] == expected, 'Inconsistent source versions')
        sources[name] = expected
        return path

    def audit(name, expected, stage, count, stream_count):
        p = pin(name, expected)
        a = json.loads(p.read_text())
        suffix = 'quality' if stage == 'quality' else 'result'
        require(a['format'] == f'qbridge-coat-policy-workflow-v5-{suffix}-audit'
                and a['status'] == 'complete'
                and all(a.get(k) is True for k in ('complete', 'passed', 'independent_review')),
                'Complete independent audit required')
        require(a['stage'] == stage and type(a['response_count']) is int
                and a['response_count'] == count and a['stream_count'] == stream_count,
                'Complete fixed census required')
        for field in ('sources_sha256', 'artifacts_sha256'):
            for n, h in a[field].items():
                pin(n, h)
        record_names = list(a['artifacts_sha256'])
        require(len(record_names) == 1 and record_names[0].endswith('_records.json'),
                'Exact complete independent records required')
        records = json.loads((ROOT/record_names[0]).read_text())
        require(records['stage'] == stage and records['response_count'] == count
                and len(records['responses']) == count and len(records['streams']) == stream_count,
                'Actual complete response and stream arrays required')
        manifest = pin(a['run']+'/manifest.json', a['run_manifest_sha256'])
        m = json.loads(manifest.read_text())
        require(m['status'] == 'complete' and m['all_inputs_and_runtime_rechecked'] is True,
                'Completed bound producer manifest required')
        return a, records

    quality, qr = audit(QUALITY, QUALITY_SHA, 'quality', 3600, 15)
    timing, tr = audit(TIMING, timing_sha, 'timing', 10800, 45)
    require(quality['input_inventory_sha256'] == timing['input_inventory_sha256'],
            'Quality/timing input inventories differ')
    require(quality['workflow_statistics'] == timing['workflow_statistics'],
            'Quality/timing catalogue statistics differ')
    qs = {(s['view'], s['method']): s for s in qr['streams']}
    ts = {(s['round'], s['view'], s['method']): s for s in tr['streams']}
    require(set(qs) == {(v, m) for v in VIEWS for m in METHODS}, 'Quality stream set differs')
    require(set(ts) == {(r, v, m) for r in range(3) for v in VIEWS for m in METHODS},
            'Timing stream set differs')

    quality_counts = []
    for view in VIEWS:
        for method in METHODS:
            rows = [r for r in qr['responses'] if r['view'] == view and r['method'] == method]
            require(len(rows) == 240 and sorted(r['query_index'] for r in rows) == list(range(240)),
                    'Incomplete online quality denominator')
            counts = Counter('true' if r['decision'] is True else 'false' if r['decision'] is False else 'unknown' for r in rows)
            require(dict(counts) == qs[view, method]['decisions'], 'Response/stream decisions differ')
            quality_counts.append({'view': view, 'method': method, 'requests': 240,
                                   **{k: counts[k] for k in ('true', 'false', 'unknown')},
                                   'refusals': sum(not r['valid_proof'] for r in rows)})

    round_costs, summaries = [], []
    for view in VIEWS:
        for rnd in range(3):
            values = {m: ts[rnd, view, m]['joint_seconds'] for m in METHODS}
            require(all(math.isfinite(x) and x > 0 for x in values.values()), 'Positive measured costs required')
            round_costs.append({'round': rnd, 'view': view, 'joint_seconds': values,
                                'optimized_current_ratio': values[METHODS[1]]/values[METHODS[0]],
                                'native_current_ratio': values[METHODS[2]]/values[METHODS[0]],
                                'optimized_native_ratio': values[METHODS[1]]/values[METHODS[2]]})
        for method in METHODS:
            rows = [ts[r, view, method] for r in range(3)]
            item = {'view': view, 'method': method, 'requests_per_round': 240}
            for output, field, factor in (
                    ('joint_ms_per_request', 'joint_seconds', 1000/240),
                    ('producer_ms_per_request', 'producer_seconds', 1000/240),
                    ('receiver_ms_per_request', 'receiver_seconds', 1000/240),
                    ('cold_ms_per_request', 'cold_receiver_seconds', 1000/240),
                    ('bytes_per_request', 'uncompressed_response_bytes', 1/240)):
                x = [r[field]*factor for r in rows]
                item[output] = {'median': statistics.median(x), 'minimum': min(x), 'maximum': max(x)}
            summaries.append(item)

    ratios = []
    for rnd in range(3):
        values = [r['optimized_current_ratio'] for r in round_costs if r['round'] == rnd]
        ratios.append(math.exp(sum(map(math.log, values))/5))
    require(all(math.isclose(a, b, rel_tol=1e-14, abs_tol=0) for a, b in zip(ratios, timing['timing']['h2_ratios'])),
            'Independent arithmetic and audit engineering ratios differ')
    online = quality['online_comparisons']
    native_rows = [r for r in online if r['comparison'] == METHODS[2]]
    lost = sum(len(r['lost_query_indices']) for r in native_rows)
    gained = sum(len(r['gained_query_indices']) for r in native_rows)
    same = all(r['counts']['decision_agreements'] == 240 for r in online if r['comparison'] == METHODS[1])
    slower_view_rounds = sum(r['optimized_current_ratio'] > 1 for r in round_costs)
    payload_comparisons = []
    for rnd in range(3):
        for view in VIEWS:
            a = ts[rnd, view, METHODS[0]]['state_payload']['engine_retained_exact_payload_bytes']
            b = ts[rnd, view, METHODS[1]]['state_payload']['engine_retained_exact_payload_bytes']
            payload_comparisons.append({'round': rnd, 'view': view, 'current_bytes': a,
                                        'optimized_bytes': b, 'optimized_larger': b > a})
    native_faster_same_quality = []
    for view, label in zip(VIEWS, LABELS):
        cost_rows = [r for r in round_costs if r['view'] == view]
        comparisons = [r for r in timing['online_comparisons']
                       if r['view'] == view and r['comparison'] == METHODS[2]]
        if (all(r['optimized_native_ratio'] > 1 for r in cost_rows)
                and len(comparisons) == 3
                and all(r['counts']['decision_agreements'] == 240 for r in comparisons)):
            native_faster_same_quality.append(label)

    facts = {'format': 'qbridge-coat-workflow-v5-paper-facts',
             'scope': 'Complete fixed retrospective catalogue workload; no new timing, policy selection or statistical confirmation.',
             'quality_audit_sha256': QUALITY_SHA, 'timing_audit_sha256': timing_sha,
             'script_sha256': sha(__file__), 'workflow_statistics': quality['workflow_statistics'],
             'quality_online_counts': quality_counts, 'h1': quality['h1'],
             'quality_online_comparisons': online, 'timing_online_comparisons': timing['online_comparisons'],
             'all_round_view_costs': round_costs, 'all_view_method_cost_summaries': summaries,
             'optimized_slower_than_current_view_rounds': slower_view_rounds,
             'retained_exact_state_payload_comparisons': payload_comparisons,
             'native_faster_with_same_decisions_all_rounds': native_faster_same_quality,
             'complete_timing_comparisons': timing['timing'], 'all_timing_stream_details': tr['streams'],
             'sources_sha256': sources}
    target = ROOT/'icde2027/paper'
    (target/'coat_workflow_facts_v5.json').write_text(json.dumps(facts, indent=2, allow_nan=False)+'\n')

    table = [r'\begin{table}[tb]', r'\centering\small',
             r'\caption{Catalogue decisions and costs. C/O/H denote current prepared, optimized exact and native hot execution. Decided counts are out of 240 quality requests; no answer is true. Each time is joint cost divided by 240, then median over three rounds, in ms/request. Native hot has different answer quality. Complete round-wise costs and ratios remain in the artifact.}',
             r'\label{tab:catalogue-workflow}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'View & Decided C/O/H & C & O & H \\', r'\midrule']
    for view, label in zip(VIEWS, LABELS):
        counts = [next(r for r in quality_counts if r['view'] == view and r['method'] == m) for m in METHODS]
        decided = '/'.join(str(r['true']+r['false']) for r in counts)
        require(all(r['true'] == 0 for r in counts), 'Update the explicit no-true caption if findings change')
        costs = [next(r for r in summaries if r['view'] == view and r['method'] == method)
                 ['joint_ms_per_request']['median'] for method in METHODS]
        table.append(f'{label} & {decided} & '+' & '.join(f'{v:.2f}' for v in costs)+r' \\')
    table += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (target/'coat_workflow_table_v5.tex').write_text('\n'.join(table)+'\n')

    h2 = timing['timing']['h2_supported']
    h1 = quality['h1']['h1_any_ideal_policy_supported']
    require(not h1, 'Write a result-supported positive H1 explanation if the audited evidence changes')
    require(same, 'Write explicit quality differences instead of the fixed equality sentence')
    require(all(r['optimized_larger'] for r in payload_comparisons),
            'Revise the memory summary if retained-state comparisons change')
    result = [r'% Generated from both complete independently audited workflow stages.',
              r'\input{coat_workflow_table_v5}',
              r'The sequence contains 50 distinct objectives and 150 objective--margin pairs per view. Its ideal objective rank is 14 and its represented rank 15; 12 distinct objectives lie in the moment row space under either encoding. The error transfer changes no answer. No view certifies a positive policy effect, so neither fusion-only information target is supported.',
              f'Table~\\ref{{tab:catalogue-workflow}} reports all five views. Current and optimized execution agree on all 1,200 quality requests. Their three round-wise geometric-mean joint ratios are '+', '.join(f'${r:.3f}$' for r in ratios)+f'; the registered ten-percent engineering target is {"supported" if h2 else "not supported"}. Native hot loses {lost} decided answers and gains {gained} relative to current execution; its cost is therefore not an equal-service comparison. False answers identify feasible counterexamples to a robust guarantee, not negative measured policy effects. Complete three-round outcomes, prefixes, first occurrences and duplicates remain in the artifact.']
    result.append(f'Optimized execution is slower than current in {slower_view_rounds}/15 view--rounds and retains more exact-state payload in every view. These are recorded Python payload estimates, not isolated RSS measurements.'+
                  (' Native hot is faster than optimized in all three '+', '.join(native_faster_same_quality)+
                   ' rounds, with the same decisions there.' if native_faster_same_quality else ''))
    (target/'coat_workflow_results_v5.tex').write_text('\n\n'.join(result)+'\n')
    for name, expected in sources.items():
        require(sha(ROOT/name) == expected, 'Source changed during asset generation: '+name)
    print(json.dumps({'source_files': len(sources), 'timing_streams': len(tr['streams']),
                      'quality_responses': len(qr['responses']), 'timing_responses': len(tr['responses']),
                      'h1_supported': h1, 'h2_supported': h2, 'h2_ratios': ratios}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timing-audit-sha256', required=True)
    main(parser.parse_args().timing_audit_sha256)
