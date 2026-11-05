"""Reconcile the complete, independently audited full-panel Coat diagnostic.

This produces descriptive paper assets only. It never recomputes experiments,
selects a configuration, or converts numerical audit into scientific coverage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    ('observation', 'OBS'),
    ('randomized_group_score', 'R group'),
    ('randomized_cell_projection', 'R projected'),
    ('fusion_group_score', 'Fusion group'),
    ('fusion_cell_projection', 'Fusion projected'),
)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def reconcile(run, audit_path):
    manifest = read(run / 'manifest.json')
    protocol = read(run / 'protocol.json')
    audit = read(audit_path)
    require(manifest['status'] == 'complete', 'Full-panel run incomplete')
    require(audit.get('complete') is True and audit['run'] == run.relative_to(ROOT).as_posix(),
            'Independent full-panel audit incomplete or belongs to another run')
    require(sha(run / 'manifest.json') == audit['run_manifest_sha256'], 'Audit manifest binding changed')
    require(sha(run / 'protocol.json') == manifest['protocol_sha256'] == audit['protocol_sha256'],
            'Audit protocol binding changed')
    for name, expected in audit['run_file_sha256'].items():
        item = Path(name)
        require(not item.is_absolute() and '..' not in item.parts, 'Unsafe audit path')
        require(sha(run / item) == expected, 'Audited run file changed: ' + name)
    for name, expected in manifest['source_sha256'].items():
        require(sha(run / 'source_snapshot' / name) == expected, 'Source snapshot changed: ' + name)
    for name, expected in protocol['source_files_sha256'].items():
        require(sha(ROOT / name) == expected == audit['checked_source_sha256'][name], 'Raw input changed')

    queries = read(run / 'queries.json')
    results = read(run / 'results.json')
    summary = read(run / 'summary.json')
    full_cell = read(run / 'full_cell_control.json')
    wanted = set(itertools.product(protocol['sensitivity_radii'], protocol['transport_tolerances']))
    require(len(queries) == 15 and len(results) == 12 and
            {(r['radius'], r['rho']) for r in results} == wanted, 'Incomplete registered query grid')
    primary_index = next(i for i, q in enumerate(queries) if
                         all(q[k] == v for k, v in protocol['queries']['primary'].items()))
    primary = next(r for r in results if r['radius'] == protocol['primary_radius'] and
                   r['rho'] == protocol['primary_transport_tolerance'])
    require(summary['primary_query_index'] == primary_index == audit['primary']['query_index'],
            'Primary query changed')
    statuses = Counter()
    by_method = {m: Counter() for m, _ in METHODS}
    documents = proofs = refusals = 0
    for row in results:
        require(set(row['methods']) == {m for m, _ in METHODS}, 'Missing or unexpected source method')
        for method, item in row['methods'].items():
            require(len(item['answers']) == 15, 'Missing query response')
            documents += 1
            refusals += item['document']['status'] != 'ready'
            for answer in item['answers']:
                statuses[answer['status']] += 1
                by_method[method][answer['status']] += 1
                proofs += answer['certificate'] is not None
    require(len(full_cell['answers']) == 15, 'Missing full-cell control')
    documents += 1
    refusals += full_cell['document']['status'] != 'ready'
    for answer in full_cell['answers']:
        statuses[answer['status']] += 1
        proofs += answer['certificate'] is not None
    require(documents == audit['view_documents'] == 61 and sum(statuses.values()) == audit['query_responses'] == 915,
            'Audit response/document census differs')
    require(proofs == audit['proofs_valid'] and statuses == Counter(audit['answer_status_counts']),
            'Audit proof/status census differs')

    old = ROOT / protocol['original_case']
    require(sha(old / 'queries.json') == protocol['original_queries_file_sha256'], 'Original query record changed')
    thin = read(old / 'summary.json')
    rows = []
    for method, label in METHODS:
        answer = primary['methods'][method]['answers'][primary_index]
        for key in ('status', 'lower_bound'):
            require(answer[key] == summary['primary_methods'][method][key] ==
                    audit['primary']['methods'][method][key], 'Primary answer differs')
        rows.append({'method': method, 'label': label,
                     'thinned': thin['primary_methods'][method],
                     'full_panel': {k: answer.get(k) for k in ('status', 'lower_bound', 'route', 'reason')},
                     'full_panel_all_declarations_statuses': dict(by_method[method])})
    full_answer = full_cell['answers'][primary_index]
    for key in ('status', 'lower_bound'):
        require(full_answer[key] == summary['primary_full_cell_control'][key] ==
                audit['primary']['full_cell_control'][key], 'Full-cell primary answer differs')
    rows.append({'method': 'full_cell_control', 'label': 'R full-cell control',
                 'thinned': thin['primary_full_cell_control'],
                 'full_panel': {k: full_answer.get(k) for k in ('status', 'lower_bound', 'route', 'reason')}})
    rescue = (rows[3]['full_panel']['status'] == 'certified' and
              rows[0]['full_panel']['status'] == rows[1]['full_panel']['status'] == 'unresolved')
    require(rescue == summary['primary_descriptive_hypothesis_observed'] ==
            audit['primary']['descriptive_hypothesis_observed'], 'Primary diagnostic condition differs')
    controls = read(run / 'retained_direct_controls.json')
    old_queries = read(old / 'queries.json')
    direct = controls['per_query'][primary_index]['all_16_within_user']
    require(direct == old_queries[primary_index]['direct_randomized_scores']['all_16_within_user'],
            'Retained direct control changed')
    return {
        'run': run.relative_to(ROOT).as_posix(), 'manifest_sha256': sha(run / 'manifest.json'),
        'protocol_sha256': manifest['protocol_sha256'], 'audit_sha256': sha(audit_path),
        'summary_sha256': sha(run / 'summary.json'),
        'independent_users': 290, 'view_documents': documents, 'query_responses': sum(statuses.values()),
        'proofs_valid': proofs, 'refused_view_documents': refusals, 'answer_status_counts': dict(statuses),
        'primary_query_index': primary_index, 'primary_descriptive_hypothesis_observed': rescue,
        'primary_comparisons': rows, 'retained_all16_direct_control': direct,
        'scope': 'Retrospective diagnostic on the same users, with explicit SRSWOR, iid, static-rating, sensitivity and transport conditions. No known population truth or measured coverage.'}


def formatted(value):
    return '--' if value is None else f'${value:.5f}$'


def table(facts):
    lines = [r'\begin{tabular}{lrr}', r'\toprule',
             r'Construction & \shortstack{One item\\per source} & \shortstack{Full user\\blocks}\\', r'\midrule']
    for row in facts['primary_comparisons']:
        lines.append(' & '.join((row['label'], formatted(row['thinned']['lower_bound']),
                                formatted(row['full_panel']['lower_bound']))) + r'\\')
    return '\n'.join([*lines, r'\bottomrule', r'\end{tabular}', ''])


def prose(facts):
    rows = facts['primary_comparisons']
    f = rows[3]['full_panel']; r = rows[1]['full_panel']
    counts = facts['answer_status_counts']
    comparison = ''
    if f['lower_bound'] is not None and r['lower_bound'] is not None:
        gap = 100 * (f['lower_bound'] - r['lower_bound'])
        comparison = f"The fused lower bound is {abs(gap):.3f} percentage points {'higher' if gap >= 0 else 'lower'}; this compares bounds, not measured policy effects. "
    if facts['primary_descriptive_hypothesis_observed']:
        conclusion = 'Fusion certifies the fixed primary policy while both source-alone views remain unresolved.'
    elif f['status'] == 'unresolved':
        conclusion = 'The bound remains below the requested 0.001 improvement margin.'
    elif f['status'] == 'uncertain':
        conclusion = 'The primary query has no verified fused answer.'
    else:
        conclusion = 'The primary certification is not unique to fusion.'
    census = (f"All {facts['query_responses']} responses are unresolved." if
              counts.get('unresolved', 0) == facts['query_responses'] else
              f"The full-panel diagnostic returns {counts.get('certified', 0)} certified, "
              f"{counts.get('unresolved', 0)} unresolved and {counts.get('uncertain', 0)} uncertain responses.")
    return rf'''% Generated from complete full-panel results and independent audit.
{census} At the primary declaration, the full-panel group-fusion lower bound is {formatted(f['lower_bound'])}, versus {formatted(r['lower_bound'])} for full-budget R group. {comparison}{conclusion} Table~\ref{{tab:coat-blocks}} retains both protocols and all source constructions. Independent audit reconstructs user-level statistics, all 61 view documents and {facts['proofs_valid']} numerical proofs; {facts['refused_view_documents']} view documents are refused. This does not establish the sampling or transport assumptions empirically.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/icde2027/coat_full_block_v1')
    parser.add_argument('--audit', type=Path, default=ROOT / 'runs/icde2027/audit/coat_full_block_result_audit.json')
    parser.add_argument('--paper', type=Path, default=ROOT / 'icde2027/paper')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    facts = reconcile(args.run.resolve(), args.audit.resolve())
    if args.check_only:
        print(json.dumps({'complete': True, 'responses': facts['query_responses'],
                          'primary_diagnostic_observed': facts['primary_descriptive_hypothesis_observed']}))
        return
    args.paper.mkdir(parents=True, exist_ok=True)
    (args.paper / 'coat_block_figure_data.json').write_text(json.dumps(facts, indent=2, allow_nan=False) + '\n')
    (args.paper / 'coat_full_block_table.tex').write_text(table(facts))
    (args.paper / 'coat_full_block_results.tex').write_text(prose(facts))
    print('Generated descriptive Coat comparison table, results and provenance.')


if __name__ == '__main__':
    main()
