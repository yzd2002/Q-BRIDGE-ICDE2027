"""Paper assets from independently audited V5 full panels and retained thinning."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/icde2027/analysis/coat_representation_v5_run'
AUDIT = ROOT/'runs/icde2027/audit/coat_representation_v5_result_audit.json'
AUDIT_SHA = '2db138f6c1b64f75c48b565637dca18f6c9c0ed18ebee3df3477237321f27f4b'
THIN = ROOT/'icde2027/paper/repair_summary_facts_v4.json'
THIN_SHA = '9412f97ebc01b64accef25c99978c0ef29fca18b44f7fe8024cae2d896a61d7c'
PAPER = ROOT/'icde2027/paper'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def main():
    require(sha(AUDIT) == AUDIT_SHA and sha(THIN) == THIN_SHA, 'Audited data identity changed')
    audit, thin = read(AUDIT), read(THIN)
    require(all(audit.get(k) is True for k in ('complete', 'passed', 'independent_review')),
            'Actual independent full result audit required')
    require(audit['run'] == str(RUN.relative_to(ROOT)) and
            sha(RUN/'manifest.json') == audit['run_manifest_sha256'], 'Wrong audited run')
    require((audit['views_checked'], audit['responses_checked'], audit['direct_controls_checked']) == (61, 915, 15),
            'Full audited census required')
    sources = dict(audit['sources_sha256'])
    for name, h in thin['sources_sha256'].items():
        require(name not in sources or sources[name] == h, 'Conflicting source identity')
        sources[name] = h
    sources.update({str(AUDIT.relative_to(ROOT)): AUDIT_SHA, str(THIN.relative_to(ROOT)): THIN_SHA})
    for name, h in sources.items():
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts and (ROOT/p).resolve().is_relative_to(ROOT), 'Unsafe evidence path')
        require(sha(ROOT/p) == h, 'Audited source changed: '+name)
    manifest = read(RUN/'manifest.json'); summary = read(RUN/'summary.json')
    require(manifest['status'] == 'complete' and summary['status_counts'] == audit['status_counts'],
            'Run/audit status differs')
    primary_index = summary['primary_query_index']
    require(primary_index == audit['primary_query_index'] == thin['coat_comparison']['primary_query_index'],
            'Primary query identity differs')
    protocol = read(RUN/'original_protocol.json')
    journal = [json.loads(s) for s in (RUN/'method_journal.jsonl').read_text().splitlines()]
    require(len(journal) == 60 and len({r['file'] for r in journal}) == 60, 'Incomplete method journal')
    all_results = []
    primary = {}
    for item in journal:
        require(sha(RUN/item['file']) == item['sha256'], 'Changed journal result')
        r = read(RUN/item['file']); all_results.append(r)
        if r['radius'] == protocol['primary_radius'] and r['rho'] == protocol['primary_transport_tolerance']:
            require(r['method'] not in primary, 'Duplicate primary method')
            primary[r['method']] = r['answers'][primary_index]
    control = read(RUN/'full_cell_control.json'); all_results.append(control)
    primary['full_cell_control'] = control['answers'][primary_index]
    require(len(primary) == 6, 'Missing primary construction')
    statuses = Counter(a['status'] for r in all_results for a in r['answers'])
    require(sum(statuses.values()) == 915 and dict(statuses) == audit['status_counts'], 'Answer census differs')
    rows = []
    for old in thin['coat_comparison']['primary_comparisons']:
        name = old['method']; current = primary[name]
        audit_name = 'randomized_full_cell_control' if name == 'full_cell_control' else name
        require(all(current.get(k) == audit['primary_results'][audit_name].get(k) for k in ('status', 'lower_bound', 'reason')),
                'Primary audited answer differs')
        rows.append({'method': name, 'label': old['label'], 'thinned': old['thinned'],
                     'thinned_provenance': old['thinned_provenance'],
                     'full_panel': {k: current.get(k) for k in ('status', 'lower_bound', 'reason')},
                     'full_panel_provenance': 'Complete independently audited V5 fixed-data representation repair.'})
    direct = read(RUN/'direct_controls.json')
    require(len(direct) == 15 and [r['query_index'] for r in direct] == list(range(15)), 'Incomplete direct controls')
    require(all(r['new_passes_margin'] is (r['new']['lower_bound'] > r['decision_threshold']) for r in direct),
            'Direct threshold flags differ')
    f = primary['fusion_group_score']['lower_bound']; r = primary['randomized_group_score']['lower_bound']
    gap = 100*(Fraction(f)-Fraction(r))
    require(statuses == Counter({'unresolved': 915}) and all(not d['new_passes_margin'] for d in direct),
            'This prose must be revised for changed outcomes')
    facts = {'status': 'complete', 'sources_sha256': sources, 'generator_sha256': sha(__file__),
             'full_panel_audit_sha256': AUDIT_SHA, 'primary_comparisons': rows,
             'full_panel_status_counts': dict(statuses), 'direct_controls': direct,
             'primary_query_index': primary_index, 'group_fusion_minus_R_pp_exact': [str(gap.numerator), str(gap.denominator)],
             'old_new_lower_directions': audit['old_new_lower_directions'],
             'scope': 'Same fixed users/items/queries, conditional scientific assumptions; no new data or empirical coverage.'}
    table = [r'\begin{tabular}{lrr}', r'\toprule',
             r'Construction & \shortstack{One item\\per source} & \shortstack{Full user\\blocks}\\', r'\midrule']
    table += [f"{x['label']} & ${x['thinned']['lower_bound']:.5f}$ & ${x['full_panel']['lower_bound']:.5f}$"+r'\\' for x in rows]
    table += [r'\bottomrule', r'\end{tabular}']
    (PAPER/'coat_table_v5.tex').write_text('\n'.join(table)+'\n')
    (PAPER/'coat_full_block_results_v5.tex').write_text(
        rf'''% Generated from independently audited fixed-data V5 repair.
All 915 full-panel lower-bound answers remain unresolved, with no refused view. At the primary declaration, group fusion gives ${f:.5f}$ versus ${r:.5f}$ for full-budget R group: a {float(gap):.3f}-percentage-point improvement in the bound, not a measured policy effect. Neither passes $\delta=.001$. Table~\ref{{tab:coat-blocks}} retains every source construction. The independent audit reconstructs user-level statistics and all 61 views, 915 proofs and 15 direct controls. It does not establish the scientific source assumptions.
''')
    d = direct[primary_index]['new']
    (PAPER/'coat_direct_results_v5.tex').write_text(
        rf'''% Generated from all 15 independently audited direct controls.
The full-cell premise also permits direct randomized scores, averaged over all 16 ratings within each user, followed by a one-sided Hoeffding bound with $N=290$ for each fixed query. The primary estimate is ${d['estimate']:.5f}$ and its lower bound ${d['lower_bound']:.5f}$. None of the 15 controls passes its margin. These per-query bounds do not support adaptive selection; failure to certify does not establish zero effect or validate transport.
''')
    (PAPER/'coat_facts_v5.json').write_text(json.dumps(facts, indent=2, allow_nan=False)+'\n')
    for name, h in sources.items():
        require(sha(ROOT/name) == h, 'Source changed during asset generation')
    print(json.dumps({'primary_rows': len(rows), 'views': 61, 'responses': 915, 'direct_controls': 15,
                      'source_closure_files': len(sources)}))


if __name__ == '__main__':
    main()
