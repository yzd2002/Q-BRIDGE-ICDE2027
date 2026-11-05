"""Reaggregate saved, independently audited costs; never run a query/timer."""
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT/'runs/icde2027/audit/checking_schedule_v4_result_audit.json'
EXPECTED_AUDIT_SHA = 'd0368340190d9639f07045acaa98ec3351c3f8e16afc0c1801a4d68741d44a30'
RUN = ROOT/'runs/icde2027/checking_schedule_v4'
GROUPS = ('original_main', 'threshold_engine', 'source_derived')
METHODS = ('s0_p0', 's0_p1', 's1_p0', 's1_p1', 'screened_auto', 'original_prepared')


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    assert sha(AUDIT) == EXPECTED_AUDIT_SHA
    audit = json.loads(AUDIT.read_text())
    assert all(audit.get(k) is True for k in ('complete', 'passed', 'independent_review'))
    sources = {str(AUDIT.relative_to(ROOT)): sha(AUDIT)}
    records = []
    for round_index in range(3):
        for group, count in zip(GROUPS, (3, 54, 6)):
            conditions = sorted(p for p in (RUN/f'round_{round_index}'/group).iterdir() if p.is_dir())
            assert len(conditions) == count
            for method in METHODS:
                values = []
                for directory in conditions:
                    path = directory/f'{method}_summary.json'; name = str(path.relative_to(ROOT))
                    assert sha(path) == audit['sources_sha256'][name]
                    sources[name] = sha(path)
                    s = json.loads(path.read_text())
                    assert s['method'] == method and s['queries'] == (1000 if group == 'original_main' else 100)
                    consumer = s['persistent_replay']['total_consumer_seconds']
                    assert math.isclose(s['producer_seconds']+consumer, s['joint_seconds'], rel_tol=1e-12, abs_tol=1e-12)
                    values.append({'producer': s['producer_seconds'], 'consumer': consumer,
                                   'joint': s['joint_seconds'], 'queries': s['queries'],
                                   'bytes': s['uncompressed_bytes']})
                sums = {key: sum(v[key] for v in values) for key in values[0]}
                records.append({'round': round_index, 'group': group, 'method': method,
                                'inputs': count, 'queries': sums['queries'],
                                'producer_ms_per_request': 1000*sums['producer']/sums['queries'],
                                'receiver_ms_per_request': 1000*sums['consumer']/sums['queries'],
                                'joint_ms_per_request': 1000*sums['joint']/sums['queries'],
                                'receiver_fraction_of_joint': sums['consumer']/sums['joint'],
                                'wire_bytes_per_request': sums['bytes']/sums['queries']})
    assert len(sources)-1 == 1134 and len(records) == 54
    fields = ('producer_ms_per_request', 'receiver_ms_per_request', 'joint_ms_per_request',
              'receiver_fraction_of_joint', 'wire_bytes_per_request')
    aggregate = []
    for group in GROUPS:
        for method in METHODS:
            selected = [r for r in records if r['group'] == group and r['method'] == method]
            aggregate.append({'group': group, 'method': method,
                              **{key: {'median': statistics.median(r[key] for r in selected),
                                       'minimum': min(r[key] for r in selected),
                                       'maximum': max(r[key] for r in selected)} for key in fields}})
    facts = {'scope': 'Descriptive arithmetic reaggregation of saved complete M2 timings; no new timing or query.',
             'records': records, 'three_round_summaries': aggregate, 'sources_sha256': sources,
             'script_sha256': sha(__file__)}
    path = ROOT/'runs/icde2027/analysis/execution_cost_context_v5.json'
    path.write_text(json.dumps(facts, indent=2, allow_nan=False)+'\n')
    table = [r'\begin{table}[t]', r'\centering\small',
             r'\caption{Absolute checking costs on the three primary streams. In each round, joint time and uncompressed response bytes are divided by 3,000 requests; receiver share is receiver time divided by joint time. Entries are medians of three rounds. Joint cost includes the persistent receiver; 1 KiB is 1,024 bytes.}',
             r'\label{tab:absolute-checking}', r'\begin{tabular}{lrrr}', r'\toprule',
             r'Plan & Joint ms/request & Receiver & KiB/request \\', r'\midrule']
    labels = {'s0_p0': r'$S_0P_0$', 's0_p1': r'$S_0P_1$', 's1_p0': r'$S_1P_0$', 's1_p1': r'$S_1P_1$'}
    for method, label in labels.items():
        item = next(r for r in aggregate if r['group'] == 'original_main' and r['method'] == method)
        table.append(f"{label} & {item['joint_ms_per_request']['median']:.2f} & "
                     f"{100*item['receiver_fraction_of_joint']['median']:.1f}\\% & "
                     f"{item['wire_bytes_per_request']['median']/1024:.2f} \\\\")
    table += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (ROOT/'icde2027/paper/execution_cost_context_v5.tex').write_text('\n'.join(table)+'\n')
    for name, h in sources.items():
        assert sha(ROOT/name) == h
    print(json.dumps({'summary_files': 1134, 'aggregate_rows': 18, 'output': str(path.relative_to(ROOT))}))


if __name__ == '__main__':
    main()
