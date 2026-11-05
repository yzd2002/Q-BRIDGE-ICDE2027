"""Reconcile the complete threshold follow-up without selecting conditions."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def same(a, b):
    return math.isclose(a, b, rel_tol=2e-10, abs_tol=2e-9)


def geometric(values):
    need(all(math.isfinite(v) and v > 0 for v in values), 'Invalid timing ratios')
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/icde2027/threshold_workload_v2')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/icde2027/threshold_workload_summary_v2')
    args = parser.parse_args()
    directory = args.run.resolve(); manifest = read(directory / 'manifest.json')
    need(manifest['status'] == 'complete' and manifest['completed_conditions'] == 60, 'Study is not complete')
    reg = read(directory / 'protocol.json')
    need(digest(directory / 'protocol.json') == manifest['protocol_sha256'], 'Protocol changed')
    for name, expected in manifest['source_sha256'].items():
        need(digest(directory / 'source_snapshot' / name) == expected, f'Source snapshot changed: {name}')
    need(reg['queries'] == 100 and reg['repetitions'] == 3 and len(reg['methods']) == 6, 'Unregistered grid')
    cfg = reg['query_engine']; n = reg['queries']; prefixes = reg['query_prefixes']
    expected = {('engine', f'd{d}_{s}_{q}_r{r}') for d, s, q, r in itertools.product(
        cfg['dimensions'], cfg['structures'], cfg['streams'], range(3))}
    expected.update(('end_to_end', f'{s}_r{r}') for s, r in itertools.product(reg['end_to_end']['randomized_designs'], range(3)))
    completed = read(directory / 'completed_conditions.json')
    need(len(completed) == len(expected) == 60 and {(c['stage'], c['name']) for c in completed} == expected,
         'Missing, repeated or unexpected condition')
    names = [m['name'] for m in reg['methods']]
    conditions = []; flat = []; decisions = Counter(); valid = missing = 0
    for condition in completed:
        loc = directory / condition['stage'] / condition['name']
        need(condition == read(loc / 'condition.json'), 'Condition index differs from saved record')
        need(digest(loc / 'queries.npy') == condition['queries_sha256'], 'Saved query input changed')
        need(sorted(condition['method_order']) == sorted(names), 'Plan list differs')
        methods = {}
        for name in names:
            result = read(loc / f'{name}_summary.json')
            need(result['method'] == name and result['queries'] == n, 'Plan/query identity differs')
            rows = result['per_query']
            need([r['query'] for r in rows] == list(range(1, n + 1)), 'Query trace is incomplete')
            total = sum(result['input_stage_seconds'].values()) + sum(r['query_seconds'] + r['serialization_seconds'] for r in rows)
            need(same(total, result['producer_seconds']) and same(rows[-1]['cumulative_seconds'], total), 'Producer accounting differs')
            need(all(math.isfinite(r['query_seconds']) and r['query_seconds'] >= 0 and
                     math.isfinite(r['serialization_seconds']) and r['serialization_seconds'] >= 0 for r in rows), 'Invalid duration')
            replay = result['persistent_replay']; standalone = result['stateless_replay']
            for checks in (replay, standalone):
                need(checks['valid_proofs'] + checks['no_bound_responses'] == n, 'Incomplete proof accounting')
                need([r['query'] for r in checks['per_query']] == list(range(1, n + 1)), 'Incomplete receiver trace')
                costs = checks['checker_initialization_seconds'] + checks['decode_seconds'] + checks['verify_seconds']
                need(same(costs, checks['total_consumer_seconds']) and same(costs, checks['per_query'][-1]['cumulative_seconds']), 'Receiver accounting differs')
                need(checks['decisions'] == result['decisions'], 'Receiver-mode decisions differ')
            need(same(result['joint_seconds'], total + replay['total_consumer_seconds']), 'Joint accounting differs')
            need(digest(loc / result['proof_file']) == result['proof_file_sha256'], 'Proof stream changed')
            need((loc / result['proof_file']).stat().st_size == result['compressed_bytes'], 'Compressed size differs')
            for q in prefixes:
                key = str(q)
                need(result['prefixes'][key] == rows[q - 1]['cumulative_seconds'], 'Producer prefix differs')
                for checks in (replay, standalone):
                    need(checks['prefixes'][key] == checks['per_query'][q - 1]['cumulative_seconds'], 'Receiver prefix differs')
                need(same(result['joint_prefixes'][key], result['prefixes'][key] + replay['prefixes'][key]), 'Joint prefix differs')
                flat.append({'stage': condition['stage'], 'condition': condition['name'], 'dimension': condition['dimension'],
                             'structure': condition['structure'], 'stream': condition['stream'], 'repeat': condition['repeat'],
                             'method': name, 'queries': q, 'producer_seconds': result['prefixes'][key],
                             'consumer_seconds': replay['prefixes'][key], 'joint_seconds': result['joint_prefixes'][key]})
            tally = Counter('true' if r['decision'] is True else 'false' if r['decision'] is False else 'unknown' for r in rows)
            need(dict(tally) == result['decisions'], 'Decision counts differ')
            valid += replay['valid_proofs']; missing += replay['no_bound_responses']; decisions.update(tally)
            methods[name] = result
        for i in range(n):
            need(len({r['per_query'][i]['decision'] for r in methods.values() if r['per_query'][i]['decision'] is not None}) <= 1,
                 'Opposite decided answers among plans')
        for baseline in reg['primary']['baselines']:
            base = methods[baseline]
            for name in names:
                item = methods[name]; saved = condition['comparisons'][baseline][name]
                need(same(saved['producer_ratio'], item['producer_seconds'] / base['producer_seconds']) and
                     same(saved['joint_ratio'], item['joint_seconds'] / base['joint_seconds']), 'Paired ratio differs')
                paired = zip(item['per_query'], base['per_query'])
                loss = sum(b['decision'] is not None and a['decision'] is not b['decision'] for a, b in paired)
                need(loss == saved['lost_decided_answers'], 'Decided-answer loss differs')
        conditions.append({**condition, 'methods': methods})
    need(valid + missing == 36000 and len(flat) == 1080, 'Final response/prefix census differs')
    primary = {}; proposed = reg['primary']['method']
    for baseline in reg['primary']['baselines']:
        ratios = []
        for rep in range(3):
            group = [c for c in conditions if c['stage'] == 'engine' and c['repeat'] == rep]
            need(len(group) == 18, 'Engine repeat lacks a grid condition')
            ratios.append(geometric([c['methods'][proposed]['joint_seconds'] / c['methods'][baseline]['joint_seconds'] for c in group]))
        losses = sum(c['comparisons'][baseline][proposed]['lost_decided_answers'] for c in conditions)
        primary[baseline] = {'joint_geometric_ratios_by_repeat': ratios, 'median_geometric_ratio': median(ratios),
                             'lost_decided_answers_all_conditions': losses,
                             'engineering_target_supported': all(v <= reg['primary']['maximum_ratio'] for v in ratios) and losses == 0}
    result = {'manifest': manifest, 'protocol': reg, 'conditions': conditions,
              'primary': primary, 'both_comparisons_supported': all(x['engineering_target_supported'] for x in primary.values()),
              'proof_totals': {'valid': valid, 'missing': missing}, 'decision_totals': dict(decisions),
              'p_value': None, 'scope': 'Complete measurement reconciliation; independent proof audit is separate.'}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (args.out / 'all_prefixes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    print(json.dumps({k: result[k] for k in ('primary', 'both_comparisons_supported', 'proof_totals', 'decision_totals')}, indent=2))


if __name__ == '__main__':
    main()
