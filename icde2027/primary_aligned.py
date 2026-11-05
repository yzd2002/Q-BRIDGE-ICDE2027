"""Matched original 1,000-query streams; no changes to either saved executor.

This retrospective timing driver requires a protocol-bound independent gate.
Its lower-only adapter changes the wire envelope, not numerical evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import traceback

import highspy
import numpy as np
import scipy
from scipy.optimize._highspy import _core

from .experiment import json_write
from .proof_check import CheckedView
from .query_workload import engine_for, fixture_view
from .research_io import ROOT, sha256
from .threshold_extension.engine import ThresholdEngine
from .threshold_extension.proofs import ThresholdChecker, bundle_proof
from .threshold_extension.workload import compare, read, require, source_files, utc


METHODS = {
    'original_prepared': ('original', 'prepared_cache', None),
    'original_hot': ('original', 'highs_warm', None),
    'threshold_hot': ('threshold', 'threshold_hot', 1),
    'threshold_hot_auto': ('threshold', 'threshold_hot', 0),
    'both_histories': ('threshold', 'both_pools', 1),
}
RUNTIME = {
    '__init__.py', 'primary_aligned.py', 'aggregate.py', 'confidence_view.py',
    'experiment.py', 'highs_reopt.py', 'proof_check.py', 'query_workload.py',
    'research_io.py', 'source_intervals.py', 'threshold_extension/__init__.py',
    'threshold_extension/engine.py', 'threshold_extension/proofs.py',
    'threshold_extension/workload.py',
}
SERVICE_FIELDS = ('status', 'route', 'lower_bound', 'upper_bound',
                  'robust_threshold_decision', 'certificate', 'reason')


def service_response(answer):
    """One shared wire contract; native diagnostics are archived separately."""
    result = {key: answer.get(key) for key in SERVICE_FIELDS}
    result['reason'] = answer.get('reason') or answer.get('decision_reason')
    return result


def method_order(reg, repeat):
    base = np.random.default_rng(reg['order_seed']).permutation(reg['methods']).tolist()
    shift = 2 * repeat
    return base[shift:] + base[:shift]


def adapt_lower(answer, objective, threshold, digest):
    """Wrap already-checked lower evidence; never invent an upper witness."""
    result = dict(answer)
    proof = answer.get('certificate')
    if proof is None:
        require(answer.get('status') == 'uncertain' and answer.get('lower_bound') is None
                and isinstance(answer.get('reason'), str) and bool(answer['reason']),
                'A missing lower proof must be an explicit no-bound response')
        result.update(upper_bound=None, robust_threshold_decision=None)
        return result
    require(answer.get('lower_bound') == proof.get('claimed_lower'), 'Answer/proof lower mismatch')
    require(answer.get('status') in {'certified', 'unresolved'}, 'Invalid lower-only status')
    bundle = bundle_proof(digest, objective, threshold, proof)
    require((answer['status'] == 'certified') is (bundle['robust_threshold_decision'] is True),
            'Lower-only status disagrees with its threshold evidence')
    result.update(certificate=bundle, upper_bound=None,
                  robust_threshold_decision=bundle['robust_threshold_decision'])
    return result


def make_engine(name, view, bank_size):
    kind, plan, strategy = METHODS[name]
    if kind == 'original':
        return engine_for(view, plan, bank_size)
    return ThresholdEngine(view, plan=plan, bank_size=bank_size, simplex_strategy=strategy)


def checked_response(checker, item, index, objective, threshold):
    require(type(item.get('query_index')) is int and item['query_index'] == index,
            'Response is not bound to its query position')
    answer = item['answer']; value = answer.get('robust_threshold_decision')
    require(value is None or type(value) is bool, 'Decision is not three-valued')
    proof = answer.get('certificate')
    if proof is None:
        require(answer.get('status') == 'uncertain' and answer.get('lower_bound') is None
                and answer.get('upper_bound') is None and value is None
                and isinstance(answer.get('reason'), str) and bool(answer['reason']),
                'Missing proof cannot claim a bound or decision')
        return False
    checked = checker.verify(proof, expected_objective=objective, expected_threshold=threshold)
    require(checked.get('valid') is True, 'Invalid saved proof')
    require(answer.get('lower_bound') == checked['lower_bound']
            and answer.get('upper_bound') == checked['upper_bound']
            and value is checked['robust_threshold_decision'], 'Saved answer differs from proof')
    require(answer.get('status') in {'certified', 'unresolved'}
            and (answer['status'] == 'certified') is (value is True), 'Incorrect response status')
    return True


def replay(path, view, queries, reg, *, stateless=False, view_text=None):
    if view_text is None:
        view_text = json.dumps(view, allow_nan=False)
    tick = time.perf_counter()
    checker = None if stateless else ThresholdChecker(json.loads(view_text), cache_size=reg['bank_size'])
    initialization = 0. if stateless else time.perf_counter() - tick
    decode = verification = 0.; valid = missing = 0
    decisions = Counter(); rows = []
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        for i, line in enumerate(stream):
            require(i < len(queries), 'Extra response')
            tick = time.perf_counter(); item = json.loads(line)
            decode += time.perf_counter() - tick
            if stateless:
                tick = time.perf_counter(); checker = ThresholdChecker(json.loads(view_text), cache_size=0)
                initialization += time.perf_counter() - tick
            tick = time.perf_counter()
            has_proof = checked_response(checker, item, i, queries[i].tolist(), reg['threshold'])
            verification += time.perf_counter() - tick
            valid += int(has_proof); missing += int(not has_proof)
            value = item['answer']['robust_threshold_decision']
            decisions['true' if value is True else 'false' if value is False else 'unknown'] += 1
            rows.append({'query': i + 1, 'cumulative_seconds': initialization + decode + verification})
    require(len(rows) == len(queries), 'Incomplete proof stream')
    return {'mode': 'stateless' if stateless else 'persistent_fifo',
            'cache_size': 0 if stateless else reg['bank_size'],
            'checker_initialization_seconds': initialization, 'decode_seconds': decode,
            'verify_seconds': verification, 'total_consumer_seconds': initialization + decode + verification,
            'valid_proofs': valid, 'no_bound_responses': missing, 'decisions': dict(decisions),
            'prefixes': {str(q): rows[q - 1]['cumulative_seconds'] for q in reg['query_prefixes']},
            'per_query': rows, 'scope': 'Excludes compressed-file reads, decompression and network.'}


def measure(directory, name, view, queries, reg, common_seconds, *, view_text=None):
    retained = []
    try:
        if view_text is None:
            view_text = json.dumps(view, allow_nan=False)
        tick = time.perf_counter(); fresh_view = json.loads(view_text)
        require(fresh_view == view, 'Fresh view decode differs from trusted input')
        view_decode = time.perf_counter() - tick
        tick = time.perf_counter(); engine = make_engine(name, fresh_view, reg['bank_size'])
        initialization = time.perf_counter() - tick
        cumulative = common_seconds + view_decode + initialization
        rows = []; decisions = Counter(); routes = Counter(); total_bytes = 0
        diagnostics = []
        for i, objective in enumerate(queries):
            tick = time.perf_counter(); native = engine.query(objective, reg['threshold'])
            answer = native
            if METHODS[name][0] == 'original':
                answer = adapt_lower(answer, objective.tolist(), reg['threshold'], engine.digest)
            answer = service_response(answer)
            query_seconds = time.perf_counter() - tick
            value = answer['robust_threshold_decision']
            require(value is None or type(value) is bool, 'Non-three-valued response')
            tick = time.perf_counter()
            encoded = json.dumps({'query_index': i, 'answer': answer}, separators=(',', ':'), allow_nan=False)
            serialization = time.perf_counter() - tick
            retained.append(encoded); total_bytes += len(encoded.encode('utf8')) + 1
            cumulative += query_seconds + serialization
            decisions['true' if value is True else 'false' if value is False else 'unknown'] += 1
            routes[answer['route']] += 1
            rows.append({'query': i + 1, 'query_seconds': query_seconds,
                         'serialization_seconds': serialization, 'cumulative_seconds': cumulative,
                         'decision': value, 'lower_bound': answer['lower_bound'],
                         'upper_bound': answer['upper_bound'], 'route': answer['route']})
            diagnostics.append({'query_index': i, 'native_answer': native})
        path = directory / f'{name}.jsonl.gz'
        tick = time.perf_counter()
        with gzip.open(path, 'wt', encoding='utf8', compresslevel=3) as stream:
            for line in retained:
                stream.write(line + '\n')
        compression = time.perf_counter() - tick
        persistent = replay(path, view, queries, reg, view_text=view_text)
        stateless = replay(path, view, queries, reg, stateless=True, view_text=view_text)
        require(persistent['decisions'] == stateless['decisions'] == dict(decisions), 'Replay differs')
        # Diagnostic exports occur after every timed producer query and replay.
        # Their serialization/compression costs do not masquerade as wire cost.
        diagnostic_lines = []; diagnostic_serialization = 0.; diagnostic_bytes = 0
        for item in diagnostics:
            tick = time.perf_counter(); line = json.dumps(item, separators=(',', ':'), allow_nan=False)
            diagnostic_serialization += time.perf_counter() - tick
            diagnostic_lines.append(line); diagnostic_bytes += len(line.encode('utf8')) + 1
        diagnostic_path = directory / f'{name}_native.jsonl.gz'
        tick = time.perf_counter()
        with gzip.open(diagnostic_path, 'wt', encoding='utf8', compresslevel=3) as stream:
            for line in diagnostic_lines:
                stream.write(line + '\n')
        diagnostic_write = time.perf_counter() - tick
        result = {
            'method': name, 'executor': METHODS[name], 'queries': len(queries),
            'common_view_seconds': common_seconds, 'view_decode_seconds': view_decode,
            'plan_initialization_seconds': initialization,
            'producer_seconds': cumulative, 'joint_seconds': cumulative + persistent['total_consumer_seconds'],
            'prefixes': {str(q): rows[q-1]['cumulative_seconds'] for q in reg['query_prefixes']},
            'joint_prefixes': {str(q): rows[q-1]['cumulative_seconds'] + persistent['prefixes'][str(q)]
                               for q in reg['query_prefixes']},
            'per_query': rows, 'decisions': dict(decisions), 'routes': dict(routes),
            'persistent_replay': persistent, 'stateless_replay': stateless,
            'proof_file': path.name, 'proof_file_sha256': sha256(path),
            'uncompressed_bytes': total_bytes, 'compressed_bytes': path.stat().st_size,
            'gzip_write_seconds': compression,
            'native_diagnostics': {'file': diagnostic_path.name, 'sha256': sha256(diagnostic_path),
                'uncompressed_bytes': diagnostic_bytes, 'compressed_bytes': diagnostic_path.stat().st_size,
                'serialization_seconds': diagnostic_serialization, 'gzip_write_seconds': diagnostic_write,
                'included_in_primary_cost': False},
            'process_peak_rss_kib_not_method_isolated': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        json_write(directory / f'{name}_summary.json', result)
        return result
    except Exception as error:
        with gzip.open(directory / f'{name}_partial.jsonl.gz', 'wt', encoding='utf8') as stream:
            for line in retained:
                stream.write(line + '\n')
        json_write(directory / f'{name}_failure.json', {'completed_responses': len(retained),
                   'error': str(error), 'traceback': traceback.format_exc()})
        raise


def validate_inputs(reg):
    require(reg['status_at_registration'] == 'registered_before_new_timing_outcomes', 'Protocol status')
    require(reg['queries'] == 1000 and reg['repetitions'] == 3 and reg['bank_size'] == 16
            and reg['threshold'] == .001 and reg['query_prefixes'] == [1, 10, 100, 1000], 'Wrong grid')
    require(reg['methods'] == list(METHODS), 'Method roster or order changed')
    require(reg['order_scheme'] == 'one_seeded_permutation_rotated_left_0_2_4', 'Unregistered order scheme')
    require(len(reg['conditions']) == 3, 'Three original repeated streams required')
    old = ROOT / reg['source_run']
    require(old == ROOT / 'runs/icde2027/query_engine_v1', 'Not the original execution run')
    require(sha256(old/'manifest.json') == reg['source_manifest_sha256']
            and read(old/'manifest.json')['status'] == 'complete', 'Original run identity/status mismatch')
    loaded = []
    for rep, condition in enumerate(reg['conditions']):
        require(condition['name'] == f'd512_connected_nearby_nonproportional_r{rep}', 'Wrong primary condition')
        directory = old / condition['name']
        for name in ('view.json', 'queries.npy', 'condition.json'):
            require(sha256(directory/name) == condition['input_sha256'][name], 'Changed original ' + name)
        queries = np.load(directory/'queries.npy', allow_pickle=False)
        require(queries.shape == (1000, 512) and queries.dtype == np.dtype('float64')
                and np.isfinite(queries).all(), 'Wrong original query matrix')
        view = read(directory/'view.json')
        checked = CheckedView(view)
        require(checked.d == 512 and checked.digest == condition['view_digest'], 'Original view mismatch')
        loaded.append((directory, view, queries))
    return loaded


def run(args):
    reg = read(args.protocol); loaded = validate_inputs(reg)
    require(not args.out.exists(), 'Refusing existing output')
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')),
            'Single-thread environment required')
    gate = read(args.gate)
    require(gate.get('passed') is True and gate.get('independent_review') is True
            and gate.get('protocol_sha256') == sha256(args.protocol), 'Missing/current independent gate')
    paths = source_files(); hashes = {str(p.relative_to(ROOT/'icde2027')): sha256(p) for p in paths}
    require(RUNTIME <= set(gate.get('implementation_sha256', {})), 'Gate lacks transitive runtime modules')
    for name, value in gate['implementation_sha256'].items():
        require(hashes.get(name) == value, 'Stale gate source: ' + name)
    inherited = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, {inherited[0]})
    args.out.mkdir(parents=True, exist_ok=False)
    for source in paths:
        relative = source.relative_to(ROOT/'icde2027'); destination = args.out/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
        require(sha256(destination) == hashes[str(relative)] == sha256(source), 'Snapshot changed during copy')
    shutil.copy2(args.protocol, args.out/'protocol.json'); shutil.copy2(args.gate, args.out/'implementation_gate.json')
    manifest = {'status': 'running', 'study': reg['study'], 'started_utc': utc(),
                'protocol_sha256': sha256(args.protocol), 'source_sha256': hashes,
                'implementation_gate_sha256': sha256(args.out/'implementation_gate.json'),
                'source_run': reg['source_run'], 'source_manifest_sha256': reg['source_manifest_sha256'],
                'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                'highspy': highspy.Highs().version(), 'host': platform.node(), 'platform': platform.platform(),
                'scipy_bundled_highs': '.'.join(str(getattr(_core, k)) for k in
                    ('HIGHS_VERSION_MAJOR', 'HIGHS_VERSION_MINOR', 'HIGHS_VERSION_PATCH')),
                'cpu': next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                             if s.startswith('model name')), 'unknown'),
                'cpu_affinity': sorted(os.sched_getaffinity(0)), 'inherited_cpu_affinity': inherited,
                'load_at_start': list(os.getloadavg()), 'native_threads': 1, 'gpu_used': False,
                'thread_environment': {k: os.environ[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                'command': [sys.executable, *sys.argv], 'source_control': 'No git; byte-exact source snapshots.'}
    json_write(args.out/'manifest.json', manifest); completed = []
    try:
        for rep, (old, view, queries) in enumerate(loaded):
            directory = args.out / old.name; directory.mkdir()
            for filename in ('view.json', 'queries.npy'):
                shutil.copy2(old/filename, directory/filename)
                require(sha256(directory/filename) == reg['conditions'][rep]['input_sha256'][filename]
                        == sha256(old/filename), 'Original input changed during copy: ' + filename)
            document, common, shape = fixture_view(512, 'connected')
            require(document['view'] == view and CheckedView(document['view']).digest == CheckedView(view).digest,
                    'Current common construction differs from original trusted view')
            order = method_order(reg, rep)
            condition = {'name': old.name, 'repeat': rep, 'method_order': order, 'shape': shape,
                         'source_input_sha256': reg['conditions'][rep]['input_sha256'],
                         'queries_sha256': sha256(directory/'queries.npy'), 'view_sha256': sha256(directory/'view.json'),
                         'view_digest': CheckedView(view).digest, 'common_view_seconds': common}
            json_write(directory/'condition.json', condition); results = {}
            for name in order:
                results[name] = measure(directory, name, view, queries, reg, common,
                                        view_text=(directory/'view.json').read_text())
                print(f'{utc()} {old.name} {name}: producer={results[name]["producer_seconds"]:.6f}, '
                      f'joint={results[name]["joint_seconds"]:.6f}, decisions={results[name]["decisions"]}', flush=True)
            condition['comparisons'] = {b: compare(results, b) for b in reg['methods']}
            require(all(v['opposite_decisions'] == 0 for row in condition['comparisons'].values() for v in row.values()),
                    'Opposite proved decisions between plans')
            json_write(directory/'condition.json', condition); completed.append(condition)
            json_write(args.out/'completed_conditions.json', completed)
        contrasts = {}
        for baseline in ('threshold_hot', 'threshold_hot_auto'):
            rows = [c['comparisons'][baseline]['original_prepared'] for c in completed]
            contrasts[baseline] = {'joint_ratios': [r['joint_ratio'] for r in rows],
                                   'producer_ratios': [r['producer_ratio'] for r in rows],
                                   'lost_decided_answers': sum(r['lost_decided_answers'] for r in rows),
                                   'all_three_joint_ratios_at_most_0_8': all(r['joint_ratio'] <= .8 for r in rows)}
        support = all(c['all_three_joint_ratios_at_most_0_8'] and c['lost_decided_answers'] == 0 for c in contrasts.values())
        validate_inputs(reg)
        json_write(args.out/'summary.json', {'complete': True, 'conditions': 3, 'methods': list(METHODS),
                   'responses': 15000, 'primary_contrasts': contrasts, 'engineering_target_supported': support,
                   'interpretation': 'Descriptive matched repeat on observed constructed streams; no significance claim.'})
    except Exception as error:
        manifest.update(status='failed', finished_utc=utc(), error=str(error), traceback=traceback.format_exc(),
                        completed_conditions=len(completed))
        json_write(args.out/'manifest.json', manifest); raise
    manifest.update(status='complete', finished_utc=utc(), completed_conditions=3, query_responses=15000,
                    load_at_end=list(os.getloadavg()))
    json_write(args.out/'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
