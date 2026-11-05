"""Registered follow-up for verified threshold queries; keep earlier runs intact.

This module is an execution driver, not the numerical checker.  The independent
gate and the formal protocol must be complete before a registered run starts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
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

from ..experiment import json_write
from ..proof_check import CheckedView
from ..query_workload import fixture_view, query_stream, event_provider
from ..research_io import ROOT, sha256
from .engine import ThresholdEngine
from .proofs import ThresholdChecker


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_files():
    package = ROOT / 'icde2027'
    paths = list(package.glob('*.py'))
    paths.extend((package / 'threshold_extension').glob('*.py'))
    return sorted(paths)


def start_run(out, protocol, gate):
    require(not out.exists(), f'Refusing existing run directory: {out}')
    require(gate.get('passed') is True, 'Independent implementation gate has not passed')
    required = {'__init__.py', 'aggregate.py', 'confidence_view.py', 'experiment.py',
                'highs_reopt.py', 'proof_check.py', 'query_workload.py', 'research_io.py',
                'source_intervals.py', 'threshold_extension/__init__.py',
                'threshold_extension/proofs.py', 'threshold_extension/engine.py',
                'threshold_extension/workload.py'}
    require(required <= set(gate.get('implementation_sha256', {})),
            'Implementation gate must bind every required runtime module')
    current = {str(p.relative_to(ROOT / 'icde2027')): sha256(p) for p in source_files()}
    for name, expected in gate['implementation_sha256'].items():
        require(current.get(name) == expected, f'Gate source no longer matches: {name}')
    out.mkdir(parents=True)
    snapshot = out / 'source_snapshot'
    for source in source_files():
        relative = source.relative_to(ROOT / 'icde2027')
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        require(sha256(destination) == current[str(relative)] == sha256(source),
                f'Source changed during snapshot: {relative}')
    shutil.copy2(protocol, out / 'protocol.json')
    json_write(out / 'implementation_gate.json', gate)
    cpu = next((line.partition(':')[2].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), 'unknown')
    manifest = {
        'status': 'running', 'study': 'verified_threshold_followup_v2', 'started_utc': utc(),
        'protocol_sha256': sha256(protocol), 'source_sha256': current,
        'implementation_gate_sha256': sha256(out / 'implementation_gate.json'),
        'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'highspy': highspy.Highs().version(), 'host': platform.node(), 'platform': platform.platform(),
        'cpu': cpu, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'load_at_start': list(os.getloadavg()), 'native_highs_threads': 1, 'gpu_used': False,
        'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'source_control': 'No git repository; exact source snapshot and hashes retained.',
        'command': [sys.executable, *sys.argv],
    }
    json_write(out / 'manifest.json', manifest)
    return manifest


def replay(path, view, queries, threshold, cache_size, *, stateless, prefixes=(1, 10, 100), refusal_path=None):
    begin = time.perf_counter()
    checker = None if stateless or view is None else ThresholdChecker(view, cache_size=cache_size)
    initialization = time.perf_counter() - begin if checker is not None else 0.
    decode = verification = 0.
    valid = missing = 0
    decisions = Counter()
    rows = []
    refusal = read(refusal_path) if refusal_path is not None else None
    refusal_hash = sha256(refusal_path) if refusal_path is not None else None
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        for i, line in enumerate(stream):
            tick = time.perf_counter()
            item = json.loads(line)
            decode += time.perf_counter() - tick
            require(i < len(queries) and item['query_index'] == i, 'Saved query order differs')
            answer = item['answer']; proof = answer['certificate']
            value = answer['robust_threshold_decision']
            require(value is None or type(value) is bool, 'Threshold decision must be three-valued')
            decisions['true' if value is True else 'false' if value is False else 'unknown'] += 1
            if proof is None:
                require(view is None and refusal is not None and refusal.get('status') == 'uncertain',
                        'A ready view or missing refusal record cannot authorize a proof-free response')
                require(answer['status'] == 'uncertain' and answer['lower_bound'] is None
                        and answer['upper_bound'] is None and value is None,
                        'Only an explicit no-bound refusal may omit its proof')
                require(answer.get('route') == 'materialization_refusal'
                        and answer.get('reason') == refusal.get('reason')
                        and answer.get('refusal_sha256') == refusal_hash,
                        'No-bound response is not bound to its recorded materialization refusal')
                missing += 1
                cumulative = initialization + decode + verification
                rows.append({'query': i + 1, 'cumulative_seconds': cumulative})
                continue
            if stateless:
                require(view is not None, 'A proof requires a trusted ready view')
                tick = time.perf_counter()
                checker = ThresholdChecker(view, cache_size=0)
                initialization += time.perf_counter() - tick
            require(checker is not None, 'A proof requires a trusted ready view')
            tick = time.perf_counter()
            checked = checker.verify(proof, expected_objective=queries[i].tolist(), expected_threshold=threshold)
            verification += time.perf_counter() - tick
            require(checked.get('valid') is True, 'Checker did not validate the proof')
            require(answer['lower_bound'] == checked['lower_bound']
                    and answer['upper_bound'] == checked['upper_bound'],
                    'Saved answer bounds differ from the checked proof values')
            require(checked['robust_threshold_decision'] is value, 'Saved decision differs from its checked evidence')
            require((answer['status'] == 'certified') == (value is True), 'Status and positive decision disagree')
            # The independent implementation gate checks all evidence fields;
            # this replay additionally binds every response to its input stream.
            valid += 1
            rows.append({'query': i + 1, 'cumulative_seconds': initialization + decode + verification})
    require(valid + missing == len(queries), 'Incomplete saved proof stream')
    return {'checker_mode': 'stateless' if stateless else 'persistent_fifo',
            'cache_size': 0 if stateless else cache_size,
            'checker_initialization_seconds': initialization, 'decode_seconds': decode,
            'verify_seconds': verification, 'total_consumer_seconds': initialization + decode + verification,
            'valid_proofs': valid, 'no_bound_responses': missing, 'decisions': dict(decisions),
            'per_query': rows,
            'prefixes': {str(q): rows[q - 1]['cumulative_seconds'] for q in prefixes if q <= len(rows)},
            'scope': 'Excludes compressed-file I/O/decompression and unmeasured network transfer.'}


def measure(directory, method, queries, reg, *, view=None, common_seconds=0., provider=None):
    name = method['name']; retained = []
    try:
        stages = {}
        refusal = None; refusal_path = None; refusal_hash = None
        if provider is not None:
            document, stages = provider()
            if document['status'] == 'ready':
                view = document['view']
            else:
                refusal = document
                view = None
                refusal_path = directory / f'{name}_refused_document.json'
                json_write(refusal_path, document)
                refusal_hash = sha256(refusal_path)
        else:
            stages['common_view'] = common_seconds
        if view is not None and (directory / 'view.json').is_file():
            require(CheckedView(view).digest == CheckedView(read(directory / 'view.json')).digest,
                    'Plans received different views')
        elif view is not None:
            json_write(directory / 'view.json', view)
        tick = time.perf_counter()
        engine = (ThresholdEngine(view, plan=method['plan'], bank_size=reg['bank_size'],
                                  simplex_strategy=method['simplex_strategy']) if view is not None else None)
        stages['plan_checker_initialization'] = time.perf_counter() - tick if engine is not None else 0.
        cumulative = sum(stages.values())
        rows = []; routes = Counter(); decisions = Counter(); encoded_bytes = 0
        for i, objective in enumerate(queries):
            tick = time.perf_counter()
            answer = (engine.query(objective, reg['threshold']) if engine is not None else {
                'certificate': None, 'status': 'uncertain', 'lower_bound': None, 'upper_bound': None,
                'robust_threshold_decision': None, 'route': 'materialization_refusal',
                'reason': refusal['reason'], 'refusal_sha256': refusal_hash,
            })
            query_seconds = time.perf_counter() - tick
            value = answer['robust_threshold_decision']
            require(value is None or type(value) is bool, 'Threshold decision must be three-valued')
            key = 'true' if value is True else 'false' if value is False else 'unknown'
            tick = time.perf_counter()
            encoded = json.dumps({'query_index': i, 'answer': answer}, separators=(',', ':'), allow_nan=False)
            serialization_seconds = time.perf_counter() - tick
            cumulative += query_seconds + serialization_seconds
            encoded_bytes += len(encoded.encode('utf8')) + 1
            retained.append(encoded)
            routes[answer['route']] += 1; decisions[key] += 1
            rows.append({'query': i + 1, 'query_seconds': query_seconds,
                         'serialization_seconds': serialization_seconds, 'cumulative_seconds': cumulative,
                         'decision': value, 'lower_bound': answer['lower_bound'], 'upper_bound': answer['upper_bound'],
                         'route': answer['route']})
        proof_path = directory / f'{name}.jsonl.gz'
        tick = time.perf_counter()
        with gzip.open(proof_path, 'wt', encoding='utf8', compresslevel=3) as stream:
            for item in retained:
                stream.write(item + '\n')
        gzip_seconds = time.perf_counter() - tick
        persistent = replay(proof_path, view, queries, reg['threshold'], reg['bank_size'], stateless=False,
                            prefixes=reg['query_prefixes'], refusal_path=refusal_path)
        stateless = replay(proof_path, view, queries, reg['threshold'], reg['bank_size'], stateless=True,
                           prefixes=reg['query_prefixes'], refusal_path=refusal_path)
        require(persistent['decisions'] == stateless['decisions'] == dict(decisions), 'Receiver-mode decisions differ')
        summary = {
            'method': name, 'plan': method['plan'], 'simplex_strategy': method['simplex_strategy'],
            'queries': len(queries), 'producer_seconds': cumulative,
            'joint_seconds': cumulative + persistent['total_consumer_seconds'],
            'input_stage_seconds': stages, 'prefixes': {str(q): rows[q - 1]['cumulative_seconds'] for q in reg['query_prefixes']},
            'joint_prefixes': {str(q): rows[q - 1]['cumulative_seconds'] + persistent['prefixes'][str(q)]
                               for q in reg['query_prefixes']},
            'per_query': rows, 'routes': dict(routes), 'decisions': dict(decisions),
            'persistent_replay': persistent, 'stateless_replay': stateless,
            'proof_file': proof_path.name, 'proof_file_sha256': sha256(proof_path),
            'compressed_bytes': proof_path.stat().st_size, 'uncompressed_bytes': encoded_bytes,
            'gzip_write_seconds': gzip_seconds,
            'process_peak_rss_kib_not_method_isolated': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'final_memory': engine.memory_info() if engine is not None else None,
        }
        json_write(directory / f'{name}_summary.json', summary)
        return summary
    except Exception as error:
        with gzip.open(directory / f'{name}_partial.jsonl.gz', 'wt', encoding='utf8') as stream:
            for item in retained:
                stream.write(item + '\n')
        json_write(directory / f'{name}_failure.json', {'completed_responses': len(retained),
                   'error': str(error), 'traceback': traceback.format_exc()})
        raise


def compare(methods, baseline):
    reference = methods[baseline]
    output = {}
    for name, result in methods.items():
        paired = list(zip(result['per_query'], reference['per_query']))
        loss = sum(b['decision'] is not None and a['decision'] is not b['decision'] for a, b in paired)
        opposition = sum(a['decision'] is not None and b['decision'] is not None and
                         a['decision'] is not b['decision'] for a, b in paired)
        differences = [a['lower_bound'] - b['lower_bound'] for a, b in paired
                       if a['lower_bound'] is not None and b['lower_bound'] is not None]
        output[name] = {
            'producer_ratio': result['producer_seconds'] / reference['producer_seconds'],
            'joint_ratio': result['joint_seconds'] / reference['joint_seconds'],
            'lost_decided_answers': loss, 'opposite_decisions': opposition,
            'lower_minus_baseline_min': min(differences, default=None),
            'lower_minus_baseline_max': max(differences, default=None),
        }
    return output


def run(args):
    protocol = args.protocol.resolve(); reg = read(protocol)
    require(reg['status_at_registration'] == 'registered_before_new_outcomes', 'Protocol is not registered')
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')),
            'Registered timing requires OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1')
    require(reg['queries'] == 100 and reg['query_prefixes'] == [1, 10, 100], 'Wrong registered query grid')
    old = ROOT / reg['source_input_run']
    old_manifest = read(old / 'manifest.json')
    require(old_manifest['status'] == 'complete', 'Original source-to-answer chain must complete first')
    gate = read(args.gate)
    require(gate['protocol_sha256'] == sha256(protocol), 'Implementation gate refers to a different protocol')
    inherited = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, {inherited[0]})
    manifest = start_run(args.out, protocol, gate)
    manifest.update(inherited_affinity_before_pin=inherited, source_input_run=reg['source_input_run'],
                    source_input_manifest_sha256=sha256(old / 'manifest.json'))
    json_write(args.out / 'manifest.json', manifest)
    completed = []
    try:
        cfg = reg['query_engine']
        specifications = []
        for d in cfg['dimensions']:
            for si, structure in enumerate(cfg['structures']):
                for qi, stream in enumerate(cfg['streams']):
                    for rep in range(reg['repetitions']):
                        specifications.append(('engine', f'd{d}_{structure}_{stream}_r{rep}', d, structure, stream, si, qi, rep))
        for di, design in enumerate(reg['end_to_end']['randomized_designs']):
            for rep in range(reg['repetitions']):
                specifications.append(('end_to_end', f'{design}_r{rep}', 64, 'source_to_answer', design, di, 0, rep))
        for stage, name, d, structure, stream, si, qi, rep in specifications:
            directory = args.out / stage / name; directory.mkdir(parents=True)
            condition = {'stage': stage, 'name': name, 'dimension': d, 'structure': structure, 'stream': stream, 'repeat': rep}
            provider = None; view = None; common = 0.
            if stage == 'engine':
                document, common, shape = fixture_view(d, structure)
                view = document['view']; condition['shape'] = shape
                queries = query_stream(d, reg['queries'], stream, [reg['master_seed'], 1, d, si, qi, rep])
            else:
                source = old / name
                generation = read(source / 'event_generation.json')
                for filename, record in generation['files'].items():
                    require(sha256(source / filename) == record['sha256'], f'Input event columns changed: {name}/{filename}')
                condition['input_files'] = generation['files']; condition['source_directory'] = str(source.relative_to(ROOT))
                q = np.array(generation['randomized_q'])
                source_cfg = read(old / 'protocol.json')['end_to_end']
                provider = event_provider(source, source_cfg, q, {'fixture': 'registered_end_to_end', 'design': stream, 'rep': rep})
                queries = query_stream(d, reg['queries'], 'nearby_nonproportional', [reg['master_seed'], 4, si, rep])
            np.save(directory / 'queries.npy', queries)
            condition['queries_sha256'] = sha256(directory / 'queries.npy')
            order = np.random.default_rng(np.random.SeedSequence([reg['master_seed'], 9, 0 if stage == 'engine' else 1, d, si, qi, rep])).permutation(len(reg['methods'])).tolist()
            condition['method_order'] = [reg['methods'][i]['name'] for i in order]
            condition['common_view_seconds'] = common
            json_write(directory / 'condition.json', condition)
            methods = {}
            for index in order:
                method = reg['methods'][index]
                methods[method['name']] = measure(directory, method, queries, reg, view=view, common_seconds=common, provider=provider)
            for i in range(len(queries)):
                decided = {m['per_query'][i]['decision'] for m in methods.values()
                           if m['per_query'][i]['decision'] is not None}
                require(len(decided) <= 1, 'Opposite proved decisions among the six plans')
            condition['comparisons'] = {name: compare(methods, name) for name in reg['primary']['baselines']}
            require(all(v['opposite_decisions'] == 0 for group in condition['comparisons'].values() for v in group.values()),
                    'Opposite proved decisions on an identical view and query')
            completed.append(condition)
            json_write(directory / 'condition.json', condition)
            json_write(args.out / 'completed_conditions.json', completed)
            print('completed ' + stage + '/' + name + ': ' + ', '.join(f'{m}={v["joint_seconds"]:.4f}s joint' for m, v in methods.items()), flush=True)
        for filename, expected in manifest['source_sha256'].items():
            require(sha256(args.out / 'source_snapshot' / filename) == expected, 'Source snapshot changed')
        require(sha256(old / 'manifest.json') == manifest['source_input_manifest_sha256'], 'Completed input manifest changed')
        manifest.update(status='complete', finished_utc=utc(), completed_conditions=len(completed))
        json_write(args.out / 'manifest.json', manifest)
    except Exception as error:
        manifest.update(status='failed', finished_utc=utc(), completed_conditions=len(completed),
                        error=str(error), traceback=traceback.format_exc())
        json_write(args.out / 'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'icde2027/configs/threshold_workload_v2.json')
    parser.add_argument('--gate', type=Path, default=ROOT / 'runs/icde2027/reviews/threshold_implementation_gate.json')
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
