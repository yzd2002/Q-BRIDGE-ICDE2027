"""Fixed-input V6 exact-arithmetic factorial and last-first reference.

Old V4/V5 implementations are unchanged. No formal execution is available
without a new, source-bound independent review gate and the full quality pass.
The last-first query method is a source copy of CheckingScheduleEngine.query,
with only candidate order and equal-lower tie-breaking changed. Its full query
context and optimized exact checker are inherited unchanged from V5.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

import highspy
import numpy as np
import scipy
from scipy.optimize._highspy import _core as scipy_highs

from .checking_schedule_v4 import (
    CheckingScheduleEngine, FractionReplay, NUMERICAL_ERRORS,
    archive_snapshot, validate_snapshot, load_input, sha256, write_json, require,
)
from .optimized_exact_v5 import (
    OptimizedCheckingScheduleEngine, OptimizedThresholdChecker,
    _QueryContextEngine, _initialize,
)
from .primary_aligned import RUNTIME, checked_response, service_response
from .threshold_extension.proofs import ThresholdChecker, _floats

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'icde2027/configs/optimized_schedule_v6.json'
ORIGINAL_CONFIG = ROOT / 'icde2027/configs/checking_schedule_v4.json'
CORE = ('s0_p0', 's0_p1', 's1_p0', 's1_p1')
METHODS = (*CORE, 'last_first')
FACTOR_CONTRASTS = (('s1_p0', 's0_p0'), ('s1_p1', 's0_p1'),
                    ('s0_p1', 's0_p0'), ('s1_p1', 's1_p0'))
DIRECTIONAL_CHECKS = {
    'retention_saves_without_screening': {'ratio': 's0_p1/s0_p0', 'direction': 'less_than_one'},
    'screening_saves_without_retention': {'ratio': 's1_p0/s0_p0', 'direction': 'less_than_one'},
    'screening_costs_with_retention': {'ratio': 's1_p1/s0_p1', 'direction': 'greater_than_one'},
}
STATE_FIELDS = ('last_primal_generation', 'last_dual_generation',
                'primal_total_retained_entries', 'dual_total_retained_entries',
                'cutoff_retries', 'rejected_dual_proposals', 'rejected_primal_proposals')


class OptimizedFactorialEngine(OptimizedCheckingScheduleEngine):
    """Only S/P differ: query contexts and decoded exact views remain shared."""
    def __init__(self, view, *, screening, prepared):
        require(type(screening) is bool and type(prepared) is bool,
                'Boolean screening and prepared factors required')
        super().__init__(view, simplex_strategy=1)
        self.screening, self.prepared_reuse = screening, prepared


class _LastFirstSchedule(CheckingScheduleEngine):
    """Different-order bridge; no screening, same native state transitions."""
    def query(self, objective, threshold=.001):
        begin = time.perf_counter()
        c, delta = _floats(objective), _floats([threshold])[0]
        require(len(c) == self.d, 'Objective dimension differs from view')
        with np.errstate(over='ignore', invalid='ignore'):
            norm = float(np.sum(np.abs(c)))
        require(math.isfinite(norm), 'Objective norm exceeds finite implementation range')
        metrics = self._new_metrics()
        initial = self._ensure_initial(metrics)
        lower = upper = None
        deferred_duals, deferred_primals = [], []

        def check_dual(candidate, scale, route):
            nonlocal lower
            try:
                checked = self._exact_candidate(candidate, c, delta, scale, metrics)
            except NUMERICAL_ERRORS:
                metrics['rejected_cached_duals'] += 1
                return None
            if (lower is None or checked['claimed_lower'] > lower['claimed_lower']
                    or (route == 'box_shortcut' and checked['claimed_lower'] == lower['claimed_lower'])):
                lower = checked
            if lower['claimed_lower'] > delta:
                return self._finish(c, delta, lower, None, route, metrics, begin)
            return None

        def replay_duals():
            while deferred_duals:
                candidate, scale, route = deferred_duals.pop(0)
                metrics['replayed_duals'] += 1
                answer = check_dual(candidate, scale, route)
                if answer is not None:
                    return answer
            return None

        def check_primal(state, route):
            nonlocal upper
            try:
                checked = self._upper_proof(c, state, metrics)
            except NUMERICAL_ERRORS:
                metrics['rejected_cached_primals'] += 1
                return None
            if upper is None or checked['claimed_upper'] < upper['claimed_upper']:
                upper = checked
            if checked['claimed_upper'] <= delta:
                positive = replay_duals()
                require(positive is None, 'Contradictory exact lower/upper evidence')
                if lower is None:
                    # Preserve the upper; native work may still recover the
                    # lower required by this existing service contract.
                    return None
                return self._finish(c, delta, lower, checked, route, metrics, begin)
            return None

        duals = []
        if self.last_dual is not None:
            duals.append((self.last_dual, norm, 'last_dual'))
        duals.append((self.zero, None, 'box_shortcut'))
        for candidate, scale, route in duals:
            if self._defer_dual(candidate, c, delta, scale, metrics):
                deferred_duals.append((candidate, scale, route)); metrics['deferred_duals'] += 1
            else:
                answer = check_dual(candidate, scale, route)
                if answer is not None:
                    return answer
        points = [('initial_primal', initial)]
        if self.last_primal is not None:
            points.append(('last_primal', self.last_primal))
        seen = set()
        for route, state in points:
            if not self.checker.owns_primal(state):
                metrics['invalid_bank_entries'] += 1; metrics['rejected_cached_primals'] += 1
                continue
            if state.raw in seen:
                continue
            seen.add(state.raw)
            if self._defer_primal(state, c, delta, metrics):
                deferred_primals.append((state, route)); metrics['deferred_primals'] += 1
            else:
                answer = check_primal(state, route)
                if answer is not None:
                    return answer
        # No proposal reaches the solver until every deferred candidate is checked.
        answer = replay_duals()
        if answer is not None:
            return answer
        for state, route in deferred_primals:
            metrics['replayed_primals'] += 1
            answer = check_primal(state, route)
            if answer is not None:
                return answer

        cutoff = delta
        for _ in range(2):
            proposal = self._native_run(c, cutoff, metrics)
            if proposal['lower'] is not None:
                try:
                    checked = self._lower_proof(c, delta, proposal['lower'], proposal['upper'], metrics)
                    if lower is None or checked['claimed_lower'] > lower['claimed_lower']:
                        lower = checked
                    self._save_dual(proposal['lower'], proposal['upper'], norm, metrics)
                except NUMERICAL_ERRORS:
                    metrics['rejected_dual_proposals'] += 1
            positive = lower is not None and lower['claimed_lower'] > delta
            if proposal['primal'] is not None:
                state = self._checked_point(proposal['primal'], metrics)
                if state is not None:
                    self._save_point(state, metrics)
                    if not positive:
                        checked = self._upper_proof(c, state, metrics)
                        if upper is None or checked['claimed_upper'] < upper['claimed_upper']:
                            upper = checked
            if positive:
                route = ('solver_cutoff' if proposal['status'] == highspy.HighsModelStatus.kObjectiveBound
                         else 'solver')
                return self._finish(c, delta, lower, None, route, metrics, begin)
            if lower is not None and upper is not None and upper['claimed_upper'] <= delta:
                return self._finish(c, delta, lower, upper, 'solver', metrics, begin)
            if cutoff is not None and proposal['status'] != highspy.HighsModelStatus.kOptimal:
                metrics['cutoff_retries'] += 1; cutoff = None
                continue
            break
        if lower is None:
            return {**metrics, **self._retention_metrics(), 'status': 'uncertain',
                    'route': 'numerical_refusal', 'lower_bound': None, 'upper_bound': None,
                    'certificate': None, 'robust_threshold_decision': None,
                    'reason': 'No finite exact lower certificate could be constructed',
                    'seconds': time.perf_counter() - begin}
        return self._finish(c, delta, lower, upper, 'solver_unknown', metrics, begin)

class LastFirstEngine(_QueryContextEngine, _LastFirstSchedule):
    def __init__(self, view):
        _initialize(self, view, last_only=True, simplex_strategy=1)


def make_engine(name, view):
    require(name in METHODS, 'Unknown registered method')
    if name == 'last_first':
        return LastFirstEngine(view)
    return OptimizedFactorialEngine(view, screening=name.startswith('s1'),
                                    prepared=name.endswith('p1'))


class ResourceLimit(RuntimeError):
    """The registered wall budget expired; no partial result is promoted."""


class Budget:
    def __init__(self, seconds):
        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
                'A positive finite wall-clock limit is required')
        self.started = time.monotonic()
        self.limit = float(seconds)

    def check(self):
        if time.monotonic() - self.started >= self.limit:
            raise ResourceLimit('Registered quality-plus-timing wall budget exhausted')


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def native_state(engine):
    """Audit snapshot outside service timing; excludes prepared cache internals."""
    last = engine.last_dual
    return {
        'last_dual': None if last is None else [list(last.lower), list(last.upper)],
        'last_primal': None if engine.last_primal is None else list(engine.last_primal.raw),
        'last_dual_generation': engine.last_dual_generation,
        'last_primal_generation': engine.last_primal_generation,
    }


def state_payload(engine):
    retained = [engine.initial_primal, engine.last_primal]
    retained.extend(x.prepared for x in [engine.zero, *engine.dual_bank]
                    if x.prepared is not None)
    return {'retained_exact_python_payload_bytes': engine.checker.cache_payload_bytes(retained),
            'payload_scope': 'Approximate Python payload; excludes native model, view, registry overhead and RSS.',
            'retention': engine.memory_info(), 'checker_stats': dict(engine.checker.stats)}


def replay_stream(path, view_text, queries, reg, budget, *, kind='optimized', cold=False):
    """A fixed consumer, with all view/response decoding and checks charged.

    Compressed reads/decompression are outside the declared CPU service cost.
    Legacy and dense Fraction checks are independent audits, not main service.
    """
    require(kind in ('optimized', 'legacy'), 'Known receiver required')
    constructor = OptimizedThresholdChecker if kind == 'optimized' else ThresholdChecker
    tick = time.perf_counter()
    checker = None if cold else constructor(json.loads(view_text), cache_size=reg['bank_size'])
    initialization = 0. if cold else time.perf_counter() - tick
    decode = verification = 0.
    valid = missing = 0
    decisions, rows = Counter(), []
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        for i, line in enumerate(stream):
            budget.check()
            require(i < len(queries), 'Extra receiver response')
            tick = time.perf_counter()
            item = json.loads(line)
            decode += time.perf_counter() - tick
            if cold:
                tick = time.perf_counter()
                checker = constructor(json.loads(view_text), cache_size=0)
                initialization += time.perf_counter() - tick
            tick = time.perf_counter()
            has_proof = checked_response(checker, item, i, queries[i].tolist(), reg['threshold'])
            verification += time.perf_counter() - tick
            valid += int(has_proof)
            missing += int(not has_proof)
            value = item['answer']['robust_threshold_decision']
            decisions['true' if value is True else 'false' if value is False else 'unknown'] += 1
            rows.append({'query': i+1, 'cumulative_seconds': initialization + decode + verification})
    require(len(rows) == len(queries), 'Incomplete receiver response stream')
    payload = (checker.cache_payload_bytes() if kind == 'optimized' and checker is not None else None)
    return {'checker': kind, 'mode': 'cold' if cold else 'persistent_fifo',
            'cache_size': 0 if cold else reg['bank_size'],
            'checker_initialization_seconds': initialization, 'decode_seconds': decode,
            'verify_seconds': verification,
            'total_consumer_seconds': initialization + decode + verification,
            'valid_proofs': valid, 'no_bound_responses': missing, 'decisions': dict(decisions),
            'prefixes': {str(q): rows[q-1]['cumulative_seconds'] for q in reg['query_prefixes']},
            'per_query': rows, 'retained_exact_python_payload_bytes': payload,
            'scope': 'Excludes compressed-file reads, decompression and network; payload is not RSS.'}


def write_gzip(path, records):
    with gzip.open(path, 'wt', encoding='utf8', compresslevel=3) as stream:
        for record in records:
            stream.write(record + '\n')


def measure(directory, name, view_text, queries, reg, budget):
    """One fresh producer/receiver stream; used by quality and timing stages."""
    directory = Path(directory)
    retained, diagnostics, rows = [], [], []
    wall_begin = time.perf_counter()
    try:
        budget.check()
        tick = time.perf_counter()
        view = json.loads(view_text)
        view_decode = time.perf_counter() - tick
        tick = time.perf_counter()
        engine = make_engine(name, view)
        initialization = time.perf_counter() - tick
        cumulative = view_decode + initialization
        decisions, routes = Counter(), Counter()
        for i, objective in enumerate(queries):
            budget.check()
            tick = time.perf_counter()
            native = engine.query(objective, reg['threshold'])
            answer = service_response(native)
            query_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            line = json.dumps({'query_index': i, 'answer': answer}, separators=(',', ':'), allow_nan=False)
            serialization = time.perf_counter() - tick
            retained.append(line)
            # Native-state observation is diagnostic, not hidden service work.
            diagnostics.append({'query_index': i, 'native_answer': native,
                                'last_native_state': native_state(engine)})
            cumulative += query_seconds + serialization
            decision = answer['robust_threshold_decision']
            decisions['true' if decision is True else 'false' if decision is False else 'unknown'] += 1
            routes[answer['route']] += 1
            rows.append({'query': i+1, 'query_seconds': query_seconds,
                         'serialization_seconds': serialization, 'cumulative_seconds': cumulative,
                         'decision': decision, 'route': answer['route']})
        path = directory / f'{name}.jsonl.gz'
        tick = time.perf_counter()
        write_gzip(path, retained)
        gzip_seconds = time.perf_counter() - tick
        rr = {**reg, 'query_prefixes': [q for q in reg['query_prefixes'] if q <= len(queries)]}
        persistent = replay_stream(path, view_text, queries, rr, budget)
        cold = replay_stream(path, view_text, queries, rr, budget, cold=True)
        legacy = replay_stream(path, view_text, queries, rr, budget, kind='legacy')
        require(persistent['decisions'] == cold['decisions'] == legacy['decisions'] == dict(decisions),
                'Receiver decision mismatch')
        tick = time.perf_counter()
        oracle = FractionReplay(view)
        valid = 0
        for i, line in enumerate(retained):
            budget.check()
            valid += oracle.verify(json.loads(line), i, queries[i].tolist(), reg['threshold'])
        independent_seconds = time.perf_counter() - tick
        require(valid == persistent['valid_proofs'] == legacy['valid_proofs'], 'Independent replay differs')
        native_path = directory / f'{name}_native.jsonl.gz'
        tick = time.perf_counter()
        native_lines = [json.dumps(row, separators=(',', ':'), allow_nan=False) for row in diagnostics]
        diagnostic_serialization = time.perf_counter() - tick
        tick = time.perf_counter()
        write_gzip(native_path, native_lines)
        diagnostic_write = time.perf_counter() - tick
        metric_keys = ('screen_seconds', 'verify_seconds', 'solve_seconds', 'preparation_seconds',
                       'query_context_seconds', 'exact_dual_preparation_seconds', 'primal_preparation_seconds',
                       'dual_candidates_screened', 'primal_candidates_screened',
                       'dual_candidates_exact_checked', 'primal_candidates_exact_checked',
                       'exact_dual_preparations', 'exact_dual_reuses', 'screen_preparations',
                       'deferred_duals', 'deferred_primals', 'replayed_duals', 'replayed_primals',
                       'rejected_cached_duals', 'rejected_cached_primals', 'rejected_dual_proposals',
                       'rejected_primal_proposals', 'invalid_bank_entries', 'cutoff_retries')
        totals = {k: sum(row['native_answer'].get(k, 0) for row in diagnostics) for k in metric_keys}
        totals['native_calls'] = sum(len(row['native_answer']['native_calls']) for row in diagnostics)
        memory = state_payload(engine)
        if name.endswith('p0'):
            require(memory['retention']['retained_exact_duals'] == 0
                    and engine.checker.cache_size == 0 and not engine.checker._dual_cache,
                    'P0 retained a hidden exact-dual state')
        result = {'method': name, 'queries': len(queries),
                  'view_decode_seconds': view_decode, 'plan_initialization_seconds': initialization,
                  'producer_seconds': cumulative,
                  'joint_seconds': cumulative + persistent['total_consumer_seconds'],
                  'prefixes': {str(q): rows[q-1]['cumulative_seconds'] for q in rr['query_prefixes']},
                  'joint_prefixes': {str(q): rows[q-1]['cumulative_seconds'] + persistent['prefixes'][str(q)]
                                     for q in rr['query_prefixes']},
                  'per_query': rows, 'decisions': dict(decisions), 'routes': dict(routes),
                  'persistent_replay': persistent, 'cold_replay': cold,
                  'legacy_replay_audit': {**legacy, 'included_in_primary_cost': False},
                  'independent_fraction_replay': {'valid': valid, 'refusals': len(queries)-valid,
                      'seconds': independent_seconds, 'included_in_primary_cost': False},
                  'proof_file': path.name, 'proof_file_sha256': sha256(path),
                  'uncompressed_bytes': sum(len(x.encode())+1 for x in retained),
                  'compressed_bytes': path.stat().st_size, 'gzip_write_seconds': gzip_seconds,
                  'native_diagnostics': {'file': native_path.name, 'sha256': sha256(native_path),
                      'serialization_seconds': diagnostic_serialization,
                      'gzip_write_seconds': diagnostic_write, 'included_in_primary_cost': False},
                  'diagnostic_totals': totals, 'state_payload': memory,
                  'diagnostic_totals_scope': 'Nested timers overlap and are not added into joint cost. '
                       'No individual skipped-candidate counterfactual times are measured.',
                  'process_peak_rss_kib_not_method_isolated': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        write_json(directory / f'{name}_summary.json', result)
        budget.check()
        return result
    except Exception as error:
        write_gzip(directory / f'{name}_partial.jsonl.gz', retained)
        write_gzip(directory / f'{name}_partial_native.jsonl.gz',
                   [json.dumps(row, allow_nan=False) for row in diagnostics])
        write_json(directory / f'{name}_failure.json', {
            'completed_responses': len(retained), 'failed_or_verifying_query_index': locals().get('i'),
            'elapsed_seconds_including_failed_work': time.perf_counter()-wall_begin,
            'error': repr(error), 'traceback': traceback.format_exc()})
        raise


def read_records(path, expected_queries):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        records = [json.loads(line) for line in stream]
    require(len(records) == expected_queries, 'Incomplete or extended registered response stream')
    require(all(type(r.get('query_index')) is int and r['query_index'] == i
                for i, r in enumerate(records)), 'Response position mismatch')
    return records


def call_signature(native):
    return [{k: v for k, v in call.items() if k != 'seconds'} for call in native['native_calls']]


def compare_streams(directory, expected_queries):
    """Full core equivalence and declared bridge equivalence; no sample audit."""
    directory = Path(directory)
    wire = {m: read_records(directory/f'{m}.jsonl.gz', expected_queries) for m in METHODS}
    native = {m: read_records(directory/f'{m}_native.jsonl.gz', expected_queries) for m in METHODS}
    for m in CORE[1:]:
        require(wire[m] == wire[CORE[0]], f'Core service evidence mismatch: {m}')
    differences = []
    for i in range(expected_queries):
        base = native['s0_p1'][i]
        base_answer = base['native_answer']
        decision = wire['s0_p1'][i]['answer']['robust_threshold_decision']
        for m in METHODS:
            other = native[m][i]
            require(other['last_native_state'] == base['last_native_state'],
                    f'Last native state mismatch: {m}, query {i}')
            require(call_signature(other['native_answer']) == call_signature(base_answer),
                    f'Native path mismatch: {m}, query {i}')
            for key in STATE_FIELDS:
                require(other['native_answer'][key] == base_answer[key],
                        f'Native/state counter mismatch: {m}, query {i}, {key}')
            require(wire[m][i]['answer']['robust_threshold_decision'] is decision,
                    f'Bridge/core decision mismatch: {m}, query {i}')
        bridge = wire['last_first'][i]['answer']
        original = wire['s0_p1'][i]['answer']
        if decision is not True:
            require(bridge == original, f'Non-TRUE bridge service evidence differs: query {i}')
        if bridge != original:
            require(decision is True and original['route'] == 'box_shortcut'
                    and bridge['route'] == 'last_dual',
                    f'Undeclared TRUE bridge difference: query {i}')
            require({k: v for k, v in bridge.items() if k not in ('route', 'lower_bound', 'certificate')}
                    == {k: v for k, v in original.items() if k not in ('route', 'lower_bound', 'certificate')},
                    f'Unrelated TRUE service fields differ: query {i}')
            original_proof, bridge_proof = original['certificate'], bridge['certificate']
            require({k: v for k, v in original_proof.items() if k != 'lower_proof'}
                    == {k: v for k, v in bridge_proof.items() if k != 'lower_proof'},
                    f'Unrelated TRUE proof fields differ: query {i}')
            zero, last = original_proof['lower_proof'], bridge_proof['lower_proof']
            state = native['last_first'][i]['last_native_state']['last_dual']
            require(zero['format'] == 'qbridge-dual-proof-v1'
                    and all(x == 0 for x in zero['lower_multipliers'] + zero['upper_multipliers'])
                    and last['format'] == 'qbridge-scaled-dual-proof-v1'
                    and state is not None
                    and last['lower_multipliers'] == state[0] and last['upper_multipliers'] == state[1],
                    f'TRUE bridge difference is not zero-versus-last evidence: query {i}')
            differences.append({'query_index': i, 'decision': decision,
                'route_original': original['route'], 'route_last_first': bridge['route'],
                'lower_original': original['lower_bound'], 'lower_last_first': bridge['lower_bound'],
                'lower_direction': ('greater' if bridge['lower_bound'] > original['lower_bound']
                                    else 'less' if bridge['lower_bound'] < original['lower_bound'] else 'equal'),
                'original_certificate_json_bytes': len(canonical(original_proof).encode()),
                'bridge_certificate_json_bytes': len(canonical(bridge_proof).encode()),
                'original_service_sha256': hashlib.sha256(canonical(original).encode()).hexdigest(),
                'bridge_service_sha256': hashlib.sha256(canonical(bridge).encode()).hexdigest()})
    return {'core_same_evidence_and_paths': True, 'bridge_same_decisions_and_native_state': True,
            'bridge_nontrue_same_evidence': True, 'bridge_true_differences': differences,
            'queries_checked': expected_queries}


def method_order(reg, input_id, timing_round):
    require(type(timing_round) is int and 0 <= timing_round < 5, 'Five fixed timing positions required')
    seed = int(hashlib.sha256(f"{reg['order_seed']}:{input_id}".encode()).hexdigest()[:16], 16)
    base = np.random.default_rng(seed).permutation(reg['methods']).tolist()
    shift = timing_round % len(base)
    return base[shift:] + base[:shift]


def validate_inputs(reg):
    require(tuple(reg['methods']) == METHODS and tuple(reg['core_methods']) == CORE,
            'Five registered methods and four fixed core arms required')
    require(type(reg['timing_rounds']) is int and reg['timing_rounds'] == 5
            and reg['threshold'] == .001 and reg['bank_size'] == 16
            and reg['receiver'] == 'optimized_fifo16', 'Registered execution settings changed')
    require(reg['wall_clock_limit_seconds'] == 36000 and reg['threads'] == 1
            and type(reg['cpu']) is int and reg['cpu'] >= 0, 'Registered resource settings changed')
    require(reg['query_prefixes'] == [1, 10, 100, 1000], 'Registered prefixes changed')
    require(reg['factor_contrasts'] == [list(x) for x in FACTOR_CONTRASTS]
            and reg['bridge_contrasts'] == [['last_first', m] for m in CORE]
            and reg['directional_checks'] == DIRECTIONAL_CHECKS,
            'Predeclared contrasts or directional checks changed')
    require(reg['source_protocol'] == str(ORIGINAL_CONFIG.relative_to(ROOT))
            and sha256(ORIGINAL_CONFIG) == reg['source_protocol_sha256'], 'Original protocol identity changed')
    original = json.loads(ORIGINAL_CONFIG.read_text())
    require(reg['inputs'] == original['inputs'], 'Selection/reordering/change of original inputs is forbidden')
    expected_groups = {'original_main': 3, 'threshold_engine': 54, 'source_derived': 6}
    require(reg['input_counts'] == expected_groups
            and Counter(x['group'] for x in reg['inputs']) == expected_groups,
            'All original input groups are required')
    require(len(reg['inputs']) == len({x['id'] for x in reg['inputs']}) == 63,
            'Duplicate or missing registered inputs')
    total = 0
    for entry in reg['inputs']:
        view_text, queries = load_input(entry)
        require(entry['threshold'] == reg['threshold'], 'Per-input margin changed')
        require(len(queries) in (100, 1000), 'Unexpected input length')
        # The structural observation is registered, not a new selection rule.
        require(len({tuple(map(float, q)) for q in queries}) == len(queries),
                'An original input no longer has unique objectives')
        require(bool(view_text), 'Empty trusted view')
        total += len(queries)
    require(total == 9000 and reg['expected_responses'] == total*5*5 == 225000
            and reg['quality_expected_responses'] == total*5 == 45000,
            'Complete registered quality/timing grid required')


def inside_root(relative):
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts,
            'A project-relative evidence path is required')
    result = (ROOT/path).resolve()
    require(result.is_relative_to(ROOT) and result.is_file(), 'Missing or external evidence file')
    return result


def gate_payload(config=CONFIG):
    config = Path(config)
    reg = json.loads(config.read_text())
    paths = {ROOT/'icde2027'/p for p in RUNTIME}
    paths.update(ROOT/'icde2027'/p for p in (
        'checking_schedule_v4.py', 'test_checking_schedule_v4.py',
        'optimized_exact_v5.py', 'test_optimized_exact_v5.py',
        'optimized_schedule_v6.py', 'test_optimized_schedule_v6.py'))
    paths.add(ORIGINAL_CONFIG)
    return {'protocol_sha256': sha256(config),
            'design_sha256': sha256(inside_root(reg['design'])),
            'environment': {'python': platform.python_version(), 'python_executable': str(Path(sys.executable).resolve()),
                'numpy': np.__version__, 'scipy': scipy.__version__,
                'highspy': highspy.Highs().version(), 'scipy_highs': scipy_highs._Highs().version(),
                'requirements_sha256': sha256(ROOT/'.deployment/requirements-installed.txt')},
            'runtime_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)},
            'input_index_sha256': hashlib.sha256(canonical(reg['inputs']).encode()).hexdigest()}


def validate_gate(config, gate):
    expected = gate_payload(config)
    require(gate.get('status') == 'approved' and gate.get('independent_review') is True
            and isinstance(gate.get('reviewer'), str) and bool(gate['reviewer'])
            and isinstance(gate.get('reviewed_at_utc'), str) and bool(gate['reviewed_at_utc']),
            'Independent source-bound approved gate required')
    for key, value in expected.items():
        require(gate.get(key) == value, f'Gate mismatch: {key}')
    report = inside_root(gate.get('review_report', ''))
    require(sha256(report) == gate.get('review_report_sha256'), 'Independent review report changed')
    return expected


def preserve_input_bytes(output, reg):
    """All exact source bytes are retained once, not merely described by paths."""
    manifest = {}
    for entry in reg['inputs']:
        source = ROOT/entry['directory']
        for name, expected in entry['sha256'].items():
            content = (source/name).read_bytes()
            require(hashlib.sha256(content).hexdigest() == expected, 'Input changed during snapshot')
            relative = Path('input_snapshot')/entry['id']/name
            destination = output/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            manifest[str(relative)] = {'sha256': expected, 'bytes': len(content)}
    return manifest


def snapshot_sources(output, config, gate_path, payload):
    saved = archive_snapshot(output, config, gate_path, payload)
    gate = json.loads(Path(gate_path).read_text())
    report = inside_root(gate['review_report'])
    content = report.read_bytes()
    relative = 'independent_review_report'+report.suffix
    (output/relative).write_bytes(content)
    saved[relative] = {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    return saved


def validate_integrity(config, gate_path, gate, payload, output, snapshot):
    require(validate_gate(config, gate) == payload, 'Runtime approval changed')
    require(json.loads(Path(gate_path).read_text()) == gate, 'Gate content changed')
    require(sha256(gate_path) == snapshot['gate.json']['sha256'], 'Gate bytes changed')
    validate_snapshot(output, snapshot)


def stage_index(directory):
    """Hash every retained stage member; the index itself is stored elsewhere."""
    return {str(p.relative_to(directory)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for p in sorted(directory.rglob('*')) if p.is_file()}


def verify_stage_index(directory, index, budget):
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    require(actual == set(index), 'Quality-stage file membership changed')
    for relative, identity in index.items():
        budget.check()
        path = directory/relative
        require(path.stat().st_size == identity['bytes'] and sha256(path) == identity['sha256'],
                f'Quality-stage member changed: {relative}')


def execute_condition(stage, entry, order, reg, budget, timing_round):
    text, queries = load_input(entry)
    directory = stage/entry['id']
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/'order.json', {'methods': order, 'input': entry,
                                        'timing_round': timing_round})
    methods = {}
    for name in order:
        budget.check()
        methods[name] = measure(directory, name, text, queries, reg, budget)
    comparison = compare_streams(directory, len(queries))
    record = {'input_id': entry['id'], 'group': entry['group'], 'timing_round': timing_round,
              'methods': methods, 'comparison': comparison}
    write_json(directory/'condition.json', record)
    return record


def summarize(reg, records):
    expected = {(entry['id'], repeat) for entry in reg['inputs'] for repeat in range(5)}
    require(len(records) == len(expected) and
            {(r['input_id'], r['timing_round']) for r in records} == expected,
            'Cannot summarize a partial or duplicated timing grid')
    output = {}
    contrasts = reg['factor_contrasts'] + reg['bridge_contrasts']
    for group, count in reg['input_counts'].items():
        output[group] = []
        for repeat in range(5):
            rows = [r for r in records if r['group'] == group and r['timing_round'] == repeat]
            require(len(rows) == count, 'Group completion mismatch')
            values = {}
            for a, b in contrasts:
                ratios = [r['methods'][a]['joint_seconds']/r['methods'][b]['joint_seconds'] for r in rows]
                require(all(math.isfinite(v) and v > 0 for v in ratios), 'Invalid measured ratio')
                values[f'{a}/{b}'] = {'geometric_mean': math.exp(sum(map(math.log, ratios))/len(ratios)),
                                      'min': min(ratios), 'max': max(ratios), 'per_input': ratios}
            values['mean_log_interaction'] = (
                math.log(values['s1_p1/s0_p1']['geometric_mean'])
                - math.log(values['s1_p0/s0_p0']['geometric_mean']))
            values['input_ids'] = [r['input_id'] for r in rows]
            output[group].append(values)
    main = output['original_main']
    directions = {}
    for name, definition in reg['directional_checks'].items():
        ratios = [r[definition['ratio']]['geometric_mean'] for r in main]
        passed = [x < 1 if definition['direction'] == 'less_than_one' else x > 1 for x in ratios]
        directions[name] = {'ratios_by_round': ratios, 'expected_direction': definition['direction'],
                            'direction_holds_by_round': passed, 'direction_holds_all_five': all(passed)}
    return {'groups': output, 'all_records': records,
            'primary_directional_robustness': directions,
            'all_original_directions_hold_all_five': all(v['direction_holds_all_five'] for v in directions.values()),
            'scope': 'Fixed-input arithmetic robustness; bridge order differs. No p-value, '
                     'performance stopping rule, old-target replacement or workload generalization.'}


def execute(config, gate_path, output):
    config, gate_path, output = Path(config), Path(gate_path), Path(output)
    reg = json.loads(config.read_text())
    budget = Budget(reg['wall_clock_limit_seconds'])
    gate = json.loads(gate_path.read_text())
    payload = validate_gate(config, gate)
    validate_inputs(reg)
    budget.check()
    require(not output.exists(), 'Output exists: prior/partial studies must never be overwritten')
    require(reg['cpu'] in os.sched_getaffinity(0), 'Registered allowed CPU unavailable')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        require(os.environ.get(name) == '1', f'{name} must be set to 1 before Python starts')
    os.sched_setaffinity(0, {reg['cpu']})
    output.mkdir(parents=True)
    manifest = {**payload, 'status': 'running', 'stage': 'quality', 'started_at_utc': utc(),
                'host': platform.node(), 'affinity': sorted(os.sched_getaffinity(0)),
                'wall_clock_limit_seconds': reg['wall_clock_limit_seconds'],
                'formal_timing_started': False, 'quality_responses_planned': 45000,
                'timing_responses_planned': 225000, 'completed_quality_conditions': 0,
                'completed_timing_conditions': 0}
    write_json(output/'manifest.json', manifest)
    timing_records, quality_records = [], []
    try:
        snapshot = snapshot_sources(output, config, gate_path, payload)
        snapshot.update(preserve_input_bytes(output, reg))
        manifest['snapshot_files'] = snapshot
        validate_snapshot(output, snapshot)
        write_json(output/'manifest.json', manifest)
        quality = output/'quality'
        quality.mkdir()
        ids = np.random.default_rng(reg['order_seed']-1).permutation(len(reg['inputs']))
        for ix in ids:
            entry = reg['inputs'][int(ix)]
            quality_records.append(execute_condition(quality, entry, method_order(reg, entry['id'], 0),
                                                      reg, budget, None))
            manifest['completed_quality_conditions'] = len(quality_records)
            write_json(output/'manifest.json', manifest)
            validate_integrity(config, gate_path, gate, payload, output, snapshot)
            print(f"quality {len(quality_records)}/63: {entry['id']}", flush=True)
        require(len(quality_records) == 63 and
                sum(sum(m['queries'] for m in r['methods'].values()) for r in quality_records) == 45000,
                'Incomplete actual quality stage')
        quality_index = stage_index(quality)
        quality_report = {'status': 'complete', 'completed_at_utc': utc(), 'responses': 45000,
            'protocol_sha256': payload['protocol_sha256'], 'runtime_sha256': payload['runtime_sha256'],
            'inputs_sha256': payload['input_index_sha256'],
            'checks': {'all_core_evidence_and_paths': True, 'all_bridge_decisions_and_native_states': True,
                       'all_optimized_legacy_and_fraction_replays': True},
            'bridge_true_differences': sum(len(r['comparison']['bridge_true_differences']) for r in quality_records),
            'files': quality_index,
            'scope': 'Actual full quality pass. Its clock values are not formal timing outcomes.'}
        write_json(output/'quality_completion.json', quality_report)
        quality_report_sha = sha256(output/'quality_completion.json')
        # This checks retained data, not an externally supplied "passed" flag.
        verify_stage_index(quality, quality_index, budget)
        validate_integrity(config, gate_path, gate, payload, output, snapshot)
        manifest.update(stage='timing', formal_timing_started=True,
                        quality_completion_sha256=quality_report_sha,
                        timing_started_at_utc=utc())
        write_json(output/'manifest.json', manifest)
        for repeat in range(5):
            ids = np.random.default_rng(reg['order_seed']+repeat).permutation(len(reg['inputs']))
            stage = output/'timing'/f'round_{repeat}'
            stage.mkdir(parents=True)
            for ix in ids:
                budget.check()
                entry = reg['inputs'][int(ix)]
                timing_records.append(execute_condition(stage, entry, method_order(reg, entry['id'], repeat),
                                                        reg, budget, repeat))
                manifest['completed_timing_conditions'] = len(timing_records)
                write_json(output/'manifest.json', manifest)
                validate_integrity(config, gate_path, gate, payload, output, snapshot)
                require(sha256(output/'quality_completion.json') == quality_report_sha,
                        'Quality receipt changed during timing')
                print(f"timing {len(timing_records)}/315: round {repeat}, {entry['id']}", flush=True)
        budget.check()
        result = summarize(reg, timing_records)
        validate_inputs(reg)
        validate_integrity(config, gate_path, gate, payload, output, snapshot)
        verify_stage_index(quality, quality_index, budget)
        require(sha256(output/'quality_completion.json') == quality_report_sha, 'Quality receipt changed')
        write_json(output/'summary.json', result)
        budget.check()
        manifest.update(status='complete', stage='complete', finished_at_utc=utc(),
                        quality_responses=45000, timing_responses=225000,
                        timing_arm_streams=1575, final_integrity_check='passed',
                        elapsed_wall_seconds=time.monotonic()-budget.started)
        write_json(output/'completion.json', manifest)
        write_json(output/'manifest.json', manifest)
    except Exception as error:
        manifest.update(status='resource_limit' if isinstance(error, ResourceLimit) else 'failed',
                        finished_at_utc=utc(), completed_quality_conditions=len(quality_records),
                        completed_timing_conditions=len(timing_records),
                        elapsed_wall_seconds=time.monotonic()-budget.started,
                        error=repr(error), traceback=traceback.format_exc())
        write_json(output/'failure.json', manifest)
        write_json(output/'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--print-gate-payload', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.print_gate_payload:
        require(not args.execute, 'Choose inspection or formal execution')
        print(json.dumps(gate_payload(args.config), indent=2))
    elif args.execute:
        require(args.gate is not None and args.output is not None, 'Independent gate and fresh output required')
        execute(args.config, args.gate, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
