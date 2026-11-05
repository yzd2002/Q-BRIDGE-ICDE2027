"""Gated retrospective 2x2: conservative screening x exact dual retention.

The four core arms share candidates, order, native solver and receiver.  Only
synthetic development is permitted until an independent hash-bound approval.
Old engines, proofs and saved outcomes are imported without modification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import time
import traceback

import highspy
import numpy as np
import scipy
from scipy.optimize._highspy import _core as scipy_highs

from .primary_aligned import (RUNTIME, SERVICE_FIELDS, adapt_lower,
                              make_engine as original_engine, replay,
                              service_response)
from .threshold_extension.engine import ThresholdEngine
from .threshold_extension.proofs import ThresholdChecker, _floats


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'icde2027/configs/checking_schedule_v4.json'
CORE = ('s0_p0', 's0_p1', 's1_p0', 's1_p1')
METHODS = (*CORE, 'screened_auto', 'original_prepared')
FULL_INTERVAL = (-math.inf, math.inf)
NUMERICAL_ERRORS = (AttributeError, TypeError, ValueError, ArithmeticError)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def _add(x, y):
    if x == (0., 0.):
        return y
    if y == (0., 0.):
        return x
    lo, hi = x[0] + y[0], x[1] + y[1]
    if not math.isfinite(lo) or not math.isfinite(hi):
        return FULL_INTERVAL
    return math.nextafter(lo, -math.inf), math.nextafter(hi, math.inf)


def _neg(x):
    return -x[1], -x[0]


def _mul(x, y):
    if x == (0., 0.) or y == (0., 0.):
        return (0., 0.)
    products = [a * b for a in x for b in y]
    if not all(math.isfinite(v) for v in products):
        return FULL_INTERVAL
    return math.nextafter(min(products), -math.inf), math.nextafter(max(products), math.inf)


def _point(x):
    return (float(x), float(x))


def dot_interval(c, point):
    """Conservative IEEE binary64 interval; no approximate result is a proof."""
    result = (0., 0.)
    for a, b in zip(c, point):
        result = _add(result, _mul(_point(a), _point(b)))
    return result


def prepare_interval(view, lo, hi):
    constant = (0., 0.)
    residual = [(0., 0.) for _ in view['lower']]
    matrix = view['matrix']
    for j, (eta_lo, eta_hi) in enumerate(zip(lo, hi)):
        constant = _add(constant, _mul(_point(view['rhs_lower'][j]), _point(eta_lo)))
        constant = _add(constant, _neg(_mul(_point(view['rhs_upper'][j]), _point(eta_hi))))
        difference = _add(_point(eta_hi), _neg(_point(eta_lo)))
        for k in range(matrix['indptr'][j], matrix['indptr'][j + 1]):
            i = matrix['indices'][k]
            residual[i] = _add(residual[i], _mul(_point(matrix['data'][k]), difference))
    return constant, tuple(residual)


def lower_interval(view, c, scale, terms):
    constant, residual = terms
    result = _mul(_point(scale), constant)
    for ci, ri, lo, hi in zip(c, residual, view['lower'], view['upper']):
        coefficient = _add(_point(ci), _mul(_point(scale), ri))
        a, b = _mul(coefficient, _point(lo)), _mul(coefficient, _point(hi))
        result = _add(result, (min(a[0], b[0]), min(a[1], b[1])))
    return result


@dataclass
class Candidate:
    lower: tuple
    upper: tuple
    prepared: object = None
    screen_terms: object = None


class CheckingScheduleEngine(ThresholdEngine):
    """Last-only fixed schedule; does not call the old eager-preparing init.

    Reuses only the unchanged native call, checked-primal factories, counters
    and wire bundler. P0 builds identical exact terms for each use and drops
    them; P1 retains locally issued terms. Float terms are a separate cache.
    """
    def __init__(self, view, *, screening, prepared, simplex_strategy=1):
        begin = time.perf_counter()
        require(type(screening) is bool and type(prepared) is bool, 'Boolean factors required')
        require(simplex_strategy in (0, 1), 'Only public choose/serial dual strategies')
        self.screening, self.prepared_reuse = screening, prepared
        self._view = json.loads(json.dumps(view, allow_nan=False))
        self.checker = ThresholdChecker(self._view, cache_size=0)
        self.digest, self.d, self.j = self.checker.digest, self.checker.d, self.checker.j
        self.plan, self.bank_size, self.simplex_strategy = 'threshold_hot', 1, simplex_strategy
        self._use_primal_pool = self._use_dual_pool = False
        self.primal_retention_capacity = self.dual_retention_capacity = 1
        self._initial_raw = tuple(self._view['feasible_point'])
        self.initial_primal = self.checker.primal_state(self._initial_raw)
        self.last_primal = self.last_dual = None
        self.last_primal_generation = self.last_dual_generation = 0
        self.primal_bank, self.dual_bank = [], []
        self.maximum_primal_bank_entries = self.maximum_dual_bank_entries = 0
        self._zeros = (0.,) * self.j
        self.zero = Candidate(self._zeros, self._zeros)
        self._lower, self._upper = np.asarray(view['lower']), np.asarray(view['upper'])
        self.indices = np.arange(self.d, dtype=np.int32)
        self.h = highspy.Highs()
        self.options = {'output_flag': False, 'threads': 1, 'solver': 'simplex',
                        'simplex_strategy': simplex_strategy,
                        'primal_feasibility_tolerance': 1e-10, 'dual_feasibility_tolerance': 1e-10}
        for key, value in self.options.items():
            require(self.h.setOptionValue(key, value) == highspy.HighsStatus.kOk,
                    f'HiGHS rejected {key}')
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = self.d, self.j
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = np.zeros(self.d), self._lower, self._upper
        lp.row_lower_ = np.asarray(view['rhs_lower'], dtype=float)
        lp.row_upper_ = np.asarray(view['rhs_upper'], dtype=float)
        source = view['matrix']; matrix = highspy.HighsSparseMatrix()
        matrix.num_col_, matrix.num_row_, matrix.format_ = self.d, self.j, highspy.MatrixFormat.kRowwise
        matrix.start_ = np.asarray(source['indptr'], dtype=np.int32)
        matrix.index_ = np.asarray(source['indices'], dtype=np.int32)
        matrix.value_ = np.asarray(source['data'], dtype=float)
        lp.a_matrix_ = matrix
        require(self.h.passModel(lp) == highspy.HighsStatus.kOk, 'HiGHS rejected trusted view')
        self.native_version = self.h.version()
        self.initialization_seconds = time.perf_counter() - begin

    def _new_metrics(self):
        result = super()._new_metrics()
        result.update(exact_dual_preparations=0, exact_dual_reuses=0,
                      exact_dual_preparation_seconds=0., screen_preparations=0,
                      deferred_duals=0, deferred_primals=0, replayed_duals=0,
                      replayed_primals=0, rejected_cached_duals=0,
                      rejected_cached_primals=0)
        return result

    def _exact_candidate(self, candidate, c, delta, scale, metrics):
        tick = time.perf_counter()
        try:
            state = candidate.prepared if self.prepared_reuse else None
            if state is not None and not self.checker.owns_dual(state, candidate.lower, candidate.upper):
                metrics['invalid_bank_entries'] += 1
                state = None
            if state is None:
                prep = time.perf_counter()
                try:
                    state = self.checker.dual_state(candidate.lower, candidate.upper)
                finally:
                    metrics['exact_dual_preparation_seconds'] += time.perf_counter() - prep
                    metrics['exact_dual_preparations'] += 1
                if self.prepared_reuse:
                    candidate.prepared = state
            else:
                metrics['exact_dual_reuses'] += 1
            return self.checker.lower_proof(c, candidate.lower, candidate.upper,
                                            threshold=delta, scale=scale, prepared=state)
        finally:
            metrics['verify_seconds'] += time.perf_counter() - tick
            metrics['dual_candidates_exact_checked'] += 1

    def _defer_dual(self, candidate, c, delta, scale, metrics):
        if not self.screening:
            return False
        tick = time.perf_counter()
        try:
            # Validate raw coefficients even if no exact state is needed.
            lo, hi = _floats(candidate.lower), _floats(candidate.upper)
            require(len(lo) == len(hi) == self.j and all(v >= 0 for v in lo + hi),
                    'Invalid cached dual')
            if candidate.screen_terms is None:
                candidate.screen_terms = prepare_interval(self._view, lo, hi)
                metrics['screen_preparations'] += 1
            value = lower_interval(self._view, c, 1. if scale is None else scale,
                                   candidate.screen_terms)
            return math.isfinite(value[1]) and value[1] <= delta
        except NUMERICAL_ERRORS:
            # A failed screen requires the same exact attempt as S0.
            metrics['invalid_bank_entries'] += 1
            return False
        finally:
            metrics['dual_candidates_screened'] += 1
            metrics['screen_seconds'] += time.perf_counter() - tick

    def _defer_primal(self, state, c, delta, metrics):
        if not self.screening:
            return False
        tick = time.perf_counter()
        try:
            interval = dot_interval(c, state.raw)
            return math.isfinite(interval[0]) and interval[0] > delta
        finally:
            metrics['primal_candidates_screened'] += 1
            metrics['screen_seconds'] += time.perf_counter() - tick

    def _save_dual(self, lo, hi, norm, metrics):
        tick = time.perf_counter(); metrics['native_dual_candidates'] += 1
        try:
            require(norm > 0, 'No normalized candidate for zero objective')
            lower, upper = _floats([x / norm for x in lo]), _floats([x / norm for x in hi])
            require(len(lower) == len(upper) == self.j and all(x >= 0 for x in lower + upper),
                    'Invalid normalized multipliers')
            old = self.last_dual
            if old is not None and old.lower == lower and old.upper == upper:
                candidate = old
                metrics['native_dual_reuses'] += 1
            else:
                candidate = Candidate(lower, upper)
            self.last_dual = candidate
            self.dual_bank[:] = [candidate]
            self.last_dual_generation += 1
            self.maximum_dual_bank_entries = 1
            metrics['last_dual_updates'] += 1
        except NUMERICAL_ERRORS:
            metrics['rejected_dual_proposals'] += 1
        finally:
            metrics['preparation_seconds'] += time.perf_counter() - tick

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
            if lower is None or checked['claimed_lower'] > lower['claimed_lower']:
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

        duals = [(self.zero, None, 'box_shortcut')]
        if self.last_dual is not None:
            duals.append((self.last_dual, norm, 'last_dual'))
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

    def memory_info(self):
        candidates = [self.zero, *self.dual_bank]
        return {**self._retention_metrics(),
                'retained_exact_duals': sum(x.prepared is not None for x in candidates),
                'retained_float_duals': sum(x.screen_terms is not None for x in candidates),
                'checker_cache': self.checker.cache_info()}


def make_engine(name, view):
    require(name in METHODS, 'Unknown registered arm')
    if name == 'original_prepared':
        return original_engine(name, view, 16)
    return CheckingScheduleEngine(view, screening=name.startswith('s1') or name == 'screened_auto',
                                  prepared=name.endswith('p1') or name == 'screened_auto',
                                  simplex_strategy=0 if name == 'screened_auto' else 1)


class FractionReplay:
    """Independent exported-evidence audit, without producer/checker caches."""
    def __init__(self, view):
        self.digest = hashlib.sha256(json.dumps(view, sort_keys=True, separators=(',', ':'),
                                               allow_nan=False).encode()).hexdigest()
        self.lo, self.hi = self.numbers(view['lower']), self.numbers(view['upper'])
        self.a, self.b = self.numbers(view['rhs_lower']), self.numbers(view['rhs_upper'])
        matrix = view['matrix']; values = self.numbers(matrix['data'])
        self.rows = [tuple((matrix['indices'][k], values[k])
                           for k in range(matrix['indptr'][j], matrix['indptr'][j + 1]))
                     for j in range(len(self.a))]
        self.feasible(self.numbers(view['feasible_point']))

    @staticmethod
    def numbers(values):
        require(all(type(x) in (int, float) and math.isfinite(x) for x in values), 'Finite numbers required')
        return tuple(Fraction.from_float(float(x)) for x in values)

    def feasible(self, point):
        require(len(point) == len(self.lo), 'Point dimension')
        require(all(a <= x <= b for a, x, b in zip(self.lo, point, self.hi)), 'Box infeasible')
        for a, row, b in zip(self.a, self.rows, self.b):
            require(a <= sum((v * point[i] for i, v in row), Fraction(0)) <= b, 'Row infeasible')

    def exact_lower(self, c, lo, hi, scale):
        require(len(lo) == len(hi) == len(self.a) and all(x >= 0 for x in lo + hi)
                and scale >= 0, 'Invalid dual proposal')
        result = scale * sum((a*x - b*y for a,b,x,y in zip(self.a,self.b,lo,hi)), Fraction(0))
        residual = list(c)
        for row, x, y in zip(self.rows, lo, hi):
            for i, value in row:
                residual[i] += scale * value * (y - x)
        return result + sum((min(v*a, v*b) for v,a,b in zip(residual,self.lo,self.hi)), Fraction(0))

    def verify(self, item, index, objective, threshold):
        require(type(item.get('query_index')) is int and item['query_index'] == index, 'Query index')
        answer = item['answer']; proof = answer['certificate']
        require(set(answer) == set(SERVICE_FIELDS), 'Common service envelope')
        if proof is None:
            require(answer['status'] == 'uncertain' and all(answer[k] is None for k in
                    ('lower_bound','upper_bound','robust_threshold_decision'))
                    and isinstance(answer['reason'], str) and bool(answer['reason']), 'Invalid refusal')
            return False
        c, delta = self.numbers(objective), self.numbers([threshold])[0]
        require(len(c) == len(self.lo), 'Objective dimension')
        require(proof['format'] == 'qbridge-threshold-proof-v1' and proof['view_digest'] == self.digest
                and self.numbers(proof['objective']) == c
                and self.numbers([proof['decision_threshold']])[0] == delta, 'Bundle binding')
        lp = proof['lower_proof']
        require(lp['view_digest'] == self.digest and self.numbers(lp['objective']) == c
                and self.numbers([lp['decision_threshold']])[0] == delta
                and lp['format'] in ('qbridge-dual-proof-v1','qbridge-scaled-dual-proof-v1'), 'Lower binding')
        scale = self.numbers([lp['dual_scale'] if lp['format'] == 'qbridge-scaled-dual-proof-v1' else 1.])[0]
        exact = self.exact_lower(c, self.numbers(lp['lower_multipliers']), self.numbers(lp['upper_multipliers']), scale)
        lower = self.numbers([lp['claimed_lower']])[0]
        require(lower <= exact and answer['lower_bound'] == lp['claimed_lower'], 'Invalid lower')
        upper = None; up = proof['upper_proof']
        if up is not None:
            # The existing protocol binds this point through the checked
            # outer bundle, not through duplicated inner binding fields.
            require(isinstance(up, dict) and set(up) == {'point','claimed_upper'}, 'Upper schema')
            point = self.numbers(up['point']); self.feasible(point)
            value = sum((a*b for a,b in zip(c,point)), Fraction(0))
            represented = float(value)
            if Fraction.from_float(represented) < value:
                represented = math.nextafter(represented, math.inf)
            upper = self.numbers([up['claimed_upper']])[0]
            require(up['claimed_upper'] == represented and lower <= value
                    and answer['upper_bound'] == up['claimed_upper'], 'Invalid upper')
        else:
            require(answer['upper_bound'] is None, 'Unproved upper')
        decision = True if lower > delta else False if upper is not None and upper <= delta else None
        require(proof['robust_threshold_decision'] is decision and answer['robust_threshold_decision'] is decision
                and answer['status'] == ('certified' if decision is True else 'unresolved'), 'Wrong verdict')
        return True


def measure(directory, name, view_text, queries, reg):
    """Fresh fixed-view producer plus unchanged persistent/stateless receivers."""
    directory = Path(directory); retained, diagnostics, rows = [], [], []
    wall_begin = time.perf_counter()
    try:
        tick = time.perf_counter(); view = json.loads(view_text)
        view_decode = time.perf_counter() - tick
        tick = time.perf_counter(); engine = make_engine(name, view)
        initialization = time.perf_counter() - tick
        cumulative = view_decode + initialization
        decisions, routes, refusals = Counter(), Counter(), 0
        for i, objective in enumerate(queries):
            tick = time.perf_counter(); native = engine.query(objective, reg['threshold'])
            answer = (adapt_lower(native, objective.tolist(), reg['threshold'], engine.digest)
                      if name == 'original_prepared' else native)
            answer = service_response(answer)
            query_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            line = json.dumps({'query_index': i, 'answer': answer}, separators=(',', ':'), allow_nan=False)
            serialization = time.perf_counter() - tick
            retained.append(line); diagnostics.append({'query_index': i, 'native_answer': native})
            cumulative += query_seconds + serialization
            decision = answer['robust_threshold_decision']
            decisions['true' if decision is True else 'false' if decision is False else 'unknown'] += 1
            routes[answer['route']] += 1; refusals += int(answer['certificate'] is None)
            rows.append({'query': i+1, 'query_seconds': query_seconds, 'serialization_seconds': serialization,
                         'cumulative_seconds': cumulative, 'decision': decision, 'route': answer['route']})
        path = directory / f'{name}.jsonl.gz'; tick = time.perf_counter()
        with gzip.open(path, 'wt', encoding='utf8', compresslevel=3) as stream:
            stream.write('\n'.join(retained) + '\n')
        gzip_write = time.perf_counter() - tick
        rr = {**reg, 'query_prefixes': [q for q in reg['query_prefixes'] if q <= len(queries)]}
        persistent = replay(path, view, queries, rr, view_text=view_text)
        stateless = replay(path, view, queries, rr, stateless=True, view_text=view_text)
        require(persistent['decisions'] == stateless['decisions'] == dict(decisions), 'Receiver mismatch')
        tick = time.perf_counter(); independent = FractionReplay(view)
        valid = sum(independent.verify(json.loads(line), i, queries[i].tolist(), reg['threshold'])
                    for i, line in enumerate(retained))
        independent_seconds = time.perf_counter() - tick
        require(valid == persistent['valid_proofs'], 'Independent replay differs')
        tick = time.perf_counter()
        native_lines = [json.dumps(row, separators=(',', ':'), allow_nan=False) for row in diagnostics]
        diagnostic_serialization = time.perf_counter() - tick
        native_path = directory / f'{name}_native.jsonl.gz'; tick = time.perf_counter()
        with gzip.open(native_path, 'wt', encoding='utf8', compresslevel=3) as stream:
            stream.write('\n'.join(native_lines) + '\n')
        diagnostic_write = time.perf_counter() - tick
        result = {'method': name, 'queries': len(queries), 'view_decode_seconds': view_decode,
                  'plan_initialization_seconds': initialization, 'producer_seconds': cumulative,
                  'joint_seconds': cumulative + persistent['total_consumer_seconds'],
                  'prefixes': {str(q): rows[q-1]['cumulative_seconds'] for q in rr['query_prefixes']},
                  'joint_prefixes': {str(q): rows[q-1]['cumulative_seconds'] + persistent['prefixes'][str(q)]
                                     for q in rr['query_prefixes']},
                  'per_query': rows, 'decisions': dict(decisions), 'routes': dict(routes), 'refusals': refusals,
                  'persistent_replay': persistent, 'stateless_replay': stateless,
                  'independent_fraction_replay': {'valid': valid, 'refusals': len(queries)-valid,
                                                   'seconds': independent_seconds, 'included_in_primary_cost': False},
                  'proof_file': path.name, 'proof_file_sha256': sha256(path),
                  'uncompressed_bytes': sum(len(x.encode())+1 for x in retained),
                  'compressed_bytes': path.stat().st_size, 'gzip_write_seconds': gzip_write,
                  'native_diagnostics': {'file': native_path.name, 'sha256': sha256(native_path),
                                         'serialization_seconds': diagnostic_serialization,
                                         'gzip_write_seconds': diagnostic_write, 'included_in_primary_cost': False},
                  'process_peak_rss_kib_not_method_isolated': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        metric_keys = ('screen_seconds','verify_seconds','solve_seconds','preparation_seconds',
                       'exact_dual_preparation_seconds','primal_preparation_seconds',
                       'dual_candidates_screened','primal_candidates_screened',
                       'dual_candidates_exact_checked','primal_candidates_exact_checked',
                       'exact_dual_preparations','exact_dual_reuses','screen_preparations',
                       'deferred_duals','deferred_primals','replayed_duals','replayed_primals',
                       'rejected_cached_duals','rejected_cached_primals','rejected_dual_proposals',
                       'rejected_primal_proposals','invalid_bank_entries','cutoff_retries')
        result['diagnostic_totals'] = {key: sum(row['native_answer'].get(key,0) for row in diagnostics)
                                      for key in metric_keys}
        result['diagnostic_totals']['native_calls'] = sum(len(row['native_answer'].get('native_calls',[]))
                                                         for row in diagnostics)
        result['diagnostic_totals_scope'] = ('Absent legacy counters are zero/not instrumented, not evidence of zero work. '
            'Exact-dual preparation and primal preparation are subsets of verify_seconds; '
            'diagnostic times are not added to primary outer elapsed costs.')
        write_json(directory / f'{name}_summary.json', result)
        return result
    except Exception as error:
        with gzip.open(directory / f'{name}_partial.jsonl.gz', 'wt', encoding='utf8') as stream:
            stream.write('\n'.join(retained) + ('\n' if retained else ''))
        with gzip.open(directory / f'{name}_partial_native.jsonl.gz', 'wt', encoding='utf8') as stream:
            for row in diagnostics:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        write_json(directory / f'{name}_failure.json', {'completed_responses': len(retained),
                   'failed_or_verifying_query_index': locals().get('i'),
                   'elapsed_seconds_including_failed_work': time.perf_counter()-wall_begin,
                   'error': repr(error), 'traceback': traceback.format_exc()})
        raise


def method_order(reg, input_id, timing_round):
    seed = int(hashlib.sha256(f"{reg['order_seed']}:{input_id}".encode()).hexdigest()[:16], 16)
    base = np.random.default_rng(seed).permutation(reg['methods']).tolist()
    shift = 2 * timing_round % len(base)
    return base[shift:] + base[:shift]


def gate_payload(config=CONFIG):
    config = Path(config); reg = json.loads(config.read_text())
    paths = {ROOT / 'icde2027' / p for p in RUNTIME}
    paths.update({Path(__file__), ROOT/'icde2027/test_checking_schedule_v4.py'})
    return {'protocol_sha256': sha256(config), 'design_sha256': sha256(ROOT/reg['design']),
            'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                            'scipy': scipy.__version__, 'highspy': highspy.Highs().version(),
                            'scipy_highs': scipy_highs._Highs().version(),
                            'requirements_sha256': sha256(ROOT/'.deployment/requirements-installed.txt')},
            'runtime_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}}


def validate_gate(config, gate):
    expected = gate_payload(config)
    require(gate.get('status') == 'approved' and isinstance(gate.get('reviewer'), str)
            and bool(gate['reviewer']) and bool(gate.get('reviewed_at_utc')), 'Independent approved gate required')
    for key, value in expected.items():
        require(gate.get(key) == value, f'Gate mismatch: {key}')
    return expected


def archive_snapshot(output, config, gate_path, payload):
    """Persist exact approved bytes, not only references to mutable paths."""
    output, config, gate_path = Path(output), Path(config), Path(gate_path)
    reg = json.loads(config.read_text())
    sources = {ROOT/name: ('source_snapshot/'+name, digest)
               for name,digest in payload['runtime_sha256'].items()}
    sources.update({config: ('protocol.json',payload['protocol_sha256']),
                    ROOT/reg['design']: ('design.md',payload['design_sha256']),
                    gate_path: ('gate.json',sha256(gate_path)),
                    ROOT/'.deployment/requirements-installed.txt':
                    ('source_snapshot/.deployment/requirements-installed.txt',
                     payload['environment']['requirements_sha256'])})
    saved = {}
    for source, (name, expected) in sources.items():
        content = source.read_bytes()
        require(hashlib.sha256(content).hexdigest() == expected, f'Snapshot source changed: {source}')
        destination = output/name; destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(content)
        saved[name] = {'sha256':expected,'bytes':len(content)}
    return saved


def validate_snapshot(output, saved):
    for name, identity in saved.items():
        content = (Path(output)/name).read_bytes()
        require(len(content) == identity['bytes'] and hashlib.sha256(content).hexdigest() == identity['sha256'],
                f'Saved snapshot changed: {name}')


def validate_inputs(reg):
    require(tuple(reg['methods']) == METHODS and tuple(reg['core_methods']) == CORE, 'Arm selection changed')
    require(reg['timing_rounds'] == 3 and reg['threshold'] == .001 and reg['bank_size'] == 16,
            'Registered timing/threshold/receiver settings changed')
    require(Counter(x['group'] for x in reg['inputs']) == reg['input_counts'], 'Incomplete input grid')
    require(len({x['id'] for x in reg['inputs']}) == 63, 'Duplicate or missing inputs')
    for entry in reg['inputs']:
        directory = ROOT / entry['directory']
        for name, expected in entry['sha256'].items():
            require(sha256(directory/name) == expected, f'Input hash mismatch: {directory/name}')
        q = np.load(directory/'queries.npy', mmap_mode='r', allow_pickle=False)
        require(list(q.shape) == entry['query_shape'] and entry['threshold'] == reg['threshold'], 'Input shape/threshold')
    require(sum(e['query_shape'][0] for e in reg['inputs'])*3*6 == reg['expected_responses'] == 162000,
            'Incomplete response grid')


def load_input(entry):
    """Hash the actual bytes to be decoded, also when files change after gate."""
    directory = ROOT / entry['directory']
    captured = {name: (directory/name).read_bytes() for name in ('view.json','queries.npy','condition.json')}
    require({name: hashlib.sha256(value).hexdigest() for name,value in captured.items()} == entry['sha256'],
            f'Changed registered input: {entry["id"]}')
    queries = np.load(io.BytesIO(captured['queries.npy']), allow_pickle=False)
    require(list(queries.shape) == entry['query_shape'] and np.isfinite(queries).all(), 'Invalid loaded query array')
    return captured['view.json'].decode('utf8'), queries


def compare_streams(directory, *, expected_queries, methods=METHODS):
    require(type(expected_queries) is int and expected_queries > 0, 'Positive planned query count required')
    records, native = {}, {}
    for name in methods:
        with gzip.open(Path(directory)/f'{name}.jsonl.gz', 'rt') as stream:
            records[name] = [json.loads(line) for line in stream]
        require(len(records[name]) == expected_queries, f'Service stream length differs from plan: {name}')
        for i, item in enumerate(records[name]):
            require(isinstance(item,dict) and type(item.get('query_index')) is int
                    and item['query_index'] == i, f'Service query index mismatch: {name} at {i}')
        with gzip.open(Path(directory)/f'{name}_native.jsonl.gz', 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        require(len(rows) == expected_queries, f'Native stream length differs from plan: {name}')
        for i, item in enumerate(rows):
            require(isinstance(item,dict) and type(item.get('query_index')) is int
                    and item['query_index'] == i, f'Native query index mismatch: {name} at {i}')
            require(isinstance(item.get('native_answer'),dict), f'Malformed native answer: {name} at {i}')
        native[name] = [item['native_answer'] for item in rows]
    baseline = records[CORE[0]]; differences = {}
    for name in CORE[1:]:
        require(records[name] == baseline, f'Core evidence/decision/route mismatch: {name}')
        for i, (a, b) in enumerate(zip(native[CORE[0]], native[name])):
            for key in ('last_primal_generation','last_dual_generation','primal_total_retained_entries',
                        'dual_total_retained_entries','cutoff_retries','rejected_dual_proposals',
                        'rejected_primal_proposals'):
                require(a[key] == b[key], f'Core state mismatch {name} query {i}: {key}')
            calls = lambda x: [{k:v for k,v in r.items() if k != 'seconds'} for r in x['native_calls']]
            require(calls(a) == calls(b), f'Core native path mismatch {name} query {i}')
    for name in methods:
        require(len(records[name]) == len(baseline), 'Different response counts')
        counts = Counter()
        for a, b in zip(records['s1_p1'], records[name]):
            x, y = a['answer']['robust_threshold_decision'], b['answer']['robust_threshold_decision']
            require(x is None or y is None or x is y, f'Opposite proved decisions: {name}')
            counts['s1_p1_loses_decided'] += int(x is None and y is not None)
            counts['s1_p1_gains_decided'] += int(x is not None and y is None)
        differences[name] = dict(counts)
    return {'core_same_evidence_and_paths': True, 'relative_to_s1_p1': differences}


def summarize(reg, records):
    output = {}; contrasts = reg['factor_contrasts'] + reg['bridge_contrasts']
    for group in reg['input_counts']:
        output[group] = []
        for repeat in range(reg['timing_rounds']):
            rows = [r for r in records if r['group'] == group and r['timing_round'] == repeat]
            require(len(rows) == reg['input_counts'][group], 'Cannot summarize an incomplete registered grid')
            result = {}
            for numerator, denominator in contrasts:
                values = [r['methods'][numerator]['joint_seconds']/r['methods'][denominator]['joint_seconds'] for r in rows]
                result[f'{numerator}/{denominator}'] = {'geometric_mean': math.exp(sum(map(math.log,values))/len(values)),
                                                       'min': min(values), 'max': max(values), 'per_input': values}
            result['mean_log_interaction'] = (math.log(result['s1_p1/s0_p1']['geometric_mean'])
                                               - math.log(result['s1_p0/s0_p0']['geometric_mean']))
            result['input_ids'] = [r['input_id'] for r in rows]
            output[group].append(result)
    target = reg['primary']
    supported = all(r[f'{a}/{b}']['geometric_mean'] <= target['maximum_ratio']
                    for r in output[target['group']] for a,b in target['contrasts'])
    return {'groups': output, 'both_factors_meet_main_engineering_target': supported,
            'interpretation': 'Retrospective fixed-input timing; no p-value or independent workload generalization.',
            'all_records': records}


def execute(config, gate_path, output):
    config, gate_path, output = Path(config), Path(gate_path), Path(output)
    gate = json.loads(gate_path.read_text()); payload = validate_gate(config, gate)
    reg = json.loads(config.read_text()); validate_inputs(reg)
    require(not output.exists(), 'Output already exists; registered runs are never overwritten')
    require(reg['cpu'] == 4 and 4 in os.sched_getaffinity(0), 'Authorized CPU4 unavailable')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        require(os.environ.get(key) == '1', f'Set {key}=1 before Python startup')
    os.sched_setaffinity(0, {4})
    output.mkdir(parents=True)
    manifest = {**payload, 'gate': gate, 'gate_file_sha256':sha256(gate_path), 'started_at_utc': utc(),
                'affinity': sorted(os.sched_getaffinity(0)), 'host':platform.node(),
                'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                'highspy': highspy.Highs().version(), 'scipy_highs':scipy_highs._Highs().version(),
                'thread_environment': {k:os.environ.get(k) for k in
                                       ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                'status': 'running', 'formal_scope': 'retrospective fixed-view serving'}
    write_json(output/'manifest.json', manifest)
    records = []
    try:
        manifest['snapshot_files'] = archive_snapshot(output,config,gate_path,payload)
        validate_snapshot(output,manifest['snapshot_files'])
        write_json(output/'manifest.json',manifest)
        for repeat in range(reg['timing_rounds']):
            ids = np.random.default_rng(reg['order_seed']+repeat).permutation(len(reg['inputs']))
            for ix in ids:
                entry = reg['inputs'][int(ix)]
                text, queries = load_input(entry)
                directory = output/f'round_{repeat}'/entry['id']; directory.mkdir(parents=True)
                order = method_order(reg, entry['id'], repeat)
                write_json(directory/'order.json', {'methods': order, 'input': entry})
                methods = {name: measure(directory, name, text, queries, reg) for name in order}
                comparison = compare_streams(directory,expected_queries=len(queries))
                record = {'input_id': entry['id'], 'group': entry['group'], 'timing_round': repeat,
                          'methods': methods, 'comparison': comparison}
                write_json(directory/'condition.json', record); records.append(record)
        result = summarize(reg, records)
        # A changed implementation is a failed run, never a mixed-version result.
        validate_gate(config, gate)
        require(sha256(gate_path) == manifest['gate_file_sha256'], 'Gate bytes changed during run')
        validate_snapshot(output,manifest['snapshot_files'])
        validate_inputs(reg)
        write_json(output/'summary.json', result)
        completion = {'status': 'complete', 'finished_at_utc': utc(),
                      'arm_streams': len(records)*len(METHODS), 'responses': reg['expected_responses'],
                      'final_integrity_check': 'passed'}
        manifest.update(completion)
        write_json(output/'completion.json',completion)
        write_json(output/'manifest.json',manifest)
    except Exception:
        failure = {'status':'failed', 'finished_at_utc':utc(), 'completed_conditions':len(records),
                   'traceback':traceback.format_exc()}
        manifest.update(failure)
        write_json(output/'failure.json',failure)
        write_json(output/'manifest.json',manifest)
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
        require(not args.execute, 'Choose gate inspection or execution')
        print(json.dumps(gate_payload(args.config), indent=2))
    elif args.execute:
        require(args.gate is not None and args.output is not None, 'Independent --gate and fresh --output required')
        execute(args.config, args.gate, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
