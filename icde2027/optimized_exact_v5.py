"""Exact reference kernels; no new statistical assumptions or execution policy.

The legacy standard-library ThresholdChecker verifies every exported proof.
The two adapters inherit the existing query schedules and native calls. Only
their initialization and exact arithmetic differ. There is no objective-result
cache, planner, source-data access, or global monkeypatch in this module.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
from fractions import Fraction
import json
import math
import sys
import time

import highspy
import numpy as np

from .checking_schedule_v4 import Candidate, CheckingScheduleEngine
from .threshold_extension.engine import ThresholdEngine
from .threshold_extension.proofs import (
    FORMAT, LOWER_FORMATS, CheckedPrimal, ThresholdChecker, _floats, upper_float,
)


# Never expose a shared mutable Fraction through issued state. Fraction's
# nominal immutability can be bypassed with object.__setattr__; integers cannot.
ZERO = 0


def _fraction_signature(value):
    """Immutable primitive snapshot, not an alias or Fraction equality check."""
    if type(value) is not Fraction:
        raise ValueError('exact state must contain a Fraction')
    numerator, denominator = value.numerator, value.denominator
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise ValueError('invalid Fraction internals')
    return numerator, denominator


def _vector_signature(values):
    return tuple(_fraction_signature(value) for value in values)


@dataclass(frozen=True)
class QueryContext:
    """Locally issued exact objective; its lazy box cache is issuer-private."""
    view_digest: str
    raw: tuple
    exact: tuple


@dataclass(frozen=True)
class ExactDualState:
    """Locally issued constant/residual; all raw multipliers remain on wire."""
    view_digest: str
    base_lower: tuple
    base_upper: tuple
    constant: Fraction
    support: tuple
    dense_offset: tuple | None
    mode: str


class OptimizedThresholdChecker(ThresholdChecker):
    """Same proof contract, whole-view checks and bounded state ownership.

    A context only saves Fraction conversion/box arithmetic. Every use still
    validates the complete requested objective and its binding to this issuer.
    An explicit invalid context is rejected; an invalid prepared dual is
    rebuilt from the fully validated raw multipliers, as in ThresholdChecker.
    """
    def __init__(self, view, cache_size=16):
        # This performs the unchanged complete CSR/box/witness validation.
        super().__init__(view, cache_size=cache_size)
        self._issued_queries = {}
        self._active_query = None
        self.stats.update(query_context_builds=0, box_evaluations=0,
                          box_plus_constant_evaluations=0,
                          sparse_dual_evaluations=0, dense_dual_evaluations=0,
                          skipped_zero_multiplier_rows=0, residual_matrix_terms=0)

    def query_context(self, objective):
        raw = _floats(objective)
        if len(raw) != self.d:
            raise ValueError('objective dimension differs from trusted view')
        exact = tuple(Fraction.from_float(x) for x in raw)
        context = QueryContext(self.digest, raw, exact)
        self._register('_issued_queries', context, (raw, exact, None, _vector_signature(exact)))
        self.stats['query_context_builds'] += 1
        return context

    def owns_query(self, context, objective):
        if type(context) is not QueryContext:
            return False
        saved = self._issued_queries.get(id(context))
        if saved is None or saved[0]() is not context:
            return False
        raw, exact, _, signature = saved[1]
        try:
            requested = _floats(objective)
            intact = _vector_signature(context.exact) == signature
        except (AttributeError, TypeError, ValueError, ArithmeticError):
            return False
        return (context.view_digest == self.digest and context.raw is raw
                and context.exact is exact
                and intact and requested == raw)

    def _box(self, context):
        # Called only after the complete context ownership/binding check. Keep
        # this cache in the issuer record, not in a caller-supplied token. A
        # full-hot dense dual need not evaluate an unused box first.
        reference, (raw, exact, cached, signature) = self._issued_queries[id(context)]
        if cached is None:
            view = self.view_checker
            cached = sum((c * (lo if c >= 0 else hi)
                          for c, lo, hi in zip(exact, view.lower, view.upper)), ZERO)
            self._issued_queries[id(context)] = (reference, (raw, exact, cached, signature))
            self.stats['box_evaluations'] += 1
        return cached

    def _context(self, objective, context=None):
        # Consume a possible one-shot iterator once, then compare all entries.
        raw = _floats(objective)
        if len(raw) != self.d:
            raise ValueError('objective dimension differs from trusted view')
        selected = self._active_query if context is None else context
        if selected is None:
            return self.query_context(raw)
        if not self.owns_query(selected, raw):
            raise ValueError('query context is not locally issued and bound to this objective')
        return selected

    @contextmanager
    def query_scope(self, objective):
        """One synchronous request; nested use is rejected, not thread-safe."""
        if self._active_query is not None:
            raise RuntimeError('query context scope is already active')
        context = self.query_context(objective)
        self._active_query = context
        try:
            yield context
        finally:
            self._active_query = None

    @staticmethod
    def _dual_record(state):
        return (state.base_lower, state.base_upper, state.constant,
                state.support, state.dense_offset, state.mode)

    @staticmethod
    def _dual_signature(state):
        return (_fraction_signature(state.constant),
                tuple((i, _fraction_signature(value)) for i, value in state.support),
                None if state.dense_offset is None else _vector_signature(state.dense_offset))

    def owns_dual(self, state, lower_multipliers, upper_multipliers):
        if type(state) is not ExactDualState:
            return False
        saved = self._issued_duals.get(id(state))
        if saved is None or saved[0]() is not state:
            return False
        try:
            lo, hi = _floats(lower_multipliers), _floats(upper_multipliers)
            original, signature = saved[1]
            intact = self._dual_signature(state) == signature
        except (AttributeError, TypeError, ValueError, ArithmeticError):
            return False
        return (state.view_digest == self.digest
                and all(current is original for current, original
                        in zip(self._dual_record(state), original))
                and intact and state.base_lower == lo and state.base_upper == hi)

    def dual_state(self, lower_multipliers, upper_multipliers):
        lo, hi = _floats(lower_multipliers), _floats(upper_multipliers)
        if len(lo) != self.j or len(hi) != self.j or any(x < 0 for x in lo + hi):
            raise ValueError('matching nonnegative dual multipliers required')
        key = (lo, hi)
        state = self._dual_cache.get(key)
        if state is not None:
            if self.owns_dual(state, lo, hi):
                self.stats['dual_cache_hits'] += 1
                return state
            self.stats['invalid_cache_entries'] += 1
            del self._dual_cache[key]
        constant, residual = ZERO, {}
        view = self.view_checker
        for row, a, b, low, high in zip(view.rows, view.rhs_lower, view.rhs_upper, lo, hi):
            if low == high == 0:
                self.stats['skipped_zero_multiplier_rows'] += 1
                continue
            l = ZERO if low == 0 else Fraction.from_float(low)
            h = ZERO if high == 0 else Fraction.from_float(high)
            # Equal nonzero multipliers still contribute a*l-b*h.
            constant += a * l - b * h
            difference = h - l
            if not difference:
                continue
            for i, coefficient in row:
                residual[i] = residual.get(i, ZERO) + difference * coefficient
                self.stats['residual_matrix_terms'] += 1
        support = tuple((i, value) for i, value in sorted(residual.items()) if value)
        # A fixed arithmetic dispatch, not a learned/cost-planning policy:
        # sparse correction uses two box terms per retained coordinate.
        if not support:
            mode, dense = 'box', None
        elif 2 * len(support) < self.d:
            mode, dense = 'sparse', None
        else:
            mode = 'dense'
            dense = tuple(residual[i] if i in residual else Fraction(0) for i in range(self.d))
        # A zero-row/all-zero dual otherwise retains the shared start value.
        # Each emitted constant is a fresh Fraction, including exact zero.
        constant = Fraction(constant.numerator, constant.denominator)
        state = ExactDualState(self.digest, lo, hi, constant, support, dense, mode)
        self._register('_issued_duals', state, (self._dual_record(state), self._dual_signature(state)))
        self._remember(self._dual_cache, key, state)
        self.stats['dual_preparations'] += 1
        return state

    def owns_primal(self, state, point=None):
        if type(state) is not CheckedPrimal:
            return False
        saved = self._issued_points.get(id(state))
        if saved is None or saved[0]() is not state:
            return False
        raw, exact, signature = saved[1]
        try:
            intact = _vector_signature(state.point) == signature
            requested = None if point is None else _floats(point)
        except (AttributeError, TypeError, ValueError, ArithmeticError):
            return False
        return (state.view_digest == self.digest and state.raw is raw
                and state.point is exact and intact
                and (requested is None or requested == raw))

    def primal_state(self, point):
        """Original whole-view primal check plus primitive mutation guards."""
        raw = _floats(point)
        if len(raw) != self.d:
            raise ValueError('primal dimension differs from trusted view')
        state = self._point_cache.get(raw)
        if state is not None:
            if self.owns_primal(state, raw):
                self.stats['primal_cache_hits'] += 1
                return state
            self.stats['invalid_cache_entries'] += 1
            del self._point_cache[raw]
        exact = tuple(Fraction.from_float(value) for value in raw)
        self.stats['primal_exact_checks'] += 1
        if not self.view_checker.is_feasible(exact, already_rational=True):
            raise ValueError('primal point does not exactly satisfy the whole trusted view')
        state = CheckedPrimal(self.digest, raw, exact)
        self._register('_issued_points', state, (raw, exact, _vector_signature(exact)))
        self._remember(self._point_cache, raw, state)
        return state

    def _evaluate(self, context, state, factor):
        view = self.view_checker
        if factor == 0 or not state.support:
            self.stats['box_plus_constant_evaluations'] += 1
            return self._box(context) + factor * state.constant
        if state.mode == 'sparse':
            self.stats['sparse_dual_evaluations'] += 1
            value = self._box(context) + factor * state.constant
            for i, residual in state.support:
                ci = context.exact[i]
                z = ci + factor * residual
                value += z * (view.lower[i] if z >= 0 else view.upper[i])
                value -= ci * (view.lower[i] if ci >= 0 else view.upper[i])
            return value
        self.stats['dense_dual_evaluations'] += 1
        value = factor * state.constant
        for ci, residual, lo, hi in zip(context.exact, state.dense_offset, view.lower, view.upper):
            z = ci + factor * residual
            value += z * (lo if z >= 0 else hi)
        return value

    def exact_bound(self, objective, state, scale=1., *, query_context=None):
        """An exact Fraction value; rejects foreign/modified local-state tokens."""
        context = self._context(objective, query_context)
        if (type(state) is not ExactDualState
                or not self.owns_dual(state, state.base_lower, state.base_upper)):
            raise ValueError('locally issued unmodified dual state required')
        factor = _floats([scale])[0]
        if factor < 0:
            raise ValueError('nonnegative dual scale required')
        return self._evaluate(context, state, Fraction.from_float(factor))

    def lower_proof(self, objective, lower_multipliers, upper_multipliers, *, threshold,
                    scale=None, prepared=None, query_context=None):
        context = self._context(objective, query_context)
        delta = _floats([threshold])[0]
        lo, hi = _floats(lower_multipliers), _floats(upper_multipliers)
        if len(lo) != self.j or len(hi) != self.j or any(x < 0 for x in lo + hi):
            raise ValueError('matching nonnegative dual multipliers required')
        factor = 1. if scale is None else _floats([scale])[0]
        if factor < 0:
            raise ValueError('nonnegative dual scale required')
        if prepared is not None and self.owns_dual(prepared, lo, hi):
            state = prepared
            self.stats['prepared_state_reuses'] += 1
        else:
            if prepared is not None:
                self.stats['invalid_cache_entries'] += 1
            state = self.dual_state(lo, hi)
        exact = self._evaluate(context, state, Fraction.from_float(factor))
        value = math.nextafter(float(exact), -math.inf)
        if not math.isfinite(value):
            raise ValueError('finite lower bound required')
        result = {'format': 'qbridge-dual-proof-v1' if scale is None else 'qbridge-scaled-dual-proof-v1',
                  'view_digest': self.digest, 'objective': list(context.raw),
                  'lower_multipliers': list(state.base_lower), 'upper_multipliers': list(state.base_upper),
                  'claimed_lower': value, 'decision_threshold': delta}
        if scale is not None:
            result['dual_scale'] = factor
        return result

    def upper_proof(self, objective, state, *, query_context=None):
        context = self._context(objective, query_context)
        if not self.owns_primal(state):
            raise ValueError('locally issued whole-view feasible primal state required')
        exact = sum((c * p for c, p in zip(context.exact, state.point)), ZERO)
        return {'point': list(state.raw), 'claimed_upper': upper_float(exact)}

    def verify(self, proof, *, expected_objective, expected_threshold):
        """Same wire schema; reconstruct evidence without producer tokens."""
        if not isinstance(proof, dict) or proof.get('format') != FORMAT or proof.get('view_digest') != self.digest:
            raise ValueError('threshold proof is not bound to the trusted view')
        context = self._context(expected_objective)
        delta = _floats([expected_threshold])[0]
        if (_floats(proof['objective']) != context.raw
                or _floats([proof['decision_threshold']])[0] != delta):
            raise ValueError("threshold proof differs from the caller's requested query")
        lower_record = proof['lower_proof']
        if (not isinstance(lower_record, dict) or lower_record.get('format') not in LOWER_FORMATS
                or lower_record.get('view_digest') != self.digest
                or _floats(lower_record['objective']) != context.raw
                or _floats([lower_record['decision_threshold']])[0] != delta):
            raise ValueError('lower proof is not bound to the requested view/query/threshold')
        lower = Fraction.from_float(_floats([lower_record['claimed_lower']])[0])
        state = self.dual_state(lower_record['lower_multipliers'], lower_record['upper_multipliers'])
        scale = lower_record['dual_scale'] if lower_record['format'] == 'qbridge-scaled-dual-proof-v1' else 1.
        exact_lower = self.exact_bound(context.raw, state, scale, query_context=context)
        if lower > exact_lower:
            raise ValueError('claimed lower bound exceeds the exact weak dual')
        upper = None
        upper_record = proof.get('upper_proof')
        if upper_record is not None:
            if not isinstance(upper_record, dict):
                raise ValueError('upper proof must contain a point and its bound')
            primal = self.primal_state(upper_record['point'])
            exact_upper = sum((c * p for c, p in zip(context.exact, primal.point)), ZERO)
            upper = Fraction.from_float(_floats([upper_record['claimed_upper']])[0])
            if upper != Fraction.from_float(upper_float(exact_upper)):
                raise ValueError('upper bound is not the outward-rounded exact feasible objective')
            if lower > exact_upper:
                raise ValueError('lower and feasible upper evidence contradict each other')
        exact_delta = Fraction.from_float(delta)
        decision = True if lower > exact_delta else False if upper is not None and upper <= exact_delta else None
        declared = proof.get('robust_threshold_decision', 'missing')
        if declared is not None and type(declared) is not bool:
            raise ValueError('threshold decision must be true, false, or null')
        if declared is not decision:
            raise ValueError('declared threshold decision does not follow from its evidence')
        return {'valid': True, 'valid_lp_lower_bound': True, 'valid_lp_upper_bound': upper is not None,
                'view_digest': self.digest, 'lower_bound': float(lower),
                'upper_bound': None if upper is None else float(upper),
                'robust_threshold_decision': decision, 'scientific_assumptions_verified': False,
                'negative_scope': 'minimum over this represented confidence view only',
                'requested_query_checked': True}

    def cache_payload_bytes(self, retained_states=()):
        """Approximate retained Python payload, excluding view/native/registries."""
        seen = set()
        def size(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            total = sys.getsizeof(value)
            if isinstance(value, Fraction):
                total += size(value.numerator) + size(value.denominator)
            elif isinstance(value, (tuple, list)):
                total += sum(size(x) for x in value)
            elif isinstance(value, dict):
                total += sum(size(k) + size(v) for k, v in value.items())
            elif isinstance(value, (CheckedPrimal, ExactDualState, QueryContext)):
                total += sum(size(getattr(value, field.name)) for field in fields(value))
                table = (self._issued_queries if type(value) is QueryContext else
                         self._issued_duals if type(value) is ExactDualState else self._issued_points)
                issued = table.get(id(value))
                if issued is not None and issued[0]() is value:
                    # Count immutable mutation snapshots and lazy box payload;
                    # weak-reference/dictionary management is excluded.
                    total += size(issued[1])
            return total
        return size(self._point_cache) + size(self._dual_cache) + size(retained_states)


def _initialize(engine, view, *, last_only, simplex_strategy):
    """Local initialization only; matches the two inherited native schedules."""
    begin = time.perf_counter()
    if type(simplex_strategy) is not int or simplex_strategy not in (0, 1):
        raise ValueError('simplex_strategy must be choose=0 or serial dual=1')
    engine._view = json.loads(json.dumps(view, allow_nan=False))
    engine.checker = OptimizedThresholdChecker(engine._view, cache_size=0)
    engine.digest, engine.d, engine.j = engine.checker.digest, engine.checker.d, engine.checker.j
    engine.plan, engine.bank_size = ('threshold_hot', 1) if last_only else ('full_hot', 0)
    engine.simplex_strategy = simplex_strategy
    engine.screening, engine.prepared_reuse = False, True
    engine._use_primal_pool = engine._use_dual_pool = False
    engine.primal_retention_capacity = engine.dual_retention_capacity = int(last_only)
    engine._initial_raw = tuple(engine._view['feasible_point'])
    engine.initial_primal = engine.checker.primal_state(engine._initial_raw)
    engine.last_primal = engine.last_dual = None
    engine.last_primal_generation = engine.last_dual_generation = 0
    engine.primal_bank, engine.dual_bank = [], []
    engine.maximum_primal_bank_entries = engine.maximum_dual_bank_entries = 0
    engine._zeros = (0.,) * engine.j
    if last_only:
        engine.zero = Candidate(engine._zeros, engine._zeros)
    else:
        engine._zero_dual = engine.checker.dual_state(engine._zeros, engine._zeros)
    engine._lower = np.asarray(engine._view['lower'], dtype=float)
    engine._upper = np.asarray(engine._view['upper'], dtype=float)
    engine.indices = np.arange(engine.d, dtype=np.int32)
    engine.h = highspy.Highs()
    engine.options = {'output_flag': False, 'threads': 1, 'solver': 'simplex',
                      'simplex_strategy': simplex_strategy,
                      'primal_feasibility_tolerance': 1e-10, 'dual_feasibility_tolerance': 1e-10}
    for key, value in engine.options.items():
        if engine.h.setOptionValue(key, value) != highspy.HighsStatus.kOk:
            raise RuntimeError(f'HiGHS rejected {key}')
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = engine.d, engine.j
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = np.zeros(engine.d), engine._lower, engine._upper
    lp.row_lower_ = np.asarray(engine._view['rhs_lower'], dtype=float)
    lp.row_upper_ = np.asarray(engine._view['rhs_upper'], dtype=float)
    source = engine._view['matrix']
    matrix = highspy.HighsSparseMatrix()
    matrix.num_col_, matrix.num_row_, matrix.format_ = engine.d, engine.j, highspy.MatrixFormat.kRowwise
    matrix.start_ = np.asarray(source['indptr'], dtype=np.int32)
    matrix.index_ = np.asarray(source['indices'], dtype=np.int32)
    matrix.value_ = np.asarray(source['data'], dtype=float)
    lp.a_matrix_ = matrix
    if engine.h.passModel(lp) != highspy.HighsStatus.kOk:
        raise RuntimeError('HiGHS rejected trusted view')
    engine.native_version = engine.h.version()
    engine.initialization_seconds = time.perf_counter() - begin


class _QueryContextEngine:
    def query(self, objective, threshold=.001):
        begin = time.perf_counter()
        with self.checker.query_scope(objective) as context:
            preparation = time.perf_counter() - begin
            answer = super().query(context.raw, threshold)
        # The inherited stopwatch starts after context preparation. Charge it
        # explicitly; no setup work is moved outside reported query cost.
        answer['query_context_seconds'] = preparation
        answer['verify_seconds'] += preparation
        answer['seconds'] = time.perf_counter() - begin
        return answer


class OptimizedCheckingScheduleEngine(_QueryContextEngine, CheckingScheduleEngine):
    """Original S0P1 candidate order, last-only retention and native settings."""
    def __init__(self, view, *, simplex_strategy=1):
        _initialize(self, view, last_only=True, simplex_strategy=simplex_strategy)


class OptimizedNativeFullHotEngine(_QueryContextEngine, ThresholdEngine):
    """Persistent native basis, full solve, no cross-objective candidates.

    Inherited full_hot semantics disable objective cutoff. The initial checked
    witness and zero dual remain fallback evidence, not a query-result cache.
    """
    def __init__(self, view, *, simplex_strategy=1):
        _initialize(self, view, last_only=False, simplex_strategy=simplex_strategy)


def make_engine(name, view, *, simplex_strategy=1):
    constructors = {'optimized_last': OptimizedCheckingScheduleEngine,
                    'native_full_hot': OptimizedNativeFullHotEngine}
    if name not in constructors:
        raise ValueError('name must be optimized_last or native_full_hot')
    return constructors[name](view, simplex_strategy=simplex_strategy)
