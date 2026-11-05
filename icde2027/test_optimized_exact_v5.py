"""Toy-only exact arithmetic/security/API checks; no real input or timing study."""
import copy
from dataclasses import replace
from fractions import Fraction
import gc
import math
import random
import weakref

import pytest

from icde2027.checking_schedule_v4 import CheckingScheduleEngine
from icde2027.optimized_exact_v5 import (
    ExactDualState, OptimizedCheckingScheduleEngine, OptimizedNativeFullHotEngine,
    OptimizedThresholdChecker, make_engine,
)
from icde2027.primary_aligned import service_response
from icde2027.threshold_extension.engine import ThresholdEngine
from icde2027.threshold_extension.proofs import ThresholdChecker, bundle_proof


def matrix_view(d, rows, point=None):
    """Build dyadic toy constraints with an exactly feasible supplied witness."""
    point = [.5] * d if point is None else point
    ids, values, ptr, lower, upper = [], [], [0], [], []
    for row in rows:
        center = sum(Fraction.from_float(float(a)) * Fraction.from_float(float(point[i]))
                     for i, a in row)
        lower.append(float(center - Fraction(1, 4)))
        upper.append(float(center + Fraction(1, 4)))
        ids.extend(i for i, _ in row)
        values.extend(float(a) for _, a in row)
        ptr.append(len(ids))
    return {'format': 'qbridge-confidence-view-v1', 'lower': [0.] * d, 'upper': [1.] * d,
            'rhs_lower': lower, 'rhs_upper': upper,
            'matrix': {'indptr': ptr, 'indices': ids, 'data': values},
            'feasible_point': list(point)}


@pytest.fixture
def view():
    return matrix_view(2, [[(0, 1), (1, 1)]])


def independent_bound(view, objective, lower, upper, scale):
    """Dense Fraction oracle, independent of prepared/query-context helpers."""
    f = lambda x: Fraction.from_float(float(x))
    c = list(map(f, objective))
    constant = Fraction(0)
    matrix = view['matrix']
    for row in range(len(lower)):
        lo, hi = f(lower[row]), f(upper[row])
        constant += f(view['rhs_lower'][row]) * lo - f(view['rhs_upper'][row]) * hi
        for k in range(matrix['indptr'][row], matrix['indptr'][row + 1]):
            c[matrix['indices'][k]] += f(scale) * f(matrix['data'][k]) * (hi - lo)
    value = f(scale) * constant
    for z, lo, hi in zip(c, view['lower'], view['upper']):
        value += min(z * f(lo), z * f(hi))
    return value


def verified_by_both(view, answer, objective, threshold):
    proof = answer['certificate']
    assert proof is not None
    a = ThresholdChecker(view, cache_size=0).verify(
        proof, expected_objective=objective, expected_threshold=threshold)
    b = OptimizedThresholdChecker(view, cache_size=0).verify(
        proof, expected_objective=objective, expected_threshold=threshold)
    assert a == b
    return a


def test_zero_dual_skips_all_matrix_terms_and_equal_nonzero_keeps_constant():
    v = matrix_view(8, [[(0, 1)], [(2, .5), (2, -.5)], [(i, .25) for i in range(8)]])
    checker = OptimizedThresholdChecker(v, cache_size=0)
    c = [.1, -.2, .3, 0., 1., -1., .125, -.125]
    context = checker.query_context(c)
    zero = checker.dual_state([0, 0, 0], [0, 0, 0])
    assert zero.support == () and zero.mode == 'box'
    assert checker.stats['residual_matrix_terms'] == 0
    assert checker.stats['skipped_zero_multiplier_rows'] == 3
    box = independent_bound(v, c, [0, 0, 0], [0, 0, 0], 1)
    assert checker.exact_bound(c, zero, query_context=context) == box
    equal = checker.dual_state([2, 0, 0], [2, 0, 0])
    assert equal.support == () and equal.constant == -1
    assert checker.exact_bound(c, equal, query_context=context) == box - 1
    assert checker.stats['residual_matrix_terms'] == 0
    assert checker.stats['box_evaluations'] == 1


def test_sparse_dense_and_exact_cancellation_dispatch():
    v = matrix_view(8, [[(0, 1)], [(2, .5), (2, -.5)], [(i, .25) for i in range(8)]])
    checker = OptimizedThresholdChecker(v)
    c = [.3, -.1, 0., 1., -1., .5, -.25, .2]
    context = checker.query_context(c)
    sparse = checker.dual_state([1, 1, 0], [0, 0, 0])
    assert sparse.mode == 'sparse' and sparse.support == ((0, Fraction(-1)),)
    assert sparse.dense_offset is None
    dense = checker.dual_state([0, 0, 1], [0, 0, 0])
    assert dense.mode == 'dense' and len(dense.dense_offset) == 8
    for state in (sparse, dense):
        for scale in (0, .125, 1, 1.1, 4):
            got = checker.exact_bound(c, state, scale, query_context=context)
            want = independent_bound(v, c, state.base_lower, state.base_upper, scale)
            assert got == want
    assert checker.stats['query_context_builds'] == checker.stats['box_evaluations'] == 1


def test_seeded_small_random_programs_match_independent_fraction_oracle():
    rng = random.Random(50108)
    for _ in range(80):
        d, j = rng.randint(1, 12), rng.randint(0, 6)
        point = [rng.choice([0., .25, .5, .75, 1.]) for _ in range(d)]
        rows = [[(rng.randrange(d), rng.choice([-.5, 0., .25, .5, 1.]))
                 for _ in range(rng.randint(0, d + 2))] for _ in range(j)]
        v = matrix_view(d, rows, point)
        checker = OptimizedThresholdChecker(v, cache_size=0)
        c = [rng.choice([-.7, -.125, -0., .1, .25, .9]) for _ in range(d)]
        lo = [rng.choice([0., 0., .1, .25, 1.]) for _ in range(j)]
        hi = [rng.choice([0., 0., .2, .5, 1.]) for _ in range(j)]
        context, state = checker.query_context(c), checker.dual_state(lo, hi)
        for scale in (0., .25, 1., 1.1, 4.):
            expected = independent_bound(v, c, lo, hi, scale)
            assert checker.exact_bound(c, state, scale, query_context=context) == expected
            proof = checker.lower_proof(c, lo, hi, threshold=.001, scale=scale,
                                        prepared=state, query_context=context)
            assert proof['claimed_lower'] == math.nextafter(float(expected), -math.inf)
        assert checker.stats['query_context_builds'] == 1
        assert checker.stats['dual_preparations'] == 1


@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf, True, '1', 10 ** 400])
def test_reject_nonfinite_or_nonnumeric_objective_multiplier_threshold_scale(view, bad):
    checker = OptimizedThresholdChecker(view)
    with pytest.raises((ValueError, TypeError, ArithmeticError)):
        checker.query_context([bad, 1.])
    with pytest.raises((ValueError, TypeError, ArithmeticError)):
        checker.dual_state([bad], [0])
    with pytest.raises((ValueError, TypeError, ArithmeticError)):
        checker.lower_proof([1, 1], [0], [0], threshold=bad)
    with pytest.raises((ValueError, TypeError, ArithmeticError)):
        checker.lower_proof([1, 1], [0], [0], threshold=0, scale=bad)


def test_reject_wrong_dimensions_and_negative_multipliers_even_at_zero_scale(view):
    checker = OptimizedThresholdChecker(view)
    for c in ([], [1], [1, 1, 1]):
        with pytest.raises(ValueError):
            checker.query_context(c)
    for lo, hi in (([], []), ([0, 0], [0, 0]), ([-1], [0])):
        with pytest.raises(ValueError):
            checker.lower_proof([1, 1], lo, hi, threshold=0, scale=0)
    with pytest.raises(ValueError):
        checker.lower_proof([1, 1], [0], [0], threshold=0, scale=-1)


def test_whole_view_witness_and_upper_feasibility_are_not_skipped(view):
    bad = copy.deepcopy(view)
    bad['feasible_point'] = [0., 0.]  # Inside the box, outside the moment slab.
    with pytest.raises(ValueError, match='whole-view'):
        OptimizedThresholdChecker(bad)
    checker = OptimizedThresholdChecker(view)
    with pytest.raises(ValueError, match='whole trusted view'):
        checker.primal_state([0, 0])
    malformed = copy.deepcopy(view)
    malformed['matrix']['indices'][0] = 2
    with pytest.raises(ValueError, match='sparse matrix'):
        OptimizedThresholdChecker(malformed)


@pytest.mark.parametrize('field,value', [
    ('raw', (9., 9.)), ('exact', (Fraction(9), Fraction(9))),
    ('view_digest', 'forged'),
])
def test_modified_query_context_rejected(view, field, value):
    checker = OptimizedThresholdChecker(view)
    context = checker.query_context([1., 1.])
    object.__setattr__(context, field, value)
    assert not checker.owns_query(context, [1, 1])
    with pytest.raises(ValueError, match='query context'):
        checker.lower_proof([1, 1], [0], [0], threshold=0, query_context=context)


def test_context_requires_owner_complete_objective_and_survives_input_mutation(view):
    checker, other = OptimizedThresholdChecker(view), OptimizedThresholdChecker(view)
    original = [1., -1.]
    context = checker.query_context(original)
    original[1] = 1.
    assert checker.owns_query(context, [1, -1])
    for invalid in (context, replace(context)):
        with pytest.raises(ValueError, match='query context'):
            other.lower_proof([1, -1], [0], [0], threshold=0, query_context=invalid)
    with pytest.raises(ValueError, match='query context'):
        checker.lower_proof(original, [0], [0], threshold=0, query_context=context)
    with checker.query_scope([1., -1.]) as active:
        checker.lower_proof([1, -1], [0], [0], threshold=0)
        checker.upper_proof([1, -1], checker.primal_state([.5, .5]))
        assert checker._active_query is active
        with pytest.raises(ValueError, match='query context'):
            checker.lower_proof([1, 1], [0], [0], threshold=0)
        with pytest.raises(RuntimeError, match='already active'):
            with checker.query_scope([1, -1]):
                pass
    assert checker._active_query is None


@pytest.mark.parametrize('field,value', [
    ('constant', Fraction(99)), ('support', ((0, Fraction(99)),)),
    ('dense_offset', (Fraction(99), Fraction(99))), ('mode', 'forged'),
    ('base_lower', (99.,)), ('base_upper', (99.,)), ('view_digest', 'forged'),
])
def test_modified_dual_token_rebuilt_from_checked_raw_multipliers(view, field, value):
    checker = OptimizedThresholdChecker(view, cache_size=1)
    state = checker.dual_state([1], [0])
    object.__setattr__(state, field, value)
    assert not checker.owns_dual(state, [1], [0])
    with pytest.raises(ValueError, match='locally issued'):
        checker.exact_bound([1, 1], state)
    proof = checker.lower_proof([1, 1], [1], [0], threshold=0, prepared=state)
    expected = independent_bound(view, [1, 1], [1], [0], 1)
    assert proof['claimed_lower'] == math.nextafter(float(expected), -math.inf)
    assert checker.stats['invalid_cache_entries'] >= 1
    assert checker.dual_state([1], [0]) is not state


def test_foreign_and_unissued_dual_primal_states_cannot_authorize(view):
    checker, other = OptimizedThresholdChecker(view), OptimizedThresholdChecker(view)
    state = other.dual_state([1], [0])
    for foreign in (state, replace(state), ThresholdChecker(view).dual_state([1], [0])):
        proof = checker.lower_proof([1, 1], [1], [0], threshold=0, prepared=foreign)
        assert proof['claimed_lower'] == math.nextafter(.75, -math.inf)
    with pytest.raises(ValueError, match='locally issued'):
        checker.upper_proof([1, 1], other.primal_state([.5, .5]))
    point = checker.primal_state([.5, .5])
    object.__setattr__(point, 'point', (Fraction(0), Fraction(0)))
    with pytest.raises(ValueError, match='locally issued'):
        checker.upper_proof([1, 1], point)


@pytest.mark.parametrize('c,lo,hi,threshold,expected', [
    ([1, 1], [1], [0], .1, True),
    ([-1, -1], [0], [0], 0, False),
    ([0, 0], [0], [0], 0, False),
    ([1, 1], [0], [0], .1, None),
])
def test_wire_interoperability_all_three_decisions_and_boundary(view, c, lo, hi, threshold, expected):
    checker = OptimizedThresholdChecker(view)
    with checker.query_scope(c):
        lower = checker.lower_proof(c, lo, hi, threshold=threshold, scale=1.)
        upper = checker.upper_proof(c, checker.primal_state([.5, .5]))
    proof = bundle_proof(checker.digest, c, threshold, lower, upper)
    for receiver in (ThresholdChecker(view), OptimizedThresholdChecker(view)):
        result = receiver.verify(proof, expected_objective=c, expected_threshold=threshold)
        assert result['robust_threshold_decision'] is expected
    legacy = ThresholdChecker(view)
    proof = bundle_proof(legacy.digest, c, threshold,
                         legacy.lower_proof(c, lo, hi, threshold=threshold),
                         legacy.upper_proof(c, legacy.primal_state([.5, .5])))
    result = checker.verify(proof, expected_objective=c, expected_threshold=threshold)
    assert result['robust_threshold_decision'] is expected


@pytest.mark.parametrize('attack', [
    'outer_objective', 'outer_threshold', 'outer_view', 'lower_objective',
    'lower_threshold', 'lower_view', 'lower_claim', 'negative_multiplier',
    'negative_scale', 'upper_point', 'upper_claim', 'decision', 'bool_decision',
])
def test_wire_tampering_rejected_by_optimized_receiver(view, attack):
    producer = OptimizedThresholdChecker(view)
    c, delta = [1., 1.], .1
    lower = producer.lower_proof(c, [0], [0], threshold=delta, scale=1.)
    upper = producer.upper_proof(c, producer.primal_state([.5, .5]))
    proof = bundle_proof(producer.digest, c, delta, lower, upper)
    if attack == 'outer_objective': proof['objective'][1] = 2.
    elif attack == 'outer_threshold': proof['decision_threshold'] = 2.
    elif attack == 'outer_view': proof['view_digest'] = 'bad'
    elif attack == 'lower_objective': proof['lower_proof']['objective'][1] = 2.
    elif attack == 'lower_threshold': proof['lower_proof']['decision_threshold'] = 2.
    elif attack == 'lower_view': proof['lower_proof']['view_digest'] = 'bad'
    elif attack == 'lower_claim': proof['lower_proof']['claimed_lower'] = 99.
    elif attack == 'negative_multiplier': proof['lower_proof']['lower_multipliers'][0] = -1.
    elif attack == 'negative_scale': proof['lower_proof']['dual_scale'] = -1.
    elif attack == 'upper_point': proof['upper_proof']['point'] = [0., 0.]
    elif attack == 'upper_claim': proof['upper_proof']['claimed_upper'] = 0.
    elif attack == 'decision': proof['robust_threshold_decision'] = True
    elif attack == 'bool_decision': proof['robust_threshold_decision'] = 0
    with pytest.raises(ValueError):
        OptimizedThresholdChecker(view).verify(proof, expected_objective=c, expected_threshold=delta)


def test_state_registry_weak_ownership_and_fifo_capacity(view):
    checker = OptimizedThresholdChecker(view, cache_size=1)
    context = checker.query_context([1, 1])
    first = checker.dual_state([1], [0])
    old_ref, query_ref = weakref.ref(first), weakref.ref(context)
    second = checker.dual_state([2], [0])
    assert checker.cache_info()['dual_cache_entries'] == 1
    assert checker.owns_dual(first, [1], [0])  # Caller-owned token remains valid.
    del first, context
    gc.collect()
    assert old_ref() is None and query_ref() is None
    assert len(checker._issued_queries) == 0 and len(checker._issued_duals) == 1
    assert checker.cache_payload_bytes([second]) > 0


def test_dense_dual_and_upper_witness_do_not_eagerly_compute_unused_box(view):
    checker = OptimizedThresholdChecker(view)
    with checker.query_scope([1., 1.]) as context:
        state = checker.dual_state([1], [0])
        assert state.mode == 'dense'
        checker.lower_proof([1, 1], [1], [0], threshold=0, prepared=state)
        checker.upper_proof([1, 1], checker.primal_state([.5, .5]))
        assert checker.stats['query_context_builds'] == 1
        assert checker.stats['box_evaluations'] == 0
        checker.lower_proof([1, 1], [0], [0], threshold=0)
        checker.lower_proof([1, 1], [0], [0], threshold=0)
        assert checker.stats['box_evaluations'] == 1
        assert checker.owns_query(context, [1, 1])


def native_diagnostics_without_times(answer):
    return [{k: v for k, v in row.items() if k != 'seconds'} for row in answer['native_calls']]


def test_last_only_adapter_preserves_legacy_service_schedule_and_native_configuration(view):
    legacy = CheckingScheduleEngine(view, screening=False, prepared=True)
    optimized = OptimizedCheckingScheduleEngine(view)
    assert optimized.options == legacy.options
    assert optimized.checker.stats['dual_preparations'] == 0
    assert optimized.checker.stats['primal_exact_checks'] == 1
    objectives = [[1, 1], [1, 1.1], [1.1, 1], [-1, -1], [0, 0], [1, -1], [1, 1]]
    for c in objectives:
        before = optimized.checker.stats['query_context_builds']
        a, b = legacy.query(c, .001), optimized.query(c, .001)
        assert service_response(a) == service_response(b)
        assert native_diagnostics_without_times(a) == native_diagnostics_without_times(b)
        assert a['last_primal_generation'] == b['last_primal_generation']
        assert a['last_dual_generation'] == b['last_dual_generation']
        assert optimized.checker.stats['query_context_builds'] == before + 1
        assert b['dual_candidates_screened'] == b['primal_candidates_screened'] == 0
        assert optimized.checker._active_query is None
        verified_by_both(view, b, c, .001)
    assert optimized.memory_info()['retained_float_duals'] == 0


def test_native_full_hot_adapter_preserves_full_solve_basis_and_no_candidate_retention(view):
    legacy = ThresholdEngine(view, plan='full_hot', bank_size=0)
    optimized = OptimizedNativeFullHotEngine(view)
    assert optimized.options == legacy.options
    for c, delta in [([1, 1], .1), ([1, 1], .2), ([-1, -1], 0.), ([0, 0], 0.)]:
        a, b = legacy.query(c, delta), optimized.query(c, delta)
        assert service_response(a) == service_response(b)
        assert native_diagnostics_without_times(a) == native_diagnostics_without_times(b)
        assert len(b['native_calls']) == 1
        assert b['native_calls'][0]['objective_bound'] is None
        assert optimized.last_dual is optimized.last_primal is None
        assert optimized.dual_bank == optimized.primal_bank == []
        verified_by_both(view, b, c, delta)
    assert optimized.memory_info()['dual_total_retained_entries'] == 0


def test_factory_and_query_scope_cleanup_on_exception(view):
    for name in ('optimized_last', 'native_full_hot'):
        engine = make_engine(name, view)
        with pytest.raises(ValueError):
            engine.query([1, 1], math.nan)
        assert engine.checker._active_query is None
        answer = engine.query([0, 0], 0)
        verified_by_both(view, answer, [0, 0], 0)
    with pytest.raises(ValueError): make_engine('other', view)
    with pytest.raises(ValueError): make_engine('optimized_last', view, simplex_strategy=True)


def test_initial_view_is_detached_from_caller_mutation(view):
    checker = OptimizedThresholdChecker(view)
    view['lower'][0] = .9
    proof = checker.lower_proof([1, 1], [0], [0], threshold=0)
    assert proof['claimed_lower'] == math.nextafter(0., -math.inf)


def test_overflow_cannot_emit_nonfinite_proof():
    v = matrix_view(2, [], [1., 1.])
    v['lower'] = [1., 1.]
    checker = OptimizedThresholdChecker(v)
    with pytest.raises((ValueError, ArithmeticError)):
        checker.lower_proof([float(2 ** 1023)] * 2, [], [], threshold=0)


@pytest.mark.parametrize('field', ['constant', 'sparse_support', 'dense_offset', 'dense_zero', 'zero_constant'])
@pytest.mark.parametrize('slot,value', [('_numerator', 100), ('_denominator', 3), ('_denominator', 0)])
def test_nested_dual_fraction_mutation_rejected_and_rebuilt(field, slot, value):
    d = 8 if field == 'sparse_support' else 2
    v = matrix_view(d, [[(0, 1)]])
    checker = OptimizedThresholdChecker(v)
    lo = [0.] if field == 'zero_constant' else [1.]
    state = checker.dual_state(lo, [0.])
    target = (state.constant if field in ('constant', 'zero_constant') else
              state.support[0][1] if field == 'sparse_support' else
              state.dense_offset[-1] if field == 'dense_zero' else state.dense_offset[0])
    object.__setattr__(target, slot, value)
    assert not checker.owns_dual(state, lo, [0.])
    c = [1.] * d
    with pytest.raises(ValueError, match='unmodified'):
        checker.exact_bound(c, state)
    proof = checker.lower_proof(c, lo, [0.], threshold=.1, prepared=state)
    expected = independent_bound(v, c, lo, [0.], 1.)
    assert proof['claimed_lower'] == math.nextafter(float(expected), -math.inf)
    ThresholdChecker(v).verify(bundle_proof(checker.digest, c, .1, proof),
                               expected_objective=c, expected_threshold=.1)
    assert checker.dual_state(lo, [0.]) is not state
    other = OptimizedThresholdChecker(v)
    assert other.exact_bound(c, other.dual_state([0.], [0.])) == 0


@pytest.mark.parametrize('slot,value', [('_numerator', 100), ('_denominator', 3), ('_denominator', 0)])
def test_nested_query_fraction_mutation_rejected_even_after_lazy_box(slot, value):
    v = matrix_view(2, [[(0, 1)]])
    v['lower'] = [.25, .25]
    checker = OptimizedThresholdChecker(v)
    zero = checker.dual_state([0.], [0.])
    with checker.query_scope([1., 1.]) as context:
        expected = checker.exact_bound([1., 1.], zero)
        assert expected == Fraction(1, 2) and checker.stats['box_evaluations'] == 1
        object.__setattr__(context.exact[0], slot, value)
        assert not checker.owns_query(context, [1., 1.])
        with pytest.raises(ValueError, match='query context'):
            checker.exact_bound([1., 1.], zero)
        with pytest.raises(ValueError, match='query context'):
            checker.upper_proof([1., 1.], checker.primal_state([.5, .5]), query_context=context)
    assert checker.exact_bound([1., 1.], zero) == Fraction(1, 2)


@pytest.mark.parametrize('slot,value', [('_numerator', 0), ('_denominator', 3), ('_denominator', 0)])
def test_nested_primal_fraction_mutation_cannot_authorize_upper(view, slot, value):
    checker = OptimizedThresholdChecker(view)
    state = checker.primal_state([.5, .5])
    object.__setattr__(state.point[0], slot, value)
    assert not checker.owns_primal(state)
    with pytest.raises(ValueError, match='whole-view feasible'):
        checker.upper_proof([1., 1.], state)
    rebuilt = checker.primal_state([.5, .5])
    assert rebuilt is not state and checker.owns_primal(rebuilt)
    assert checker.upper_proof([1., 1.], rebuilt)['claimed_upper'] == 1.


def test_returned_exact_value_does_not_alias_private_box_or_other_tokens(view):
    checker = OptimizedThresholdChecker(view, cache_size=0)
    first = checker.dual_state([0.], [0.])
    second = checker.dual_state([0.], [0.])
    assert first.constant is not second.constant
    context = checker.query_context([1., 1.])
    result = checker.exact_bound([1., 1.], first, query_context=context)
    object.__setattr__(result, '_numerator', 100)
    assert checker.exact_bound([1., 1.], first, query_context=context) == 0
    object.__setattr__(first.constant, '_numerator', 100)
    assert not checker.owns_dual(first, [0.], [0.])
    assert checker.owns_dual(second, [0.], [0.])
    assert checker.exact_bound([1., 1.], second, query_context=context) == 0
