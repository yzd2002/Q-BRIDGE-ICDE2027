"""Development checks of the opt-in representation map, not a paper experiment."""
from fractions import Fraction as F
import itertools
import json
import math

import numpy as np
import pytest
from scipy import sparse

from . import represented_intervals_v4 as v4
from .proof_check import CheckedView


def rational(value):
    return F.from_float(float(value))


@pytest.mark.parametrize('value', [F(1, 3), F(-1, 3), F(1), F(0),
                                  F(1, 2**1075), F(-1, 2**1075),
                                  F(2**53 + 1, 2**53), F(2**53 - 1, 2**53)])
def test_outward_conversion_is_tight_for_one_exact_endpoint(value):
    lo = v4.outward_float(value, lower=True)
    hi = v4.outward_float(value, lower=False)
    assert rational(lo) <= value <= rational(hi)
    assert rational(math.nextafter(lo, math.inf)) > value
    assert rational(math.nextafter(hi, -math.inf)) < value


@pytest.mark.parametrize('value', [F(2**1025), F(-2**1025), float('inf'), float('nan')])
def test_unrepresentable_endpoints_refused(value):
    with pytest.raises(ValueError):
        v4.outward_float(value, lower=True)


def test_point_two_row_has_exact_kappa_and_normalized_ideal_row():
    matrix, (row,) = v4.row_representations([[.2]*5])
    assert row.kappa == F(18014398509481985, 18014398509481984)
    assert row.ideal_weights == (F(1, 5),)*5
    assert np.array_equal(matrix.toarray(), [[.2]*5])
    assert matrix.data is not np.asarray([[.2]*5])


@pytest.mark.parametrize('all_success', [False, True])
@pytest.mark.parametrize('method', ['group', 'projection'])
def test_five_point_two_boundary_counterexample_is_enclosed(all_success, method):
    counts = np.zeros((1, 5, 2), dtype=int)
    counts[:, :, int(all_success)] = 10
    rows = np.array([[.2]*5])
    if method == 'group':
        mapped, info = v4.randomized_group_intervals_v4(
            counts, rows, [[.2]*5], .05, q_semantics='normalize_weights')
        assert info['rows'][0]['construction'] == 'clopper_pearson'
    else:
        mapped, _ = v4.randomized_cell_intervals_v4(counts, rows, .05)
    truth = sum((rational(x)*int(all_success) for x in rows[0]), F())
    assert rational(mapped.lower[0]) <= truth <= rational(mapped.upper[0])
    if all_success:
        assert truth > 1
        assert mapped.upper[0] == math.nextafter(1., math.inf)
    else:
        assert mapped.lower[0] == 0


@pytest.mark.parametrize('all_success', [False, True])
@pytest.mark.parametrize('method', ['group', 'projection'])
def test_sparse_overlapping_nonuniform_rows_at_binary_boundaries(all_success, method):
    rows = np.array([[0, .7, .3, 0, 0], [.1, .2, .3, .4, 0],
                     [0, 0, 0, 0, 0], [0, 2, .1, 0, 0]])
    counts = np.zeros((1, 5, 2), int)
    counts[:, :, int(all_success)] = 10
    if method == 'group':
        mapped, _ = v4.randomized_group_intervals_v4(counts, rows, [[F(1, 5)]*5], .05,
                                                     q_semantics='exact_probabilities')
    else:
        mapped, _ = v4.randomized_cell_intervals_v4(counts, rows, .05)
    for j, row in enumerate(rows):
        truth = sum((rational(w)*int(all_success) for w in row), F())
        assert rational(mapped.lower[j]) <= truth <= rational(mapped.upper[j])
    assert mapped.lower[2] == mapped.upper[2] == 0


def test_exact_vertex_enumeration_sparse_overlapping_nonuniform_projection():
    # Includes a zero row, overlapping support, unnormalized and tiny weights.
    rows = sparse.csr_matrix([[.2, .2, .2, .2], [0, .7, .3, 0],
                              [2., 0, .1, 0], [0, 0, 0, 0],
                              [2.**-1074, 0, 0, 0]])
    lower = [F(1, 7), F(1, 9), F(1, 11), F(1, 13)]
    upper = [F(6, 7), F(8, 9), F(10, 11), F(12, 13)]
    rho = [F(1, 17), F(0), F(2, 19), F(1), F(0)]
    mapped = v4.project_cell_intervals(rows, lower, upper, rho=rho)
    dense = [[rational(w) for w in row] for row in rows.toarray()]
    values = [[] for _ in dense]
    for vertex in itertools.product(*zip(lower, upper)):
        for j, weights in enumerate(dense):
            value = sum((w*x for w, x in zip(weights, vertex)), F())
            for sign in (-1, 1):
                transported = value + sign*sum(weights)*rho[j]
                values[j].append(transported)
                assert rational(mapped.lower[j]) <= transported <= rational(mapped.upper[j])
    for j in range(len(dense)):
        assert mapped.exact_lower[j] == min(values[j])
        assert mapped.exact_upper[j] == max(values[j])
    assert mapped.lower[3] == mapped.upper[3] == 0


def test_transport_units_are_scaled_by_exact_kappa():
    mapped = v4.map_normalized_intervals([[.2]*5, [2, 0, 0, 0, 0]],
                                          [F(1, 4)]*2, [F(3, 4)]*2, rho=F(1, 7))
    for j, kappa in enumerate(mapped.kappas):
        assert mapped.exact_lower[j] == kappa*F(3, 28)
        assert mapped.exact_upper[j] == kappa*F(25, 28)
    assert mapped.kappas[1] == 2
    json.dumps(mapped.record(), allow_nan=False)


def test_zero_rows_and_empty_row_set_have_deterministic_images():
    mapped = v4.map_normalized_intervals([[0, 0]], 0, 1, rho=F(7))
    assert mapped.lower == mapped.upper == (0.,)
    empty, _ = v4.randomized_cell_intervals_v4(np.zeros((1, 2, 2), int),
                                              sparse.csr_matrix((0, 2)), .05)
    assert empty.lower == empty.upper == ()


@pytest.mark.parametrize('rows', [[[-1., 1.]], [[float('nan'), 1.]],
                                 [[float('inf'), 1.]], [[1+1j, 0]], [["x", "y"]]])
def test_invalid_stored_rows_refused(rows):
    with pytest.raises(ValueError):
        v4.row_representations(rows)


def test_duplicate_sparse_entries_are_not_silently_rounded_into_one():
    duplicate = sparse.csr_matrix(([.2, .2], [0, 0], [0, 2]), shape=(1, 2))
    assert not duplicate.has_canonical_format
    with pytest.raises(ValueError, match='canonical CSR'):
        v4.row_representations(duplicate)


@pytest.mark.parametrize('method', [v4.project_cell_intervals, v4.map_normalized_intervals])
def test_empty_statistical_intersection_is_not_repaired_by_transport(method):
    with pytest.raises(v4.UnusableStatisticalInterval):
        method([[1.]], [.8], [.2], rho=1)
    with pytest.raises(ValueError):
        method([[1.]], [0], [1], rho=-.1)


def test_probability_semantics_are_explicit_and_exact():
    with pytest.raises(TypeError):
        v4.canonical_probabilities([[.5, .5]])
    with pytest.raises(ValueError, match='sum exactly'):
        v4.canonical_probabilities([[.2]*5], semantics='exact_probabilities')
    assert v4.canonical_probabilities([[.2]*5], semantics='normalize_weights') == ((F(1, 5),)*5,)
    assert v4.canonical_probabilities([[F(1, 3), F(2, 3)]], semantics='exact_probabilities') == ((F(1, 3), F(2, 3)),)
    with pytest.raises(ValueError):
        v4.canonical_probabilities([[0, 0]], semantics='normalize_weights')


def test_enumerated_score_expectations_for_nonuniform_overlapping_groups():
    rows = np.array([[.7, .2, .1], [.3, 0, .7], [0, .2, .8], [0, 0, 0]])
    q = [[F(1, 2), F(1, 3), F(1, 6)]]
    statistics = v4.exact_group_score_statistics(np.zeros((1, 3, 2), int), rows, q,
                                                q_semantics='exact_probabilities')
    for outcomes in itertools.product((0, 1), repeat=3):
        for raw, record in zip(rows, statistics):
            active = [i for i, w in enumerate(raw) if w]
            if not active:
                assert record['zero_row']
                continue
            # Independently enumerate original assignment probabilities and
            # condition on inclusion in this group.
            mass = sum(q[0][i] for i in active)
            expectation = sum(q[0][i]/mass * weight*outcomes[i]
                              for i, weight in zip(active, record['multipliers']))
            target = sum(rational(raw[i])*outcomes[i] for i in active)/sum(rational(raw[i]) for i in active)
            assert expectation == target


def test_score_statistics_match_individually_expanded_observations():
    record, = v4.exact_group_score_statistics([[[1, 2], [1, 1]]], [[.25, .75]],
                                             [[.5, .5]], q_semantics='exact_probabilities')
    sample = [F(0), F(1, 2), F(1, 2), F(0), F(3, 2)]
    mean = sum(sample)/len(sample)
    variance = sum((x-mean)**2 for x in sample)/(len(sample)-1)
    assert record['mean'] == mean == F(1, 2)
    assert record['variance'] == variance == F(3, 8)
    assert record['total_sq'] == sum(x*x for x in sample)


def test_cp_branch_uses_exact_normalized_multipliers_not_float_equality():
    counts = np.array([[[10, 10], [10, 10]]])
    _, cp = v4.randomized_group_intervals_v4(counts, [[.3, .7]], [[.3, .7]], .05,
                                            q_semantics='normalize_weights')
    _, eb = v4.randomized_group_intervals_v4(counts, [[.3, .7]], [[.5, .5]], .05,
                                            q_semantics='exact_probabilities')
    assert cp['rows'][0]['construction'] == 'clopper_pearson'
    assert eb['rows'][0]['construction'] == 'empirical_bernstein'


@pytest.mark.parametrize('n', [0, 1])
def test_empty_cells_and_small_n_eb_return_complete_normalized_range(n):
    counts = np.array([[[0, n], [0, 0]]])
    mapped, info = v4.randomized_group_intervals_v4(counts, [[.25, .75]], [[.5, .5]], .05,
                                                  q_semantics='exact_probabilities')
    assert mapped.lower == (0.,) and mapped.upper == (1.,)
    assert info['rows'][0]['construction'] == 'insufficient_eb'
    cell, _ = v4.randomized_cell_intervals_v4(np.zeros_like(counts), [[.2, .2]], .05)
    assert cell.exact_lower == (F(0),)
    assert cell.exact_upper == (2*rational(.2),)


def test_group_context_and_randomization_support_are_required():
    with pytest.raises(ValueError, match='one context'):
        v4.exact_group_score_statistics(np.zeros((2, 2, 2), int), [[.5, 0, 0, .5]],
                                        [[.5, .5], [.5, .5]], q_semantics='exact_probabilities')
    with pytest.raises(ValueError, match='Positive known'):
        v4.exact_group_score_statistics(np.zeros((1, 2, 2), int), [[.5, .5]],
                                        [[1, 0]], q_semantics='exact_probabilities')
    # The deterministic projection itself has no one-context restriction.
    mapped = v4.project_cell_intervals([[.5, 0, 0, .5]], [0]*4, [1]*4)
    assert mapped.upper == (1.,)


@pytest.mark.parametrize('bad', [[[[1, .5]]], [[[-1, 1]]], [[[float('nan'), 1]]],
                                [[[2**52+1, 0]]], [[[1+1j, 0]]]])
def test_invalid_counts_refused(bad):
    with pytest.raises(ValueError):
        v4.observation_intervals_v4(bad, 2., .05)


def test_cp_probability_arguments_are_conservative_but_ppf_is_trusted(monkeypatch):
    observed = []
    def fake_ppf(probability, a, b):
        observed.append(probability)
        return .25
    monkeypatch.setattr(v4.beta, 'ppf', fake_ppf)
    failure = F(1, 19)
    v4.trusted_binomial_interval(2, 7, failure)
    assert rational(observed[0]) <= failure/2
    assert rational(observed[1]) >= 1-failure/2


def test_sensitivity_image_is_exact_before_final_conversion(monkeypatch):
    monkeypatch.setattr(v4, 'trusted_binomial_interval', lambda *args, **kwargs: (F(1, 3), F(2, 3)))
    lower, upper, info = v4.observation_intervals_v4([[[3, 4]]], F(7, 5), F(1, 20), consistency=False)
    expected_lo = F(5, 21)
    expected_hi = F(16, 21)
    assert rational(lower[0]) <= expected_lo <= rational(upper[0])
    assert rational(upper[0]) >= expected_hi
    assert F(*map(int, info['exact_lower'][0])) == expected_lo
    assert F(*map(int, info['exact_upper'][0])) == expected_hi


def test_joint_consistency_complement_and_budget_are_exact(monkeypatch):
    calls = []
    values = iter([(F(1, 3), F(2, 3)), (F(1, 4), F(1)), (F(1, 3), F(1))])
    def fake_interval(k, n, failure, **kwargs):
        calls.append(failure)
        return next(values)
    monkeypatch.setattr(v4, 'trusted_binomial_interval', fake_interval)
    _, _, info = v4.observation_intervals_v4([[[3, 4]]], 2, F(1, 20))
    assert calls == [F(1, 40), F(1, 80), F(1, 80)]
    assert F(*map(int, info['exact_lower'][0])) == F(1, 4)
    assert F(*map(int, info['exact_upper'][0])) == F(2, 3)


def test_bad_quantile_and_empty_group_interval_fail_closed(monkeypatch):
    monkeypatch.setattr(v4.beta, 'ppf', lambda *args: float('nan'))
    with pytest.raises(v4.UnusableStatisticalInterval):
        v4.trusted_binomial_interval(1, 2, .05)
    # An adversarial trusted half-width illustrates why invalid clipping may
    # not silently become a positive authorization.
    monkeypatch.setattr(v4, '_trusted_eb_radius', lambda *args: F(0))
    with pytest.raises(v4.UnusableStatisticalInterval):
        v4.randomized_group_intervals_v4([[[0, 0], [0, 2]]], [[.25, .75]],
                                         [[.5, .5]], .05, q_semantics='exact_probabilities')


@pytest.mark.parametrize('method', ['group_score', 'cell_projection'])
def test_opt_in_materialization_encloses_boundary_and_binds_semantics(method):
    counts = np.array([[[0, 10]]*5])
    result = v4.materialize_counts_v4(counts, counts, rows=[[.2]*5],
                                     randomized_q=[[.2]*5], q_semantics='normalize_weights',
                                     randomized_method=method, highs_threads=1)
    assert result['status'] == 'ready'
    checked = CheckedView(result['view'])
    assert checked.is_feasible([1.]*5)
    metadata = result['view']['metadata']
    assert metadata['q_semantics'] == 'normalize_weights'
    assert metadata['assignment_probabilities_exact'] == [[['1', '5']]*5]
    assert metadata['rho_units'] == 'normalized ideal group mean'
    assert metadata['counts_shape'] == [1, 5, 2]
    assert len(metadata['observational_counts_sha256']) == 64
    assert result['view']['matrix']['data'] == [.2]*5
    json.dumps(result, allow_nan=False)


def test_materialization_refuses_unusable_interval_without_a_view(monkeypatch):
    def unusable(*args, **kwargs):
        raise v4.UnusableStatisticalInterval('deliberate test failure')
    monkeypatch.setattr(v4, 'observation_intervals_v4', unusable)
    result = v4.materialize_counts_v4([[[0, 1]]], [[[0, 1]]], rows=[[1]],
                                     randomized_q=[[1]], q_semantics='exact_probabilities')
    assert result['status'] == 'uncertain'
    assert 'view' not in result
    assert result['reason'] == 'unusable_statistical_interval_v4'
