from fractions import Fraction as F
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse

from icde2027 import coat_representation_v5 as m


def rows(n):
    return sparse.csr_matrix(np.full((1, n), 1/n))


def test_outward_complement_and_empty_intersection_are_preserved():
    e = m.endpoints([F(0.7)], [1-F(0.1)])
    assert F(e['upper'][0]) >= 1-F(0.1)
    assert F(e['lower'][0]) <= F(0.7)
    empty = m.endpoints([F(3, 4)], [F(1, 4)])
    assert not empty['statistical_intersection_nonempty']
    assert empty['lower'][0] > empty['upper'][0]


def test_observation_counts_joint_denominator_and_rational_transform():
    calls = []
    def cp(k, n, delta, *, lower_only=False):
        calls.append((k, n, delta, lower_only))
        return (F(0.1), F(1)) if lower_only else (F(0.2), F(0.8))
    data = np.array([[5, 0], [1, 4], [0, 0]])
    with patch.object(m, 'trusted_binomial_interval', cp):
        result = m.observation_base(data, radius=1.5, alpha=.05)
    assert [n for _, n, _, low in calls if low] == [3]*4
    assert [n for _, n, _, low in calls if not low] == [2, 1]
    assert all(delta == F(.05)/8 for _, _, delta, low in calls if low)
    assert all(delta == F(.05)/4 for _, _, delta, low in calls if not low)
    target_lo = max(F(.2)/F(1.5), 1-F(1.5)*(1-F(.2)), F(.1))
    target_hi = min(F(1.5)*F(.8), 1-(1-F(.8))/F(1.5), 1-F(.1))
    assert m.unpack(result['exact_lower'][0]) == target_lo
    assert m.unpack(result['exact_upper'][0]) == target_hi
    assert F(result['lower'][0]) <= target_lo
    assert F(result['upper'][0]) >= target_hi


@pytest.mark.parametrize('data', [np.array([[0, 0], [0, 0]]), np.array([[5, 5], [5, 5]]),
                                np.array([[1, 2], [3, 1]])])
def test_empty_all_success_all_failure_items(data):
    result = m.observation_base(data, radius=2, alpha=.05)
    assert result['n_independent_users'] == 2
    assert all(0 <= l <= u <= 1 for l, u in zip(result['lower'], result['upper']))
    cell = m.randomized_cell_base(data, alpha=.05)
    if not np.any(data):
        assert cell['lower'] == [0., 0.] and cell['upper'] == [1., 1.]


def test_exact_active_user_scores_and_unbiased_variance():
    data = np.array([[5, 1, 1], [5, 4, 1], [1, 2, 3], [0, 0, 0]])
    stats = m.group_statistics(data, rows(3), ['group'])
    g = stats['groups'][0]
    assert g['active_user_indices'] == [0, 1, 2] and g['n_active'] == 3
    assert list(map(m.unpack, g['active_scores_exact'])) == [F(1, 3), F(2, 3), F()]
    assert m.unpack(g['sum_Z_exact']) == 1
    assert m.unpack(g['sum_Z_squared_exact']) == F(5, 9)
    assert m.unpack(g['mean_exact']) == F(1, 3)
    assert m.unpack(g['unbiased_variance_exact']) == F(1, 9)
    assert m.unpack(g['kappa_exact']) == 3*F(1/3) != 1
    with patch.object(m, '_trusted_eb_radius', return_value=F(1, 10)) as radius:
        base = m.randomized_group_base(stats, alpha=.05)
    assert radius.call_args.args == ({'n': 3, 'variance': F(1, 9), 'bound': F(1)}, F(.05))
    assert m.unpack(base['exact_lower'][0]) == F(1, 3)-F(1, 10)
    mapped = m.map_moments(rows(3), base, construction='group_score', rho=.02)
    exact = 3*F(1/3)*(F(1, 3)-F(1, 10)-F(.02))
    assert F(mapped['lower'][0]) <= exact
    assert m.unpack(mapped['exact_lower'][0]) == exact


def test_fixed_group_branch_does_not_depend_on_observed_binary_scores():
    stats = m.group_statistics(np.array([[5, 5], [1, 1]]), rows(2), ['two'])
    with patch.object(m, 'trusted_binomial_interval', side_effect=AssertionError('must use EB')):
        base = m.randomized_group_base(stats, alpha=.05)
    assert base['groups'][0]['construction'] == 'active_user_empirical_bernstein'


def test_small_sample_branch_preserves_singleton_cp():
    stats = m.group_statistics(np.array([[5]]), rows(1), ['one'])
    result = m.randomized_group_base(stats, alpha=.05)
    assert result['groups'][0]['construction'] == 'singleton_cp'
    assert 0 < result['lower'][0] < 1 and result['upper'][0] == 1
    stats = m.group_statistics(np.array([[5, 1]]), rows(2), ['two'])
    result = m.randomized_group_base(stats, alpha=.05)
    assert result['groups'][0]['construction'] == 'insufficient_eb'
    assert result['lower'] == [0.] and result['upper'] == [1.]
    stats = m.group_statistics(np.array([[0]]), rows(1), ['empty'])
    result = m.randomized_group_base(stats, alpha=.05)
    assert result['lower'] == [0.] and result['upper'] == [1.]


def test_projected_cells_keep_exact_sum_and_transport_mass():
    base = m.endpoints([F(1, 5), F(1, 3), F(1, 7)], [F(1)]*3)
    result = m.map_moments(rows(3), base, construction='cell_projection', rho=.05)
    lo = F(1/3)*(F(1, 5)+F(1, 3)+F(1, 7))-3*F(1/3)*F(.05)
    assert m.unpack(result['exact_lower'][0]) == lo
    assert F(result['lower'][0]) <= lo


def test_direct_scores_use_exact_stored_objectives_and_user_units():
    data = np.array([[5, 1, 4], [1, 5, 1]])
    c = [.1, -.2, .3]
    result = m.direct_all_scores(data, c, alpha=.05)
    expected = [F(.1)+F(.3), -F(.2)]
    assert list(map(m.unpack, result['scores_exact'])) == expected
    mean = sum(expected)/2
    assert m.unpack(result['estimate_exact']) == mean
    assert list(map(m.unpack, result['range_exact'])) == [-3*F(.2), 3*F(.3)]
    assert result['n_independent_user_blocks'] == 2
    assert result['adaptive_query_guarantee'] is False
    assert F(result['lower_bound']) <= m.unpack(result['lower_exact'])
    assert m.unpack(result['lower_exact']) == mean-m.unpack(result['trusted_hoeffding_radius_exact'])
    zero = m.direct_all_scores(data, [0, 0, 0], alpha=.05)
    assert zero['lower_bound'] == 0


@pytest.mark.parametrize('bad', [True, float('nan'), float('inf'), 0, 1, -1])
def test_invalid_source_budget_is_rejected(bad):
    with pytest.raises(ValueError):
        m.observation_base(np.array([[5]]), radius=2, alpha=bad)


def test_invalid_radius_shape_and_group_weights_rejected():
    with pytest.raises(ValueError):
        m.observation_base(np.array([[5]]), radius=.5, alpha=.05)
    with pytest.raises(ValueError):
        m.group_statistics(np.array([[5, 1]]), sparse.csr_matrix([[.2, .8]]), ['bad'])
    with pytest.raises(ValueError):
        m.direct_all_scores(np.array([[5, 1]]), [1], alpha=.05)
    with pytest.raises(ValueError):
        m.direct_all_scores(np.array([[0, 0]]), [1, -1], alpha=.05)
