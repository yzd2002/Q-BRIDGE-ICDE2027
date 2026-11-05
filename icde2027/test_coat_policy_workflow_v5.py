"""Semantic toy tests; no released-data workflow or timing is executed."""
import copy
from fractions import Fraction as F

import numpy as np
import pytest
from scipy import sparse

from . import coat_policy_workflow_v5 as w
from .confidence_view import create_view
from .threshold_extension.proofs import ThresholdChecker, bundle_proof


def view():
    return create_view([.2, .4], [.8, .9], sparse.csr_matrix((0, 2)), [], [],
                       highs_threads=1)['view']


class Engine:
    def __init__(self, source):
        self.checker = ThresholdChecker(source, 0)
        self.digest = self.checker.digest
        self.calls = 0
        self.refuse_after_first = False

    def query(self, c, threshold):
        self.calls += 1
        if self.refuse_after_first and self.calls > 1:
            return dict(status='uncertain', route='toy_refusal', lower_bound=None, upper_bound=None,
                        robust_threshold_decision=None, certificate=None, reason='synthetic refusal')
        lower = self.checker.lower_proof(c, [], [], threshold=threshold)
        upper = (self.checker.upper_proof(c, self.checker.primal_state([.8, .9]))
                 if self.calls > 1 else None)
        return w.from_bundle(bundle_proof(self.digest, c, threshold, lower, upper), 'toy')


def test_changed_threshold_is_redecided_and_unknown_calls_engine():
    source = view(); engine = Engine(source); cache = w.ExactObjectiveCache(engine, source)
    first, _ = cache.query([1., 1.], .5)
    assert first['robust_threshold_decision'] is True
    unknown, _ = cache.query([1., 1.], 1.)
    assert unknown['robust_threshold_decision'] is None and engine.calls == 2
    false, diagnostic = cache.query([1., 1.], 2.)
    assert false['robust_threshold_decision'] is False and engine.calls == 2
    assert not diagnostic['engine_invoked']
    assert false['certificate']['decision_threshold'] == 2.
    assert false['certificate']['lower_proof']['decision_threshold'] == 2.
    assert w.check_answer(ThresholdChecker(source), false, [1., 1.], 2.)
    cache.query([2., 1.], .5)
    assert engine.calls == 3


def test_returned_mutation_does_not_modify_cached_evidence():
    source = view(); engine = Engine(source); cache = w.ExactObjectiveCache(engine, source)
    first, _ = cache.query([1., 1.], .1)
    first['certificate']['lower_proof']['claimed_lower'] = 9.
    second, _ = cache.query([1., 1.], .1)
    assert engine.calls == 1 and second['lower_bound'] < 1.
    assert w.check_answer(ThresholdChecker(source), second, [1., 1.], .1)


@pytest.mark.parametrize('attack', ['foreign', 'modified', 'replaced'])
def test_unissued_or_modified_cache_state_never_authorizes(attack):
    source = view(); engine = Engine(source); cache = w.ExactObjectiveCache(engine, source)
    cache.query([1., 1.], .1)
    key = (1., 1.)
    if attack == 'foreign':
        other = w.ExactObjectiveCache(Engine(source), source)
        other.query(key, .1); cache._cache[key] = other._cache[key]
    elif attack == 'modified':
        object.__setattr__(cache._cache[key], 'wire', '{}')
    else:
        cache._cache[key] = w.ObjectiveEvidence(key, '{}')
    answer, _ = cache.query(key, .1)
    assert engine.calls == 2 and cache.stats['invalid_cached_entries'] == 1
    assert w.check_answer(ThresholdChecker(source), answer, key, .1)


def test_native_refusal_keeps_old_valid_unknown_without_claiming_a_decision():
    source = view(); engine = Engine(source); cache = w.ExactObjectiveCache(engine, source)
    cache.query([1., 1.], .1); engine.refuse_after_first = True
    answer, diagnostic = cache.query([1., 1.], 1.)
    assert answer['status'] == 'unresolved' and answer['reason'] is None
    assert answer['robust_threshold_decision'] is None and answer['certificate'] is not None
    assert diagnostic['native']['reason'] == 'synthetic refusal'
    assert w.check_answer(ThresholdChecker(source), answer, [1., 1.], 1.)


@pytest.mark.parametrize('field,value', [('reason', 'failed'), ('lower_bound', 10.),
                                        ('robust_threshold_decision', 1), ('status', 'uncertain')])
def test_outer_contract_tampering_is_rejected(field, value):
    source = view(); engine = Engine(source)
    answer = engine.query([1., 1.], .1); answer[field] = value
    with pytest.raises(ValueError):
        w.check_answer(ThresholdChecker(source), answer, [1., 1.], .1)


def test_changed_inner_query_threshold_or_view_is_rejected():
    source = view(); answer = Engine(source).query([1., 1.], .1)
    for field, value in [('objective', [2., 1.]), ('decision_threshold', .2), ('view_digest', '0'*64)]:
        bad = copy.deepcopy(answer); bad['certificate']['lower_proof'][field] = value
        with pytest.raises(ValueError):
            w.check_answer(ThresholdChecker(source), bad, [1., 1.], .1)


def test_whole_view_witness_is_checked_for_common_cache():
    source = view(); engine = Engine(source); source['feasible_point'] = [100., 100.]
    with pytest.raises(ValueError):
        w.ExactObjectiveCache(engine, source)


def test_toy_metadata_sequence_and_noncollinear_interactions():
    features = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Single-attribute group averages span constants/main effects, not interaction.
    rows = sparse.csr_matrix([[.5, .5, 0., 0.], [0., 0., .5, .5],
                              [.5, 0., .5, 0.], [0., .5, 0., .5]])
    result = w.catalogue_queries(features, ['a', 'b'], rows, feature_names=('a', 'b'))
    assert result['statistics']['policy_positions'] == 8
    assert result['statistics']['requests_per_view'] == 24
    assert result['statistics']['ideal_objective_rank'] == 3
    assert [p['multipliers'] for p in result['policies']] == [
        [.5, .5], [.5, 1.], [.5, 2.], [1., 2.], [1., .5], [2., .5], [2., 1.], [2., 2.]]
    assert all(p['ideal_in_moment_row_span'] == (1. in p['multipliers']) for p in result['policies'])
    for p in result['policies']:
        assert sum(F(int(a), int(b)) for a, b in p['ideal_objective_exact']) == 0
    assert [r['query_index'] for r in result['requests']] == list(range(24))
    assert [r['threshold'] for r in result['requests']] == [0.]*8+[.001]*8+[.01]*8


def test_exact_span_does_not_drop_small_nonzero_residual():
    basis = w.row_basis([[F(1), F(0), F(1)], [F(0), F(1), F(1)]])
    assert w.in_span([F(1), F(2), F(3)], basis)
    assert not w.in_span([F(1), F(2), F(3)+F(1, 2**1000)], basis)


def test_tiny_represented_positive_is_not_an_ideal_positive_by_default():
    policy = {'ideal_transfer_lower_exact': w.pair(-F(1, 2**53)),
              'ideal_transfer_upper_exact': w.pair(F(1, 2**54))}
    positive = {'lower_bound': 2.**-54, 'upper_bound': None, 'robust_threshold_decision': True}
    assert w.ideal_transfer(positive, policy, 0.)['ideal_robust_threshold_decision'] is None
    negative = {'lower_bound': -.5, 'upper_bound': 0., 'robust_threshold_decision': False}
    assert w.ideal_transfer(negative, policy, 0.)['ideal_robust_threshold_decision'] is None
    positive['lower_bound'] = .01
    assert w.ideal_transfer(positive, policy, 0.)['ideal_robust_threshold_decision'] is True
