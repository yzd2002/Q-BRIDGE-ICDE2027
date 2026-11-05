"""Synthetic interface checks; no measurements from the formal input streams."""
import copy
import gzip
import json

import numpy as np
import pytest
from scipy import sparse

from icde2027.primary_aligned import (
    METHODS, SERVICE_FIELDS, adapt_lower, checked_response, make_engine, measure,
    method_order, replay, service_response,
)
from icde2027.confidence_view import create_view
from icde2027.threshold_extension.proofs import ThresholdChecker


@pytest.fixture
def view():
    matrix = sparse.csr_matrix([[.5, 0., .5, 0.], [0., .5, 0., .5]])
    return create_view([.125, .25, .25, .375], [.625, .6875, .8125, .875],
                       matrix, [.65625, .71875], [.6875, .75], highs_threads=1)['view']


def lower_answer(view, c, threshold):
    engine = make_engine('original_prepared', view, 16)
    return engine.query(c, threshold), engine.digest


def test_original_engines_are_original_classes(view):
    from icde2027.confidence_view import QueryEngine
    from icde2027.highs_reopt import HighsReoptimization
    prepared = make_engine('original_prepared', view, 16)
    hot = make_engine('original_hot', view, 16)
    assert type(prepared) is QueryEngine and prepared.compiled_reuse and prepared.reuse
    assert type(hot) is HighsReoptimization and hot.retain_basis
    assert make_engine('threshold_hot', view, 16).simplex_strategy == 1
    assert make_engine('threshold_hot_auto', view, 16).simplex_strategy == 0
    assert make_engine('both_histories', view, 16).plan == 'both_pools'


def test_balanced_order_before_timing():
    reg = {'order_seed': 2026110501, 'methods': list(METHODS)}
    orders = [method_order(reg, r) for r in range(3)]
    assert all(sorted(row) == sorted(METHODS) for row in orders)
    assert all(len({row.index(name) for row in orders}) == 3 for name in METHODS)


def test_service_projection_removes_only_diagnostics():
    native = {key: key for key in SERVICE_FIELDS}
    native.update(metrics={'solve_seconds': 1.}, primal_diagnostic=-100.)
    assert service_response(native) == {key: key for key in SERVICE_FIELDS}
    assert native['metrics'] == {'solve_seconds': 1.}


@pytest.mark.parametrize('threshold', [-.1, .001, .1])
def test_lower_adapter_replays_and_never_claims_false(view, threshold):
    c = [-.1, -.1, .1, .1]
    original, digest = lower_answer(view, c, threshold)
    frozen = copy.deepcopy(original)
    adapted = adapt_lower(original, c, threshold, digest)
    assert original == frozen
    assert adapted['upper_bound'] is None
    assert adapted['certificate']['upper_proof'] is None
    assert adapted['robust_threshold_decision'] is (True if original['lower_bound'] > threshold else None)
    assert checked_response(ThresholdChecker(view), {'query_index': 0, 'answer': adapted}, 0, c, threshold)


def test_equal_lower_is_unknown_without_upper(view):
    c = [0., 0., 0., 0.]
    first, _ = lower_answer(view, c, 0.)
    threshold = first['lower_bound']  # The checker may round zero outward.
    original, digest = lower_answer(view, c, threshold)
    adapted = adapt_lower(original, c, threshold, digest)
    assert adapted['lower_bound'] == threshold and adapted['robust_threshold_decision'] is None
    assert checked_response(ThresholdChecker(view), {'query_index': 0, 'answer': adapted}, 0, c, threshold)


@pytest.mark.parametrize('attack', ['lower', 'status', 'objective', 'threshold', 'view'])
def test_adapter_rejects_mismatched_evidence(view, attack):
    c = [-.1, -.1, .1, .1]; threshold = .001
    answer, digest = lower_answer(view, c, threshold)
    if attack == 'lower': answer['lower_bound'] += .01
    elif attack == 'status': answer['status'] = 'unresolved'
    elif attack == 'objective': c[0] += .1
    elif attack == 'threshold': threshold += 1.
    else: digest = '0' * 64
    with pytest.raises(ValueError):
        adapt_lower(answer, c, threshold, digest)


def test_explicit_ready_view_numerical_refusal_remains_no_bound(view):
    answer = {'status': 'uncertain', 'lower_bound': None, 'certificate': None,
              'reason': 'synthetic numerical failure', 'route': 'solver'}
    out = adapt_lower(answer, [0.] * 4, .001, ThresholdChecker(view).digest)
    assert out['robust_threshold_decision'] is None and out['upper_bound'] is None
    assert not checked_response(ThresholdChecker(view), {'query_index': 0, 'answer': out}, 0, [0.] * 4, .001)


@pytest.mark.parametrize('attack', ['status', 'lower', 'upper', 'decision', 'reason'])
def test_refusal_cannot_hide_a_decision(view, attack):
    answer = {'status': 'uncertain', 'lower_bound': None, 'upper_bound': None,
              'certificate': None, 'robust_threshold_decision': None, 'reason': 'synthetic'}
    key = {'lower': 'lower_bound', 'upper': 'upper_bound', 'decision': 'robust_threshold_decision'}.get(attack, attack)
    answer[key] = {'status': 'certified', 'lower': .1, 'upper': -.1, 'decision': False, 'reason': ''}[attack]
    with pytest.raises(ValueError):
        checked_response(ThresholdChecker(view), {'query_index': 0, 'answer': answer}, 0, [0.] * 4, .001)


@pytest.mark.parametrize('attack', ['index', 'bool_index', 'outer_lower', 'outer_upper', 'decision', 'status', 'query'])
def test_receiver_rejects_changed_outer_binding(view, attack):
    c = [-.1, -.1, .1, .1]; original, digest = lower_answer(view, c, .001)
    adapted = adapt_lower(original, c, .001, digest); item = {'query_index': 0, 'answer': adapted}
    if attack == 'index': item['query_index'] = 1
    elif attack == 'bool_index': item['query_index'] = False
    elif attack == 'outer_lower': adapted['lower_bound'] += .01
    elif attack == 'outer_upper': adapted['upper_bound'] = 0.
    elif attack == 'decision': adapted['robust_threshold_decision'] = False
    elif attack == 'status': adapted['status'] = 'unresolved'
    else: c[0] += .01
    with pytest.raises(ValueError):
        checked_response(ThresholdChecker(view), item, 0, c, .001)


@pytest.mark.parametrize('name', list(METHODS))
def test_common_measurement_and_receiver_on_two_toy_queries(view, name, tmp_path):
    queries = np.array([[-.1, -.1, .1, .1], [0., 0., 0., 0.]])
    reg = {'bank_size': 16, 'threshold': .001, 'query_prefixes': [1, 2]}
    result = measure(tmp_path, name, view, queries, reg, .125)
    assert result['queries'] == 2 and result['common_view_seconds'] == .125
    assert result['producer_seconds'] >= .125 + result['plan_initialization_seconds']
    assert result['joint_seconds'] == result['producer_seconds'] + result['persistent_replay']['total_consumer_seconds']
    assert result['persistent_replay']['valid_proofs'] == result['stateless_replay']['valid_proofs'] == 2
    with gzip.open(tmp_path/f'{name}.jsonl.gz', 'rt') as stream:
        items = [json.loads(line) for line in stream]
    assert [i['query_index'] for i in items] == [0, 1]
    assert all(set(i['answer']) == set(SERVICE_FIELDS) for i in items)
    with gzip.open(tmp_path/f'{name}_native.jsonl.gz', 'rt') as stream:
        diagnostics = [json.loads(line) for line in stream]
    assert [i['query_index'] for i in diagnostics] == [0, 1]
    assert result['native_diagnostics']['included_in_primary_cost'] is False
    assert result['native_diagnostics']['uncompressed_bytes'] > 0
    if name.startswith('original'):
        assert items[1]['answer']['robust_threshold_decision'] is None
    else:
        assert items[1]['answer']['robust_threshold_decision'] is False
    # Every prefix includes the same first-use cost, not an extrapolated rate.
    assert result['prefixes']['1'] >= .125
    assert result['prefixes']['2'] == result['producer_seconds']
    assert result['joint_prefixes']['2'] == result['joint_seconds']


def test_new_view_decode_is_checked_before_measurement(view, tmp_path):
    queries = np.array([[-.1, -.1, .1, .1]])
    reg = {'bank_size': 16, 'threshold': .001, 'query_prefixes': [1]}
    changed = copy.deepcopy(view); changed['lower'][0] += .001
    with pytest.raises(ValueError, match='Fresh view decode'):
        measure(tmp_path, 'original_prepared', view, queries, reg, 0., view_text=json.dumps(changed))


def test_replay_rejects_missing_and_extra_responses(view, tmp_path):
    queries = np.array([[-.1, -.1, .1, .1], [0., 0., 0., 0.]])
    reg = {'bank_size': 16, 'threshold': .001, 'query_prefixes': [1, 2]}
    measure(tmp_path, 'original_prepared', view, queries, reg, 0.)
    path = tmp_path/'original_prepared.jsonl.gz'
    with gzip.open(path, 'rt') as stream: lines = list(stream)
    for changed in (lines[:1], lines + lines[:1]):
        with gzip.open(path, 'wt') as stream: stream.writelines(changed)
        with pytest.raises(ValueError): replay(path, view, queries, reg)
