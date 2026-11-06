"""Small synthetic correctness/registration tests; no formal workload timing."""
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path

import highspy
import numpy as np
import pytest

from icde2027.checking_schedule_v4 import Candidate, FractionReplay
from icde2027.optimized_exact_v5 import OptimizedThresholdChecker
from icde2027.optimized_schedule_v6 import (
    Budget, CONFIG, CORE, METHODS, OptimizedFactorialEngine, ResourceLimit,
    call_signature, compare_streams, gate_payload, make_engine, measure,
    method_order, native_state, replay_stream, service_response, stage_index,
    summarize, validate_gate, verify_stage_index,
)
from icde2027.threshold_extension.proofs import ThresholdChecker


@pytest.fixture
def view():
    return {'format': 'qbridge-confidence-view-v1', 'lower': [0., 0.],
            'upper': [1., 1.], 'rhs_lower': [.75], 'rhs_upper': [1.25],
            'matrix': {'indptr': [0, 2], 'indices': [0, 1], 'data': [1., 1.]},
            'feasible_point': [.5, .5]}


def verify(view, native, c, delta):
    item = {'query_index': 0, 'answer': service_response(native)}
    assert FractionReplay(view).verify(item, 0, c, delta)
    for ctor in (ThresholdChecker, OptimizedThresholdChecker):
        assert ctor(view, cache_size=0).verify(native['certificate'],
            expected_objective=c, expected_threshold=delta)['valid']


@pytest.mark.parametrize('bad', [0, 1, None, 'true', np.bool_(True)])
def test_factors_are_actual_booleans(view, bad):
    with pytest.raises(ValueError, match='Boolean'):
        OptimizedFactorialEngine(view, screening=bad, prepared=True)
    with pytest.raises(ValueError, match='Boolean'):
        OptimizedFactorialEngine(view, screening=False, prepared=bad)


def test_constructor_has_no_hidden_dual_cache(view):
    for name in METHODS:
        engine = make_engine(name, view)
        assert engine.checker.cache_size == 0 and not engine.checker._dual_cache
        assert engine.checker.stats['dual_preparations'] == 0
        assert engine.memory_info()['retained_exact_duals'] == 0
        assert engine.simplex_strategy == 1
        assert engine.checker._active_query is None
    with pytest.raises(ValueError, match='Unknown'):
        make_engine('unregistered', view)


def test_full_core_and_bridge_equivalence_on_mixed_toy_sequence(view):
    queries = [[1., 1.], [1., 1.1], [1.1, 1.], [-1., -1.],
               [0., 0.], [1., -1.], [1., 1.], [0., 0.]]
    sequences = {}
    for method in METHODS:
        engine = make_engine(method, view)
        records = []
        for c in queries:
            native = engine.query(c, .001)
            verify(view, native, c, .001)
            records.append((service_response(native), call_signature(native), native_state(engine)))
            assert engine.checker._active_query is None
        sequences[method] = records
        assert engine.checker.stats['query_context_builds'] == len(queries)
        if method.endswith('p0'):
            assert engine.memory_info()['retained_exact_duals'] == 0
            assert not engine.checker._dual_cache
    for method in CORE:
        assert sequences[method] == sequences[CORE[0]]
    for a, b in zip(sequences['s0_p1'], sequences['last_first']):
        assert a[0]['robust_threshold_decision'] is b[0]['robust_threshold_decision']
        assert a[1:] == b[1:]
        if a[0]['robust_threshold_decision'] is not True:
            assert a[0] == b[0]


def test_factors_only_change_intended_preparation_and_screen_counts(view):
    answers = {}
    for name in CORE:
        engine = make_engine(name, view)
        engine.query([1., 1.], .001)
        engine.query([1., 1.1], .001)
        answers[name] = engine.query([1., 1.1], .001)
        assert engine.checker.stats['query_context_builds'] == 3
    assert answers['s0_p0']['exact_dual_preparations'] == 2
    assert answers['s1_p0']['exact_dual_preparations'] == 1
    assert answers['s0_p1']['exact_dual_reuses'] == 2
    assert answers['s1_p1']['exact_dual_reuses'] == 1
    assert answers['s1_p1']['deferred_duals'] == 1
    assert answers['s1_p1']['replayed_duals'] == 0


def seed_dual(engine, lo=(.5,), hi=(0.,)):
    candidate = Candidate(lo, hi)
    engine.last_dual = candidate
    engine.dual_bank[:] = [candidate]
    engine.last_dual_generation = 1
    return candidate


def test_last_first_two_true_candidates_can_export_different_evidence(view):
    ordinary, bridge = make_engine('s0_p1', view), make_engine('last_first', view)
    for engine in (ordinary, bridge):
        seed_dual(engine)
        engine._native_run = lambda *a: pytest.fail('Both candidates already prove TRUE')
    a, b = ordinary.query([1., 1.], -.1), bridge.query([1., 1.], -.1)
    assert a['route'] == 'box_shortcut' and b['route'] == 'last_dual'
    assert a['robust_threshold_decision'] is b['robust_threshold_decision'] is True
    assert a['certificate'] != b['certificate']
    assert native_state(ordinary) == native_state(bridge)
    for native in (a, b):
        verify(view, native, [1., 1.], -.1)


def test_last_first_can_do_more_work_when_box_already_true(view):
    ordinary, bridge = make_engine('s0_p1', view), make_engine('last_first', view)
    for engine in (ordinary, bridge):
        seed_dual(engine, (0.,), (2.,))
        engine._native_run = lambda *a: pytest.fail('Box proves TRUE')
    a, b = ordinary.query([1., 1.], -.1), bridge.query([1., 1.], -.1)
    assert service_response(a) == service_response(b)
    assert a['dual_candidates_exact_checked'] == 1
    assert b['dual_candidates_exact_checked'] == 2


def test_equal_nontrue_lower_prefers_zero_evidence(view):
    ordinary, bridge = make_engine('s0_p1', view), make_engine('last_first', view)
    for engine in (ordinary, bridge):
        seed_dual(engine, (0.,), (0.,))
    a, b = ordinary.query([-1., -1.], .001), bridge.query([-1., -1.], .001)
    assert a['robust_threshold_decision'] is b['robust_threshold_decision'] is False
    assert service_response(a) == service_response(b)
    assert a['certificate']['lower_proof']['format'] == 'qbridge-dual-proof-v1'
    verify(view, b, [-1., -1.], .001)


@pytest.mark.parametrize('name', METHODS)
def test_exact_equality_belongs_to_false(view, name):
    a = make_engine(name, view).query([0., 0.], 0.)
    assert a['robust_threshold_decision'] is False and a['upper_bound'] == 0.
    verify(view, a, [0., 0.], 0.)


@pytest.mark.parametrize('name', METHODS)
def test_unknown_replays_and_preserves_native_cutoff_retry(view, name):
    engine = make_engine(name, view)
    seed_dual(engine, (0.,), (0.,))
    calls = []
    def native(c, cutoff, metrics):
        assert metrics['deferred_duals'] == metrics['replayed_duals']
        assert metrics['deferred_primals'] == metrics['replayed_primals']
        calls.append(cutoff)
        status = highspy.HighsModelStatus.kObjectiveBound if cutoff is not None else highspy.HighsModelStatus.kOptimal
        metrics['native_calls'].append({'seconds': 0., 'simplex_iterations': 0,
                                        'objective_bound': cutoff, 'model_status_code': int(status)})
        return {'lower': None, 'upper': None, 'primal': None, 'status': status}
    engine._native_run = native
    a = engine.query([1., 1.], .1)
    assert calls == [.1, None]
    assert a['robust_threshold_decision'] is None and a['cutoff_retries'] == 1
    assert a['certificate']['lower_proof']['format'] == 'qbridge-dual-proof-v1'
    verify(view, a, [1., 1.], .1)


@pytest.mark.parametrize('name', METHODS)
def test_invalid_prepared_and_candidate_states_do_not_authorize(view, name):
    engine = make_engine(name, view)
    engine.zero.prepared = ThresholdChecker(view).dual_state([0.], [0.])
    seed_dual(engine, (-1.,), (0.,))
    a = engine.query([-1., -1.], .001)
    assert a['rejected_cached_duals'] == 1
    assert a['robust_threshold_decision'] is False
    if not name.endswith('p0'):
        assert engine.checker.owns_dual(engine.zero.prepared, (0.,), (0.,))
    verify(view, a, [-1., -1.], .001)


@pytest.mark.parametrize('bad', [math.nan, math.inf, True, '1', 10**400])
def test_scope_is_cleared_after_invalid_query(view, bad):
    for name in METHODS:
        engine = make_engine(name, view)
        with pytest.raises((ValueError, TypeError, ArithmeticError)):
            engine.query([1., bad], .001)
        assert engine.checker._active_query is None
        with pytest.raises((ValueError, TypeError, ArithmeticError)):
            engine.query([1., 1.], bad)
        assert engine.checker._active_query is None
        verify(view, engine.query([0., 0.], 0.), [0., 0.], 0.)


def test_measured_toy_stream_has_all_three_verifiers_and_exact_cost_sums(view, tmp_path):
    queries = np.array([[1., 1.], [1., 1.1], [-1., -1.], [0., 0.]])
    reg = {'bank_size': 16, 'threshold': .001, 'query_prefixes': [1, 2, 4]}
    for method in METHODS:
        result = measure(tmp_path, method, json.dumps(view), queries, reg, Budget(60))
        assert result['persistent_replay']['checker'] == 'optimized'
        assert result['legacy_replay_audit']['checker'] == 'legacy'
        assert result['independent_fraction_replay']['valid'] == 4
        assert result['joint_seconds'] == result['producer_seconds'] + result['persistent_replay']['total_consumer_seconds']
        assert result['prefixes']['4'] == result['producer_seconds']
        assert result['joint_prefixes']['4'] == result['joint_seconds']
    compared = compare_streams(tmp_path, len(queries))
    assert compared['core_same_evidence_and_paths']
    assert compared['bridge_same_decisions_and_native_state']
    # A real saved-service mutation must be caught, not only a summary flag.
    p = tmp_path/'last_first.jsonl.gz'
    with gzip.open(p, 'rt') as f:
        rows = [json.loads(line) for line in f]
    rows[0]['answer']['robust_threshold_decision'] = False
    with gzip.open(p, 'wt') as f:
        for row in rows:
            f.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='decision mismatch'):
        compare_streams(tmp_path, len(queries))


def test_replay_rejects_extra_or_missing_wire_records(view, tmp_path):
    queries = np.array([[0., 0.]])
    reg = {'bank_size': 16, 'threshold': .001, 'query_prefixes': [1]}
    p = tmp_path/'empty.gz'
    with gzip.open(p, 'wt'):
        pass
    with pytest.raises(ValueError, match='Incomplete'):
        replay_stream(p, json.dumps(view), queries, reg, Budget(60))


@pytest.mark.parametrize('change', ['route', 'reason', 'proof'])
def test_undeclared_true_bridge_differences_fail(view, tmp_path, change):
    queries = np.array([[1., 1.], [1., 1.1]])
    reg = {'bank_size': 16, 'threshold': .001, 'query_prefixes': [1, 2]}
    for method in METHODS:
        measure(tmp_path, method, json.dumps(view), queries, reg, Budget(60))
    p = tmp_path/'last_first.jsonl.gz'
    with gzip.open(p, 'rt') as f:
        rows = [json.loads(line) for line in f]
    assert rows[1]['answer']['robust_threshold_decision'] is True
    if change == 'route':
        rows[1]['answer']['route'] = 'solver'
    elif change == 'reason':
        rows[1]['answer']['reason'] = 'changed without permission'
    else:
        rows[1]['answer']['certificate']['view_digest'] = 'forged'
    with gzip.open(p, 'wt') as f:
        for row in rows:
            f.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='Undeclared TRUE'):
        compare_streams(tmp_path, len(queries))


@pytest.mark.parametrize('queries,delta,direction', [
    ([[-1., -1.], [-.1, -.1]], -1.1, 'greater'),
    ([[-1., -.1], [1., 1.]], -.8, 'less'),
])
def test_allowed_both_true_bridge_change_is_measured_and_bounded(view, tmp_path, queries, delta, direction):
    q = np.array(queries)
    reg = {'bank_size': 16, 'threshold': delta, 'query_prefixes': [1, 2]}
    for method in METHODS:
        measure(tmp_path, method, json.dumps(view), q, reg, Budget(60))
    comparison = compare_streams(tmp_path, 2)
    changed = comparison['bridge_true_differences']
    assert len(changed) == 1 and changed[0]['query_index'] == 1
    assert changed[0]['route_original'] == 'box_shortcut'
    assert changed[0]['route_last_first'] == 'last_dual'
    assert changed[0]['lower_direction'] == direction
    assert changed[0]['original_certificate_json_bytes'] > 0
    assert changed[0]['bridge_certificate_json_bytes'] > 0
    # Even an allowed route pair cannot excuse unrelated envelope mutations.
    path = tmp_path/'last_first.jsonl.gz'
    with gzip.open(path, 'rt') as f:
        rows = [json.loads(line) for line in f]
    rows[1]['answer']['reason'] = 'unrelated change'
    with gzip.open(path, 'wt') as f:
        for row in rows:
            f.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='Unrelated TRUE service'):
        compare_streams(tmp_path, 2)


def test_order_has_each_arm_in_each_position():
    reg = json.loads(CONFIG.read_text())
    for entry in reg['inputs']:
        orders = [method_order(reg, entry['id'], r) for r in range(5)]
        for position in range(5):
            assert {o[position] for o in orders} == set(METHODS)
    with pytest.raises(ValueError):
        method_order(reg, 'x', 5)


def test_no_flag_only_gate_and_changed_gate_are_accepted():
    with pytest.raises(ValueError, match='Independent'):
        validate_gate(CONFIG, {'status': 'approved'})
    payload = gate_payload(CONFIG)
    gate = {**payload, 'status': 'approved', 'independent_review': True,
            'reviewer': 'synthetic-test-only', 'reviewed_at_utc': 'not-a-real-review',
            'review_report': 'runs/icde2027/OPTIMIZED_SCHEDULE_V6_DESIGN.md',
            'review_report_sha256': 'wrong'}
    with pytest.raises(ValueError, match='report changed'):
        validate_gate(CONFIG, gate)
    gate['runtime_sha256'] = {}
    with pytest.raises(ValueError, match='runtime_sha256'):
        validate_gate(CONFIG, gate)


def test_quality_index_cannot_be_flagged_complete_after_tampering(tmp_path):
    (tmp_path/'result.json').write_text('{"status":"complete"}')
    index = stage_index(tmp_path)
    verify_stage_index(tmp_path, index, Budget(60))
    (tmp_path/'result.json').write_text('{"status":"complete","fake":true}')
    with pytest.raises(ValueError, match='member changed'):
        verify_stage_index(tmp_path, index, Budget(60))
    (tmp_path/'extra').write_text('x')
    with pytest.raises(ValueError, match='membership'):
        verify_stage_index(tmp_path, index, Budget(60))


def test_budget_and_partial_grid_cannot_complete():
    budget = Budget(60)
    budget.started -= 61
    with pytest.raises(ResourceLimit):
        budget.check()
    with pytest.raises(ValueError, match='partial'):
        summarize(json.loads(CONFIG.read_text()), [])


def test_budget_failure_preserves_partial_stream(view, tmp_path):
    budget = Budget(1)
    budget.started -= 2
    with pytest.raises(ResourceLimit):
        measure(tmp_path, 's0_p0', json.dumps(view), np.array([[0., 0.]]),
                {'bank_size': 16, 'threshold': .001, 'query_prefixes': [1]}, budget)
    assert (tmp_path/'s0_p0_partial.jsonl.gz').is_file()
    failure = json.loads((tmp_path/'s0_p0_failure.json').read_text())
    assert failure['completed_responses'] == 0 and 'ResourceLimit' in failure['error']


def test_injected_quality_failure_never_enters_timing(tmp_path, monkeypatch):
    """Negative driver probe only: no trusted gate, formal query, or timing run."""
    import icde2027.optimized_schedule_v6 as module
    gate = tmp_path/'unit-test-not-an-approval.json'
    gate.write_text('{}')
    output = tmp_path/'negative-driver-probe'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.setenv(name, '1')
    monkeypatch.setattr(module, 'validate_gate', lambda *args: {})
    monkeypatch.setattr(module, 'validate_inputs', lambda *args: None)
    monkeypatch.setattr(module, 'snapshot_sources', lambda *args: {})
    monkeypatch.setattr(module, 'preserve_input_bytes', lambda *args: {})
    monkeypatch.setattr(module, 'validate_snapshot', lambda *args: None)
    monkeypatch.setattr(module.os, 'sched_getaffinity', lambda *args: {4})
    monkeypatch.setattr(module.os, 'sched_setaffinity', lambda *args: None)
    reached = []
    def fail_quality(stage, entry, order, reg, budget, timing_round):
        reached.append((stage.name, timing_round))
        raise ValueError('injected unit-test quality failure')
    monkeypatch.setattr(module, 'execute_condition', fail_quality)
    with pytest.raises(ValueError, match='injected unit-test'):
        module.execute(CONFIG, gate, output)
    assert reached == [('quality', None)]
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['status'] == 'failed' and manifest['formal_timing_started'] is False
    assert manifest['completed_quality_conditions'] == manifest['completed_timing_conditions'] == 0
    assert not (output/'timing').exists() and not (output/'quality_completion.json').exists()
