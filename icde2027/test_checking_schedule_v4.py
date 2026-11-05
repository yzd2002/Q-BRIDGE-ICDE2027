"""Synthetic development only: never execute registered workload objectives."""
import copy
from fractions import Fraction
import gzip
import json
import math
from pathlib import Path

import highspy
import numpy as np
import pytest

from icde2027.checking_schedule_v4 import (
    CONFIG, CORE, METHODS, Candidate, CheckingScheduleEngine, FractionReplay,
    compare_streams, dot_interval, gate_payload, lower_interval, make_engine,
    load_input, measure, method_order, prepare_interval, service_response, summarize,
    validate_gate, validate_snapshot,
)
from icde2027.threshold_extension.proofs import ThresholdChecker


@pytest.fixture
def view():
    # Every number and the witness are exactly represented, no source data.
    return {'format': 'qbridge-confidence-view-v1', 'lower': [0., 0.],
            'upper': [1., 1.], 'rhs_lower': [.75], 'rhs_upper': [1.25],
            'matrix': {'indptr': [0, 2], 'indices': [0, 1], 'data': [1., 1.]},
            'feasible_point': [.5, .5]}


def check_answer(view, answer, c, delta):
    item = {'query_index': 0, 'answer': service_response(answer)}
    assert FractionReplay(view).verify(item, 0, c, delta)
    checked = ThresholdChecker(view, cache_size=0).verify(
        answer['certificate'], expected_objective=c, expected_threshold=delta)
    assert checked['valid']


def test_constructor_does_not_hide_prepared_state(view):
    for name in CORE:
        engine = make_engine(name, view)
        assert engine.checker.stats['dual_preparations'] == 0
        assert engine.checker.cache_size == 0
        assert engine.memory_info()['retained_exact_duals'] == 0
        assert engine.simplex_strategy == 1
    assert make_engine('screened_auto', view).simplex_strategy == 0
    from icde2027.confidence_view import QueryEngine
    assert type(make_engine('original_prepared', view)) is QueryEngine


def test_four_arms_preserve_all_service_bytes_and_native_path(view):
    objectives = [[1., 1.], [1., 1.1], [1.1, 1.], [-1., -1.],
                  [0., 0.], [1., -1.], [1., 1.], [0., 0.]]
    reference = None
    for name in CORE:
        engine = make_engine(name, view); rows = []
        for c in objectives:
            answer = engine.query(c, .001); check_answer(view, answer, c, .001)
            rows.append((service_response(answer), answer['last_dual_generation'],
                         answer['last_primal_generation'],
                         [{k:v for k,v in r.items() if k != 'seconds'} for r in answer['native_calls']]))
        if reference is None:
            reference = rows
        else:
            assert rows == reference
        if name.endswith('p0'):
            assert engine.memory_info()['retained_exact_duals'] == 0
    assert {r[0]['robust_threshold_decision'] for r in reference} == {False, True}


@pytest.mark.parametrize('name', CORE)
def test_exact_equality_is_false(view, name):
    answer = make_engine(name, view).query([0., 0.], 0.)
    assert answer['robust_threshold_decision'] is False
    assert answer['upper_bound'] == 0.
    check_answer(view, answer, [0., 0.], 0.)


def test_factors_change_only_expected_check_counts(view):
    records = {}
    for name in CORE:
        engine = make_engine(name, view)
        engine.query([1., 1.], .001)  # Seed the same last native proposal.
        first = engine.query([1., 1.1], .001)
        second = engine.query([1., 1.1], .001)
        records[name] = second
        assert first['route'] == second['route'] == 'last_dual'
        assert not second['native_calls']
        assert second['exact_dual_reuses'] == (1 if name == 's1_p1' else 2 if name == 's0_p1' else 0)
    assert records['s1_p1']['dual_candidates_exact_checked'] == 1
    assert records['s0_p1']['dual_candidates_exact_checked'] == 2
    assert records['s1_p0']['exact_dual_preparations'] == 1
    assert records['s0_p0']['exact_dual_preparations'] == 2
    assert records['s1_p1']['exact_dual_preparations'] == 0
    assert records['s1_p1']['deferred_duals'] == 1
    assert records['s1_p1']['replayed_duals'] == 0


def unknown_native(engine, bound_status=highspy.HighsModelStatus.kObjectiveBound):
    calls = []
    def native(c, bound, metrics):
        calls.append(bound)
        # All deferred candidates must have been completed before first call.
        assert metrics['replayed_duals'] == metrics['deferred_duals']
        assert metrics['replayed_primals'] == metrics['deferred_primals']
        status = bound_status if bound is not None else highspy.HighsModelStatus.kOptimal
        metrics['native_calls'].append({'seconds': 0., 'simplex_iterations': 0,
                                        'objective_bound': bound, 'model_status_code': int(status)})
        return {'lower': None, 'upper': None, 'primal': None, 'status': status}
    engine._native_run = native
    return calls


@pytest.mark.parametrize('name', CORE)
def test_unknown_replays_everything_and_clears_cutoff(view, name):
    engine = make_engine(name, view); calls = unknown_native(engine)
    answer = engine.query([1., 1.], .1)
    assert calls == [.1, None]
    assert answer['robust_threshold_decision'] is None
    assert answer['route'] == 'solver_unknown' and answer['cutoff_retries'] == 1
    assert answer['lower_bound'] < .1 < answer['upper_bound']
    assert answer['dual_candidates_exact_checked'] == 1
    assert answer['primal_candidates_exact_checked'] == 1
    check_answer(view, answer, [1., 1.], .1)


def test_false_replays_lower_before_answer(view):
    plain, screened = make_engine('s0_p1', view), make_engine('s1_p1', view)
    c = [-1., -1.]
    a, b = plain.query(c, .001), screened.query(c, .001)
    assert service_response(a) == service_response(b)
    assert b['deferred_duals'] == b['replayed_duals'] == 1
    assert not b['native_calls']


def test_overaggressive_screen_still_cannot_lose_a_decided_answer(view):
    engine = make_engine('s1_p1', view)
    # Fault injection: every screen lies. Complete fallback must recover true.
    engine._defer_dual = lambda *args: True
    engine._defer_primal = lambda *args: True
    engine._native_run = lambda *args: pytest.fail('Native solve before complete fallback')
    answer = engine.query([1., 1.], -.1)
    assert answer['robust_threshold_decision'] is True and answer['route'] == 'box_shortcut'
    assert answer['replayed_duals'] == 1
    check_answer(view, answer, [1., 1.], -.1)


@pytest.mark.parametrize('name', CORE)
def test_rejected_cached_dual_and_invalid_last_primal_are_counted(view, name):
    engine = make_engine(name, view)
    bad = Candidate((-1.,), (0.,)); engine.last_dual = bad; engine.dual_bank = [bad]
    # A state issued by another checker must not be accepted.
    engine.last_primal = ThresholdChecker(view).primal_state([.5,.5])
    answer = engine.query([-1., -1.], .001)
    assert answer['robust_threshold_decision'] is False
    assert answer['rejected_cached_duals'] == 1
    # Initial witness decides before last-primal is reached; no spurious count.
    assert answer['primal_candidates_exact_checked'] == 1
    check_answer(view, answer, [-1., -1.], .001)


def test_foreign_prepared_state_is_rebuilt(view):
    engine = make_engine('s0_p1', view)
    engine.zero.prepared = ThresholdChecker(view).dual_state([0.], [0.])
    answer = engine.query([-1., -1.], .001)
    assert answer['invalid_bank_entries'] == 1 and answer['exact_dual_preparations'] == 1
    assert engine.checker.owns_dual(engine.zero.prepared, (0.,), (0.,))
    check_answer(view, answer, [-1., -1.], .001)


@pytest.mark.parametrize('name', CORE)
def test_failed_native_primal_never_authorizes_false(view, name):
    engine = make_engine(name, view)
    def native(c, bound, metrics):
        metrics['native_calls'].append({'seconds':0., 'simplex_iterations':0})
        return {'lower': [0.], 'upper': [0.], 'primal': [0.,0.],
                'status': highspy.HighsModelStatus.kOptimal}
    engine._native_run = native
    answer = engine.query([1.,1.], .1)
    assert answer['robust_threshold_decision'] is None
    assert answer['rejected_primal_proposals'] == 1
    assert answer['last_primal_generation'] == 0
    check_answer(view, answer, [1.,1.], .1)


def test_interval_encloses_independent_fraction_lower_and_dot(view):
    rng = np.random.default_rng(52363); checker = FractionReplay(view)
    cases = [([0.,0.], [0.], [0.], 0.),
             ([1e-320,-1e-320], [1e-320], [0.], 1e-320),
             ([1e100,-1e100], [1e100], [1e100], 1.),
             ([1.,-1.], [1e308], [0.], 1e308)]
    cases += [(rng.normal(size=2).tolist(), rng.random(1).tolist(), rng.random(1).tolist(),
               float(rng.random()*10)) for _ in range(100)]
    for c, lo, hi, scale in cases:
        interval = lower_interval(view, c, scale, prepare_interval(view, lo, hi))
        exact = checker.exact_lower(checker.numbers(c), checker.numbers(lo), checker.numbers(hi),
                                    checker.numbers([scale])[0])
        assert interval[0] == -math.inf or Fraction.from_float(interval[0]) <= exact
        assert interval[1] == math.inf or exact <= Fraction.from_float(interval[1])
        dot = dot_interval(c, [.5,.5]); rational = sum(a*b for a,b in zip(checker.numbers(c),checker.numbers([.5,.5])))
        assert dot[0] == -math.inf or Fraction.from_float(dot[0]) <= rational
        assert dot[1] == math.inf or rational <= Fraction.from_float(dot[1])


def synthetic_view(seed, d=7, j=4):
    rng = np.random.default_rng(seed)
    matrix = rng.integers(-3,4,size=(j,d))/8
    center = matrix @ np.full(d,.5)
    return {'format':'qbridge-confidence-view-v1','lower':[0.]*d,'upper':[1.]*d,
            'rhs_lower':(center-.125).tolist(),'rhs_upper':(center+.125).tolist(),
            'feasible_point':[.5]*d,
            'matrix':{'indptr':[r*d for r in range(j+1)],'indices':list(range(d))*j,
                      'data':matrix.reshape(-1).tolist()}}


def test_signed_sparse_interval_and_four_arm_property_checks():
    rng = np.random.default_rng(137)
    for seed in range(8):
        view = synthetic_view(seed); audit = FractionReplay(view)
        engines = [make_engine(name,view) for name in CORE]
        for _ in range(20):
            c = (rng.integers(-8,9,size=7)/8).tolist()
            lo, hi = rng.random(4).tolist(), rng.random(4).tolist()
            scale = float(rng.random()*4)
            interval = lower_interval(view,c,scale,prepare_interval(view,lo,hi))
            exact = audit.exact_lower(audit.numbers(c),audit.numbers(lo),audit.numbers(hi),audit.numbers([scale])[0])
            assert Fraction.from_float(interval[0]) <= exact <= Fraction.from_float(interval[1])
            delta = float(rng.choice([-.5,0.,.001,.5]))
            answers = [engine.query(c,delta) for engine in engines]
            assert all(service_response(a) == service_response(answers[0]) for a in answers)
            for answer in answers: check_answer(view,answer,c,delta)


@pytest.mark.parametrize('delta,expected', [
    (math.nextafter(math.nextafter(.375,-math.inf),-math.inf),True),
    (math.nextafter(.375,-math.inf),None), (.375,False),
    (math.nextafter(.375,math.inf),False)])
def test_adjacent_float_thresholds_and_repeat_reuse(view,delta,expected):
    reference = None
    for name in CORE:
        engine = make_engine(name,view)
        answers = [engine.query([.5,.5],delta),engine.query([.5,.5],delta)]
        assert all(a['robust_threshold_decision'] is expected for a in answers)
        for answer in answers: check_answer(view,answer,[.5,.5],delta)
        current = [service_response(a) for a in answers]
        if reference is None: reference = current
        else: assert current == reference


def test_numerical_refusal_keeps_failed_check_cost(view):
    engine = make_engine('s0_p0',view)
    engine.zero = Candidate((-1.,),(0.,))  # Inject failed retained candidate validation.
    unknown_native(engine,highspy.HighsModelStatus.kOptimal)
    answer = engine.query([1.,1.],.1)
    assert answer['status'] == 'uncertain' and answer['certificate'] is None
    assert answer['rejected_cached_duals'] == 1
    assert answer['exact_dual_preparations'] == answer['dual_candidates_exact_checked'] == 1
    assert answer['verify_seconds'] >= answer['exact_dual_preparation_seconds'] > 0
    assert not FractionReplay(view).verify({'query_index':0,'answer':service_response(answer)},0,[1.,1.],.1)


def test_invalid_last_primal_is_rejected_before_unknown(view):
    engine = make_engine('s1_p1',view)
    engine.last_primal = ThresholdChecker(view).primal_state([.5,.5])
    unknown_native(engine,highspy.HighsModelStatus.kOptimal)
    answer = engine.query([1.,1.],.1)
    assert answer['robust_threshold_decision'] is None
    assert answer['rejected_cached_primals'] == 1 and answer['invalid_bank_entries'] == 1


def test_loaded_input_bytes_are_bound_before_decoding(view,tmp_path):
    import hashlib
    (tmp_path/'view.json').write_text(json.dumps(view))
    np.save(tmp_path/'queries.npy',np.array([[0.,0.]]))
    (tmp_path/'condition.json').write_text('{}')
    entry = {'id':'synthetic','directory':str(tmp_path),'query_shape':[1,2],
             'sha256':{name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
                       for name in ('view.json','queries.npy','condition.json')}}
    text, queries = load_input(entry)
    assert json.loads(text) == view and queries.tolist() == [[0.,0.]]
    (tmp_path/'view.json').write_text('{}')
    with pytest.raises(ValueError,match='Changed registered input'): load_input(entry)


def test_scaled_prepared_and_fresh_terms_are_identical(view):
    outputs = []
    for flag in (False, True):
        engine = CheckingScheduleEngine(view, screening=False, prepared=flag)
        candidate = Candidate((.1,), (.03,)); metrics = engine._new_metrics()
        rows = [engine._exact_candidate(candidate, [.2,-.3], .001, scale, metrics)
                for scale in (.1, .3, 1.7, 0.)]
        assert metrics['exact_dual_preparations'] == (1 if flag else 4)
        outputs.append(rows)
    assert outputs[0] == outputs[1]
    assert outputs[0][0]['dual_scale'] == .1
    assert outputs[0][0]['lower_multipliers'] == [.1]


@pytest.mark.parametrize('attack', ['index','objective','threshold','view','lower','upper','decision','outer','point','scale'])
def test_independent_replay_detects_corruption(view, attack):
    c, delta = [-1.,-1.], .001
    answer = make_engine('s0_p1', view).query(c, delta)
    item = {'query_index':0, 'answer':service_response(answer)}
    proof = item['answer']['certificate']
    if attack == 'index': item['query_index'] = False
    elif attack == 'objective': proof['objective'][0] = 2.
    elif attack == 'threshold': proof['decision_threshold'] = 2.
    elif attack == 'view': proof['view_digest'] = 'bad'
    elif attack == 'lower': proof['lower_proof']['claimed_lower'] = 3.
    elif attack == 'upper': proof['upper_proof']['claimed_upper'] = -3.
    elif attack == 'decision': proof['robust_threshold_decision'] = True
    elif attack == 'outer': item['answer']['lower_bound'] = 3.
    elif attack == 'point': proof['upper_proof']['point'] = [0.,0.]
    else:
        proof['lower_proof']['format'] = 'qbridge-scaled-dual-proof-v1'
        proof['lower_proof']['dual_scale'] = -1.
    with pytest.raises(ValueError): FractionReplay(view).verify(item, 0, c, delta)


@pytest.mark.parametrize('key', ['format','view_digest','objective','decision_threshold'])
def test_upper_record_rejects_unregistered_binding_fields(view,key):
    c = [-1.,-1.]; answer = make_engine('s0_p1',view).query(c,.001)
    item = {'query_index':0,'answer':service_response(answer)}
    item['answer']['certificate']['upper_proof'][key] = 'forged'
    with pytest.raises(ValueError,match='Upper schema'):
        FractionReplay(view).verify(item,0,c,.001)


def test_upper_witness_is_checked_against_current_bound_outer_objective(view):
    c = [-1.,-1.]; answer = make_engine('s0_p1',view).query(c,.001)
    item = {'query_index':0,'answer':service_response(answer)}
    # Copying a feasible witness is fine only after recomputing its exact dot
    # for the current objective. Rebind all outer/lower metadata, keep old U.
    changed = [-2.,-2.]; proof = item['answer']['certificate']
    proof['objective'] = changed; proof['lower_proof']['objective'] = changed
    proof['lower_proof']['claimed_lower'] = -5.; item['answer']['lower_bound'] = -5.
    with pytest.raises(ValueError,match='Invalid upper'):
        FractionReplay(view).verify(item,0,changed,.001)


@pytest.mark.parametrize('name', METHODS)
def test_toy_measurement_charges_initialization_and_all_replays(view, name, tmp_path):
    queries = np.array([[1.,1.], [1.,1.1], [-1.,-1.], [0.,0.]])
    reg = {'threshold':.001,'bank_size':16,'query_prefixes':[1,4]}
    result = measure(tmp_path, name, json.dumps(view), queries, reg)
    assert result['producer_seconds'] >= result['plan_initialization_seconds'] + result['view_decode_seconds']
    assert result['joint_seconds'] == result['producer_seconds'] + result['persistent_replay']['total_consumer_seconds']
    assert result['independent_fraction_replay']['valid'] == result['persistent_replay']['valid_proofs'] == 4
    assert result['stateless_replay']['valid_proofs'] == 4
    assert result['joint_prefixes']['4'] == result['joint_seconds']
    with gzip.open(tmp_path/f'{name}.jsonl.gz','rt') as stream:
        assert all(set(json.loads(line)['answer']) == {'status','route','lower_bound','upper_bound',
            'robust_threshold_decision','certificate','reason'} for line in stream)


def test_full_synthetic_comparison_and_bridge_availability(view, tmp_path):
    queries = np.array([[1.,1.], [1.,1.1], [-1.,-1.], [0.,0.]])
    reg = {'threshold':.001,'bank_size':16,'query_prefixes':[1,4]}
    for name in METHODS: measure(tmp_path, name, json.dumps(view), queries, reg)
    compared = compare_streams(tmp_path,expected_queries=len(queries))
    assert compared['core_same_evidence_and_paths']
    assert compared['relative_to_s1_p1']['original_prepared']['s1_p1_gains_decided'] == 2


@pytest.mark.parametrize('method,stream_kind,attack', [
    ('s0_p0','native','empty'), ('s0_p0','native','truncate'),
    ('all','native','empty'), ('all','service','empty'),
    ('s1_p1','native','reorder'), ('s1_p0','native','duplicate_index'),
    ('s0_p1','native','bool_index'), ('s1_p1','native','extra_row'),
    ('s1_p0','service','truncate'), ('s0_p0','service','duplicate_index'),
    ('s1_p1','service','reorder'), ('s0_p1','service','bool_index'),
    ('original_prepared','native','empty'), ('screened_auto','native','truncate'),
    ('original_prepared','service','duplicate_index'), ('screened_auto','native','reorder'),
])
def test_stream_comparison_rejects_missing_or_misbound_records(view,tmp_path,method,stream_kind,attack):
    queries = np.array([[1.,1.], [1.,1.1], [-1.,-1.], [0.,0.]])
    reg = {'threshold':.001,'bank_size':16,'query_prefixes':[1,4]}
    for name in METHODS: measure(tmp_path,name,json.dumps(view),queries,reg)
    for name in METHODS if method == 'all' else [method]:
        suffix = '_native' if stream_kind == 'native' else ''
        path = tmp_path/f'{name}{suffix}.jsonl.gz'
        with gzip.open(path,'rt') as stream: rows = [json.loads(line) for line in stream]
        if attack == 'empty': rows = []
        elif attack == 'truncate': rows = rows[:-1]
        elif attack == 'reorder': rows[0],rows[1] = rows[1],rows[0]
        elif attack == 'duplicate_index': rows[1]['query_index'] = rows[0]['query_index']
        elif attack == 'bool_index': rows[0]['query_index'] = False
        elif attack == 'extra_row': rows.append(copy.deepcopy(rows[-1]))
        with gzip.open(path,'wt') as stream:
            for row in rows: stream.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='stream length differs from plan|query index mismatch'):
        compare_streams(tmp_path,expected_queries=len(queries))


@pytest.mark.parametrize('expected', [0,False,-1,None])
def test_stream_comparison_requires_positive_planned_query_count(tmp_path,expected):
    with pytest.raises(ValueError,match='Positive planned query count required'):
        compare_streams(tmp_path,expected_queries=expected)


def test_failure_preserves_partial_records_and_elapsed_cost(view, tmp_path, monkeypatch):
    import icde2027.checking_schedule_v4 as module
    engine = make_engine('s0_p0',view); original = engine.query; calls = 0
    def broken(c, threshold):
        nonlocal calls
        calls += 1
        if calls == 2: raise ArithmeticError('synthetic failed second check')
        return original(c,threshold)
    engine.query = broken; monkeypatch.setattr(module,'make_engine',lambda *a:engine)
    with pytest.raises(ArithmeticError):
        measure(tmp_path,'s0_p0',json.dumps(view),np.array([[-1.,-1.],[1.,1.]]),
                {'threshold':.001,'bank_size':16,'query_prefixes':[1,2]})
    failure = json.loads((tmp_path/'s0_p0_failure.json').read_text())
    assert failure['completed_responses'] == 1 and failure['elapsed_seconds_including_failed_work'] > 0
    with gzip.open(tmp_path/'s0_p0_partial.jsonl.gz','rt') as stream: assert len(list(stream)) == 1


def test_orders_balanced_and_stable():
    reg = {'order_seed':2026110504,'methods':list(METHODS)}
    orders = [method_order(reg,'synthetic',r) for r in range(3)]
    assert all(sorted(o) == sorted(METHODS) for o in orders)
    assert all(len({o.index(name) for o in orders}) == 3 for name in METHODS)
    assert orders == [method_order(reg,'synthetic',r) for r in range(3)]


def test_gate_hashes_and_refuses_unsigned_or_modified_runtime():
    payload = gate_payload(CONFIG)
    with pytest.raises(ValueError): validate_gate(CONFIG,payload)
    gate = {**payload,'status':'approved','reviewer':'synthetic-test-only','reviewed_at_utc':'2000-01-01T00:00:00Z'}
    assert validate_gate(CONFIG,gate) == payload
    bad = copy.deepcopy(gate); bad['runtime_sha256']['icde2027/checking_schedule_v4.py'] = 'wrong'
    with pytest.raises(ValueError): validate_gate(CONFIG,bad)
    bad = copy.deepcopy(gate); bad['protocol_sha256'] = 'wrong'
    with pytest.raises(ValueError): validate_gate(CONFIG,bad)


def test_summary_requires_entire_grid_and_prespecified_both_factor_rule():
    reg = {'input_counts':{'toy':2},'timing_rounds':3,
           'factor_contrasts':[['s1_p0','s0_p0'],['s1_p1','s0_p1'],['s0_p1','s0_p0'],['s1_p1','s1_p0']],
           'bridge_contrasts':[['s1_p1','original_prepared'],['s1_p1','screened_auto']],
           'primary':{'group':'toy','contrasts':[['s1_p1','s0_p1'],['s1_p1','s1_p0']],'maximum_ratio':.8}}
    rows = [{'group':'toy','timing_round':r,'input_id':str(i),
             'methods':{m:{'joint_seconds':1. if m == 's1_p1' else 2.} for m in METHODS}}
             for r in range(3) for i in range(2)]
    assert summarize(reg,rows)['both_factors_meet_main_engineering_target']
    with pytest.raises(ValueError): summarize(reg,rows[:-1])
    rows[-1]['methods']['s0_p1']['joint_seconds'] = .1
    assert not summarize(reg,rows)['both_factors_meet_main_engineering_target']


def test_registered_scope_is_metadata_only():
    reg = json.loads(Path(CONFIG).read_text())
    assert len(reg['inputs']) == 63
    assert sum(e['query_shape'][0] for e in reg['inputs']) == 9000
    assert reg['expected_responses'] == 162000
    assert reg['input_counts'] == {'original_main':3,'threshold_engine':54,'source_derived':6}
    assert all(set(e['sha256']) == {'queries.npy','view.json','condition.json'} for e in reg['inputs'])


@pytest.mark.parametrize('fail', [False,True])
def test_synthetic_execute_snapshot_and_terminal_manifest(view,tmp_path,monkeypatch,fail):
    """Exercise the driver with ONE synthetic input; formal grid is never read."""
    import hashlib
    import icde2027.checking_schedule_v4 as module
    source = tmp_path/'synthetic_input'; source.mkdir()
    (source/'view.json').write_text(json.dumps(view))
    (source/'condition.json').write_text('{}')
    np.save(source/'queries.npy',np.array([[-1.,-1.]]))
    entry = {'id':'synthetic/only','group':'toy','directory':str(source),'query_shape':[1,2],
             'threshold':.001,'sha256':{name:hashlib.sha256((source/name).read_bytes()).hexdigest()
             for name in ('view.json','queries.npy','condition.json')}}
    reg = json.loads(CONFIG.read_text())
    reg.update(inputs=[entry],input_counts={'toy':1},timing_rounds=1,expected_responses=6,query_prefixes=[1])
    reg['primary']['group'] = 'toy'
    config = tmp_path/'synthetic_config.json'; config.write_text(json.dumps(reg))
    gate = {**gate_payload(config),'status':'approved','reviewer':'SYNTHETIC TEST ONLY',
            'reviewed_at_utc':'2000-01-01T00:00:00Z'}
    gate_path = tmp_path/'synthetic_gate.json'; gate_path.write_text(json.dumps(gate))
    checks = []
    def validate_synthetic_only(protocol):
        assert protocol['inputs'] == [entry]
        assert Path(protocol['inputs'][0]['directory']).is_relative_to(tmp_path)
        checks.append(True)
    monkeypatch.setattr(module,'validate_inputs',validate_synthetic_only)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        monkeypatch.setenv(key,'1')
    monkeypatch.setattr(module.os,'sched_getaffinity',lambda pid:{4})
    monkeypatch.setattr(module.os,'sched_setaffinity',lambda pid,cpus:None)
    output = tmp_path/'synthetic_output'
    if fail:
        def failed_measure(*args): raise RuntimeError('synthetic execution failure')
        monkeypatch.setattr(module,'measure',failed_measure)
        with pytest.raises(RuntimeError,match='synthetic execution failure'):
            module.execute(config,gate_path,output)
    else:
        module.execute(config,gate_path,output)
        assert len(checks) == 2  # Input integrity checked at start and finish.
    manifest = json.loads((output/'manifest.json').read_text())
    terminal = json.loads((output/('failure.json' if fail else 'completion.json')).read_text())
    assert manifest['status'] == terminal['status'] == ('failed' if fail else 'complete')
    assert manifest['finished_at_utc'] == terminal['finished_at_utc']
    assert manifest['started_at_utc'] <= manifest['finished_at_utc']
    assert (output/'protocol.json').read_bytes() == config.read_bytes()
    assert (output/'gate.json').read_bytes() == gate_path.read_bytes()
    assert (output/'design.md').read_bytes() == (module.ROOT/reg['design']).read_bytes()
    assert len(manifest['snapshot_files']) == len(gate['runtime_sha256'])+4
    validate_snapshot(output,manifest['snapshot_files'])
    copied = output/'source_snapshot/icde2027/checking_schedule_v4.py'
    assert copied.read_bytes() == Path(module.__file__).read_bytes()
    copied.write_text('synthetic corruption')
    with pytest.raises(ValueError,match='Saved snapshot changed'):
        validate_snapshot(output,manifest['snapshot_files'])
