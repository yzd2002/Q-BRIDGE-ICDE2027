"""Synthetic checks only: no new Coat confidence intervals or outcomes."""
from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from fractions import Fraction
import itertools
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
from scipy import sparse
from scipy.stats import beta

from . import coat_full_block as block
from . import research_io
from .coat_case import metadata_rows, fixed_queries
from .confidence_view import create_view
from .proof_check import CheckedView


def uniform_rows(groups, d):
    return sparse.csr_matrix([[1. / len(g) if i in g else 0. for i in range(d)] for g in groups])


def decode_fraction(value):
    return Fraction(value['numerator'], value['denominator'])


@pytest.mark.parametrize('d,sample', [(2, 1), (3, 1), (3, 2), (4, 2), (5, 3)])
def test_srs_conditional_group_identity_exact(d, sample):
    """Exhaust all potential binary vectors, groups, and assigned subsets."""
    groups = [g for size in range(1, d + 1) for g in itertools.combinations(range(d), size)]
    selected = list(itertools.combinations(range(d), sample))
    rows = uniform_rows(groups, d)
    for outcomes in itertools.product((0, 1), repeat=d):
        ratings = np.array([[4 if outcomes[i] else 1 for i in range(d)] for _ in selected])
        for u, assignment in enumerate(selected):
            ratings[u, [i for i in range(d) if i not in assignment]] = 0
        calculated = block.group_statistics(ratings, rows)['groups']
        for group, record in zip(groups, calculated):
            expected = Fraction(sum(outcomes[i] for i in group), len(group))
            by_m = {}
            for u, assignment in enumerate(selected):
                intersection = set(group).intersection(assignment)
                m = len(intersection)
                if m:
                    score = Fraction(sum(outcomes[i] for i in intersection), m)
                    by_m.setdefault(m, []).append(score)
                    index = record['active_user_indices'].index(u)
                    assert record['active_scores'][index] == float(score)
            for values in by_m.values():
                assert sum(values, Fraction()) / len(values) == expected
            all_active = [z for values in by_m.values() for z in values]
            assert sum(all_active, Fraction()) / len(all_active) == expected
            assert record['n_active'] <= len(selected)


def test_uniform_marginal_inclusion_without_srs_is_insufficient():
    # {0,1} or {2,3} gives every item inclusion 1/2 but biased active means.
    assignments = ({0, 1}, {2, 3})
    y, group = (0, 0, 1, 0), {0, 1, 2}
    inclusion = [Fraction(sum(i in a for a in assignments), 2) for i in range(4)]
    scores = [Fraction(sum(y[i] for i in a & group), len(a & group)) for a in assignments]
    assert inclusion == [Fraction(1, 2)] * 4
    assert sum(scores) / 2 == Fraction(1, 2)
    assert Fraction(sum(y[i] for i in group), len(group)) == Fraction(1, 3)


def test_fixed_size_thinning_preserves_weight_ratio_not_joint_probability():
    # Two latent types, positive mass on every size-two subset of three items.
    latent_mass = (Fraction(1, 3), Fraction(2, 3))
    assignments = ((0, 1), (0, 2), (1, 2))
    probabilities = ((Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)),
                     (Fraction(1, 6), Fraction(1, 2), Fraction(1, 3)))
    ratings = ((0, 1, 1), (1, 0, 1))
    size = 2
    for item in range(3):
        g = [sum((p for a, p in zip(assignments, row) if item in a), Fraction())
             for row in probabilities]
        e = sum(m * x for m, x in zip(latent_mass, g))
        thin_e, thin_g = e / size, [x / size for x in g]
        assert sum(latent_mass[u] * g[u] * ratings[u][item] for u in range(2)) == size * sum(
            latent_mass[u] * thin_g[u] * ratings[u][item] for u in range(2))
        for outcome in (0, 1):
            joint = sum(latent_mass[u] * g[u] for u in range(2) if ratings[u][item] == outcome)
            if not joint:
                continue
            conditional = [latent_mass[u] * g[u] / joint if ratings[u][item] == outcome else Fraction()
                           for u in range(2)]
            thin_joint = joint / size
            thin_conditional = [latent_mass[u] * thin_g[u] / thin_joint if ratings[u][item] == outcome else Fraction()
                                for u in range(2)]
            assert conditional == thin_conditional
            assert sum(conditional[u] * e / g[u] for u in range(2)) == sum(
                thin_conditional[u] * thin_e / thin_g[u] for u in range(2))


@pytest.mark.parametrize('bad', [[], [1, 2], [[[1]]], [[1.2]], [[-1]], [[6]], [[float('nan')]],
                               [[float('inf')]], [['4']], [[1 + 1j]]])
def test_invalid_rating_domain_rejected(bad):
    with pytest.raises(ValueError):
        block.validate_ratings(bad)


def test_selected_counts_and_shape_checked():
    with pytest.raises(ValueError, match='shape'):
        block.validate_ratings([[4, 0]], expected_shape=(2, 2))
    with pytest.raises(ValueError, match='selected'):
        block.validate_ratings([[4, 0], [4, 1]], items_per_user=1)
    with pytest.raises(ValueError):
        block.validate_ratings(sparse.csr_matrix([[4, 1]]))


def test_missing_not_negative_and_item_users_not_all_events():
    ratings = [[4, 1, 0], [1, 5, 0], [4, 0, 0]]
    stats = block.item_statistics(ratings)
    assert stats['n_independent_users'] == 3
    assert stats['included_users'] == [3, 2, 0]
    assert stats['successes'] == [2, 1, 0]
    assert stats['failures'] == [1, 1, 0]
    assert stats['total_ratings'] == 5
    assert block._checked_counts(stats) == [[[1, 2], [1, 1], [0, 0]]]


def test_same_item_counts_different_user_group_variance():
    first, second = np.array([[4, 4], [1, 1]]), np.array([[4, 1], [1, 4]])
    assert block.item_statistics(first) == block.item_statistics(second)
    rows = uniform_rows([(0, 1)], 2)
    a, b = [block.group_statistics(x, rows)['groups'][0] for x in (first, second)]
    assert a['active_scores'] == [1., 0.] and b['active_scores'] == [.5, .5]
    assert a['unbiased_variance'] == .5 and b['unbiased_variance'] == 0.
    assert a['sum_Z_squared'] == 1. and b['sum_Z_squared'] == .5


def test_active_user_weighting_not_zero_fill_or_pooled_weighting():
    ratings = np.array([[4, 1, 0], [4, 0, 1], [0, 0, 1]])
    record = block.group_statistics(ratings, uniform_rows([(0, 1)], 3))['groups'][0]
    assert record['M_by_user'] == [2, 1, 0]
    assert record['active_user_indices'] == [0, 1]
    assert record['active_scores'] == [.5, 1.]
    assert record['sum_Z'] / record['n_active'] == .75
    assert record['sum_Z'] / record['n_active'] != 2 / 3


def test_user_order_does_not_change_aggregate_moments():
    ratings = np.array([[4, 1, 1], [4, 4, 1], [1, 1, 1], [4, 0, 1]])
    rows = uniform_rows([(0, 1, 2), (0, 2)], 3)
    before, after = [block.group_statistics(v, rows)['groups'] for v in (ratings, ratings[[2, 0, 3, 1]])]
    for left, right in zip(before, after):
        for key in ('n_active', 'sum_Z', 'sum_Z_squared', 'unbiased_variance'):
            assert left[key] == right[key]


@pytest.mark.parametrize('n', [0, 1, 2])
def test_group_small_n_and_design_fixed_branch(n):
    values = np.zeros((3, 3), dtype=int)
    values[:n, :2] = 4
    rows = uniform_rows([(0,), (0, 1)], 3)
    statistics = block.group_statistics(values, rows)
    base = block.randomized_group_base(statistics, alpha=.05)
    singleton, larger = base['intervals']
    assert singleton['construction'] == 'singleton_cp'
    assert larger['construction'] == 'active_user_empirical_bernstein'
    assert singleton['n_active'] == larger['n_active'] == n
    assert base['failure_per_group'] == .025
    assert larger['ideal_lower'] == 0. and larger['ideal_upper'] == 1.
    if n == 0:
        assert singleton['ideal_lower'] == 0. and singleton['ideal_upper'] == 1.
    else:
        assert singleton['ideal_lower'] == beta.ppf(.025 / 2, n, 1)
    for group in statistics['groups']:
        assert (group['unbiased_variance'] is None) == (n < 2)


@pytest.mark.parametrize('all_success', [False, True])
def test_non_singleton_never_uses_cp_for_observed_binary_scores(all_success):
    values = np.full((100, 2), 4 if all_success else 1)
    result = block.randomized_group_base(block.group_statistics(values, uniform_rows([(0, 1)], 2)), alpha=.05)
    interval = result['intervals'][0]
    assert interval['construction'] == 'active_user_empirical_bernstein'
    h = 7 * math.log(4 / .05) / (3 * 99)
    assert interval['ideal_lower'] == max(0., float(all_success) - h)
    assert interval['ideal_upper'] == min(1., float(all_success) + h)


def test_eb_matches_independent_manual_calculation():
    values = np.tile(np.array([[4, 4, 1], [4, 1, 1], [4, 4, 4], [1, 1, 1]]), (40, 1))
    stats = block.group_statistics(values, uniform_rows([(0, 1, 2)], 3))
    group = stats['groups'][0]
    z, n = group['active_scores'], group['n_active']
    total, square = math.fsum(z), math.fsum(value * value for value in z)
    variance = (square - total * total / n) / (n - 1)
    radius = math.sqrt(2 * variance * math.log(4 / .025) / n) + 7 * math.log(4 / .025) / (3 * (n - 1))
    result = block.randomized_group_base(stats, alpha=.025)
    assert group['unbiased_variance'] == variance
    assert result['ideal_lower'] == [max(0., total / n - radius)]
    assert result['ideal_upper'] == [min(1., total / n + radius)]


def test_observation_cp_joint_denominator_and_failure_allocation():
    values = np.array([[4, 1, 0], [1, 4, 0], [4, 0, 0], [0, 1, 0]])
    alpha, radius = .05, 2.
    result = block.observation_base(values, radius=radius, alpha=alpha)
    assert result['joint_denominator'] == 4
    assert result['conditional_two_sided_failure_per_item'] == alpha / 6
    assert result['joint_one_sided_failure_per_endpoint'] == alpha / 12
    assert result['conditional_lower'][0] == beta.ppf(alpha / 12, 2, 2)
    assert result['conditional_upper'][0] == beta.ppf(1 - alpha / 12, 3, 1)
    assert result['joint_success_lower'][0] == beta.ppf(alpha / 12, 2, 3)
    assert result['joint_failure_lower'][0] == beta.ppf(alpha / 12, 1, 4)
    for i in range(3):
        pl, pu = result['conditional_lower'][i], result['conditional_upper'][i]
        assert result['lower'][i] == max(pl / radius, 1 - radius * (1 - pl), result['joint_success_lower'][i])
        assert result['upper'][i] == min(radius * pu, 1 - (1 - pu) / radius, 1 - result['joint_failure_lower'][i])
    assert result['lower'][2] == 0 and result['upper'][2] == 1


def test_randomized_cell_alpha_does_not_drop_empty_items():
    values = np.zeros((5, 300), dtype=int)
    values[:3, 0] = [4, 4, 1]
    result = block.randomized_cell_base(values, alpha=.05)
    assert result['n_items'] == 300 and result['failure_per_item'] == .05 / 300
    assert result['included_users'][0] == 3
    assert result['lower'][0] == beta.ppf(.05 / 600, 2, 2)
    assert result['upper'][0] == beta.ppf(1 - .05 / 600, 3, 1)
    assert result['empty_items'] == 299
    assert result['lower'][1:] == [0.] * 299 and result['upper'][1:] == [1.] * 299


@pytest.mark.parametrize('g', [3, 7, 10, 31])
@pytest.mark.parametrize('construction', ['group_score', 'cell_projection'])
@pytest.mark.parametrize('rho', [0., .02, .05])
def test_exact_kappa_and_rho_outward_mapping(g, construction, rho):
    rows = uniform_rows([tuple(range(g))], g)
    before = block.matrix_record(rows)
    lo = [.1 + .001 * i for i in range(g)]
    hi = [.8 + .001 * i for i in range(g)]
    base = {'alpha': .025, 'ideal_lower': [.1], 'ideal_upper': [.8], 'lower': lo, 'upper': hi}
    result = block.map_moments(rows, base, construction=construction, rho=rho)
    entry = result['rows'][0]
    w, kappa = Fraction(float(1 / g)), g * Fraction(float(1 / g))
    assert decode_fraction(entry['kappa']) == kappa
    if construction == 'group_score':
        exact_lo = kappa * (Fraction(.1) - Fraction(rho))
        exact_hi = kappa * (Fraction(.8) + Fraction(rho))
    else:
        exact_lo = w * sum(map(Fraction, lo)) - kappa * Fraction(rho)
        exact_hi = w * sum(map(Fraction, hi)) + kappa * Fraction(rho)
    assert decode_fraction(entry['exact_lower_before_rounding']) == exact_lo
    assert decode_fraction(entry['exact_upper_before_rounding']) == exact_hi
    assert Fraction(result['rhs_lower'][0]) <= exact_lo <= Fraction(math.nextafter(result['rhs_lower'][0], math.inf))
    assert Fraction(math.nextafter(result['rhs_upper'][0], -math.inf)) <= exact_hi <= Fraction(result['rhs_upper'][0])
    assert block.matrix_record(rows) == before
    if g in (3, 7, 10, 31):
        assert kappa != Fraction(float(g * float(1 / g)))


@pytest.mark.parametrize('values', [[[.2, .3]], [[0., 0.]], [[-.5, 1.5]], [[float('nan'), .5]]])
def test_invalid_uniform_rows_rejected(values):
    rows = sparse.csr_matrix(values)
    with pytest.raises(ValueError):
        block.row_metadata(rows, 2)


def test_duplicate_csr_item_rejected_before_normalization():
    rows = sparse.csr_matrix((np.array([.5, .5]), np.array([0, 0]), np.array([0, 2])), shape=(1, 2))
    assert not rows.has_canonical_format
    with pytest.raises(ValueError, match='Canonical'):
        block.row_metadata(rows, 2)


def test_proof_query_margin_and_view_binding():
    document = create_view([.5, 0.], [.75, .25], sparse.csr_matrix((0, 2)), [], [], highs_threads=1)
    queries = [{'objective': [1., -1.]}, {'objective': [-1., 1.]}]
    answers = block.answer_queries(document, queries, .001)
    assert [a['status'] for a in answers] == ['certified', 'unresolved']
    checker = CheckedView(document['view'])
    for index, answer in enumerate(answers):
        assert answer['query_index'] == index
        assert answer['document_sha256'] == block.canonical_hash(document)
        assert checker.verify(answer['certificate'], expected_objective=queries[index]['objective'],
                              expected_threshold=.001)['requested_query_checked']
    with pytest.raises(ValueError, match='objective'):
        checker.verify(answers[0]['certificate'], expected_objective=queries[1]['objective'], expected_threshold=.001)
    with pytest.raises(ValueError, match='threshold'):
        checker.verify(answers[0]['certificate'], expected_objective=queries[0]['objective'], expected_threshold=.002)
    changed = deepcopy(document['view']); changed['metadata'] = {'different': True}
    with pytest.raises(ValueError, match='bound'):
        CheckedView(changed).verify(answers[0]['certificate'])


def test_materialization_refusal_keeps_every_query_and_raw_candidate():
    document = create_view([.75, 0.], [.5, .25], sparse.csr_matrix((0, 2)), [], [], highs_threads=1)
    answers = block.answer_queries(document, [{'objective': [1., -1.]}] * 3, .001)
    assert document['status'] == 'uncertain' and document['candidate_view']['lower'][0] == .75
    assert len(answers) == 3
    assert all(a['status'] == 'uncertain' and a['certificate'] is None and a['lower_bound'] is None and
               a['reason'] == 'empty_statistical_intersection' and a['objective'] == [1., -1.] and
               a['decision_threshold'] == .001 and a['document_sha256'] == block.canonical_hash(document)
               for a in answers)


def test_ready_view_solver_refusal_is_retained(monkeypatch):
    document = create_view([.5, 0.], [.75, .25], sparse.csr_matrix((0, 2)), [], [], highs_threads=1)
    monkeypatch.setattr(block.QueryEngine, 'query', lambda *a, **k: {
        'status': 'uncertain', 'reason': 'synthetic_solver_failure', 'certificate': None,
        'lower_bound': None, 'route': 'solver'})
    answers = block.answer_queries(document, [{'objective': [1., -1.]}] * 2, .001)
    assert len(answers) == 2 and all(a['reason'] == 'synthetic_solver_failure' for a in answers)


def real_code_gate(tmp_path):
    protocol = tmp_path / 'protocol.json'
    protocol.write_text('{"synthetic_gate_test":true}\n')
    gate = {'format': 'qbridge-coat-full-block-gate-v1', 'passed': True,
            'independent_review': True, 'protocol_sha256': block.sha256(protocol),
            'implementation_sha256': {name: block.sha256(block.ROOT / 'icde2027' / name)
                                       for name in block.REQUIRED_RUNTIME}}
    path = tmp_path / 'gate.json'
    path.write_text(json.dumps(gate))
    return protocol, path, gate


@pytest.mark.parametrize('change', ['empty', 'false', 'not_independent', 'format', 'protocol', 'source'])
def test_gate_rejects_missing_or_stale_authority(tmp_path, change):
    protocol, path, gate = real_code_gate(tmp_path)
    if change == 'empty':
        gate['implementation_sha256'] = {}
    elif change == 'false':
        gate['passed'] = False
    elif change == 'not_independent':
        gate['independent_review'] = False
    elif change == 'format':
        gate['format'] = 'different_study'
    elif change == 'protocol':
        gate['protocol_sha256'] = '0' * 64
    elif change == 'source':
        gate['implementation_sha256']['aggregate.py'] = '0' * 64
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError):
        block.verify_gate(path, protocol)


@pytest.mark.parametrize('name', sorted(block.REQUIRED_RUNTIME))
def test_gate_requires_each_transitive_runtime_file(tmp_path, name):
    protocol, path, gate = real_code_gate(tmp_path)
    del gate['implementation_sha256'][name]
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match='every required'):
        block.verify_gate(path, protocol)


def test_gate_accepts_complete_current_source_and_rejects_loaded_drift(tmp_path, monkeypatch):
    protocol, path, gate = real_code_gate(tmp_path)
    assert block.verify_gate(path, protocol) == gate
    changed = dict(block.LOADED_RUNTIME_SHA256); changed['aggregate.py'] = '0' * 64
    monkeypatch.setattr(block, 'LOADED_RUNTIME_SHA256', changed)
    with pytest.raises(ValueError, match='Imported runtime'):
        block.verify_gate(path, protocol)


def synthetic_registered_case(tmp_path, monkeypatch):
    """A miniature filesystem fixture, never reading real rating outcomes."""
    actual_root = block.ROOT
    package = tmp_path / 'icde2027'; package.mkdir()
    for name in block.REQUIRED_RUNTIME:
        shutil.copy2(actual_root / 'icde2027' / name, package / name)
    reg = block.read(actual_root / 'icde2027/configs/coat_full_block_v1.json')
    reg['dataset'] = 'synthetic_data'; reg['original_case'] = 'synthetic_original'
    reg['unit'] = {**reg['unit'], 'n_independent_users': 4, 'n_items': 4,
                   'matrix_shape': [4, 4], 'observational_items_per_user': 2,
                   'randomized_items_per_user': 2}
    reg['compute']['cpu_affinity'] = sorted(os.sched_getaffinity(0))
    reg['groups'] = ['gender']
    reg['queries'] = {'primary': {'feature': 'front:yes', 'multiplier': 2.},
                      'features': ['front:yes'], 'multipliers': [.5, 2.]}
    reg['expected_view_documents'] = 61; reg['expected_query_responses'] = 122
    data = tmp_path / reg['dataset']; (data / 'user_item_features').mkdir(parents=True)
    values = {'train.ascii': [[4, 1, 0, 0], [0, 4, 1, 0], [0, 0, 4, 1], [1, 0, 0, 4]],
              'test.ascii': [[4, 0, 1, 0], [0, 4, 0, 1], [1, 0, 0, 4], [0, 1, 4, 0]],
              'user_item_features/item_features.ascii': [[1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]]}
    for filename, matrix in values.items():
        np.savetxt(data / filename, matrix, fmt='%d')
    names = ['gender:men', 'gender:women', 'front:yes']
    (data / 'user_item_features/item_features_map.txt').write_text('\n'.join(names) + '\n')
    sources = sorted(data.rglob('*.ascii')) + [data / 'user_item_features/item_features_map.txt']
    reg['source_files_sha256'] = {str(path.relative_to(tmp_path)): block.sha256(path) for path in sources}
    rows, _ = metadata_rows(np.array(values['user_item_features/item_features.ascii']), names, reg['groups'])
    core = fixed_queries(np.array(values['user_item_features/item_features.ascii']), names, reg)
    originals = [{**q, 'direct_randomized_scores': {'all_16_within_user': {
        'n_independent_user_blocks': 4, 'synthetic_reference': True}}} for q in core]
    old = tmp_path / reg['original_case']; old.mkdir()
    (old / 'protocol.json').write_text(json.dumps(reg))
    (old / 'queries.json').write_text(json.dumps(originals))
    reg['original_protocol_sha256'] = block.sha256(old / 'protocol.json')
    reg['original_queries_file_sha256'] = block.sha256(old / 'queries.json')
    reg['original_matrix_canonical_sha256'] = block.canonical_hash(block.matrix_record(rows))
    reg['query_core_sha256'] = block.canonical_hash(core)
    protocol = tmp_path / 'protocol.json'; protocol.write_text(json.dumps(reg))
    gate = {'format': 'qbridge-coat-full-block-gate-v1', 'passed': True, 'independent_review': True,
            'protocol_sha256': block.sha256(protocol),
            'implementation_sha256': {name: block.sha256(package / name) for name in block.REQUIRED_RUNTIME},
            'scope': 'Synthetic test fixture only; no authority for a real Coat run.'}
    gate_path = tmp_path / 'gate.json'; gate_path.write_text(json.dumps(gate))
    monkeypatch.setattr(block, 'ROOT', tmp_path)
    monkeypatch.setattr(research_io, 'ROOT', tmp_path)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.setenv(key, '1')
    return reg, Namespace(protocol=protocol, out=tmp_path / 'new_output', gate=gate_path)


def test_complete_synthetic_runner_retains_grid_provenance_and_controls(tmp_path, monkeypatch):
    reg, args = synthetic_registered_case(tmp_path, monkeypatch)
    result = block.run(args)
    assert result['view_documents'] == 61 and result['query_responses'] == 122
    assert result['proof_bearing_responses'] + result['no_proof_responses'] == 122
    manifest = block.read(args.out / 'manifest.json')
    assert manifest['status'] == 'complete' and manifest['complete_configurations'] == 12
    assert manifest['protocol_sha256'] == block.sha256(args.protocol)
    for name, digest in manifest['artifact_sha256'].items():
        assert block.sha256(args.out / name) == digest
    source_queries = block.read(tmp_path / reg['original_case'] / 'queries.json')
    retained = block.read(args.out / 'retained_direct_controls.json')
    for old, saved in zip(source_queries, retained['per_query']):
        assert saved['all_16_within_user'] == old['direct_randomized_scores']['all_16_within_user']
    base = block.read(args.out / 'base_intervals.json')
    assert set(base['observation']) == {'0.05', '0.025'}
    assert all(b['joint_denominator'] == 4 for by_radius in base['observation'].values() for b in by_radius.values())
    with pytest.raises(ValueError, match='existing'):
        block.run(args)


@pytest.mark.parametrize('field', ['source', 'original_protocol', 'original_queries', 'query_core', 'matrix'])
def test_input_preflight_binds_sources_old_design_queries_and_matrix(tmp_path, monkeypatch, field):
    reg, _ = synthetic_registered_case(tmp_path, monkeypatch)
    if field == 'source':
        reg['source_files_sha256'][next(iter(reg['source_files_sha256']))] = '0' * 64
    else:
        key = {'original_protocol': 'original_protocol_sha256', 'original_queries': 'original_queries_file_sha256',
               'query_core': 'query_core_sha256', 'matrix': 'original_matrix_canonical_sha256'}[field]
        reg[key] = '0' * 64
    with pytest.raises(ValueError):
        block.load_inputs(reg)


def test_input_preflight_requires_all_four_source_hashes_and_query_field_order(tmp_path, monkeypatch):
    reg, _ = synthetic_registered_case(tmp_path, monkeypatch)
    missing = deepcopy(reg)
    missing['source_files_sha256'].pop(next(iter(missing['source_files_sha256'])))
    with pytest.raises(ValueError, match='Every loaded source'):
        block.load_inputs(missing)
    reordered = deepcopy(reg)
    reordered['query_core_fields'] = list(reversed(block.QUERY_KEYS))
    with pytest.raises(ValueError, match='query field order'):
        block.load_inputs(reordered)


def test_runner_gate_failure_precedes_loading_or_output(tmp_path, monkeypatch):
    reg, args = synthetic_registered_case(tmp_path, monkeypatch)
    gate = block.read(args.gate); gate['implementation_sha256'] = {}
    args.gate.write_text(json.dumps(gate))
    def forbidden(_):
        raise AssertionError('Inputs must not be loaded after a bad gate')
    monkeypatch.setattr(block, 'load_inputs', forbidden)
    with pytest.raises(ValueError, match='every required'):
        block.run(args)
    assert not args.out.exists()


def test_runner_failure_keeps_partial_evidence(tmp_path, monkeypatch):
    reg, args = synthetic_registered_case(tmp_path, monkeypatch)
    original = block.answer_queries
    calls = 0
    def interrupted(*a, **k):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('synthetic interruption')
        return original(*a, **k)
    monkeypatch.setattr(block, 'answer_queries', interrupted)
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        block.run(args)
    manifest = block.read(args.out / 'manifest.json')
    assert manifest['status'] == 'failed'
    assert (args.out / 'base_intervals.json').is_file()
    partial = block.read(args.out / 'results.json')
    assert list(partial[0]['methods']) == ['observation']
    assert (args.out / 'failure_traceback.txt').is_file()
