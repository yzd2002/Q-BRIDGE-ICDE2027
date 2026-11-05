"""Registered Coat diagnostic with users, not individual ratings, as units.

This module never sends full-panel counts to the single-event materializer.
Statistical intervals assume iid users, source-static potential ratings and
the registered SRSWOR assignment. Exact LP proofs do not verify those premises.
The formal runner refuses to evaluate outcomes without an independent gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
from scipy import sparse

from .aggregate import binomial_intervals, conditional_box, empirical_bernstein_interval
from .coat_case import metadata_rows, fixed_queries
from .confidence_view import create_view, QueryEngine
from .proof_check import CheckedView
from .source_intervals import one_sided_binomial_lower
from .experiment import json_write
from .research_io import ROOT, start_run, finish_run, sha256


REQUIRED_RUNTIME = frozenset({
    '__init__.py', 'coat_full_block.py', 'coat_case.py', 'aggregate.py',
    'source_intervals.py', 'confidence_view.py', 'proof_check.py',
    'experiment.py', 'research_io.py',
})
METHODS = ('observation', 'randomized_group_score', 'randomized_cell_projection',
           'fusion_group_score', 'fusion_cell_projection')
QUERY_KEYS = ('feature', 'multiplier', 'policy', 'reference', 'objective')
LOADED_RUNTIME_SHA256 = {name: sha256(ROOT / 'icde2027' / name) for name in REQUIRED_RUNTIME}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf8')).hexdigest()


def fraction_record(value):
    value = Fraction(value)
    return {'numerator': value.numerator, 'denominator': value.denominator}


def as_fraction(value):
    value = float(value)
    require(math.isfinite(value), 'A finite binary64 value is required')
    return Fraction.from_float(value)


def outward_float(value, *, lower):
    """One outward binary64 rounding of an exact rational endpoint."""
    value = Fraction(value)
    result = float(value)
    require(math.isfinite(result), 'Mapped endpoint exceeds finite binary64 range')
    represented = Fraction.from_float(result)
    if (lower and represented > value) or (not lower and represented < value):
        result = math.nextafter(result, -math.inf if lower else math.inf)
    require(math.isfinite(result), 'Outward endpoint is not finite')
    require((Fraction.from_float(result) <= value) if lower else
            (Fraction.from_float(result) >= value), 'Outward rounding failed')
    return result


def validate_ratings(ratings, *, expected_shape=None, items_per_user=None):
    """A dense user/item matrix has at most one observation per user/item."""
    values = np.asarray(ratings)
    require(values.ndim == 2 and min(values.shape) > 0, 'Nonempty user/item matrix required')
    require(np.issubdtype(values.dtype, np.number) and not np.iscomplexobj(values),
            'Numeric integer ratings required')
    require(np.all(np.isfinite(values)) and np.all(values == np.floor(values)) and
            np.all((values >= 0) & (values <= 5)), 'Ratings must be integers in 0..5')
    if expected_shape is not None:
        require(tuple(values.shape) == tuple(expected_shape), 'Unexpected user/item shape')
    if items_per_user is not None:
        require(int(items_per_user) == items_per_user and items_per_user >= 0,
                'Integer nonnegative per-user size required')
        require(np.all((values > 0).sum(axis=1) == items_per_user),
                'Unexpected number of selected items per user')
    return values.astype(np.int64, copy=False)


def item_statistics(ratings):
    values = validate_ratings(ratings)
    included = (values > 0).sum(axis=0)
    positive = (values >= 4).sum(axis=0)
    negative = included - positive
    users, items = values.shape
    require(np.all(included <= users), 'Item count exceeds independent users')
    return {
        'n_independent_users': users, 'n_items': items,
        'included_users': included.tolist(), 'successes': positive.tolist(),
        'failures': negative.tolist(), 'total_ratings': int(included.sum()),
        'selected_items_per_user': (values > 0).sum(axis=1).tolist(),
    }


def row_metadata(rows, n_items):
    """Validate uniform supports without normalizing or changing stored T."""
    require(sparse.isspmatrix_csr(rows), 'Original CSR matrix required')
    require(rows.shape[1] == n_items and rows.shape[0] > 0, 'Moment matrix shape differs')
    require(rows.has_canonical_format and np.all(np.isfinite(rows.data)) and
            np.all(rows.data > 0), 'Canonical positive finite CSR entries required')
    result = []
    for j in range(rows.shape[0]):
        start, end = rows.indptr[j:j + 2]
        ids, weights = rows.indices[start:end], rows.data[start:end]
        require(len(ids) > 0 and np.all(weights == weights[0]),
                'Each fixed group must have identical positive stored coefficients')
        g, w = len(ids), float(weights[0])
        kappa = g * as_fraction(w)
        result.append({'index': j, 'members': ids.tolist(), 'group_size': g,
                       'stored_weight': w, 'kappa': fraction_record(kappa)})
    return result


def matrix_record(rows):
    return {'indptr': rows.indptr.tolist(), 'indices': rows.indices.tolist(),
            'data': rows.data.tolist()}


def group_statistics(ratings, rows, row_names=None):
    """Keep the within-user cross terms by forming a user score first."""
    values = validate_ratings(ratings)
    meta = row_metadata(rows, values.shape[1])
    if row_names is None:
        row_names = [str(j) for j in range(len(meta))]
    require(len(row_names) == len(meta), 'Group names differ from matrix rows')
    result = []
    for entry, name in zip(meta, row_names):
        ids = entry['members']
        m = (values[:, ids] > 0).sum(axis=1)
        k = (values[:, ids] >= 4).sum(axis=1)
        active = np.flatnonzero(m > 0)
        scores = [int(k[u]) / int(m[u]) for u in active]
        n = len(active)
        total, total_sq = math.fsum(scores), math.fsum(z * z for z in scores)
        variance = max(0., (total_sq - total * total / n) / (n - 1)) if n > 1 else None
        result.append({**entry, 'name': name, 'M_by_user': m.tolist(),
                       'successes_by_user': k.tolist(), 'active_user_indices': active.tolist(),
                       'active_scores': scores, 'n_active': n,
                       'sum_Z': total, 'sum_Z_squared': total_sq,
                       'unbiased_variance': variance,
                       'variance_null_reason': 'n_active<2' if n < 2 else None})
    return {'n_independent_users': len(values), 'groups': result,
            'score_definition': 'Active-user mean over selected members; inactive users are omitted, not zero-filled.'}


def observation_base(ratings, *, radius, alpha):
    stats = item_statistics(ratings)
    n, d = stats['n_independent_users'], stats['n_items']
    included = np.asarray(stats['included_users'])
    successes, failures = np.asarray(stats['successes']), np.asarray(stats['failures'])
    require(0 < alpha < 1, 'Source alpha must be in (0,1)')
    conditional_failure, joint_failure = alpha / (2 * d), alpha / (4 * d)
    pl, pu = binomial_intervals(successes, included, conditional_failure)
    wl, wu = conditional_box(pl, pu, radius)
    js = one_sided_binomial_lower(successes, np.full(d, n), joint_failure)
    jf = one_sided_binomial_lower(failures, np.full(d, n), joint_failure)
    lower, upper = np.maximum(wl, js), np.minimum(wu, 1 - jf)
    return {'construction': 'full_user_block_conditional_weight_plus_inclusion_consistency',
            'alpha': float(alpha), 'radius': float(radius), 'n_independent_users': n,
            'n_items': d, 'conditional_family_alpha': alpha / 2,
            'joint_family_alpha': alpha / 2,
            'conditional_two_sided_failure_per_item': conditional_failure,
            'joint_one_sided_failure_per_endpoint': joint_failure,
            'joint_denominator': n, 'included_users': stats['included_users'],
            'conditional_lower': pl.tolist(), 'conditional_upper': pu.tolist(),
            'weight_lower': wl.tolist(), 'weight_upper': wu.tolist(),
            'joint_success_lower': js.tolist(), 'joint_failure_lower': jf.tolist(),
            'lower': lower.tolist(), 'upper': upper.tolist(),
            'empty_items': int((included == 0).sum()),
            'box_contradiction': bool(np.any(lower > upper))}


def randomized_group_base(statistics, *, alpha):
    groups = statistics['groups']
    require(groups and 0 < alpha < 1, 'Nonempty groups and valid alpha required')
    delta = alpha / len(groups)
    intervals = []
    for group in groups:
        n = group['n_active']
        if group['group_size'] == 1:
            successes = sum(group['successes_by_user'])
            lo, hi = binomial_intervals(np.array([successes]), np.array([n]), delta)
            lo, hi, construction = float(lo[0]), float(hi[0]), 'singleton_cp'
        else:
            lo, hi = empirical_bernstein_interval(n, group['sum_Z'], group['sum_Z_squared'], 1., delta)
            construction = 'active_user_empirical_bernstein'
        intervals.append({'index': group['index'], 'group_size': group['group_size'],
                          'n_active': n, 'construction': construction,
                          'failure_probability': delta, 'range': [0., 1.],
                          'ideal_lower': lo, 'ideal_upper': hi})
    return {'construction': 'full_user_block_active_group_scores', 'alpha': float(alpha),
            'n_groups': len(groups), 'failure_per_group': delta, 'intervals': intervals,
            'ideal_lower': [x['ideal_lower'] for x in intervals],
            'ideal_upper': [x['ideal_upper'] for x in intervals]}


def randomized_cell_base(ratings, *, alpha):
    stats = item_statistics(ratings)
    require(0 < alpha < 1, 'Source alpha must be in (0,1)')
    d = stats['n_items']
    lower, upper = binomial_intervals(np.asarray(stats['successes']),
                                     np.asarray(stats['included_users']), alpha / d)
    return {'construction': 'full_user_block_item_cp', 'alpha': float(alpha),
            'n_items': d, 'n_independent_users': stats['n_independent_users'],
            'failure_per_item': alpha / d, 'included_users': stats['included_users'],
            'lower': lower.tolist(), 'upper': upper.tolist(),
            'empty_items': stats['included_users'].count(0)}


def map_moments(rows, base, *, construction, rho):
    """Map ideal group intervals to stored T exactly, including rho units."""
    require(math.isfinite(rho) and rho >= 0, 'Finite nonnegative normalized rho required')
    meta = row_metadata(rows, rows.shape[1])
    require(construction in {'group_score', 'cell_projection'}, 'Unknown interval construction')
    if construction == 'group_score':
        require(len(base['ideal_lower']) == len(meta) == len(base['ideal_upper']),
                'Group endpoints differ from rows')
    else:
        require(len(base['lower']) == rows.shape[1] == len(base['upper']),
                'Cell endpoints differ from columns')
    mapped = []
    for entry in meta:
        j, g = entry['index'], entry['group_size']
        w = as_fraction(entry['stored_weight']); kappa = g * w
        if construction == 'group_score':
            ideal_lo, ideal_hi = as_fraction(base['ideal_lower'][j]), as_fraction(base['ideal_upper'][j])
            unexpanded_lo, unexpanded_hi = kappa * ideal_lo, kappa * ideal_hi
        else:
            sum_lo = sum((as_fraction(base['lower'][i]) for i in entry['members']), Fraction())
            sum_hi = sum((as_fraction(base['upper'][i]) for i in entry['members']), Fraction())
            ideal_lo, ideal_hi = sum_lo / g, sum_hi / g
            unexpanded_lo, unexpanded_hi = w * sum_lo, w * sum_hi
        require(0 <= ideal_lo <= ideal_hi <= 1, 'Base intervals must lie in [0,1]')
        exact_lo = unexpanded_lo - kappa * as_fraction(rho)
        exact_hi = unexpanded_hi + kappa * as_fraction(rho)
        mapped.append({**entry, 'rho': float(rho),
                       'ideal_lower_exact': fraction_record(ideal_lo),
                       'ideal_upper_exact': fraction_record(ideal_hi),
                       'rho_exact': fraction_record(as_fraction(rho)),
                       'exact_lower_before_rounding': fraction_record(exact_lo),
                       'exact_upper_before_rounding': fraction_record(exact_hi),
                       'rhs_lower': outward_float(exact_lo, lower=True),
                       'rhs_upper': outward_float(exact_hi, lower=False)})
    return {'construction': construction, 'source_alpha': base['alpha'], 'rho': float(rho),
            'rho_units': 'ideal normalized uniform group mean',
            'rounding': 'exact binary64 rational map followed by one outward binary64 rounding',
            'rows': mapped, 'rhs_lower': [x['rhs_lower'] for x in mapped],
            'rhs_upper': [x['rhs_upper'] for x in mapped]}


def answer_queries(document, queries, threshold):
    """Keep every requested query and bind each proof or refusal to it."""
    require(math.isfinite(threshold), 'Finite threshold required')
    answers = []
    if document.get('status') != 'ready':
        require(document.get('status') == 'uncertain' and document.get('reason'),
                'Unknown materialization status')
        for i, query in enumerate(queries):
            answers.append({'query_index': i, 'objective': query['objective'],
                            'decision_threshold': float(threshold), 'status': 'uncertain',
                            'route': 'materialization_refusal', 'reason': document['reason'],
                            'document_sha256': canonical_hash(document),
                            'lower_bound': None, 'certificate': None})
        return answers
    engine = QueryEngine(document['view'], reuse=False, highs_threads=1)
    checker = CheckedView(document['view'])
    for i, query in enumerate(queries):
        answer = engine.query(query['objective'], threshold)
        certificate = answer['certificate']
        if certificate is not None:
            checked = checker.verify(certificate, expected_objective=query['objective'],
                                     expected_threshold=threshold)
            require(checked['valid_lp_lower_bound'] and checked['requested_query_checked'],
                    'Saved proof failed verification or query binding')
            require(answer['lower_bound'] == checked['claimed_lower'] and
                    (answer['status'] == 'certified') == checked['certifies_declared_threshold'],
                    'Saved answer differs from the exact checked evidence')
        else:
            require(answer['status'] == 'uncertain' and answer['lower_bound'] is None and answer.get('reason'),
                    'A missing proof requires an explicit uncertain solver response')
        answers.append({**answer, 'query_index': i, 'objective': query['objective'],
                        'decision_threshold': float(threshold), 'document_sha256': canonical_hash(document)})
    return answers


def verify_gate(gate_path, protocol_path):
    gate = read(gate_path)
    require(gate.get('format') == 'qbridge-coat-full-block-gate-v1' and
            gate.get('passed') is True and gate.get('independent_review') is True,
            'Independent full-block implementation gate has not passed')
    require(gate.get('protocol_sha256') == sha256(protocol_path), 'Gate protocol differs')
    sources = gate.get('implementation_sha256')
    require(isinstance(sources, dict) and REQUIRED_RUNTIME <= set(sources),
            'Gate must bind every required local runtime module')
    package = (ROOT / 'icde2027').resolve()
    for name, expected in sources.items():
        path = (package / name).resolve()
        require(path.is_relative_to(package) and path.is_file(), 'Invalid gate source path')
        require(isinstance(expected, str) and len(expected) == 64 and sha256(path) == expected,
                'Gate source no longer matches: ' + name)
        if name in LOADED_RUNTIME_SHA256:
            require(LOADED_RUNTIME_SHA256[name] == expected,
                    'Imported runtime differs from gate: ' + name)
    return gate


def load_inputs(registration):
    """Read-only input preflight; no intervals, policy results or optimization."""
    reg = registration
    expected_sources = {str(Path(reg['dataset']) / name) for name in (
        'train.ascii', 'test.ascii', 'user_item_features/item_features.ascii',
        'user_item_features/item_features_map.txt')}
    require(set(reg['source_files_sha256']) == expected_sources,
            'Every loaded source file must have exactly one registered hash')
    require(tuple(reg['query_core_fields']) == QUERY_KEYS, 'Registered query field order differs')
    paths = {key: ROOT / key for key in reg['source_files_sha256']}
    for name, path in paths.items():
        require(sha256(path) == reg['source_files_sha256'][name], 'Source bytes changed: ' + name)
    data = ROOT / reg['dataset']; old = ROOT / reg['original_case']
    require(sha256(old / 'protocol.json') == reg['original_protocol_sha256'], 'Original protocol changed')
    require(sha256(old / 'queries.json') == reg['original_queries_file_sha256'], 'Original queries changed')
    old_reg = read(old / 'protocol.json')
    for key in ('groups', 'queries', 'reference', 'alpha', 'threshold', 'sensitivity_radii',
                'transport_tolerances', 'primary_radius', 'primary_transport_tolerance',
                'primary_randomized_intervals'):
        require(reg[key] == old_reg[key], 'Prespecified case field changed: ' + key)
    require(tuple(reg['methods']) == METHODS, 'Method identities/order changed')
    require(reg['source_alpha'] == {'standalone': reg['alpha'],
                                  'fusion_observation': reg['alpha'] / 2,
                                  'fusion_randomized': reg['alpha'] / 2}, 'Source budgets differ')
    unit = reg['unit']
    require(unit['matrix_shape'] == [unit['n_independent_users'], unit['n_items']] and
            unit['zero_is_missing'] is True and unit['no_user_or_item_removal'] is True and
            unit['no_thinning'] is True, 'Registered sample-unit contract differs')
    observation = validate_ratings(np.loadtxt(data / 'train.ascii'), expected_shape=unit['matrix_shape'],
                                   items_per_user=unit['observational_items_per_user'])
    randomized = validate_ratings(np.loadtxt(data / 'test.ascii'), expected_shape=unit['matrix_shape'],
                                  items_per_user=unit['randomized_items_per_user'])
    features = np.loadtxt(data / 'user_item_features/item_features.ascii')
    names = [s.strip() for s in (data / 'user_item_features/item_features_map.txt').read_text().splitlines()]
    require(features.shape == (unit['n_items'], len(names)) and np.all(np.isfinite(features)) and
            np.all((features == 0) | (features == 1)) and len(set(names)) == len(names),
            'Unexpected metadata domains')
    rows, row_names = metadata_rows(features.astype(np.int64), names, reg['groups'])
    meta = row_metadata(rows, unit['n_items'])
    require(canonical_hash(matrix_record(rows)) == reg['original_matrix_canonical_sha256'],
            'Original matrix changed')
    queries = fixed_queries(features.astype(np.int64), names, reg)
    require(canonical_hash(queries) == reg['query_core_sha256'], 'Original query core changed')
    original_queries = read(old / 'queries.json')
    require([{key: q[key] for key in QUERY_KEYS} for q in original_queries] == queries,
            'Original saved query core differs')
    require(reg['expected_view_documents'] == len(reg['sensitivity_radii']) *
            len(reg['transport_tolerances']) * len(METHODS) + 1 and
            reg['expected_query_responses'] == reg['expected_view_documents'] * len(queries),
            'Registered output census differs')
    return observation, randomized, rows, row_names, meta, queries, original_queries


def _checked_counts(stats):
    return [[list(pair) for pair in zip(stats['failures'], stats['successes'])]]


def _answer_summary(answer):
    return {key: answer.get(key) for key in ('status', 'lower_bound', 'route', 'reason')}


def run(args):
    protocol = Path(args.protocol).resolve()
    out, gate_path = Path(args.out).resolve(), Path(args.gate).resolve()
    gate = verify_gate(gate_path, protocol)
    reg = read(protocol)
    require(not out.exists(), 'Refusing an existing output directory')
    require(sorted(os.sched_getaffinity(0)) == reg['compute']['cpu_affinity'],
            'CPU affinity differs from registration')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        require(os.environ.get(key) == '1', 'Set ' + key + '=1 before execution')
    observation, randomized, rows, row_names, meta, queries, old_queries = load_inputs(reg)
    # Recheck immediately before snapshot/start; no new statistical outcome exists yet.
    verify_gate(gate_path, protocol)
    manifest = start_run(out, protocol)
    try:
        for name, expected in manifest['source_sha256'].items():
            require(sha256(out / 'source_snapshot' / name) == expected, 'Snapshot mismatch: ' + name)
        for name in REQUIRED_RUNTIME:
            require(manifest['source_sha256'][name] == gate['implementation_sha256'][name],
                    'Snapshot differs from reviewed runtime: ' + name)
        require(manifest['protocol_sha256'] == gate['protocol_sha256'], 'Snapshot protocol differs from gate')
        shutil.copy2(gate_path, out / 'implementation_gate.json')
        require(read(out / 'implementation_gate.json') == gate, 'Gate changed during snapshot')
        manifest.update(study='coat_full_user_block_v1', command=[sys.executable, *sys.argv],
                        implementation_gate_sha256=sha256(out / 'implementation_gate.json'),
                        independent_statistical_units=reg['unit']['n_independent_users'],
                        complete_configurations=0)
        json_write(out / 'manifest.json', manifest)
        ostats, rstats = item_statistics(observation), item_statistics(randomized)
        gstats = group_statistics(randomized, rows, row_names)
        json_write(out / 'counts.json', {'n_independent_users': len(observation),
                   'observation': _checked_counts(ostats), 'randomized': _checked_counts(rstats),
                   'per_item': {'observation': ostats, 'randomized': rstats}})
        json_write(out / 'group_statistics.json', gstats)
        json_write(out / 'moment_rows.json', {'matrix': matrix_record(rows), 'shape': list(rows.shape),
                   'row_names': row_names, 'row_metadata': meta,
                   'canonical_sha256': canonical_hash(matrix_record(rows))})
        json_write(out / 'queries.json', queries)
        json_write(out / 'retained_direct_controls.json', {
            'source_queries_file': reg['original_case'] + '/queries.json',
            'source_queries_file_sha256': reg['original_queries_file_sha256'],
            'scope': 'Unchanged original all-16 within-user direct score. N remains 290; each fixed-query bound requires static full-cell transport and is not a simultaneous rule for selecting across queries.',
            'per_query': [{'query_index': i, 'query_core': queries[i],
                           'all_16_within_user': old['direct_randomized_scores']['all_16_within_user']}
                          for i, old in enumerate(old_queries)]})
        json_write(out / 'data_provenance.json', {
            'source_sha256': reg['source_files_sha256'], 'original_case': reg['original_case'],
            'source_url': 'https://www.cs.cornell.edu/~schnabts/mnar/coat.zip',
            'paper_url': 'https://proceedings.mlr.press/v48/schnabel16.html',
            'license': 'https://creativecommons.org/licenses/by-nc/4.0/',
            'transformations': 'All original user/item positions; binary rating>=4 on observed nonzero ratings; user-first group aggregation; no thinning or imputation.',
            'independent_units': len(observation), 'source_overlap_cells': int(((observation > 0) & (randomized > 0)).sum()),
            'ratings_per_source': [ostats['total_ratings'], rstats['total_ratings']],
            'assumptions': reg['assumptions'], 'known_population_truth': False,
            'group_source_assumptions_empirically_verified': False})
        base = {'observation': {}, 'randomized_group_score': {}, 'randomized_cells': {}}
        mappings = {}
        for alpha in (reg['alpha'], reg['alpha'] / 2):
            key = str(alpha)
            base['observation'][key] = {str(radius): observation_base(observation, radius=radius, alpha=alpha)
                                        for radius in reg['sensitivity_radii']}
            base['randomized_group_score'][key] = randomized_group_base(gstats, alpha=alpha)
            base['randomized_cells'][key] = randomized_cell_base(randomized, alpha=alpha)
            mappings[key] = {}
            for construction, family in (('group_score', 'randomized_group_score'),
                                          ('cell_projection', 'randomized_cells')):
                mappings[key][construction] = {str(rho): map_moments(rows, base[family][key],
                                                   construction=construction, rho=rho)
                                               for rho in reg['transport_tolerances']}
        json_write(out / 'base_intervals.json', base)
        json_write(out / 'mapped_moments.json', mappings)
        empty = sparse.csr_matrix((0, observation.shape[1]))
        all_results = []
        for radius in reg['sensitivity_radii']:
            for rho in reg['transport_tolerances']:
                record = {'radius': radius, 'rho': rho, 'methods': {}}
                all_results.append(record)
                for method in METHODS:
                    source_alpha = reg['alpha'] / 2 if method.startswith('fusion_') else reg['alpha']
                    if method == 'observation' or method.startswith('fusion_'):
                        ob = base['observation'][str(source_alpha)][str(radius)]
                        lo, hi = ob['lower'], ob['upper']
                    else:
                        lo, hi = np.zeros(observation.shape[1]), np.ones(observation.shape[1])
                    if method == 'observation':
                        t, rl, ru = empty, [], []
                        construction = None
                    else:
                        construction = method.split('_', 1)[1]
                        mapped = mappings[str(source_alpha)][construction][str(rho)]
                        t, rl, ru = rows, mapped['rhs_lower'], mapped['rhs_upper']
                    document = create_view(lo, hi, t, rl, ru, highs_threads=1, metadata={
                        'case': 'coat_full_user_block', 'method': method,
                        'target': reg['target'], 'n_independent_users': len(observation),
                        'radius': radius if method == 'observation' or method.startswith('fusion_') else None,
                        'rho': rho if method != 'observation' else None,
                        'source_alpha': source_alpha, 'construction': construction,
                        'protocol_sha256': manifest['protocol_sha256'],
                        'original_matrix_sha256': reg['original_matrix_canonical_sha256'],
                        'assumptions_empirically_verified': False})
                    record['methods'][method] = {'document': document,
                        'answers': answer_queries(document, queries, reg['threshold'])}
                    json_write(out / 'results.json', all_results)
                manifest['complete_configurations'] += 1
                json_write(out / 'manifest.json', manifest)
                counts = {name: sum(a['status'] == 'certified' for a in result['answers'])
                          for name, result in record['methods'].items()}
                print(f'Coat full blocks Lambda={radius}, rho={rho}: {counts}', flush=True)
        cells = base['randomized_cells'][str(reg['alpha'])]
        document = create_view(cells['lower'], cells['upper'], empty, [], [], highs_threads=1,
                               metadata={'case': 'coat_full_user_block', 'source': 'R',
                                         'assumption': 'static full-cell transport', 'alpha': reg['alpha'],
                                         'n_independent_users': len(randomized), 'target': reg['target'],
                                         'protocol_sha256': manifest['protocol_sha256']})
        full_cell = {'document': document, 'answers': answer_queries(document, queries, reg['threshold'])}
        json_write(out / 'full_cell_control.json', full_cell)
        query_index = next(i for i, q in enumerate(queries) if all(q[k] == v for k, v in reg['queries']['primary'].items()))
        primary = next(r for r in all_results if r['radius'] == reg['primary_radius'] and
                       r['rho'] == reg['primary_transport_tolerance'])
        primary_methods = {name: _answer_summary(result['answers'][query_index])
                           for name, result in primary['methods'].items()}
        outputs = [result for record in all_results for result in record['methods'].values()] + [full_cell]
        census = Counter(answer['status'] for output in outputs for answer in output['answers'])
        proofs = sum(answer['certificate'] is not None for output in outputs for answer in output['answers'])
        require(len(outputs) == reg['expected_view_documents'] and sum(census.values()) == reg['expected_query_responses'],
                'Output census differs from registration')
        rescued = (primary_methods['fusion_group_score']['status'] == 'certified' and
                   all(primary_methods[m]['status'] == 'unresolved' for m in ('observation', 'randomized_group_score')))
        summary = {'primary_query_index': query_index, 'primary_query': queries[query_index],
                   'primary_methods': primary_methods,
                   'primary_full_cell_control': _answer_summary(full_cell['answers'][query_index]),
                   'primary_descriptive_hypothesis_observed': rescued,
                   'all_settings_certified_counts': [{k: r[k] for k in ('radius', 'rho')} |
                        {name: sum(a['status'] == 'certified' for a in result['answers'])
                         for name, result in r['methods'].items()} for r in all_results],
                   'view_documents': len(outputs), 'query_responses': sum(census.values()),
                   'answer_status_counts': dict(census), 'proof_bearing_responses': proofs,
                   'no_proof_responses': sum(census.values()) - proofs,
                   'ready_view_documents': sum(x['document']['status'] == 'ready' for x in outputs),
                   'refused_view_documents': sum(x['document']['status'] == 'uncertain' for x in outputs),
                   'interpretation': 'Retrospective diagnostic, no population truth or coverage estimate; same users and fixed original queries; no result-driven method selection.'}
        json_write(out / 'summary.json', summary)
        verify_gate(gate_path, protocol)
        for name, expected in reg['source_files_sha256'].items():
            require(sha256(ROOT / name) == expected, 'Raw source changed during execution')
        old = ROOT / reg['original_case']
        require(sha256(old / 'protocol.json') == reg['original_protocol_sha256'] and
                sha256(old / 'queries.json') == reg['original_queries_file_sha256'],
                'Original case bindings changed during execution')
        for name, expected in manifest['source_sha256'].items():
            require(sha256(out / 'source_snapshot' / name) == expected, 'Frozen snapshot changed')
        finish_run(out, manifest, configurations=len(all_results), queries_per_configuration=len(queries),
                   view_documents=len(outputs), query_responses=sum(census.values()),
                   source_files_sha256=reg['source_files_sha256'],
                   artifact_sha256={p.name: sha256(p) for p in sorted(out.glob('*.json'))
                                    if p.name != 'manifest.json'})
        return summary
    except Exception as error:
        manifest.update(status='failed', error_type=type(error).__name__, error=str(error))
        json_write(out / 'manifest.json', manifest)
        (out / 'failure_traceback.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'icde2027/configs/coat_full_block_v1.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    run(parser.parse_args())
