"""Opt-in exact deterministic maps for iid Coat user blocks.

Historical full-panel code and results are untouched. CP quantiles, the
documented floating EB half-width kernel, and Hoeffding log/sqrt calls remain
trusted numerics; this module does not establish source transport or
collection assumptions.
"""
from __future__ import annotations

from fractions import Fraction as F
import math

import numpy as np

from .coat_full_block import validate_ratings, item_statistics
from .represented_intervals_v4 import (
    _trusted_eb_radius, fraction_record,
    map_normalized_intervals, outward_float, project_cell_intervals,
    row_representations, trusted_binomial_interval,
)


def rational(value):
    if isinstance(value, F):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Finite non-boolean real value required')
    return F(value)


def failure(value):
    value = rational(value)
    if not 0 < value < 1:
        raise ValueError('Failure probability must lie strictly between zero and one')
    return value


def unpack(pair):
    if (not isinstance(pair, list) or len(pair) != 2
            or any(not isinstance(x, str) for x in pair)):
        raise ValueError('Two decimal integer strings required')
    value = F(int(pair[0]), int(pair[1]))
    if fraction_record(value) != pair:
        raise ValueError('Canonical rational pair required')
    return value


def endpoints(lower, upper):
    if len(lower) != len(upper) or not lower:
        raise ValueError('Nonempty matching endpoint vectors required')
    lo = [rational(x) for x in lower]; hi = [rational(x) for x in upper]
    return {'exact_lower': [fraction_record(x) for x in lo],
            'exact_upper': [fraction_record(x) for x in hi],
            'statistical_intersection_nonempty': all(l <= u for l, u in zip(lo, hi)),
            'lower': [outward_float(x, lower=True) for x in lo],
            'upper': [outward_float(x, lower=False) for x in hi]}


def observation_base(ratings, *, radius, alpha):
    """Item-inclusion consistency uses N users, never pooled rating counts."""
    stats = item_statistics(ratings)
    alpha = failure(alpha); radius = rational(radius)
    if radius < 1:
        raise ValueError('Sensitivity radius must be at least one')
    n, d = stats['n_independent_users'], stats['n_items']
    conditional_failure, joint_failure = alpha/(2*d), alpha/(4*d)
    lower, upper, records = [], [], []
    for a in range(d):
        k, z, included = stats['successes'][a], stats['failures'][a], stats['included_users'][a]
        pl, pu = trusted_binomial_interval(k, included, conditional_failure)
        wl = max(pl/radius, 1-radius*(1-pl))
        wu = min(radius*pu, 1-(1-pu)/radius)
        js, _ = trusted_binomial_interval(k, n, joint_failure, lower_only=True)
        jf, _ = trusted_binomial_interval(z, n, joint_failure, lower_only=True)
        lower.append(max(wl, js)); upper.append(min(wu, 1-jf))
        records.append({'item': a, 'included_users': included, 'successes': k, 'failures': z,
                        **{key: fraction_record(value) for key, value in
                           [('conditional_lower', pl), ('conditional_upper', pu),
                            ('weight_lower', wl), ('weight_upper', wu),
                            ('joint_success_lower', js), ('joint_failure_lower', jf)]}})
    return {'construction': 'full_user_block_rational_observation_v5',
            'alpha_exact': fraction_record(alpha), 'radius_exact': fraction_record(radius),
            'n_independent_users': n, 'n_items': d, 'joint_denominator': n,
            'conditional_failure_exact': fraction_record(conditional_failure),
            'joint_failure_exact': fraction_record(joint_failure),
            'empty_items': stats['included_users'].count(0), 'items': records,
            'trusted_numerics': ['scipy.stats.beta.ppf'], **endpoints(lower, upper)}


def randomized_cell_base(ratings, *, alpha):
    stats = item_statistics(ratings); alpha = failure(alpha)
    n, d = stats['n_independent_users'], stats['n_items']
    pairs = [trusted_binomial_interval(k, count, alpha/d)
             for k, count in zip(stats['successes'], stats['included_users'])]
    return {'construction': 'full_user_block_rational_item_cp_v5',
            'n_independent_users': n, 'n_items': d, 'alpha_exact': fraction_record(alpha),
            'failure_per_item_exact': fraction_record(alpha/d),
            'included_users': stats['included_users'], 'successes': stats['successes'],
            'empty_items': stats['included_users'].count(0),
            'trusted_numerics': ['scipy.stats.beta.ppf'],
            **endpoints([x[0] for x in pairs], [x[1] for x in pairs])}


def group_statistics(ratings, rows, row_names):
    values = validate_ratings(ratings)
    _, representations = row_representations(rows, n_columns=values.shape[1])
    if len(row_names) != len(representations) or not representations:
        raise ValueError('One name per nonempty fixed group required')
    groups = []
    for j, (row, name) in enumerate(zip(representations, row_names)):
        if not row.indices or len(set(row.stored_weights)) != 1:
            raise ValueError('Fixed nonempty uniform groups required')
        members = list(row.indices)
        counts = (values[:, members] > 0).sum(axis=1)
        successes = (values[:, members] >= 4).sum(axis=1)
        active = [i for i, count in enumerate(counts) if count]
        scores = [F(int(successes[i]), int(counts[i])) for i in active]
        n = len(scores); total = sum(scores, F()); squared = sum((s*s for s in scores), F())
        mean = total/n if n else None
        variance = (squared-total*total/n)/(n-1) if n > 1 else None
        if variance is not None and variance < 0:
            raise ArithmeticError('Exact sample variance cannot be negative')
        groups.append({'index': j, 'name': name, 'members': members,
                       'group_size': len(members), 'kappa_exact': fraction_record(row.kappa),
                       'M_by_user': counts.tolist(), 'successes_by_user': successes.tolist(),
                       'active_user_indices': active, 'active_scores_exact': [fraction_record(s) for s in scores],
                       'n_active': n, 'sum_Z_exact': fraction_record(total),
                       'sum_Z_squared_exact': fraction_record(squared),
                       'mean_exact': fraction_record(mean) if mean is not None else None,
                       'unbiased_variance_exact': fraction_record(variance) if variance is not None else None})
    return {'n_independent_users': len(values), 'groups': groups,
            'score_definition': 'Exact k/M for active iid users; inactive users omitted, not zero-filled.'}


def randomized_group_base(statistics, *, alpha):
    groups = statistics['groups']; alpha = failure(alpha)
    if not groups:
        raise ValueError('Nonempty fixed groups required')
    delta = alpha/len(groups)
    lower, upper, diagnostics = [], [], []
    for group in groups:
        n = group['n_active']; width = None
        if group['group_size'] == 1:
            k = sum(group['successes_by_user'])
            lo, hi = trusted_binomial_interval(k, n, delta); branch = 'singleton_cp'
        elif n < 2:
            lo, hi = F(), F(1); branch = 'insufficient_eb'
        else:
            mean = unpack(group['mean_exact']); variance = unpack(group['unbiased_variance_exact'])
            width = _trusted_eb_radius({'n': n, 'variance': variance, 'bound': F(1)}, delta)
            lo, hi = max(F(), mean-width), min(F(1), mean+width)
            branch = 'active_user_empirical_bernstein'
        lower.append(lo); upper.append(hi)
        diagnostics.append({'index': group['index'], 'group_size': group['group_size'],
                            'n_active': n, 'construction': branch,
                            'failure_exact': fraction_record(delta),
                            'trusted_half_width_exact': fraction_record(width) if width is not None else None})
    return {'construction': 'full_user_block_rational_active_group_v5',
            'alpha_exact': fraction_record(alpha), 'failure_per_group_exact': fraction_record(delta),
            'n_groups': len(groups), 'n_independent_users': statistics['n_independent_users'],
            'groups': diagnostics,
            'trusted_numerics': ['scipy.stats.beta.ppf', 'math.log', 'math.sqrt', 'EB half-width floating arithmetic'],
            **endpoints(lower, upper)}


def map_moments(rows, base, *, construction, rho):
    lo = [unpack(x) for x in base['exact_lower']]
    hi = [unpack(x) for x in base['exact_upper']]
    if construction == 'group_score':
        result = map_normalized_intervals(rows, lo, hi, rho=rho)
    elif construction == 'cell_projection':
        result = project_cell_intervals(rows, lo, hi, rho=rho)
    else:
        raise ValueError('Unknown full-panel construction')
    return result.record()


def direct_all_scores(ratings, objective, *, alpha):
    """Fixed-query all-rating user averages under additional full-cell transport."""
    values = validate_ratings(ratings); alpha = failure(alpha)
    n, d = values.shape
    c = [rational(x) for x in objective]
    if len(c) != d:
        raise ValueError('Objective length differs from all items')
    count = (values > 0).sum(axis=1)
    if np.any(count == 0):
        raise ValueError('Direct all-rating control requires every user active')
    scores = [sum((d*c[a] for a in range(d) if values[u, a] >= 4), F())/int(count[u])
              for u in range(n)]
    lo, hi = min(F(), d*min(c)), max(F(), d*max(c))
    mean = sum(scores, F())/n
    variance = sum(((s-mean)**2 for s in scores), F())/(n-1) if n > 1 else None
    # Only log/sqrt are trusted here; the remaining expression stays rational.
    log_term = math.log(outward_float(1/alpha, lower=False))
    root_arg = outward_float(rational(log_term)/(2*n), lower=False)
    sqrt_term = rational(math.sqrt(root_arg))
    width = (hi-lo)*sqrt_term
    bound = mean-width
    return {'n_independent_user_blocks': n, 'scores_exact': [fraction_record(s) for s in scores],
            'estimate_exact': fraction_record(mean), 'estimate': float(mean),
            'ratings_per_user': count.tolist(),
            'sum_scores_exact': fraction_record(sum(scores, F())),
            'sum_scores_squared_exact': fraction_record(sum((s*s for s in scores), F())),
            'variance_exact': fraction_record(variance) if variance is not None else None,
            'sd_across_user_scores': math.sqrt(float(variance)) if variance is not None else None,
            'range_exact': [fraction_record(lo), fraction_record(hi)],
            'trusted_hoeffding_radius_exact': fraction_record(width),
            'trusted_log_value_exact': fraction_record(rational(log_term)),
            'sqrt_argument_upper': root_arg, 'trusted_sqrt_value_exact': fraction_record(sqrt_term),
            'lower_exact': fraction_record(bound), 'lower_bound': outward_float(bound, lower=True),
            'full_cell_static_transport_required': True, 'per_query_alpha_exact': fraction_record(alpha),
            'adaptive_query_guarantee': False,
            'trusted_numerics': ['math.log', 'math.sqrt']}
