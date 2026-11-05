"""Explicit normalized-mean semantics and outward maps for stored binary64 LPs.

Only the deterministic rational arithmetic/mapping is exact. SciPy beta
quantiles and empirical-Bernstein log/sqrt remain trusted numerical routines.
This is an opt-in v4 module; it does not replace any historical entry point.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math

import numpy as np
from scipy import sparse
from scipy.stats import beta


class UnusableStatisticalInterval(ValueError):
    """A statistical intersection is empty; it must not authorize a view."""


def _fraction(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return Fraction.from_float(float(value))
    raise ValueError('A finite real number or Fraction is required')


def fraction_record(value):
    value = _fraction(value)
    return [str(value.numerator), str(value.denominator)]


def outward_float(value, *, lower):
    """Enclose ONE already-exact rational endpoint by a finite binary64 value."""
    exact = _fraction(value)
    try:
        result = float(exact)
    except OverflowError as error:
        raise ValueError('Endpoint outside finite binary64 range') from error
    if not math.isfinite(result):
        raise ValueError('Endpoint outside finite binary64 range')
    represented = Fraction.from_float(result)
    if (lower and represented > exact) or (not lower and represented < exact):
        result = math.nextafter(result, -math.inf if lower else math.inf)
    if not math.isfinite(result):
        raise ValueError('Outward endpoint outside finite binary64 range')
    if not ((Fraction.from_float(result) <= exact) if lower else
            (Fraction.from_float(result) >= exact)):
        raise ArithmeticError('Outward conversion failed')
    return result


@dataclass(frozen=True)
class Row:
    indices: tuple[int, ...]
    stored_weights: tuple[Fraction, ...]
    kappa: Fraction

    @property
    def ideal_weights(self):
        return tuple(w / self.kappa for w in self.stored_weights) if self.kappa else ()


def row_representations(rows, *, n_columns=None):
    """Read stored coefficients without renormalizing or summing duplicates."""
    if sparse.issparse(rows):
        if not sparse.isspmatrix_csr(rows) or not rows.has_canonical_format:
            raise ValueError('Sparse rows must be canonical CSR; no silent consolidation')
        if np.iscomplexobj(rows.data):
            raise ValueError('Real stored coefficients required')
        matrix = rows.astype(float, copy=True)
    else:
        values = np.asarray(rows)
        if values.ndim != 2 or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise ValueError('A two-dimensional real stored binary64 matrix is required')
        matrix = sparse.csr_matrix(values.astype(float))
    if matrix.shape[1] < 1 or (n_columns is not None and matrix.shape[1] != n_columns):
        raise ValueError('Moment columns do not match cells')
    if not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0):
        raise ValueError('Stored coefficients must be finite and nonnegative')
    representations = []
    for j in range(matrix.shape[0]):
        start, end = matrix.indptr[j:j + 2]
        positive = [(int(matrix.indices[i]), _fraction(matrix.data[i]))
                    for i in range(start, end) if matrix.data[i] > 0]
        weights = tuple(w for _, w in positive)
        representations.append(Row(tuple(i for i, _ in positive), weights, sum(weights, Fraction())))
    return matrix, tuple(representations)


def canonical_probabilities(q, *, semantics):
    """Declare actual conditional assignment probabilities; never infer them.

    exact_probabilities requires exact row sums of one. normalize_weights
    explicitly declares the design law proportional to the supplied weights.
    Neither describes an arbitrary historical RNG/collector without evidence.
    """
    if semantics not in {'exact_probabilities', 'normalize_weights'}:
        raise ValueError('Choose exact_probabilities or normalize_weights explicitly')
    values = np.asarray(q, dtype=object)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError('q must have nonempty [context, action] shape')
    output = []
    for raw in values:
        row = tuple(_fraction(v) for v in raw)
        total = sum(row, Fraction())
        if min(row) < 0 or total <= 0:
            raise ValueError('Design weights must be nonnegative with positive total')
        if semantics == 'exact_probabilities' and total != 1:
            raise ValueError('Exact assignment probabilities must sum exactly to one')
        output.append(tuple(v / total for v in row))
    return tuple(output)


def _vector(values, size, name):
    raw = np.asarray(values, dtype=object)
    if raw.ndim == 0:
        raw = np.full(size, raw.item(), dtype=object)
    if raw.shape != (size,):
        raise ValueError(f'{name} must be scalar or have length {size}')
    return tuple(_fraction(v) for v in raw)


@dataclass(frozen=True)
class IntervalMap:
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    exact_lower: tuple[Fraction, ...]
    exact_upper: tuple[Fraction, ...]
    kappas: tuple[Fraction, ...]

    def record(self):
        return {'format': 'qbridge-represented-intervals-v4',
                'normalized_row': 'exact stored weights divided by exact kappa',
                'rho_units': 'normalized ideal group mean',
                'exact_lower': [fraction_record(x) for x in self.exact_lower],
                'exact_upper': [fraction_record(x) for x in self.exact_upper],
                'kappas': [fraction_record(x) for x in self.kappas],
                'lower': list(self.lower), 'upper': list(self.upper),
                'guarantee': 'Exact deterministic map conditional on valid supplied base intervals; not formal verification of quantiles or EB'}


def _map_result(lower, upper, rows):
    return IntervalMap(tuple(outward_float(x, lower=True) for x in lower),
                       tuple(outward_float(x, lower=False) for x in upper),
                       tuple(lower), tuple(upper), tuple(r.kappa for r in rows))


def map_normalized_intervals(rows, lower, upper, *, rho=0):
    """Map trusted intervals for exact normalized rows into unchanged stored T."""
    _, reps = row_representations(rows)
    lo = _vector(lower, len(reps), 'lower'); hi = _vector(upper, len(reps), 'upper')
    tolerances = _vector(rho, len(reps), 'rho')
    if any(v < 0 for v in tolerances):
        raise ValueError('Nonnegative normalized transport tolerances required')
    if any(l > u for l, u in zip(lo, hi)):
        raise UnusableStatisticalInterval('Empty normalized statistical interval')
    if any(l < 0 or u > 1 for l, u in zip(lo, hi)):
        raise ValueError('Normalized mean intervals must lie within [0,1]')
    low = [r.kappa * (l - tol) for r, l, tol in zip(reps, lo, tolerances)]
    high = [r.kappa * (u + tol) for r, u, tol in zip(reps, hi, tolerances)]
    return _map_result(low, high, reps)


def project_cell_intervals(rows, lower, upper, *, rho=0):
    """Exact nonnegative projection and exact kappa*rho transport expansion."""
    matrix, reps = row_representations(rows)
    lo = _vector(lower, matrix.shape[1], 'cell lower')
    hi = _vector(upper, matrix.shape[1], 'cell upper')
    tolerances = _vector(rho, len(reps), 'rho')
    if any(v < 0 for v in tolerances):
        raise ValueError('Nonnegative normalized transport tolerances required')
    if any(l > u for l, u in zip(lo, hi)):
        raise UnusableStatisticalInterval('Empty cell statistical interval')
    if any(l < 0 or u > 1 for l, u in zip(lo, hi)):
        raise ValueError('Cell mean intervals must lie within [0,1]')
    low = [sum((w*lo[i] for i,w in zip(r.indices,r.stored_weights)),Fraction())-r.kappa*tol
           for r,tol in zip(reps,tolerances)]
    high = [sum((w*hi[i] for i,w in zip(r.indices,r.stored_weights)),Fraction())+r.kappa*tol
            for r,tol in zip(reps,tolerances)]
    return _map_result(low, high, reps)


def _counts(counts):
    a = np.asarray(counts)
    if (a.ndim != 3 or a.shape[-1] != 2 or min(a.shape[:2]) < 1 or
            not np.issubdtype(a.dtype,np.number) or np.iscomplexobj(a)):
        raise ValueError('Numeric counts must have nonempty [context,action,2] shape')
    if not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(a != np.floor(a)):
        raise ValueError('Finite nonnegative integer counts required')
    if sum(int(x) for x in a.ravel()) > 2**52:
        raise ValueError('Counts exceed trusted float-quantile implementation range')
    return a.astype(np.int64, copy=False)


def _failure(alpha):
    value = _fraction(alpha)
    if not 0 < value < 1:
        raise ValueError('Failure probability must lie strictly between zero and one')
    return value


def trusted_binomial_interval(k, n, failure, *, lower_only=False):
    """CP with conservatively rounded probability arguments; ppf itself trusted."""
    failure = _failure(failure)
    if not isinstance(k,(int,np.integer)) or not isinstance(n,(int,np.integer)) or not 0 <= k <= n <= 2**52:
        raise ValueError('Integer counts must satisfy 0 <= k <= n <= 2**52')
    tail = failure if lower_only else failure/2
    lprob = outward_float(tail, lower=True)
    lo = 0. if not k else float(beta.ppf(lprob,int(k),int(n-k+1)))
    if lower_only:
        hi = 1.
    else:
        uprob = outward_float(1-tail, lower=False)
        hi = 1. if k==n else float(beta.ppf(uprob,int(k+1),int(n-k)))
    if not math.isfinite(lo) or not math.isfinite(hi) or not 0 <= lo <= hi <= 1:
        raise UnusableStatisticalInterval('Trusted quantile did not return a usable interval')
    return _fraction(lo), _fraction(hi)


def observation_intervals_v4(counts, radius, alpha, *, consistency=True):
    """Exact monotone image/intersection of trusted CP endpoints for iid events.

    Each observation contributes exactly one (context, action, binary outcome).
    Counts pooled from several correlated ratings per user do not satisfy this
    contract; full-user-block constructions must retain their own statistics.
    """
    counts = _counts(counts);alpha = _failure(alpha);radius = _fraction(radius)
    if radius < 1:
        raise ValueError('Sensitivity radius must be at least one')
    k, a = counts.shape[:2]; d = k*a
    conditional_failure = alpha/(2*d) if consistency else alpha/d
    lower=[];upper=[]
    for x in range(k):
        n_context=int(counts[x].sum())
        for action in range(a):
            failure,success=map(int,counts[x,action]);n=failure+success
            pl,pu=trusted_binomial_interval(success,n,conditional_failure)
            lo=max(pl/radius,1-radius*(1-pl));hi=min(radius*pu,1-(1-pu)/radius)
            if consistency:
                js,_=trusted_binomial_interval(success,n_context,alpha/(4*d),lower_only=True)
                jf,_=trusted_binomial_interval(failure,n_context,alpha/(4*d),lower_only=True)
                lo=max(lo,js);hi=min(hi,1-jf)
            if lo>hi:
                raise UnusableStatisticalInterval('Empty observation confidence intersection')
            lower.append(lo);upper.append(hi)
    info={'conditional_failure_exact':fraction_record(conditional_failure),
          'joint_failure_exact':fraction_record(alpha/(4*d)) if consistency else None,
          'trusted_numeric_routines':['scipy.stats.beta.ppf'],
          'deterministic_endpoint_map':'Fraction then outward binary64',
          'exact_lower':[fraction_record(x) for x in lower],
          'exact_upper':[fraction_record(x) for x in upper]}
    return (np.array([outward_float(x,lower=True) for x in lower]),
            np.array([outward_float(x,lower=False) for x in upper]),info)


def randomized_cell_intervals_v4(counts, rows, alpha, *, rho=0):
    """Known-random-assignment cell CP, projected to declared moment rows."""
    counts=_counts(counts);alpha=_failure(alpha)
    matrix,reps=row_representations(rows,n_columns=math.prod(counts.shape[:2]))
    active=sorted({i for r in reps for i in r.indices})
    lower=[Fraction()]*matrix.shape[1];upper=[Fraction(1)]*matrix.shape[1]
    flat=counts.reshape(-1,2)
    for i in active:
        failure,success=map(int,flat[i])
        lower[i],upper[i]=trusted_binomial_interval(success,failure+success,alpha/len(active))
    mapped=project_cell_intervals(matrix,lower,upper,rho=rho)
    return mapped,{'construction':'projected_cell_cp_v4','active_cells':len(active),
                   'failure_per_active_cell_exact':fraction_record(alpha/len(active)) if active else None,
                   'trusted_numeric_routines':['scipy.stats.beta.ppf']}


def exact_group_score_statistics(counts, rows, q, *, q_semantics):
    """Exact score moments under the explicitly declared assignment law."""
    counts=_counts(counts);k,a=counts.shape[:2]
    _,reps=row_representations(rows,n_columns=k*a)
    probs=canonical_probabilities(q,semantics=q_semantics)
    if len(probs)!=k or any(len(row)!=a for row in probs):
        raise ValueError('q shape does not match counts')
    records=[]
    for row in reps:
        if not row.kappa:
            records.append({'n':0,'total':Fraction(),'total_sq':Fraction(),
                            'mean':None,'variance':None,'bound':Fraction(),
                            'multipliers':(),'conditional_q':(),'zero_row':True})
            continue
        contexts={i//a for i in row.indices}
        if len(contexts)!=1:
            raise ValueError('Group scores require each row within one context')
        x=next(iter(contexts));actions=[i%a for i in row.indices]
        if any(probs[x][i]<=0 for i in actions):
            raise ValueError('Positive known randomization required on moment support')
        mass=sum((probs[x][i] for i in actions),Fraction())
        conditional=tuple(probs[x][i]/mass for i in actions)
        weights=tuple(nu/p for nu,p in zip(row.ideal_weights,conditional))
        n=sum(int(counts[x,i].sum()) for i in actions)
        total=sum((w*int(counts[x,i,1]) for i,w in zip(actions,weights)),Fraction())
        square=sum((w*w*int(counts[x,i,1]) for i,w in zip(actions,weights)),Fraction())
        variance=(square-total*total/n)/(n-1) if n>1 else None
        if variance is not None and variance<0:
            raise ArithmeticError('Exact score variance cannot be negative')
        records.append({'n':n,'total':total,'total_sq':square,
                        'mean':total/n if n else None,'variance':variance,
                        'bound':max(weights),'multipliers':weights,
                        'conditional_q':conditional,'zero_row':False})
    return tuple(records)


def _trusted_eb_radius(record, failure):
    # Variance, range and moments are exact inputs. The returned half-width is
    # still a trusted floating log/sqrt computation, not a proven enclosure.
    n=record['n']
    log_term=math.log(outward_float(4/failure,lower=False))
    variance=outward_float(record['variance'],lower=False)
    bound=outward_float(record['bound'],lower=False)
    radius=math.sqrt(2*variance*log_term/n)+7*bound*log_term/(3*(n-1))
    if not math.isfinite(radius) or radius<0:
        raise UnusableStatisticalInterval('Trusted EB radius is not finite')
    return _fraction(radius)


def randomized_group_intervals_v4(counts, rows, q, alpha, *, q_semantics, rho=0):
    """Trusted CP/EB for exact ideal scores, then exact stored-row mapping."""
    alpha=_failure(alpha)
    statistics=exact_group_score_statistics(counts,rows,q,q_semantics=q_semantics)
    lower=[];upper=[];diagnostics=[]
    for record in statistics:
        n=record['n'];failure=alpha/len(statistics)
        if record['zero_row']:
            lo=hi=Fraction();kind='deterministic_zero'
        elif all(w==1 for w in record['multipliers']):
            lo,hi=trusted_binomial_interval(int(record['total']),n,failure);kind='clopper_pearson'
        elif n<2:
            lo=Fraction();hi=Fraction(1);kind='insufficient_eb'
        else:
            radius=_trusted_eb_radius(record,failure)
            lo=max(Fraction(),record['mean']-radius);hi=min(Fraction(1),record['mean']+radius)
            kind='empirical_bernstein'
        if lo>hi:
            raise UnusableStatisticalInterval('Empty clipped group-score confidence interval')
        lower.append(lo);upper.append(hi)
        diagnostics.append({'n':n,'construction':kind,'failure_exact':fraction_record(failure),
                            'score_bound_exact':fraction_record(record['bound']),
                            'mean_exact':fraction_record(record['mean']) if n else None,
                            'variance_exact':fraction_record(record['variance']) if n>1 else None})
    return map_normalized_intervals(rows,lower,upper,rho=rho),{
        'construction':'normalized_group_scores_v4','q_semantics':q_semantics,'rows':diagnostics,
        'trusted_numeric_routines':['scipy.stats.beta.ppf','math.log','math.sqrt','EB half-width floating arithmetic'],
        'exact_arithmetic':['row normalization','q conditioning','score multipliers','score moments','mean and n-1 variance','endpoint and transport mapping']}


def materialize_counts_v4(observational,randomized,*,rows,randomized_q,q_semantics,
                          radius=2.,alpha=.05,transport_budget=0.,consistency=True,
                          randomized_method='group_score',metadata=None,highs_threads=None):
    """Explicit opt-in wrapper; unchanged matrix and existing exact view gate.

    Inputs are sufficient counts of iid single-action events, not arbitrary
    pooled block observations. No claim is made that caller q semantics match
    historical sampling code or that the caller's transport model is valid.
    The rows, q, alpha split and construction must be fixed independently of
    inference outcomes. Alpha covers this one view; it is not a budget for
    selecting the best among multiple materializations after seeing data.
    A refused statistical interval returns no usable view or bound.
    """
    from .confidence_view import create_view
    obs=_counts(observational);rand=_counts(randomized);failure=_failure(alpha)
    if obs.shape!=rand.shape:
        raise ValueError('Source count shapes differ')
    matrix,reps=row_representations(rows,n_columns=math.prod(obs.shape[:2]))
    if randomized_method not in {'group_score','cell_projection'}:
        raise ValueError('Unknown prespecified randomized construction')
    declared_radius=_fraction(radius)
    rho=_vector(transport_budget,len(reps),'transport budget')
    if declared_radius<1 or any(x<0 for x in rho):
        raise ValueError('Radius must be at least one and transport budgets nonnegative')
    probabilities=canonical_probabilities(randomized_q,semantics=q_semantics)
    # This validates the scientific support declaration even for cell CP.
    exact_group_score_statistics(np.zeros_like(rand),matrix,randomized_q,q_semantics=q_semantics)
    declaration={'format':'qbridge-materialization-v4','alpha_exact':fraction_record(failure),
                 'q_semantics':q_semantics,'rho_units':'normalized ideal group mean',
                 'assignment_probabilities_exact':[[fraction_record(x) for x in row] for row in probabilities],
                 'row_semantics':'stored binary64 weights divided by exact kappa',
                 'source_alpha_exact':[fraction_record(failure/2)]*2,
                 'radius_exact':fraction_record(declared_radius),
                 'rho_exact':[fraction_record(x) for x in rho],
                 'counts_shape':list(obs.shape),'counts_encoding':'C-order little-endian int64',
                 'observational_counts_sha256':hashlib.sha256(obs.astype('<i8').tobytes()).hexdigest(),
                 'randomized_counts_sha256':hashlib.sha256(rand.astype('<i8').tobytes()).hexdigest(),
                 'consistency':bool(consistency),'randomized_method':randomized_method,
                 'alpha_scope':'one prespecified view, not selection among materializations',
                 'caller_declaration':metadata or {}}
    try:
        lo,hi,oi=observation_intervals_v4(obs,radius,failure/2,consistency=consistency)
        if randomized_method=='group_score':
            mapped,ri=randomized_group_intervals_v4(rand,matrix,randomized_q,failure/2,
                q_semantics=q_semantics,rho=transport_budget)
        elif randomized_method=='cell_projection':
            mapped,ri=randomized_cell_intervals_v4(rand,matrix,failure/2,rho=transport_budget)
    except UnusableStatisticalInterval as error:
        return {'status':'uncertain','reason':'unusable_statistical_interval_v4',
                'detail':str(error),'declaration':declaration}
    declaration.update(observation=oi,randomized=ri,mapping=mapped.record())
    return create_view(lo,hi,matrix,mapped.lower,mapped.upper,
                       metadata=declaration,highs_threads=highs_threads)
