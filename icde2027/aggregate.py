"""Finite-sample binary policy-contrast bounds from sufficient statistics.

No observational propensity is an input. The user declares the target weights,
two fixed policies, the conditional relative-weight box, known randomized
design, and transported moment rows. Statistical guarantees require iid units
within fixed strata and honest choice of all design objects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
import time
import warnings

import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy import sparse
from scipy.stats import beta


def _finite_vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector")
    return result


def count_events(context_ids, action_ids, outcomes, n_contexts, n_actions):
    """Lossless one-pass sufficient counts for validated iid binary events.

    This validates encoded domains, not independence or causal transport.
    Add outputs across chunks; no raw reward is reconstructed or imputed.
    """
    arrays = [np.asarray(v) for v in (context_ids, action_ids, outcomes)]
    if any(v.ndim != 1 or v.shape != arrays[0].shape for v in arrays):
        raise ValueError("event columns must be equally sized vectors")
    if int(n_contexts) != n_contexts or int(n_actions) != n_actions or n_contexts < 1 or n_actions < 1:
        raise ValueError("positive integer domain sizes required")
    for values, maximum in zip(arrays, (n_contexts,n_actions,2)):
        if not np.issubdtype(values.dtype,np.integer) or np.any(values < 0) or np.any(values >= maximum):
            raise ValueError("integer event codes outside declared domain")
    x,a,y = [values.astype(np.int64,copy=False) for values in arrays]
    index = (x*n_actions+a)*2+y
    return np.bincount(index,minlength=n_contexts*n_actions*2).reshape(n_contexts,n_actions,2)


def binomial_intervals(successes, totals, alpha):
    """Two-sided Clopper-Pearson intervals, each at failure probability alpha.

    Random stratum totals are permitted under iid sampling conditional on the
    stratum. Empty strata return [0,1], never a point imputation.
    """
    k = np.asarray(successes, dtype=float)
    n = np.asarray(totals, dtype=float)
    if k.shape != n.shape or not np.all(np.isfinite(k)) or not np.all(np.isfinite(n)):
        raise ValueError("finite, equally shaped counts required")
    if np.any(n < 0) or np.any(k < 0) or np.any(k > n):
        raise ValueError("require 0 <= successes <= totals")
    if np.any(k != np.floor(k)) or np.any(n != np.floor(n)):
        raise ValueError("binomial counts must be integers")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    lo, hi = np.zeros(k.shape), np.ones(k.shape)
    mask = k > 0
    lo[mask] = beta.ppf(alpha / 2, k[mask], n[mask] - k[mask] + 1)
    mask = k < n
    hi[mask] = beta.ppf(1 - alpha / 2, k[mask] + 1, n[mask] - k[mask])
    return lo, hi


def conditional_box(probability_lo, probability_hi, radius):
    """Exact binary conditional-weight image, then monotone interval envelope."""
    lo, hi = np.asarray(probability_lo, float), np.asarray(probability_hi, float)
    if lo.shape != hi.shape or not np.isfinite(radius) or radius < 1:
        raise ValueError("valid shapes and radius >= 1 required")
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("finite probability intervals required")
    if np.any(lo < 0) or np.any(hi > 1) or np.any(lo > hi):
        raise ValueError("probabilities must satisfy 0 <= lo <= hi <= 1")
    if radius == 1:
        return lo.copy(), hi.copy()
    lower = np.maximum(lo / radius, 1 - radius * (1 - lo))
    upper = np.minimum(radius * hi, 1 - (1 - hi) / radius)
    return lower, upper


def empirical_bernstein_interval(n, total, total_sq, bound, alpha):
    """Two-sided Maurer-Pontil interval for iid Z in [0,bound].

    Each tail uses alpha/2, hence log(4/alpha). This is not a t interval.
    Counts and sums are sufficient for the sample mean/unbiased variance.
    """
    if not (0 < alpha < 1) or not np.isfinite(bound) or bound < 0:
        raise ValueError("invalid alpha/range")
    if int(n) != n or n < 0 or not np.isfinite(total) or not np.isfinite(total_sq):
        raise ValueError("invalid sufficient statistics")
    n = int(n)
    if total < -1e-10 or total > n * bound + 1e-8 or total_sq < -1e-10:
        raise ValueError("statistics outside declared range")
    if n < 2:
        return 0.0, float(bound)
    mean = total / n
    variance = max(0.0, (total_sq - total * total / n) / (n - 1))
    log_term = math.log(4 / alpha)
    radius = math.sqrt(2 * variance * log_term / n) + 7 * bound * log_term / (3 * (n - 1))
    return max(0.0, mean - radius), min(float(bound), mean + radius)


def make_group_moments(nu, group_labels):
    """Sparse normalized within-context rows, one row per nonempty group.

    Group -1 is excluded. Each included cell is stored exactly once, so this
    plan needs O(X*A) memory even for full-cell moments.
    """
    nu = np.asarray(nu, float)
    labels = np.asarray(group_labels)
    if nu.ndim != 2 or not np.all(np.isfinite(nu)) or np.any(nu < 0):
        raise ValueError("finite nonnegative nu [X,A] required")
    labels = np.broadcast_to(labels, nu.shape)
    if np.any(labels != np.floor(labels)) or np.any(labels < -1):
        raise ValueError("integer group labels >= -1 required")
    ix, iy, values = [], [], []
    j = 0
    for x in range(len(nu)):
        active = np.flatnonzero((labels[x] >= 0) & (nu[x] > 0))
        declared = np.unique(labels[x][labels[x] >= 0])
        if not np.array_equal(np.unique(labels[x,active]), declared):
            raise ValueError("moment group has no positive weight")
        ordered = active[np.argsort(labels[x,active], kind="stable")]
        boundaries = np.flatnonzero(np.diff(labels[x,ordered]))+1
        for ids in np.split(ordered, boundaries):
            if len(ids) == 0:
                continue
            weights = nu[x, ids] / nu[x, ids].sum()
            ix.extend([j] * len(ids))
            iy.extend((x * nu.shape[1] + ids).tolist())
            values.extend(weights.tolist())
            j += 1
    return sparse.csr_matrix((values, (ix, iy)), shape=(j, nu.size))


def _moment_matrix(rows, shape):
    if sparse.issparse(rows):
        t = sparse.csr_matrix(rows, dtype=float, copy=True)
    else:
        arr = np.asarray(rows, float)
        if arr.ndim == 3:
            arr = arr.reshape(len(arr), math.prod(shape))
        t = sparse.csr_matrix(arr)
    t.sum_duplicates()
    t.eliminate_zeros()
    if t.ndim != 2 or t.shape[1] != math.prod(shape) or not np.all(np.isfinite(t.data)):
        raise ValueError("invalid sparse moment matrix")
    return t


def randomized_intervals(counts, rows, q, alpha, *, method="auto"):
    """Bounds for disjoint within-context outcome moments from R counts.

    counts[x,a,y] are iid randomized observations; rows[j,x,a] is a
    nonnegative normalized within-context moment. q is the known randomized
    action distribution. Rows may overlap statistically; union bounds do not
    require independent rows. The optimizer decides whether they are disjoint.
    """
    counts = np.asarray(counts)
    q = np.asarray(q, float)
    if counts.ndim != 3 or counts.shape[-1] != 2:
        raise ValueError("counts [X,A,2] and rows [J,X,A] required")
    rows = _moment_matrix(rows, counts.shape[:2])
    if q.shape != counts.shape[:2]:
        raise ValueError("inconsistent shapes")
    if np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("nonnegative integer counts required")
    if not np.all(np.isfinite(rows.data)) or not np.all(np.isfinite(q)):
        raise ValueError("finite moment rows and q required")
    if np.any(rows.data < 0) or np.any(q < 0) or not np.allclose(q.sum(1), 1, rtol=0, atol=1e-12):
        raise ValueError("invalid randomized probabilities or rows")
    if not 0 < alpha < 1 or method not in {"auto", "eb"}:
        raise ValueError("invalid alpha or method")
    nrows = rows.shape[0]
    if nrows == 0:
        return np.zeros(0), np.zeros(0), []
    lows, highs, diagnostics = [], [], []
    for j in range(nrows):
        start, stop = rows.indptr[j:j+2]
        indices = rows.indices[start:stop]
        values = rows.data[start:stop]
        touched_context = np.unique(indices // counts.shape[1])
        if len(touched_context) != 1 or not np.isclose(values.sum(), 1, rtol=0, atol=1e-12):
            raise ValueError("each row must be normalized within exactly one context")
        x = int(touched_context[0])
        selected = indices % counts.shape[1]
        if np.any(q[x, selected] <= 0):
            raise ValueError("randomized support missing for a moment weight")
        q_selected = q[x, selected] / q[x, selected].sum()
        weight = values / q_selected
        n = int(counts[x, selected].sum())
        successes = counts[x, selected, 1]
        bound = float(weight.max())
        if method == "auto" and np.all(weight == 1.0):
            ll, uu = binomial_intervals(np.array([successes.sum()]), np.array([n]), alpha / nrows)
            lower, upper, kind = float(ll[0]), float(uu[0]), "clopper_pearson"
        else:
            lower, upper = empirical_bernstein_interval(
                n, float(weight @ successes), float(weight**2 @ successes), bound, alpha / nrows
            )
            kind = "empirical_bernstein"
        # E[Z]=nu dot m is a convex combination of Bernoulli means.
        lows.append(max(0.0, lower))
        highs.append(min(1.0, upper))
        diagnostics.append({"n": n, "range": bound, "interval": kind})
    return np.array(lows), np.array(highs), diagnostics


@dataclass
class BoundResult:
    status: str
    lower_bound: float | None
    primal_value: float | None
    duality_gap: float | None
    max_violation: float | None
    numerical_allowance: float | None
    seconds: float
    backend: str
    proposal: dict | None = None

    def to_dict(self):
        result = asdict(self)
        if self.proposal is None:
            result.pop("proposal")
        return result


def _validate_program(objective, lower, upper, matrix, rhs_lower, rhs_upper, *, allow_empty_intervals=False):
    c = _finite_vector(objective, "objective")
    lo, hi = _finite_vector(lower, "lower"), _finite_vector(upper, "upper")
    t = _moment_matrix(matrix, (len(c),))
    tl, th = _finite_vector(rhs_lower, "rhs_lower"), _finite_vector(rhs_upper, "rhs_upper")
    if lo.shape != c.shape or hi.shape != c.shape or t.shape != (len(tl), len(c)) or th.shape != tl.shape:
        raise ValueError("inconsistent program dimensions")
    if np.any(lo < 0) or np.any(lo > 1) or np.any(hi < 0) or np.any(hi > 1):
        raise ValueError("invalid program bounds")
    if not allow_empty_intervals and (np.any(lo > hi) or np.any(tl > th)):
        raise ValueError("invalid program bounds")
    return c, lo, hi, t, tl, th


def _dual_value(c, lo, hi, matrix, rhs_lo, rhs_hi, lower_multipliers, upper_multipliers):
    """Every nonnegative multiplier pair gives a weak-dual lower bound.

    Re-evaluate the linear dual expression with exact rational arithmetic for
    the represented binary64 inputs; round the final result toward -infinity.
    This certifies optimization arithmetic, not special-function evaluation
    of statistical interval endpoints or the scientific model assumptions.
    """
    fraction = lambda value: Fraction.from_float(float(value))
    residual = [fraction(value) for value in c]
    constant = Fraction(0)
    for j in range(matrix.shape[0]):
        lower = fraction(max(float(lower_multipliers[j]), 0.0))
        upper = fraction(max(float(upper_multipliers[j]), 0.0))
        constant += lower * fraction(rhs_lo[j]) - upper * fraction(rhs_hi[j])
        difference = upper - lower
        if difference:
            for offset in range(matrix.indptr[j], matrix.indptr[j+1]):
                residual[matrix.indices[offset]] += fraction(matrix.data[offset]) * difference
    value = constant
    for i, coefficient in enumerate(residual):
        value += coefficient * fraction(lo[i] if coefficient >= 0 else hi[i])
    nearest = float(value)
    downward = math.nextafter(nearest, -math.inf)
    return downward, abs(nearest - downward)


def solve_bound(objective, lower, upper, matrix, rhs_lower, rhs_upper, *, backend="auto", highs_threads=None,
                export_proposal=False, verification="exact"):
    """Minimize a contrast over box/moment intervals; return a verified dual.

    Disjoint nonnegative moment rows use fractional knapsack. Overlapping rows
    use generic HiGHS. Failed/infeasible programs return refusal (None).
    """
    begin = time.perf_counter()
    c, lo, hi, t, tl, th = _validate_program(objective, lower, upper, matrix, rhs_lower, rhs_upper)
    if backend not in {"auto", "knapsack", "highs"}:
        raise ValueError("unknown backend")
    if verification not in {"exact", "deferred"} or (verification=="deferred" and not export_proposal):
        raise ValueError("deferred verification requires an exported non-certified proposal")
    if highs_threads is not None and (not isinstance(highs_threads,int) or highs_threads < 1):
        raise ValueError("highs_threads must be a positive integer or None")
    disjoint = bool(np.all(t.data >= 0) and np.all(np.asarray(t.getnnz(axis=0)) <= 1))
    if backend == "knapsack" and not disjoint:
        raise ValueError("knapsack requires disjoint nonnegative moment rows")
    chosen = "knapsack" if backend != "highs" and disjoint else "highs"
    refusal = lambda status: BoundResult(status, None, None, None, None, None, time.perf_counter() - begin, chosen)
    if chosen == "highs":
        a = sparse.vstack([t, -t], format="csr") if t.shape[0] else None
        b = np.concatenate([th, -tl]) if t.shape[0] else None
        # SciPy passes this documented native HiGHS option through, but warns
        # because it is outside SciPy's own short option list. Suppress only
        # that expected warning; all other numerical warnings remain visible.
        options = {"dual_feasibility_tolerance":1e-10,
                   "primal_feasibility_tolerance":1e-10,
                   "ipm_optimality_tolerance":1e-12}
        if highs_threads is not None:
            options["threads"] = highs_threads
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Unrecognized options detected: \{'threads': [0-9]+\}.*",
                                    category=OptimizeWarning)
            result = linprog(c, A_ub=a, b_ub=b, bounds=list(zip(lo, hi)), method="highs",
                             options=options)
        if not result.success:
            return refusal("infeasible" if result.status == 2 else "solver_failure")
        solution = result.x
        if t.shape[0]:
            multipliers = np.maximum(-np.asarray(result.ineqlin.marginals), 0)
            upper_dual, lower_dual = multipliers[:t.shape[0]], multipliers[t.shape[0]:]
        else:
            upper_dual = lower_dual = np.zeros(0)
    else:
        solution = np.where(c < 0, hi, lo).copy()
        lower_dual, upper_dual = np.zeros(t.shape[0]), np.zeros(t.shape[0])
        for j in range(t.shape[0]):
            start, stop = t.indptr[j:j+2]
            ids, weights = t.indices[start:stop], t.data[start:stop]
            min_mass, max_mass = float(weights @ lo[ids]), float(weights @ hi[ids])
            if tl[j] > max_mass or th[j] < min_mass:
                return refusal("infeasible")
            mass = float(weights @ solution[ids])
            if tl[j] <= mass <= th[j]:
                continue
            increasing = mass < tl[j]
            target = tl[j] if increasing else th[j]
            remaining = abs(target - mass)
            ratios = c[ids] / weights
            order = np.argsort(ratios if increasing else -ratios, kind="stable")
            marginal = 0.0
            for index in order:
                i = ids[index]
                slack = hi[i] - solution[i] if increasing else solution[i] - lo[i]
                weight = weights[index]
                capacity = float(weight * slack)
                if capacity <= 0:
                    continue
                moved = min(remaining, capacity)
                solution[i] += (moved / weight) * (1 if increasing else -1)
                marginal = float(c[i] / weight)
                remaining -= moved
                if remaining <= 1e-14:
                    break
            if remaining > 1e-9:
                return refusal("solver_failure")
            if increasing:
                lower_dual[j] = max(marginal, 0.0)
            else:
                upper_dual[j] = max(-marginal, 0.0)
    moment = t @ solution
    violation = float(max(np.max(lo - solution, initial=0), np.max(solution - hi, initial=0),
                          np.max(tl - moment, initial=0), np.max(moment - th, initial=0)))
    primal = float(c @ solution)
    proposal = {"primal":solution.tolist(),"lower_multipliers":lower_dual.tolist(),
                "upper_multipliers":upper_dual.tolist()} if export_proposal else None
    if not np.isfinite(primal) or violation > 1e-7:
        return refusal("numerical_failure")
    if verification == "deferred":
        # A proposal is not a bound. The materialized-view verifier must check
        # the full global dual before it may authorize a certificate.
        return BoundResult("proposal",None,primal,None,violation,None,time.perf_counter()-begin,chosen,proposal)
    bound, allowance = _dual_value(c, lo, hi, t, tl, th, lower_dual, upper_dual)
    gap = primal - bound
    if not np.isfinite(bound) or violation > 1e-7 or gap < -1e-8:
        return refusal("numerical_failure")
    return BoundResult("ok", bound, primal, gap, violation, allowance, time.perf_counter() - begin, chosen,proposal)


def policy_objective(target_weights, policy, reference):
    w = _finite_vector(target_weights, "target_weights")
    pi, ref = np.asarray(policy, float), np.asarray(reference, float)
    if pi.ndim != 2 or ref.shape != pi.shape or len(w) != len(pi):
        raise ValueError("invalid policy dimensions")
    if np.any(w < 0) or not np.isclose(w.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("target weights must form a probability distribution")
    for p in (pi, ref):
        if not np.all(np.isfinite(p)) or np.any(p < 0) or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-12):
            raise ValueError("policies must be normalized finite probabilities")
    return (w[:, None] * (pi - ref)).ravel()


def evaluate_counts(observational, randomized, *, target_weights, policy, reference,
                    moment_rows, randomized_q, radius=2.0, alpha=.05, transport_budget=0.0,
                    backend="auto"):
    """All same-information bounds, including full-alpha source-alone controls."""
    obs, rand = np.asarray(observational), np.asarray(randomized)
    if obs.shape != rand.shape or obs.ndim != 3 or obs.shape[-1] != 2:
        raise ValueError("both sources must have shape [X,A,2]")
    if np.shape(policy) != obs.shape[:2] or np.shape(reference) != obs.shape[:2] or np.shape(target_weights) != (obs.shape[0],):
        raise ValueError("policy, reference and target weights must match count geometry")
    c = policy_objective(target_weights, policy, reference)
    rows = _moment_matrix(moment_rows, obs.shape[:2])
    t = rows
    rho = np.broadcast_to(np.asarray(transport_budget, float), (rows.shape[0],))
    if not np.all(np.isfinite(rho)) or np.any(rho < 0):
        raise ValueError("nonnegative finite transport budgets required")
    totals, successes = obs.sum(-1), obs[:, :, 1]
    pl, pu = binomial_intervals(successes, totals, alpha / (2 * len(c)))
    lo, hi = conditional_box(pl.ravel(), pu.ravel(), radius)
    rl, ru, details = randomized_intervals(rand, rows, randomized_q, alpha / 2)
    empty = np.zeros((0, len(c)))
    empty_rhs = np.zeros(0)
    output = {}
    output["fusion"] = solve_bound(c, lo, hi, t, rl-rho, ru+rho, backend=backend).to_dict()
    output["observation_matched"] = solve_bound(c, lo, hi, empty, empty_rhs, empty_rhs, backend=backend).to_dict()
    output["moment_matched"] = solve_bound(c, np.zeros(len(c)), np.ones(len(c)), t, rl-rho, ru+rho, backend=backend).to_dict()
    pl_full, pu_full = binomial_intervals(successes, totals, alpha / len(c))
    lo_full, hi_full = conditional_box(pl_full.ravel(), pu_full.ravel(), radius)
    rl_full, ru_full, _ = randomized_intervals(rand, rows, randomized_q, alpha)
    output["observation_standalone"] = solve_bound(c, lo_full, hi_full, empty, empty_rhs, empty_rhs, backend=backend).to_dict()
    output["moment_standalone"] = solve_bound(c, np.zeros(len(c)), np.ones(len(c)), t, rl_full-rho, ru_full+rho, backend=backend).to_dict()
    output["diagnostics"] = {"n_observation": int(obs.sum()), "n_randomized": int(rand.sum()),
                             "empty_observation_cells": int((totals == 0).sum()),
                             "randomized_rows": details,
                             "alpha_total": alpha, "alpha_observation_fusion": alpha/2,
                             "alpha_randomized_fusion": alpha/2}
    return output
