"""Strengthened, prespecified confidence intervals for materialized views.

The original aggregate implementation remains reproducible. This module adds
same-table consistency information and a same-transport cell-projection control.
It never receives latent propensities, hidden states, or potential-outcome truth.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import beta

from .aggregate import binomial_intervals, conditional_box, _moment_matrix


def validate_counts(counts):
    a = np.asarray(counts)
    if a.ndim != 3 or a.shape[-1] != 2 or min(a.shape[:2]) < 1:
        raise ValueError("counts must have nonempty shape [contexts, actions, 2]")
    if not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(a != np.floor(a)):
        raise ValueError("counts must be finite nonnegative integers")
    if np.any(a > 2**52) or sum(int(v) for v in a.ravel()) > 2**52:
        raise ValueError("count range exceeds this float-quantile implementation")
    return a.astype(np.int64,copy=False)


def one_sided_binomial_lower(successes, totals, alpha):
    """Clopper-Pearson lower endpoints with one-sided failure alpha each."""
    k,n = np.broadcast_arrays(np.asarray(successes,float),np.asarray(totals,float))
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(n)) or np.any(k<0) or np.any(n<k):
        raise ValueError("require finite 0 <= successes <= totals")
    if np.any(k != np.floor(k)) or np.any(n != np.floor(n)):
        raise ValueError("integer counts required")
    result=np.zeros(k.shape,float)
    use=k>0
    result[use]=beta.ppf(alpha,k[use],n[use]-k[use]+1)
    return result


def observation_intervals(counts, radius, alpha, *, consistency=True):
    """Simultaneous target-cell box, including observed joint-event constraints.

    Fixed allocation when consistency=True:
      alpha/2 -> d two-sided conditional-success intervals;
      alpha/2 -> 2d one-sided joint-event lower intervals.
    Both sources of intervals may be dependent: validity uses a union bound.
    An empty stratum gives no restriction rather than an imputed probability.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    counts=validate_counts(counts)
    d=counts.shape[0]*counts.shape[1]
    conditional_alpha=alpha/2 if consistency else alpha
    pl,pu=binomial_intervals(counts[:,:,1],counts.sum(2),conditional_alpha/d)
    lower,upper=conditional_box(pl,pu,radius)
    diagnostics={"construction":"conditional_weight_plus_consistency" if consistency else "conditional_weight_only",
                 "alpha":alpha,"conditional_family_alpha":conditional_alpha,
                 "conditional_intervals":d,"empty_cells":int((counts.sum(2)==0).sum())}
    if consistency:
        n_context=counts.sum((1,2))[:,None]
        joint_success_lower=one_sided_binomial_lower(counts[:,:,1],n_context,alpha/(4*d))
        joint_failure_lower=one_sided_binomial_lower(counts[:,:,0],n_context,alpha/(4*d))
        lower=np.maximum(lower,joint_success_lower)
        upper=np.minimum(upper,1-joint_failure_lower)
        diagnostics.update(joint_family_alpha=alpha/2,joint_one_sided_intervals=2*d,
                           joint_success_lower=joint_success_lower.tolist(),
                           joint_failure_lower=joint_failure_lower.tolist())
    diagnostics["box_contradiction"]=bool(np.any(lower>upper))
    return lower.ravel(),upper.ravel(),diagnostics


def randomized_cell_projection(counts, rows, alpha):
    """Same-information alternative: R cell CIs projected through selected T.

    Only group moments T m^R are subsequently transported. No R cell mean is
    equated to its O counterpart. Rows must be normalized and nonnegative.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    counts=validate_counts(counts)
    t=_moment_matrix(rows,counts.shape[:2])
    if np.any(t.data<0) or not np.allclose(np.asarray(t.sum(1)).ravel(),1,rtol=0,atol=1e-12):
        raise ValueError("nonnegative normalized moment rows required")
    active=np.unique(t.indices)
    lower,upper=np.zeros(counts.shape[:2]).ravel(),np.ones(counts.shape[:2]).ravel()
    if len(active):
        flat=counts.reshape(-1,2)
        ll,uu=binomial_intervals(flat[active,1],flat[active].sum(1),alpha/len(active))
        lower[active],upper[active]=ll,uu
    return np.asarray(t@lower),np.asarray(t@upper),{
        "construction":"simultaneous_randomized_cells_projected_to_groups",
        "alpha":alpha,"active_cells":len(active),"transported_cell_means":False}
