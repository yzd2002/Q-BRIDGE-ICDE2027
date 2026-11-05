"""Independent semantic checks for finite-table inference, not paper outcomes."""
import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.stats import binom
from scipy import sparse

from icde2027.aggregate import (
    binomial_intervals, conditional_box, count_events, evaluate_counts,
    make_group_moments, randomized_intervals, solve_bound,
)


@pytest.mark.parametrize("n", [0, 1, 2, 5, 10, 30, 100])
def test_binomial_interval_has_exact_finite_sample_coverage(n):
    successes = np.arange(n + 1)
    lo, hi = binomial_intervals(successes, np.full(n + 1, n), .05)
    for p in np.linspace(0, 1, 101):
        coverage = binom.pmf(successes, n, p)[(lo <= p) & (p <= hi)].sum()
        assert coverage >= .95 - 1e-12


@pytest.mark.parametrize("radius", [1., 1.05, 1.5, 2., 10.])
def test_conditional_image_equals_independent_hidden_weight_lp(radius):
    for p in np.r_[0., np.linspace(.01, .99, 53), 1.]:
        low, high = conditional_box(np.array([p]), np.array([p]), radius)
        mean_obj = np.array([0, p])
        normalization = np.array([[1-p, p]])
        result_lo = linprog(mean_obj, A_eq=normalization, b_eq=[1.], bounds=[(1/radius, radius)] * 2)
        result_hi = linprog(-mean_obj, A_eq=normalization, b_eq=[1.], bounds=[(1/radius, radius)] * 2)
        assert result_lo.success and result_hi.success
        assert low[0] == pytest.approx(result_lo.fun, abs=1e-10)
        assert high[0] == pytest.approx(-result_hi.fun, abs=1e-10)


@pytest.mark.parametrize("seed", range(40))
def test_knapsack_lower_bound_matches_independent_general_lp(seed):
    rng = np.random.default_rng(seed)
    nx, na = 3, 10
    nu = rng.uniform(.01, 1, size=(nx, na))
    rows = make_group_moments(nu, np.arange(na) % 3)
    c = rng.normal(size=nx*na)
    lo = rng.uniform(0, .4, size=nx*na)
    hi = rng.uniform(.6, 1, size=nx*na)
    witness = rng.uniform(lo, hi)
    tau = rows @ witness
    eps = rng.uniform(0, .2, size=len(tau))
    ours = solve_bound(c, lo, hi, rows, tau-eps, tau+eps, backend="knapsack")
    other = linprog(c, A_ub=sparse.vstack([rows, -rows]), b_ub=np.r_[tau+eps, -tau+eps], bounds=list(zip(lo, hi)))
    assert other.success and ours.status == "ok"
    assert ours.lower_bound <= other.fun + 1e-10
    assert ours.lower_bound == pytest.approx(other.fun, abs=1e-8)
    assert ours.duality_gap < 1e-8
    assert ours.max_violation < 1e-10


def test_overlapping_constraints_use_highs_and_verified_dual():
    t = np.array([[.5, .5, 0], [0, .5, .5]])
    result = solve_bound([-.4, .3, -.1], [0]*3, [1]*3, t, [.4, .6], [.5, .7])
    independent = linprog([-.4, .3, -.1], A_ub=np.vstack([t, -t]), b_ub=[.5,.7,-.4,-.6], bounds=[(0,1)]*3)
    assert result.backend == "highs"
    assert result.lower_bound == pytest.approx(independent.fun, abs=1e-10)
    with pytest.raises(ValueError):
        solve_bound([-.4,.3,-.1], [0]*3, [1]*3, t, [.4,.6], [.5,.7], backend="knapsack")


@pytest.mark.parametrize("backend", ["knapsack", "highs"])
def test_infeasible_program_refuses_instead_of_certifying(backend):
    result = solve_bound([-1., 1.], [0., 0.], [.1, .1], [[.5,.5]], [.4], [.6], backend=backend)
    assert result.status == "infeasible" and result.lower_bound is None


def test_randomized_group_intervals_match_raw_weighted_observations():
    counts = np.array([[[9, 11], [20, 10], [2, 5]]])
    q = np.array([[.2,.4,.4]])
    nu = np.array([[.7,.3,0.]])
    rows = make_group_moments(nu, [0,0,-1])
    ll, uu, details = randomized_intervals(counts, rows, q, .05)
    weights = np.array([.7,.3]) / (q[0,:2] / q[0,:2].sum())
    raw = np.r_[np.zeros(9), np.full(11, weights[0]), np.zeros(20), np.full(10, weights[1])]
    logterm = np.log(4/.05)
    eps = np.sqrt(2*raw.var(ddof=1)*logterm/len(raw)) + 7*max(weights)*logterm/(3*(len(raw)-1))
    assert ll[0] == pytest.approx(max(0., raw.mean()-eps))
    assert uu[0] == pytest.approx(min(1., raw.mean()+eps))
    assert details[0]["n"] == 50


def test_empty_cells_and_missing_randomized_support():
    obs = np.zeros((2,2,2), int)
    q = np.full((2,2), .5)
    rows = make_group_moments(q, [0,1])
    lo, hi, _ = randomized_intervals(obs, rows, q, .05)
    np.testing.assert_array_equal(lo, 0)
    np.testing.assert_array_equal(hi, 1)
    with pytest.raises(ValueError, match="support"):
        randomized_intervals(obs, rows, np.array([[1,0],[1,0]]), .05)
    out = evaluate_counts(obs, obs, target_weights=[.5,.5], policy=[[.2,.8],[.3,.7]],
                          reference=q, moment_rows=rows, randomized_q=q)
    assert out["fusion"]["lower_bound"] < 0
    assert out["diagnostics"]["empty_observation_cells"] == 4


def test_full_alpha_source_bounds_are_at_least_matched_bounds():
    obs = np.array([[[800,200],[100,400]], [[500,300],[250,150]]])
    rand = np.array([[[100,100],[50,150]], [[150,150],[180,120]]])
    q = np.full((2,2), .5)
    rows = make_group_moments(q, [0,0])
    out = evaluate_counts(obs, rand, target_weights=[.5,.5], policy=[[.3,.7],[.4,.6]],
                          reference=q, moment_rows=rows, randomized_q=q)
    for source in ["observation", "moment"]:
        assert out[source+"_standalone"]["lower_bound"] >= out[source+"_matched"]["lower_bound"] - 1e-10
    assert out["fusion"]["lower_bound"] >= max(out["observation_matched"]["lower_bound"], out["moment_matched"]["lower_bound"]) - 1e-10


def test_sparse_plan_memory_is_linear_in_number_of_cells():
    rows = make_group_moments(np.ones((20,100)), np.arange(100))
    assert rows.nnz == 2000 and rows.shape == (2000,2000)
    assert rows.data.nbytes + rows.indices.nbytes + rows.indptr.nbytes < 40000


def test_radius_one_preserves_exact_interval_endpoints():
    p = np.r_[0., 1e-16, .1, .3, .7, 1.]
    lo, hi = conditional_box(p, p, 1.)
    np.testing.assert_array_equal(lo, p)
    np.testing.assert_array_equal(hi, p)
    result = solve_bound(np.ones(len(p)), lo, hi, np.zeros((0,len(p))), [], [])
    assert result.status == "ok"


def test_equal_flattened_size_does_not_allow_different_context_geometry():
    counts = np.ones((2,2,2), int)
    q = np.full((2,2), .5)
    with pytest.raises(ValueError, match="geometry"):
        evaluate_counts(counts,counts,target_weights=[1.],policy=[[.1,.2,.3,.4]],reference=[[.25]*4],
                        moment_rows=make_group_moments(q,[0,0]),randomized_q=q)


def test_chunked_counts_equal_direct_independent_dictionary():
    rng=np.random.default_rng(611)
    x=rng.integers(0,5,1200);a=rng.integers(0,7,1200);y=rng.integers(0,2,1200)
    expected=np.zeros((5,7,2),int)
    for key in zip(x,a,y):
        expected[key]+=1
    actual=sum((count_events(x[i:i+73],a[i:i+73],y[i:i+73],5,7) for i in range(0,len(x),73)),start=np.zeros((5,7,2),int))
    np.testing.assert_array_equal(actual,expected)


def test_generic_solver_small_objective_coefficients_match_knapsack():
    rng=np.random.default_rng(702700+1024)
    d=1024
    rows=make_group_moments(np.ones((8,d//8)),np.arange(d//8)//4)
    lo=rng.uniform(.05,.25,d);hi=rng.uniform(.75,.95,d);tau=rows@rng.uniform(.3,.7,d)
    c=rng.normal(size=d);c*=1e-7/np.max(abs(c))
    specialized=solve_bound(c,lo,hi,rows,tau-.01,tau+.01,backend="knapsack")
    generic=solve_bound(c,lo,hi,rows,tau-.01,tau+.01,backend="highs")
    assert generic.status=="ok"
    assert abs(specialized.lower_bound-generic.lower_bound)<1e-9


@pytest.mark.parametrize("k,n", [([-1],[2]), ([3],[2]), ([.5],[2]), ([0],[np.nan])])
def test_invalid_counts_rejected(k,n):
    with pytest.raises(ValueError):
        binomial_intervals(k,n,.05)
