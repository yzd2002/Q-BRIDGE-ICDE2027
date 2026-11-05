import math
import numpy as np
import pytest
from scipy.stats import binom

from .source_intervals import observation_intervals, one_sided_binomial_lower, randomized_cell_projection
from .aggregate import make_group_moments


def test_one_sided_binomial_bound_enumerated_coverage():
    for n in [1,3,10,40]:
        k=np.arange(n+1)
        lower=one_sided_binomial_lower(k,n,.05)
        for p in [.001,.05,.2,.5,.95,.999]:
            assert binom.pmf(k,n,p)[lower<=p].sum() >= .95-1e-12


def test_full_source_set_coverage_enumerating_multinomial_counts():
    law=np.array([.6*.8,.6*.2,.4*.3,.4*.7])
    truth=np.array([.2,.7])
    for n in [1,3,8]:
        covered=0.
        for a in range(n+1):
            for b in range(n-a+1):
                for c in range(n-a-b+1):
                    count=np.array([a,b,c,n-a-b-c])
                    prob=math.factorial(n)*float(np.prod(law**count))/math.prod(math.factorial(int(k)) for k in count)
                    lo,hi,_=observation_intervals(count.reshape(1,2,2),1,.1)
                    if np.all(lo<=truth) and np.all(hi>=truth):
                        covered+=prob
        assert covered >= .9-1e-12


def test_consistency_uses_context_joint_counts_and_improves_high_exposure_cell():
    counts=np.array([[[400000,400000],[100000,100000]]])
    lo,hi,info=observation_intervals(counts,2,.05)
    weaklo,weakhi,_=observation_intervals(counts,2,.05,consistency=False)
    assert .39<lo[0]<.4 and .6<hi[0]<.61
    assert lo[0]>weaklo[0] and hi[0]<weakhi[0]
    assert info['conditional_family_alpha']+info['joint_family_alpha']==pytest.approx(.05)


def test_empty_cells_and_contexts_do_not_impute_outcomes():
    counts=np.array([[[0,0],[2,3]],[[0,0],[0,0]]])
    lo,hi,_=observation_intervals(counts,2,.05)
    np.testing.assert_array_equal(lo[[0,2,3]],0)
    np.testing.assert_array_equal(hi[[0,2,3]],1)


def test_cell_projection_transports_only_weighted_groups():
    counts=np.array([[[80,20],[10,90],[0,0]]])
    rows=make_group_moments([[.8,.2,0]],[[0,0,-1]])
    lo,hi,info=randomized_cell_projection(counts,rows,.05)
    assert lo[0]<.8*.2+.2*.9<hi[0]
    assert info['active_cells']==2 and info['transported_cell_means'] is False


@pytest.mark.parametrize('counts',[np.ones((2,3)),np.array([[[1,-1]]]),np.array([[[1,np.nan]]]),np.array([[[.1,2]]])])
def test_malformed_source_counts_rejected(counts):
    with pytest.raises(ValueError):
        observation_intervals(counts,2,.05)


@pytest.mark.parametrize('alpha',[0,1,2,float('nan')])
def test_total_confidence_budget_must_be_valid(alpha):
    counts=np.ones((1,2,2),dtype=int)
    with pytest.raises(ValueError):
        observation_intervals(counts,2,alpha)
    with pytest.raises(ValueError):
        randomized_cell_projection(counts,[[.5,.5]],alpha)
