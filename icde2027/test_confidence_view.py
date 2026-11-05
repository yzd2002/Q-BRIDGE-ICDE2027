import copy
import json
import numpy as np
import pytest
from scipy import sparse

from .aggregate import conditional_box, make_group_moments, solve_bound
from .confidence_view import create_view, materialize_counts, QueryEngine, connected_components
from .proof_check import CheckedView, verify_certificate


def teaching_view():
    p=np.array([.3,.4,.5,.6])
    lo,hi=conditional_box(p,p,2)
    t=np.array([[.5,0,.5,0],[0,.5,0,.5]])
    result=create_view(lo,hi,t,[.65,.70],[.65,.70],metadata={"kind":"illustrative_population_program_not_empirical"})
    assert result['status']=='ready',result
    return result['view']


def test_teaching_query_reuse_and_fallback_are_verified():
    view=teaching_view()
    engine=QueryEngine(view)
    reference=np.full(4,.25)
    queries=[np.array([.15,.15,.35,.35])-reference,
             np.array([.145,.16,.355,.34])-reference,
             np.array([.20,.10,.40,.30])-reference]
    outputs=[engine.query(c) for c in queries]
    assert outputs[0]['route']=='solver'
    assert outputs[1]['route']=='dual_reuse'
    assert outputs[2]['route']=='solver'
    assert all(r['status']=='certified' for r in outputs)
    assert outputs[0]['lower_bound']==pytest.approx(.01)
    assert outputs[2]['lower_bound']==pytest.approx(.005)
    for row in outputs:
        result=verify_certificate(view,json.loads(json.dumps(row['certificate'])))
        assert result['valid_lp_lower_bound'] and not result['scientific_assumptions_verified']


def test_empty_moment_system_cannot_authorize_positive_dual():
    result=create_view([0.],[1.],[[1.],[1.]],[.8,0.],[1.,.2])
    assert result['status']=='uncertain' and result.get('view') is None
    bad={"format":"qbridge-confidence-view-v1","lower":[0.],"upper":[1.],
         "rhs_lower":[.8,0.],"rhs_upper":[1.,.2],
         "matrix":{"indptr":[0,1,2],"indices":[0,0],"data":[1.,1.]},"feasible_point":[.5]}
    with pytest.raises(ValueError,match='feasibility witness'):
        CheckedView(bad)
    with pytest.raises(ValueError):
        QueryEngine(bad)


def test_empty_statistical_interval_is_retained_as_refusal():
    # A valid, very unlikely R sample outside the confidence event can yield
    # an EB lower endpoint above the known upper limit of a group mean.
    obs=np.array([[[15,15],[15,15]]])
    randomized=np.array([[[0,0],[0,30]]])
    doc=materialize_counts(obs,randomized,rows=[[.5,.5]],randomized_q=[[.75,.25]],
                           alpha=.05,highs_threads=1)
    assert doc['status']=='uncertain' and 'view' not in doc
    assert doc['reason']=='empty_statistical_intersection'
    candidate=doc['candidate_view']
    assert candidate['rhs_lower'][0]>candidate['rhs_upper'][0]==1.
    assert doc['empty_group_intervals']==[0]
    assert candidate['metadata']['source_alpha']==[.025,.025]
    with pytest.raises(ValueError):
        CheckedView(candidate)


def test_empty_cell_interval_refuses_without_solver_or_endpoint_repair(monkeypatch):
    def never(*args,**kwargs):
        raise AssertionError('An empty interval must not reach an optimizer')
    monkeypatch.setattr('icde2027.confidence_view.solve_bound',never)
    doc=create_view([.8],[.2],[[1.]],[0.],[1.])
    assert doc['status']=='uncertain' and doc['empty_cell_intervals']==[0]
    assert doc['candidate_view']['lower']==[.8] and doc['candidate_view']['upper']==[.2]
    with pytest.raises(ValueError,match='invalid program bounds'):
        create_view([1.2],[.2],[[1.]],[0.],[1.])


def test_zero_width_feasible_view_requires_an_actual_exact_witness():
    result=create_view([.5],[.5],[[1.]],[.5],[.5])
    assert result['status']=='ready'
    assert QueryEngine(result['view']).query([1.],.1)['status']=='certified'


def test_proof_cannot_substitute_a_different_view_or_claim():
    view=teaching_view()
    proof=QueryEngine(view).query([-.1,-.1,.1,.1])['certificate']
    bad=copy.deepcopy(proof);bad['claimed_lower']+=.1
    with pytest.raises(ValueError,match='exceeds'):
        verify_certificate(view,bad)
    bad=copy.deepcopy(proof);bad['lower_multipliers'][0]=-1
    with pytest.raises(ValueError,match='nonnegative'):
        verify_certificate(view,bad)
    changed=copy.deepcopy(view);changed['metadata']['extra']='different source declaration'
    with pytest.raises(ValueError,match='trusted view'):
        verify_certificate(changed,proof)
    with pytest.raises(ValueError,match='requested query'):
        verify_certificate(view,proof,expected_objective=[0.,0.,0.,0.],expected_threshold=.001)
    with pytest.raises(ValueError,match='requested query'):
        verify_certificate(view,proof,expected_objective=proof['objective'],expected_threshold=.1)
    checked=verify_certificate(view,proof,expected_objective=proof['objective'],expected_threshold=.001)
    assert checked['requested_query_checked'] and checked['certifies_declared_threshold']


def test_caller_mutation_and_bank_updates_do_not_change_engine_view():
    view=teaching_view();engine=QueryEngine(view)
    digest=engine.digest
    view['lower'][0]=.9
    result=engine.query([-.1,-.1,.1,.1])
    assert engine.digest==digest==result['certificate']['view_digest']
    assert engine.view_document()['lower'][0]<.2
    assert verify_certificate(engine.view_document(),result['certificate'])['valid_lp_lower_bound']


@pytest.mark.parametrize('seed',range(8))
def test_mixed_components_and_adaptive_duals_match_full_program(seed):
    rng=np.random.default_rng(seed)
    d=12
    t=np.zeros((5,d))
    t[0,:3]=[.2,.3,.5];t[1,2:5]=[.3,.4,.3]
    t[2,5:8]=[.1,.2,.7];t[3,8:10]=[.5,.5];t[4,9:11]=[.4,.6]
    truth=rng.uniform(.3,.7,d);lo=np.full(d,.1);hi=np.full(d,.9)
    center=t@truth
    created=create_view(lo,hi,t,center-.07,center+.07)
    assert created['status']=='ready'
    doc=created['view'];fast=QueryEngine(doc,plan='components',bank_size=3)
    full=QueryEngine(doc,plan='global',reuse=False)
    previous=np.zeros(d)
    for i in range(12):
        c=rng.normal(size=d)+.1*previous
        c-=c.mean();c/=abs(c).sum()
        # The next objective explicitly depends on the previous proof values.
        threshold=-.2 if i%3 else .01
        answer=fast.query(c,threshold)
        exact=full.query(c,threshold)
        assert answer['status']==exact['status']
        assert answer['lower_bound']<=float(c@truth)+1e-12
        assert answer['lower_bound']<=exact['primal_diagnostic']+1e-10
        assert verify_certificate(doc,answer['certificate'])['valid_lp_lower_bound']
        previous=c*answer['lower_bound']


def test_component_partition_preserves_all_rows_and_free_coordinates():
    t=sparse.csr_matrix([[.5,.5,0,0,0],[0,.5,.5,0,0],[0,0,0,1,0],[0,0,0,0,0]])
    components,zeros=connected_components(t)
    assert zeros==[3]
    assert sorted(np.concatenate([r[0] for r in components]).tolist())==list(range(5))
    assert sorted(np.concatenate([r[1] for r in components]).tolist())==[0,1,2]
    assert len(components)==3


@pytest.mark.parametrize('kind',['group_score','cell_projection'])
def test_materialized_statistical_view_has_independently_checkable_witness(kind):
    obs=np.array([[[3200,1800],[2000,3000]],[[3500,1500],[1700,3300]]])
    rand=np.array([[[3000,2000],[2000,3000]],[[3000,2000],[2000,3000]]])
    rows=make_group_moments(np.ones((2,2)),[0,0])
    result=materialize_counts(obs,rand,rows=rows,randomized_q=np.full((2,2),.5),randomized_method=kind)
    assert result['status']=='ready'
    check=CheckedView(result['view'])
    assert check.digest==result['digest']


def test_deferred_solver_output_is_explicitly_not_a_bound():
    result=solve_bound([-.1,.1],[.1,.2],[.7,.9],[[.5,.5]],[.5],[.6],export_proposal=True,verification='deferred')
    assert result.status=='proposal' and result.lower_bound is None
    assert result.proposal is not None
    with pytest.raises(ValueError):
        solve_bound([1.],[0.],[1.],np.zeros((0,1)),[],[],verification='deferred')


def test_tiny_coefficients_do_not_poison_dual_bank_or_prevent_fallback():
    view={'format':'qbridge-confidence-view-v1','lower':[.5],'upper':[1.],
          'rhs_lower':[8e-321],'rhs_upper':[1e-320],
          'matrix':{'indptr':[0,1],'indices':[0],'data':[1e-320]},'feasible_point':[1.]}
    engine=QueryEngine(view)
    first=engine.query([1e-300],7e-301)
    assert first['certificate'] is not None
    assert all(np.isfinite(x).all() for pair in engine.bank for x in pair)
    zero=engine.query([0.],0.)
    assert zero['status']=='unresolved'
    assert verify_certificate(view,zero['certificate'],expected_objective=[0.],expected_threshold=0.)['valid_lp_lower_bound']
    # A finite but enormous candidate must not prevent a valid box shortcut.
    engine.bank=[(np.array([1e308]),np.array([1e308]))]
    result=engine.query([2.],.1)
    assert result['route']=='box_shortcut' and result['status']=='certified'


def test_checker_failure_in_candidate_falls_back_without_authorizing(monkeypatch):
    view=teaching_view();engine=QueryEngine(view)
    engine.query([-.1,-.1,.1,.1])
    original=engine.checker.certificate
    calls=[0]
    def fail_once(*args,**kwargs):
        calls[0]+=1
        if calls[0]==1:
            raise ArithmeticError('injected numeric candidate failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(engine.checker,'certificate',fail_once)
    answer=engine.query([-.105,-.09,.105,.09])
    assert answer['route']=='solver' and answer['status']=='certified'
