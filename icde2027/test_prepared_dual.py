"""Prepared proof paths checked independently of their cache state."""
import copy
from fractions import Fraction
import pytest
import numpy as np
from .confidence_view import QueryEngine
from .proof_check import CheckedView,verify_certificate
from .test_confidence_view import teaching_view


def test_prepared_bound_matches_direct_fraction_expansion():
    view=teaching_view();checker=CheckedView(view)
    lo=[.251,.739];hi=[.037,.049];scale=.39;c=[-.105,-.09,.105,.09]
    prepared=checker.prepare_dual(lo,hi)
    f=Fraction.from_float;mult_lo=[f(x)*f(scale) for x in lo];mult_hi=[f(x)*f(scale) for x in hi]
    residual=[f(x) for x in c]
    exact=sum((f(a)*l-f(b)*h for a,b,l,h in zip(view['rhs_lower'],view['rhs_upper'],mult_lo,mult_hi)),Fraction(0))
    for j,row in enumerate([[.5,0,.5,0],[0,.5,0,.5]]):
        for i,coef in enumerate(row):
            residual[i]+=f(coef)*(mult_hi[j]-mult_lo[j])
    exact+=sum((z*f(l if z>=0 else u) for z,l,u in zip(residual,view['lower'],view['upper'])),Fraction(0))
    assert prepared.exact_bound(c,scale)==exact
    proof=prepared.certificate(c,scale,threshold=.001)
    assert verify_certificate(view,proof,expected_objective=c,expected_threshold=.001)['valid_lp_lower_bound']
    bad=copy.deepcopy(proof);bad['dual_scale']=-1
    with pytest.raises(ValueError,match='nonnegative'):
        verify_certificate(view,bad)
    bad=copy.deepcopy(proof);bad['claimed_lower']=.9
    with pytest.raises(ValueError,match='exceeds'):
        verify_certificate(view,bad)


def test_compiled_reuse_and_fallback_preserve_semantics():
    view=teaching_view();engine=QueryEngine(view,compiled_reuse=True)
    queries=[[-.1,-.1,.1,.1],[-.105,-.09,.105,.09],[-.05,-.15,.15,.05]]
    outputs=[engine.query(c) for c in queries]
    assert [r['route'] for r in outputs]==['solver','dual_reuse','solver']
    assert outputs[1]['certificate']['format']=='qbridge-scaled-dual-proof-v1'
    assert outputs[1]['lower_bound']==pytest.approx(.006375)
    for query,result in zip(queries,outputs):
        assert result['status']=='certified'
        assert verify_certificate(view,result['certificate'],expected_objective=query,expected_threshold=.001)['requested_query_checked']


def test_preparation_failure_falls_back_without_authorization(monkeypatch):
    view=teaching_view();engine=QueryEngine(view,compiled_reuse=True)
    engine.query([-.1,-.1,.1,.1])
    def fail(*args):
        raise ArithmeticError('injected preparation failure')
    monkeypatch.setattr(engine.checker,'prepare_dual',fail)
    result=engine.query([-.105,-.09,.105,.09])
    assert result['route']=='solver' and result['status']=='certified'


def test_prepared_state_does_not_follow_input_mutation():
    check=CheckedView(teaching_view());lo=[.2,.2];hi=[0.,0.]
    prepared=check.prepare_dual(lo,hi);value=prepared.exact_bound([-.1,-.1,.1,.1],1.)
    lo[0]=10.
    assert prepared.exact_bound([-.1,-.1,.1,.1],1.)==value


def test_foreign_or_forged_prepared_cache_cannot_authorize():
    from .confidence_view import create_view
    from dataclasses import replace
    view=create_view([.6],[.6],[[1.]],[.6],[.6])['view']
    foreign=create_view([.9],[.9],[[1.]],[.9],[.9])['view']
    engine=QueryEngine(view,compiled_reuse=True)
    lo=np.array([7e15]);hi=np.array([0.]);key=(lo.tobytes(),hi.tobytes())
    engine.bank=[(lo,hi)]
    engine.prepared_duals[key]=CheckedView(foreign).prepare_dual(lo.tolist(),hi.tolist())
    answer=engine.query([1.],.7)
    assert answer['status']=='unresolved' and answer['lower_bound']<=.6
    engine.prepared_duals[key]=None
    answer=engine.query([1.],.7)
    assert answer['status']=='unresolved' and answer['lower_bound']<=.6
    legitimate=engine.checker.prepare_dual(lo.tolist(),hi.tolist())
    forged=replace(legitimate,constant=Fraction(10)**30)
    assert not engine.checker.owns_prepared(forged,lo.tolist(),hi.tolist())
    engine.prepared_duals[key]=forged
    answer=engine.query([1.],.7)
    assert answer['status']=='unresolved' and answer['lower_bound']<=.6


def test_one_time_multiplier_iterators_snapshot_validated_values():
    view=teaching_view();check=CheckedView(view)
    prepared=check.prepare_dual((x for x in [.2,.2]),(x for x in [0.,0.]))
    proof=prepared.certificate([-.1,-.1,.1,.1],1.,threshold=.001)
    assert proof['lower_multipliers']==[.2,.2]
    assert verify_certificate(view,proof)['certifies_declared_threshold']
