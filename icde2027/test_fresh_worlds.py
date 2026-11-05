"""Structural checks use excluded test seeds, never the registered study seed."""
import numpy as np
from .fresh_worlds import generate_world,world_specs,conditional_radius
from .aggregate import conditional_box


def test_world_law_constraints_and_true_conditional_boundary():
    for a in [4,8,16]:
        world=generate_world({'id':7,'k':3,'a':a,'design':'skew','rho':.02,'unconfounded':False},917)
        assert np.isclose(world['p_observation'].sum(),1)
        assert np.isclose(world['p_randomized'].sum(),1)
        assert np.allclose(abs(world['transport_residual']),.02,atol=1e-15)
        assert np.all(world['randomized_mean']>0) and np.all(world['randomized_mean']<1)
        assert world['lambda_star']<=world['latent_ratio_bound']+1e-12
        p=world['p_observation_conditional'];m=world['target_mean'];lam=world['lambda_star']
        lo,hi=conditional_box(p,p,lam)
        assert np.all(m>=lo-1e-15) and np.all(m<=hi+1e-15)
        lo,hi=conditional_box(p,p,1+(lam-1)*.99)
        assert np.any(m<lo-1e-12) or np.any(m>hi+1e-12)


def test_unconfounded_boundary_is_one_and_sources_need_not_fully_transport():
    world=generate_world({'id':9,'k':2,'a':8,'design':'uniform','rho':0.,'unconfounded':True},914)
    assert world['lambda_star']==1
    assert np.array_equal(world['p_observation_conditional'],world['target_mean'])
    assert np.max(abs(world['target_mean']-world['randomized_mean']))>.01
    assert np.max(abs(world['transport_residual']))<1e-14


def test_registered_design_has_sixty_independent_world_ids():
    reg={'sizes':[[2,4],[8,8],[16,16]],'randomized_designs':['uniform','skew'],
         'transport_drift':[0.,.02],'worlds_per_stratum':4,'unconfounded_worlds_per_size_design':2}
    specs=world_specs(reg)
    assert len(specs)==60 and len({s['id'] for s in specs})==60
    assert sum(not s['unconfounded'] for s in specs)==48
    assert all(s['rho']==0 for s in specs if s['unconfounded'])


def test_conditional_boundary_upward_rounds_exact_ratio_counterexample():
    from fractions import Fraction
    value,exact=conditional_radius([.9244327187740513],[.4406217732985788])
    exact=Fraction(int(exact['numerator']),int(exact['denominator']))
    assert Fraction.from_float(value)>=exact
    assert Fraction.from_float(float(np.nextafter(value,-np.inf)))<exact


def test_analysis_rejects_duplicate_world_or_wrong_replication():
    import copy,json
    import pytest
    from pathlib import Path
    from .summarize_fresh import validate_census
    reg=json.loads((Path(__file__).parent/'configs/fresh_worlds_v1.json').read_text())
    bundles=[]
    for spec in world_specs(reg):
        records=[{'world_id':spec['id'],'stratum':spec['stratum'],'unconfounded':spec['unconfounded'],
                  'rep':r,'sensitivity':s,'r_events_per_cell':n} for r in range(5)
                 for s in reg['sensitivity'] for n in reg['randomized_events_per_cell']]
        bundles.append({'world':{'spec':spec},'records':records})
    validate_census(bundles,reg)
    invalid=copy.deepcopy(bundles);invalid[1]=copy.deepcopy(invalid[0])
    with pytest.raises(ValueError,match='census'):
        validate_census(invalid,reg)
    invalid=copy.deepcopy(bundles);invalid[0]['records'][0]['rep']=5
    with pytest.raises(ValueError,match='Cartesian'):
        validate_census(invalid,reg)
