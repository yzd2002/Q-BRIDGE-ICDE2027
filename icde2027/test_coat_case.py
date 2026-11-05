"""Meaningful case-unit and fixed-query checks without loading study outcomes."""
import numpy as np
from .coat_case import thin_users, direct_score, fixed_queries, metadata_rows


def test_thinning_depends_on_mask_not_rating():
    values=np.array([[0,1,3,5],[5,0,2,1],[2,1,0,4]])
    flipped=np.where(values>0,6-values,0)
    a=thin_users(values,187,0);b=thin_users(flipped,187,0)
    assert np.array_equal(a[:,:2],b[:,:2])
    assert np.array_equal(a[:,0],np.arange(3))
    assert all(values[u,i]>0 for u,i in a[:,:2])
    assert np.array_equal(a,thin_users(values,187,0))


def test_full_r_score_counts_users_not_ratings():
    ratings=np.array([[5,1,5,1],[1,5,1,5]])
    events=thin_users(ratings,182,1);c=np.array([.1,-.1,.1,-.1])
    scores=direct_score(ratings,events,c,.05)
    assert scores['all_16_within_user']['n_independent_user_blocks']==2
    assert scores['all_16_within_user']['estimate']==0
    assert scores['all_16_within_user']['range']==[-.4,.4]
    assert scores['all_16_within_user']['hoeffding_radius']==scores['one_per_user']['hoeffding_radius']


def test_group_rows_do_not_collapse_item_actions():
    features=np.array([[1,0,1],[1,0,0],[0,1,1],[0,1,0]])
    names=['gender:men','gender:women','front:yes']
    rows,labels=metadata_rows(features,names,['gender'])
    assert rows.shape==(2,4)
    assert np.allclose(rows.sum(1),1)
    reg={'queries':{'features':['front:yes'],'multipliers':[2.]}}
    query=fixed_queries(features,names,reg)[0]
    assert np.isclose(sum(query['policy']),1)
    assert np.isclose(sum(query['objective']),0)
    assert np.allclose(query['policy'],[1/3,1/6,1/3,1/6])
