import json
import numpy as np
from .query_workload import fixture_view,query_stream,measure_method,export_replay


def test_fixture_structures_and_nonproportional_probability_queries():
    sizes={}
    for structure in ['disjoint','mixed','connected']:
        doc,seconds,shape=fixture_view(16,structure)
        assert doc['status']=='ready';sizes[structure]=shape['components']
    assert sizes['disjoint']>sizes['mixed']>sizes['connected']==1
    for stream in ['nearby_nonproportional','unrelated']:
        c=query_stream(16,8,stream,[997,10])
        assert np.allclose(c.sum(1),0) and np.min(c+1/16)>=0
        assert np.linalg.matrix_rank(c)>=2


def test_full_response_serialization_and_replay_all_plans(tmp_path):
    doc,seconds,_=fixture_view(16,'mixed');c=query_stream(16,3,'nearby_nonproportional',[996,11])
    for method in ['highs_cold','highs_warm','disjoint_or_global','components','box_shortcut','dual_cache','prepared_cache']:
        summary,view=measure_method(tmp_path,method,c,.001,16,view=doc['view'],common_seconds=seconds,prefixes=[1,3])
        assert summary['replay']['proofs_valid']==3
        assert summary['prefixes']['3']==summary['cumulative_seconds']
        assert summary['cumulative_seconds']>=summary['first_view_construction_seconds']+summary['first_plan_initialization_seconds']
        assert len(summary['per_query'])==3
        assert (tmp_path/f'{method}.jsonl.gz').exists()


def test_replay_rejects_response_proof_mismatch_or_missing_proof(tmp_path):
    import pytest
    from .confidence_view import QueryEngine
    doc,_,_=fixture_view(8,'disjoint');queries=query_stream(8,1,'nearby_nonproportional',[991,11])
    answer=QueryEngine(doc['view']).query(queries[0])
    answer['lower_bound']+=.0001
    with pytest.raises(ValueError,match='differs from its proof'):
        export_replay(tmp_path/'bad.jsonl.gz',[json.dumps({'query_index':0,'answer':answer})],doc['view'],queries,.001)
    answer={'status':'unresolved','route':'solver','lower_bound':.0,'certificate':None}
    with pytest.raises(ValueError,match='requires a proof'):
        export_replay(tmp_path/'missing.jsonl.gz',[json.dumps({'query_index':0,'answer':answer})],doc['view'],queries,.001)
    answer={'status':'uncertain','route':'materialization_refusal','lower_bound':None,'certificate':None}
    checked=export_replay(tmp_path/'refused.jsonl.gz',[json.dumps({'query_index':0,'answer':answer})],None,queries,.001)
    assert checked['no_bound_responses_checked']==1 and checked['proofs_valid']==0
    with pytest.raises(ValueError,match='order'):
        export_replay(tmp_path/'order.jsonl.gz',[json.dumps({'query_index':1,'answer':answer})],None,queries,.001)
