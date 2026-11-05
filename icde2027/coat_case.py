"""Registered Coat case; outcomes never determine thinning or query features."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from .aggregate import count_events, randomized_intervals, binomial_intervals
from .source_intervals import observation_intervals, randomized_cell_projection
from .confidence_view import create_view, materialize_counts, QueryEngine
from .proof_check import CheckedView
from .experiment import json_write
from .research_io import ROOT, start_run, finish_run, sha256


def thin_users(ratings,seed,source):
    """Uniform one-item selection per user using independent labelled substreams."""
    ratings=np.asarray(ratings)
    events=[]
    for user,row in enumerate(ratings):
        observed=np.flatnonzero(row>0)
        if not len(observed):
            raise ValueError('Registered complete user block is empty')
        rng=np.random.default_rng(np.random.SeedSequence([seed,source,user]))
        action=int(rng.choice(observed))
        events.append([user,action,int(row[action]),int(row[action]>=4)])
    return np.asarray(events,dtype=np.int64)


def metadata_rows(features,names,families):
    columns=[i for i,name in enumerate(names) if name.split(':')[0] in families]
    vectors=features[:,columns].T.astype(float)
    if np.any(vectors.sum(1)==0):
        raise ValueError('Empty registered metadata category')
    return sparse.csr_matrix(vectors/vectors.sum(1)[:,None]),[names[i] for i in columns]


def fixed_queries(features,names,registration):
    a=len(features);reference=np.full(a,1/a);queries=[]
    for feature in registration['queries']['features']:
        h=features[:,names.index(feature)]
        for multiplier in registration['queries']['multipliers']:
            weights=np.where(h==1,multiplier,1.0)
            policy=weights/weights.sum()
            queries.append({'feature':feature,'multiplier':multiplier,'policy':policy.tolist(),
                            'reference':reference.tolist(),'objective':(policy-reference).tolist()})
    return queries


def direct_score(ratings,events,c,alpha):
    """Fixed-query Hoeffding comparisons under static full-cell transport.

    All-16 scores average within users. Both sample sizes remain n_users.
    These are separately calibrated per query, not an adaptive-query rule.
    """
    users,actions=ratings.shape
    a=min(0.,actions*float(np.min(c)));b=max(0.,actions*float(np.max(c)))
    selected=actions*c[events[:,1]]*events[:,3]
    all_events=(ratings>0)
    score_all=(actions*c[None,:]*(ratings>=4)*all_events).sum(1)/all_events.sum(1)
    radius=(b-a)*math.sqrt(math.log(1/alpha)/(2*users))
    return {name:{'n_independent_user_blocks':users,'estimate':float(score.mean()),
                  'sd_across_user_scores':float(score.std(ddof=1)),
                  'range':[a,b],'hoeffding_radius':radius,'lower_bound':float(score.mean()-radius),
                  'full_cell_static_transport_required':True,'per_query_alpha':alpha}
            for name,score in [('one_per_user',selected),('all_16_within_user',score_all)]}


def answer_queries(document,queries):
    if document['status']!='ready':
        return [{'status':'uncertain','reason':document['reason'],'lower_bound':None,'certificate':None} for _ in queries]
    engine=QueryEngine(document['view'],reuse=False,highs_threads=1)
    answers=[]
    for query in queries:
        c=query['objective'];answer=engine.query(c,.001)
        if answer['certificate'] is not None:
            check=engine.checker.verify(answer['certificate'],expected_objective=c,expected_threshold=.001)
            if not check['valid_lp_lower_bound']:
                raise ValueError('Invalid saved query proof')
        answers.append(answer)
    return answers


def run(args):
    protocol=ROOT/'icde2027/configs/coat_case_v1.json'
    reg=json.loads(protocol.read_text())
    manifest=start_run(args.out,protocol)
    begin=time.perf_counter();data=ROOT/reg['dataset']
    paths=[data/'train.ascii',data/'test.ascii',data/'user_item_features/item_features.ascii',
           data/'user_item_features/item_features_map.txt']
    train,test,features=[np.loadtxt(p,dtype=np.int64) for p in paths[:3]]
    names=[s.strip() for s in paths[3].read_text().splitlines()]
    read_seconds=time.perf_counter()-begin
    if train.shape!=(290,300) or test.shape!=train.shape or features.shape!=(300,33):
        raise ValueError('Unexpected official release shapes')
    if not np.all((train>0).sum(1)==24) or not np.all((test>0).sum(1)==16):
        raise ValueError('Unexpected per-user source sizes')
    if np.any(train<0) or np.any(train>5) or np.any(test<0) or np.any(test>5):
        raise ValueError('Invalid rating')
    begin=time.perf_counter()
    selected_o=thin_users(train,reg['master_seed'],0);selected_r=thin_users(test,reg['master_seed'],1)
    counts_o=count_events(np.zeros(290,dtype=np.int64),selected_o[:,1],selected_o[:,3],1,300)
    counts_r=count_events(np.zeros(290,dtype=np.int64),selected_r[:,1],selected_r[:,3],1,300)
    ingestion_seconds=time.perf_counter()-begin
    for source,events in [('observational',selected_o),('randomized',selected_r)]:
        with (args.out/f'{source}_selected_events.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['user_index','item_index','original_rating','binary_success'])
            writer.writerows(events.tolist())
    rows,row_names=metadata_rows(features,names,reg['groups'])
    q=np.full((1,300),1/300);queries=fixed_queries(features,names,reg)
    for query in queries:
        c=np.asarray(query['objective'])
        projection=rows.T@np.linalg.lstsq(rows.toarray().T,c,rcond=None)[0]
        query['distance_to_moment_row_span']=float(np.linalg.norm(c-projection))
        query['direct_randomized_scores']=direct_score(test,selected_r,c,reg['alpha'])
    json_write(args.out/'queries.json',queries)
    json_write(args.out/'counts.json',{'observation':counts_o.tolist(),'randomized':counts_r.tolist()})
    json_write(args.out/'data_provenance.json',{'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in paths},
        'source_url':'https://www.cs.cornell.edu/~schnabts/mnar/coat.zip',
        'paper_url':'https://proceedings.mlr.press/v48/schnabel16.html',
        'license':'https://creativecommons.org/licenses/by-nc/4.0/',
        'transformations':'Binary rating>=4; independently uniform one-item thinning per user and source; original row/column positions retained',
        'source_overlap_cells':int(((train>0)&(test>0)).sum()),'independent_units':290,
        'observed_counts_before_thinning':[int((train>0).sum()),int((test>0).sum())],
        'events_after_thinning':[290,290],'empty_cells_after_thinning':[int((counts_o.sum(2)==0).sum()),int((counts_r.sum(2)==0).sum())],
        'read_and_decode_seconds':read_seconds,'thinning_and_counting_seconds':ingestion_seconds,
        'moment_row_names':row_names,'moment_rows':int(rows.shape[0]),'moment_nonzeros':int(rows.nnz),
        'known_population_truth':False,'iid_study_user_blocks_assumed':True,'full_cell_transport_assumed_only_in_explicit_controls':True})
    all_results=[]
    for radius in reg['sensitivity_radii']:
        lo,hi,_=observation_intervals(counts_o,radius,reg['alpha'])
        obs=create_view(lo,hi,sparse.csr_matrix((0,300)),[],[],metadata={'source':'OBS','radius':radius,'case':'coat'},highs_threads=1)
        for rho in reg['transport_tolerances']:
            documents={'observation':obs}
            for construction in ['group_score','cell_projection']:
                if construction=='group_score':
                    rl,rh,_=randomized_intervals(counts_r,rows,q,reg['alpha'])
                else:
                    rl,rh,_=randomized_cell_projection(counts_r,rows,reg['alpha'])
                documents['randomized_'+construction]=create_view(np.zeros(300),np.ones(300),rows,rl-rho,rh+rho,
                    metadata={'source':'R selected moments','rho':rho,'construction':construction,'case':'coat'},highs_threads=1)
                documents['fusion_'+construction]=materialize_counts(counts_o,counts_r,rows=rows,randomized_q=q,
                    radius=radius,alpha=reg['alpha'],transport_budget=rho,randomized_method=construction,
                    metadata={'case':'coat','target':'iid study-user generating distribution'},highs_threads=1)
            record={'radius':radius,'rho':rho,'methods':{}}
            for name,document in documents.items():
                record['methods'][name]={'document':document,'answers':answer_queries(document,queries)}
            all_results.append(record)
            print(f'Coat Lambda={radius}, rho={rho}: '+', '.join(f'{name}={sum(a["status"]=="certified" for a in result["answers"])} / {len(queries)}' for name,result in record['methods'].items()),flush=True)
    rlo,rhi=binomial_intervals(counts_r[0,:,1],counts_r[0].sum(1),reg['alpha']/300)
    full_cell=create_view(rlo,rhi,sparse.csr_matrix((0,300)),[],[],metadata={'case':'coat','source':'R','assumption':'static full-cell transport','alpha':reg['alpha']},highs_threads=1)
    json_write(args.out/'full_cell_control.json',{'document':full_cell,'answers':answer_queries(full_cell,queries)})
    json_write(args.out/'results.json',all_results)
    primary_query=next(i for i,qry in enumerate(queries) if qry['feature']==reg['queries']['primary']['feature'] and qry['multiplier']==reg['queries']['primary']['multiplier'])
    primary=next(r for r in all_results if r['radius']==reg['primary_radius'] and r['rho']==reg['primary_transport_tolerance'])
    summary={'primary_query_index':primary_query,'primary_query':queries[primary_query],
        'primary_methods':{name:{k:result['answers'][primary_query].get(k) for k in ['status','lower_bound','route','reason']} for name,result in primary['methods'].items()},
        'primary_full_cell_control':{k:answer_queries(full_cell,[queries[primary_query]])[0].get(k) for k in ['status','lower_bound']},
        'all_settings_certified_counts':[{k:r[k] for k in ['radius','rho']}|{name:sum(a['status']=='certified' for a in result['answers']) for name,result in r['methods'].items()} for r in all_results],
        'interpretation':'Conditional case, no known population truth, no coverage estimate, no seed or query selection.'}
    json_write(args.out/'summary.json',summary)
    finish_run(args.out,manifest,configurations=len(all_results),queries_per_configuration=len(queries),
               source_files_sha256={str(p.relative_to(ROOT)):sha256(p) for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
