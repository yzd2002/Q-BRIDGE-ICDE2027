"""Registered query-engine and source-to-answer timing with full proof replay."""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import gzip
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
from scipy import sparse

from .aggregate import count_events
from .confidence_view import create_view,materialize_counts,QueryEngine,connected_components
from .highs_reopt import HighsReoptimization
from .proof_check import CheckedView
from .experiment import json_write
from .research_io import ROOT,start_run,finish_run,sha256


def fixture_matrix(d,structure):
    if d%4 or d<8 or structure not in {'disjoint','mixed','connected'}:
        raise ValueError('Registered fixture requires complete four-action blocks')
    row_ids=[];cols=[];values=[];j=0
    for block in range(d//4):
        for pair in [(4*block,4*block+2),(4*block+1,4*block+3)]:
            row_ids.extend([j,j]);cols.extend(pair);values.extend([.5,.5]);j+=1
    if structure!='disjoint':
        for block in range(d//4):
            if structure=='mixed' and block%4:
                continue
            following=(block+1)%(d//4)
            indices=[4*block+2,4*block+3,4*following,4*following+1]
            row_ids.extend([j]*4);cols.extend(indices);values.extend([.25]*4);j+=1
    return sparse.csr_matrix((values,(row_ids,cols)),shape=(j,d))


def fixture_view(d,structure):
    begin=time.perf_counter()
    t=fixture_matrix(d,structure)
    lo=np.tile([.15,.2,.25,.3],d//4);hi=np.tile([.6,.7,.75,.8],d//4)
    point=np.tile([.55,.6,.75,.8],d//4);tau=t@point
    doc=create_view(lo,hi,t,tau-.0005,tau+.0005,
         metadata={'kind':'constructed_execution_program_not_sampled_confidence_set','d':d,'structure':structure},highs_threads=1)
    common_seconds=time.perf_counter()-begin
    if doc['status']!='ready':
        raise ValueError('Controlled feasible execution fixture could not be materialized')
    components,zero=connected_components(t)
    return doc,common_seconds,{'rows':t.shape[0],'nonzeros':t.nnz,'components':len(components),
                             'largest_component_cells':max(len(c) for c,_ in components),'zero_rows':len(zero)}


def query_stream(d,count,kind,seed):
    rng=np.random.default_rng(np.random.SeedSequence(seed))
    if kind=='nearby_nonproportional':
        e=rng.uniform(-.015,.015,(count,d//4,2))
        c=np.stack([-.1-e[:,:,0],-.1-e[:,:,1],.1+e[:,:,0],.1+e[:,:,1]],axis=2).reshape(count,d)/(d//4)
    elif kind=='unrelated':
        c=rng.normal(size=(count,d));c-=c.mean(1,keepdims=True)
        c*=.8/(d*np.max(abs(c),axis=1,keepdims=True))
    else:
        raise ValueError('Unknown registered query stream')
    if np.max(abs(c.sum(1)))>1e-12 or np.min(1/d+c)<-1e-14:
        raise ValueError('Query is not a feasible probability-policy contrast')
    return c


def engine_for(view,method,bank_size):
    if method.endswith('highs_cold'):
        return HighsReoptimization(view,retain_basis=False)
    if method.endswith('highs_warm'):
        return HighsReoptimization(view,retain_basis=True)
    plan='disjoint_or_global' if method=='disjoint_or_global' else 'components'
    reuse=method in {'box_shortcut','dual_cache','prepared_cache','materialized_prepared_cache'}
    return QueryEngine(view,plan=plan,reuse=reuse,bank_size=0 if method=='box_shortcut' else bank_size,
                       compiled_reuse=method in {'prepared_cache','materialized_prepared_cache'},highs_threads=1)


def prepared_bytes(engine):
    """Python footprint of prepared-only terms; shared view endpoints excluded."""
    total=0;seen=set()
    def size(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj));n=sys.getsizeof(obj)
        if isinstance(obj,Fraction):
            return n+size(obj.numerator)+size(obj.denominator)
        if isinstance(obj,tuple):
            return n+sum(size(x) for x in obj)
        return n
    for state in getattr(engine,'prepared_duals',{}).values():
        total+=size(state.offset)+size(state.constant)+size(state.base_lower)+size(state.base_upper)
    return total


def export_replay(path,encoded,view,queries,threshold):
    write_start=time.perf_counter()
    with gzip.open(path,'wt',encoding='utf8',compresslevel=3) as f:
        for row in encoded:
            f.write(row+'\n')
    writing=time.perf_counter()-write_start
    initialize=time.perf_counter();checker=CheckedView(view) if view is not None else None
    check_init=time.perf_counter()-initialize if checker is not None else 0.
    verify_seconds=0.;decode_seconds=0.;valid=0;missing=0;routes=Counter();statuses=Counter();bounds=[]
    byformat=Counter();format_time=Counter()
    with gzip.open(path,'rt',encoding='utf8') as f:
        for i,line in enumerate(f):
            begin=time.perf_counter();record=json.loads(line);decode_seconds+=time.perf_counter()-begin
            if i>=len(queries) or record.get('query_index')!=i:
                raise ValueError('Saved query stream order differs from requested sequence')
            answer=record['answer'];routes[answer['route']]+=1;statuses[answer['status']]+=1
            bounds.append(answer['lower_bound'])
            proof=answer['certificate']
            if proof is None:
                missing+=1
                if answer['status']!='uncertain' or answer['lower_bound'] is not None:
                    raise ValueError('A finite answer requires a proof; no-proof output must be uncertain')
                continue
            if checker is None:
                raise ValueError('A proof cannot be verified without a caller-trusted ready view')
            if answer['lower_bound']!=proof['claimed_lower']:
                raise ValueError('Saved answer bound differs from its proof')
            begin=time.perf_counter()
            result=checker.verify(proof,expected_objective=queries[i].tolist(),expected_threshold=threshold)
            elapsed=time.perf_counter()-begin;verify_seconds+=elapsed
            byformat[proof['format']]+=1;format_time[proof['format']]+=elapsed
            if result['certifies_declared_threshold']!=(answer['status']=='certified'):
                raise ValueError('Saved decision does not match independently replayed proof')
            valid+=1
    if valid+missing!=len(queries):
        raise ValueError('Incomplete proof stream')
    return {'proof_file':path.name,'compressed_bytes':path.stat().st_size,'gzip_write_seconds':writing,
            'replay_checker_initialization_seconds':check_init,'replay_decode_seconds':decode_seconds,
            'replay_verify_seconds':verify_seconds,'proofs_valid':valid,'proofs_missing':missing,
            'no_bound_responses_checked':missing,
            'proof_formats':dict(byformat),'verify_seconds_by_format':dict(format_time),
            'routes':dict(routes),'statuses':dict(statuses),'lower_bounds':bounds}


def _measure_method(directory,method,queries,threshold,bank_size,*,view=None,common_seconds=0.,provider=None,rebuild=False,prefixes=(),progress=None):
    encoded=[] if progress is None else progress
    records=[];cumulative=0.;engine=None;trusted=None;cached_refusal=None
    maximum_bank=0;maximum_prepared=0;first_view_cost=0.;first_init=0.
    stage_totals=Counter();kernel_totals=Counter()
    for i,c in enumerate(queries):
        if engine is None or rebuild:
            if provider is not None:
                if cached_refusal is not None and not rebuild:
                    document,stages=cached_refusal,{}
                else:
                    document,stages=provider()
                docview=document.get('view')
                construction=sum(stages.values())
                stage_totals.update(stages)
                if document['status']!='ready':
                    if cached_refusal is None:
                        cached_refusal=document
                        json_write(directory/f'{method}_refused_document.json',document)
                    # Complete the declared queries as explicit refusals.
                    response_start=time.perf_counter()
                    refusal={'status':'uncertain','route':'materialization_refusal','reason':document['reason'],
                             'lower_bound':None,'certificate':None,'seconds':0.,'verify_seconds':0.,'solve_seconds':0.,'screen_seconds':0.}
                    response_seconds=time.perf_counter()-response_start
                    encode_start=time.perf_counter()
                    encoded.append(json.dumps({'query_index':i,'answer':refusal},separators=(',',':')))
                    serialization=time.perf_counter()-encode_start
                    cumulative+=construction+response_seconds+serialization
                    if i==0:
                        first_view_cost=construction
                    records.append({'query':i+1,'cumulative_seconds':cumulative,'query_seconds':response_seconds,'serialization_seconds':serialization,'status':'uncertain'})
                    continue
            else:
                docview=view;construction=common_seconds
            if trusted is None:
                trusted=docview;first_view_cost=construction
                if not (directory/'view.json').exists():
                    json_write(directory/'view.json',trusted)
            elif CheckedView(docview).digest!=CheckedView(trusted).digest:
                raise ValueError('Rebuilding identical inputs changed the represented trusted view')
            begin=time.perf_counter();engine=engine_for(docview,method,bank_size);initialization=time.perf_counter()-begin
            if i==0:
                first_init=initialization
            cumulative+=construction+initialization
            stage_totals['plan_checker_initialization']+=initialization
        begin=time.perf_counter();answer=engine.query(c,threshold);query_time=time.perf_counter()-begin
        encode_start=time.perf_counter();row=json.dumps({'query_index':i,'answer':answer},separators=(',',':'),allow_nan=False)
        serialization=time.perf_counter()-encode_start
        cumulative+=query_time+serialization
        records.append({'query':i+1,'cumulative_seconds':cumulative,'query_seconds':query_time,
                        'serialization_seconds':serialization,'status':answer['status']})
        encoded.append(row)
        maximum_bank=max(maximum_bank,len(getattr(engine,'bank',[])))
        for name in ['screen_seconds','solve_seconds','verify_seconds']:
            kernel_totals[name]+=answer.get(name,0.)
    if engine is not None:
        maximum_prepared=prepared_bytes(engine)
    replay=export_replay(directory/f'{method}.jsonl.gz',encoded,trusted,queries,threshold)
    if trusted is not None and not (directory/'view.json').exists():
        json_write(directory/'view.json',trusted)
    summary={'method':method,'queries':len(queries),'cumulative_seconds':cumulative,
             'prefixes':{str(n):records[n-1]['cumulative_seconds'] for n in prefixes if n<=len(records)},
             'first_view_construction_seconds':first_view_cost,'first_plan_initialization_seconds':first_init,
             'input_stage_total_seconds':dict(stage_totals),'query_kernel_total_seconds':dict(kernel_totals),
             'serialization_total_seconds':sum(r['serialization_seconds'] for r in records),
             'dual_preparation_seconds':getattr(engine,'dual_preparation_seconds',0.),
             'maximum_bank_entries':maximum_bank,'final_prepared_terms_python_bytes':maximum_prepared,
             'view_json_bytes':len(json.dumps(trusted,separators=(',',':')).encode()) if trusted is not None else None,
             'process_peak_rss_kib_not_method_isolated':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'per_query':records,'replay':replay}
    json_write(directory/f'{method}_summary.json',summary)
    return summary,trusted


def measure_method(directory,method,queries,threshold,bank_size,**kwargs):
    progress=[]
    try:
        return _measure_method(directory,method,queries,threshold,bank_size,progress=progress,**kwargs)
    except Exception as error:
        with gzip.open(directory/f'{method}_partial.jsonl.gz','wt',encoding='utf8') as f:
            for line in progress:
                f.write(line+'\n')
        json_write(directory/f'{method}_failure.json',{'method':method,'completed_responses':len(progress),
                   'exception_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()})
        raise


def comparisons(methods,warm):
    reference=methods[warm];out={}
    reference_bounds=reference['replay']['lower_bounds']
    reference_decisions=[r['status']=='certified' for r in reference['per_query']]
    for name,summary in methods.items():
        differences=[float(a-b) for a,b in zip(summary['replay']['lower_bounds'],reference_bounds) if a is not None and b is not None]
        out[name]={'time_ratio_to_warm':summary['cumulative_seconds']/reference['cumulative_seconds'],
                  'decision_disagreements_to_warm':sum((row['status']=='certified')!=ref for row,ref in zip(summary['per_query'],reference_decisions)),
                  'mean_lower_minus_warm':float(np.mean(differences)) if differences else None,
                  'min_lower_minus_warm':min(differences) if differences else None,
                  'max_lower_minus_warm':max(differences) if differences else None}
    return out


def generate_events(directory,reg,design,rep):
    begin=time.perf_counter();cfg=reg['end_to_end'];a=cfg['actions']
    ranks=1/np.arange(1,a+1,dtype=float)**.7
    q=np.full(a,1/a) if design=='uniform' else .1/a+.9*ranks/ranks.sum()
    sources={};files={}
    for source,n in enumerate([cfg['observational_events'],cfg['randomized_events']]):
        rng=np.random.default_rng(np.random.SeedSequence([reg['master_seed'],2,0 if design=='uniform' else 1,rep,source]))
        action_prob=np.full(a,1/a) if source==0 else q
        mean=np.tile([.3,.4,.5,.6] if source==0 else [.55,.6,.75,.8],a//4)
        probabilities=(action_prob[:,None]*np.stack([1-mean,mean],axis=1)).ravel()
        codes=rng.choice(2*a,size=n,p=probabilities)
        actions=(codes//2).astype(np.uint32);outcomes=(codes%2).astype(np.uint8)
        name='observation' if source==0 else 'randomized'
        for column,array in [('action',actions),('outcome',outcomes)]:
            path=directory/f'{name}_{column}.npy';np.save(path,array);files[path.name]={'sha256':sha256(path),'bytes':path.stat().st_size}
        sources[name]=n
    result={'source_rows':sources,'randomized_q':q.tolist(),'generation_seconds':time.perf_counter()-begin,'files':files,
            'scope':'Generated encoded columns; generation and integrity hashing excluded from serving latency'}
    json_write(directory/'event_generation.json',result)
    return q,result


def event_provider(directory,cfg,q,meta):
    def provide():
        begin=time.perf_counter()
        arrays={name:[np.load(directory/f'{name}_{col}.npy') for col in ['action','outcome']] for name in ['observation','randomized']}
        read_seconds=time.perf_counter()-begin
        begin=time.perf_counter();counts=[]
        for name in ['observation','randomized']:
            action,outcome=arrays[name]
            counts.append(count_events(np.zeros(len(action),dtype=np.uint8),action,outcome,1,cfg['actions']))
        count_seconds=time.perf_counter()-begin
        begin=time.perf_counter()
        document=materialize_counts(*counts,rows=fixture_matrix(cfg['actions'],'disjoint'),randomized_q=q[None,:],
            radius=cfg['radius'],alpha=cfg['alpha'],transport_budget=cfg['rho'],randomized_method=cfg['randomized_intervals'],
            metadata=meta,highs_threads=1)
        view_seconds=time.perf_counter()-begin
        stages={'read_encoded_columns':read_seconds,'validate_and_aggregate':count_seconds,
                'intervals_and_feasible_view':view_seconds}
        return document,stages
    return provide


def run(args):
    protocol=ROOT/'icde2027/configs/query_workload_v1.json';reg=json.loads(protocol.read_text())
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[0]})
    manifest=start_run(args.out,protocol);manifest['inherited_affinity_before_pin']=allowed
    manifest['stage']=args.stage;json_write(args.out/'manifest.json',manifest)
    completed=[]
    try:
        if args.stage=='query_engine':
            cfg=reg['query_engine']
            for d in cfg['dimensions']:
                for si,structure in enumerate(cfg['structures']):
                    for qi,stream in enumerate(cfg['streams']):
                        for rep in range(reg['repetitions']):
                            name=f'd{d}_{structure}_{stream}_r{rep}';directory=args.out/name;directory.mkdir()
                            doc,common,shape=fixture_view(d,structure);json_write(directory/'view.json',doc['view'])
                            queries=query_stream(d,cfg['max_queries_by_dimension'][str(d)],stream,[reg['master_seed'],1,d,si,qi,rep])
                            np.save(directory/'queries.npy',queries)
                            order=np.random.default_rng(np.random.SeedSequence([reg['master_seed'],8,d,si,qi,rep])).permutation(cfg['methods']).tolist()
                            condition={'name':name,'d':d,'structure':structure,'stream':stream,'rep':rep,'shape':shape,
                                       'common_view_seconds':common,'method_order':order,'queries_sha256':sha256(directory/'queries.npy')}
                            json_write(directory/'condition.json',condition);methods={}
                            for method in order:
                                summary,_=measure_method(directory,method,queries,reg['threshold'],reg['bank_size'],
                                    view=doc['view'],common_seconds=common,prefixes=reg['query_prefixes'])
                                methods[method]=summary
                            condition['comparisons']=comparisons(methods,'highs_warm')
                            completed.append(condition);json_write(directory/'condition.json',condition)
                            json_write(args.out/'completed_conditions.json',completed)
                            print(f'completed {name}: '+', '.join(f'{m}={v["cumulative_seconds"]:.3f}s' for m,v in methods.items()),flush=True)
        elif args.stage=='end_to_end':
            cfg=reg['end_to_end']
            for di,design in enumerate(cfg['randomized_designs']):
                for rep in range(reg['repetitions']):
                    name=f'{design}_r{rep}';directory=args.out/name;directory.mkdir()
                    q,generation=generate_events(directory,reg,design,rep)
                    queries=query_stream(cfg['actions'],cfg['queries'],cfg['stream'],[reg['master_seed'],4,di,rep])
                    np.save(directory/'queries.npy',queries)
                    order=np.random.default_rng(np.random.SeedSequence([reg['master_seed'],9,di,rep])).permutation(cfg['methods']).tolist()
                    condition={'name':name,'design':design,'rep':rep,'method_order':order,'generation':generation,
                               'queries_sha256':sha256(directory/'queries.npy')}
                    json_write(directory/'condition.json',condition);methods={};digest=None
                    provider=event_provider(directory,cfg,q,{'fixture':'registered_end_to_end','design':design,'rep':rep})
                    for method in order:
                        summary,view=measure_method(directory,method,queries,reg['threshold'],reg['bank_size'],provider=provider,
                            rebuild=method=='rebuild_highs_cold',prefixes=reg['query_prefixes'])
                        if view is not None:
                            current=CheckedView(view).digest
                            if digest is not None and current!=digest:
                                raise ValueError('Paired methods constructed different views from the same files')
                            digest=current
                        methods[method]=summary
                    condition['comparisons']=comparisons(methods,'materialized_highs_warm')
                    condition['shared_view_digest']=digest
                    completed.append(condition);json_write(directory/'condition.json',condition)
                    json_write(args.out/'completed_conditions.json',completed)
                    print(f'completed {name}: '+', '.join(f'{m}={v["cumulative_seconds"]:.3f}s' for m,v in methods.items()),flush=True)
        else:
            raise ValueError('Unregistered workload stage')
    except Exception as error:
        manifest.update(status='failed',error=str(error),traceback=traceback.format_exc(),completed_conditions=len(completed))
        json_write(args.out/'manifest.json',manifest)
        raise
    finish_run(args.out,manifest,completed_conditions=len(completed))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['query_engine','end_to_end'],required=True)
    run(parser.parse_args())
