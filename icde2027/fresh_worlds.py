"""Prespecified heterogeneous worlds and independent policy-design samples.

World generation never receives an evaluation query. All source laws, sample
counts, policies, views, proofs and unsuccessful results are retained.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import json
import math
from pathlib import Path
import traceback

import numpy as np
from scipy import sparse
from scipy.special import expit,softmax

from .aggregate import make_group_moments,policy_objective,randomized_intervals
from .source_intervals import observation_intervals,randomized_cell_projection
from .confidence_view import create_view,materialize_counts,QueryEngine
from .proof_check import CheckedView
from .experiment import draw_counts,json_write
from .research_io import ROOT,start_run,finish_run


def world_specs(reg):
    specs=[]
    for k,a in reg['sizes']:
        for design in reg['randomized_designs']:
            for rho in reg['transport_drift']:
                for j in range(reg['worlds_per_stratum']):
                    specs.append({'id':len(specs),'k':k,'a':a,'design':design,'rho':rho,
                                  'unconfounded':False,'within_stratum_id':j,
                                  'stratum':f'{k}x{a}_{design}_rho{rho}'})
    for k,a in reg['sizes']:
        for design in reg['randomized_designs']:
            for j in range(reg['unconfounded_worlds_per_size_design']):
                specs.append({'id':len(specs),'k':k,'a':a,'design':design,'rho':0.,
                              'unconfounded':True,'within_stratum_id':j,
                              'stratum':f'{k}x{a}_{design}_unconfounded'})
    return specs


def conditional_radius(probability,target):
    """Upward representable boundary for the represented binary probabilities."""
    maximum=Fraction(1)
    for p,m in zip(np.ravel(probability),np.ravel(target)):
        p,m=Fraction.from_float(float(p)),Fraction.from_float(float(m))
        if not 0<p<1 or not 0<m<1:
            raise ValueError('Interior binary means required for this world generator')
        r1=m/p;r0=(1-m)/(1-p)
        maximum=max(maximum,r1,r0,1/r1,1/r0)
    value=float(maximum)
    if Fraction.from_float(value)<maximum:
        value=float(np.nextafter(value,np.inf))
    return value,{'numerator':str(maximum.numerator),'denominator':str(maximum.denominator)}


def generate_world(spec,master):
    rng=np.random.default_rng(np.random.SeedSequence([master,0,spec['id']]))
    k,a=spec['k'],spec['a'];h=np.linspace(-.5,.5,a)
    px=rng.dirichlet(np.full(k,1.5));latent_p=rng.uniform(.25,.75,k)
    pu=np.stack([1-latent_p,latent_p],axis=1)
    intercept=rng.uniform(-.5,.5,(k,1));action_coef=rng.uniform(1,3)
    noise=rng.normal(0,.3,(k,a));reward_coef=rng.uniform(.3,1)
    reward_logits=intercept+action_coef*h[None,:]+noise
    py=expit(np.clip(reward_logits[:,None,:]+np.array([-1,1])[None,:,None]*reward_coef,-3,3))
    logits=rng.normal(0,.6,(k,a))
    gamma=0. if spec['unconfounded'] else rng.uniform(.15,.55)
    hconf=rng.uniform(-1,1,(k,a))
    e=softmax(logits[:,None,:]+np.array([-1,1])[None,:,None]*gamma*hconf[:,None,:],axis=2)
    marginal_e=np.sum(pu[:,:,None]*e,axis=1)
    target=np.sum(pu[:,:,None]*py,axis=1)
    joint_success=np.sum(pu[:,:,None]*e*py,axis=1)
    p_obs=joint_success/marginal_e
    if spec['unconfounded']:
        # The same represented conditional probability defines sampling and
        # target for the exact Lambda=1 exercise, eliminating roundoff drift.
        target=p_obs.copy()
    po=px[:,None,None]*marginal_e[:,:,None]*np.stack([1-p_obs,p_obs],axis=2)
    conditional_boundary,exact_boundary=conditional_radius(p_obs,target)
    if spec['unconfounded']:
        conditional_boundary=1.
    latent_boundary=float(np.max(np.stack([marginal_e[:,None,:]/e,e/marginal_e[:,None,:]])))
    if spec['design']=='uniform':
        q=np.full((k,a),1/a)
    else:
        q=.1/a+.9*np.stack([rng.dirichlet(np.full(a,.7)) for _ in range(k)])
    groups=np.empty((k,a),dtype=int);width=2 if a==4 else 4
    for x in range(k):
        groups[x,rng.permutation(a)]=np.arange(a)//width
    raw_nu=.7*q+.3/a
    rows=make_group_moments(raw_nu,groups)
    randomized=target.copy();drift_signs=[]
    for x in range(k):
        for label in np.unique(groups[x]):
            ids=np.flatnonzero(groups[x]==label);nu=raw_nu[x,ids]/raw_nu[x,ids].sum()
            sign=int(rng.choice([-1,1]));drift_signs.append(sign)
            base=target[x,ids]+sign*spec['rho']
            if np.min(base)<=0 or np.max(base)>=1:
                raise ValueError('Registered drift violates guaranteed outcome margin')
            direction=rng.uniform(-1,1,len(ids));direction-=nu@direction
            amplitude=float(np.max(np.abs(direction)))
            if amplitude>0:
                direction*=.4*float(np.min(np.minimum(base,1-base)))/amplitude
            randomized[x,ids]=base+direction
    residual=np.asarray(rows@(randomized-target).ravel())
    rho=spec['rho']+1e-12
    if np.max(np.abs(residual),initial=0)>rho or np.min(randomized)<=0 or np.max(randomized)>=1:
        raise ValueError('Registered source shift failed its model checks')
    pr=px[:,None,None]*q[:,:,None]*np.stack([1-randomized,randomized],axis=2)
    return {'spec':spec,'target_weights':px,'target_mean':target,'p_observation':po,
            'p_observation_conditional':p_obs,'randomized_mean':randomized,'randomized_q':q,'p_randomized':pr,
            'latent_probabilities':pu,'latent_logger':e,'latent_reward':py,'logging_gamma':gamma,
            'lambda_star':conditional_boundary,'latent_ratio_bound':latent_boundary,
            'conditional_boundary_exact':exact_boundary,
            'rows':rows,'groups':groups,'raw_nu':raw_nu,'metadata_score':h,
            'rho_declared':rho,'rho_nominal':spec['rho'],'transport_residual':residual,'drift_signs':drift_signs}


def serializable_world(world):
    out={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in world.items() if k!='rows'}
    t=world['rows'];out['moment_matrix']={'shape':list(t.shape),'indptr':t.indptr.tolist(),'indices':t.indices.tolist(),'data':t.data.tolist()}
    return out


def rng_stream(reg,family,world,rep,index):
    return np.random.default_rng(np.random.SeedSequence([reg['master_seed'],family,world,rep,index]))


def evaluate_document(document,queries,world,threshold):
    candidate=document.get('view',document.get('candidate_view'))
    if candidate is None:
        raise ValueError('Every materialization must retain its candidate set')
    try:
        # Evaluator-only, and never used by QueryEngine or create_view.
        CheckedView({**candidate,'feasible_point':world['target_mean'].ravel().tolist()})
        contains_truth=True
    except ValueError:
        contains_truth=False
    results={}
    engine=QueryEngine(document['view'],reuse=False,highs_threads=1) if document['status']=='ready' else None
    for name,query in queries.items():
        c=np.asarray(query['objective']);truth=float(c@world['target_mean'].ravel())
        if engine is None:
            answer={'status':'uncertain','reason':document['reason'],'lower_bound':None,'certificate':None}
        else:
            answer=engine.query(c,threshold)
            if answer['certificate'] is not None:
                engine.checker.verify(answer['certificate'],expected_objective=c.tolist(),expected_threshold=threshold)
        bound=answer['lower_bound']
        answer.update(true_contrast=truth,exceeds_truth=bound is not None and bound>truth+1e-12,
                      false_assertion=answer['status']=='certified' and truth<=threshold)
        if answer['exceeds_truth'] and contains_truth:
            raise ValueError('A verified bound exceeded a true mean inside the represented set')
        results[name]=answer
    return {'document':document,'set_contains_truth':contains_truth,'queries':results}


def evaluator_membership(document,mean):
    candidate=document.get('view',document.get('candidate_view'))
    if candidate is None:
        raise ValueError('Candidate set missing from coverage audit')
    try:
        CheckedView({**candidate,'feasible_point':np.ravel(mean).tolist()})
        return True
    except ValueError:
        return False


def policy_queries(world,design_o,design_r,radius,reg):
    k,a=world['spec']['k'],world['spec']['a'];reference=np.full((k,a),1/a)
    metadata=np.broadcast_to(softmax(1.5*world['metadata_score']),(k,a)).copy()
    estimates=(design_r[:,:,1]+.5)/(design_r.sum(2)+1.)
    heuristic=softmax(4*(estimates-estimates.mean(1,keepdims=True)),axis=1)
    design=materialize_counts(design_o,design_r,rows=world['rows'],randomized_q=world['randomized_q'],
        radius=radius,alpha=reg['alpha'],transport_budget=world['rho_declared'],randomized_method='cell_projection',
        metadata={'role':'independent policy design'},highs_threads=1)
    selected=reference.copy();best=0.;candidate_results=[];selected_index=-1
    if design['status']=='ready':
        engine=QueryEngine(design['view'],reuse=False,highs_threads=1)
        for score_name,score in [('linear',world['metadata_score']),('alternating',np.arange(a)%2-.5)]:
            for tilt in reg['robust_candidate_tilts']:
                policy=np.broadcast_to(softmax(tilt*score),(k,a)).copy()
                c=policy_objective(world['target_weights'],policy,reference)
                answer=engine.query(c,0.)
                candidate_results.append({'score':score_name,'tilt':tilt,'answer':answer})
                value=answer['lower_bound']
                if value is not None and value>best:
                    best=value;selected=policy;selected_index=len(candidate_results)-1
    policies={'metadata':metadata,'randomized_heuristic':heuristic,'robust_design_selection':selected}
    queries={name:{'policy':policy.tolist(),'reference':reference.tolist(),
                   'objective':policy_objective(world['target_weights'],policy,reference).tolist()}
             for name,policy in policies.items()}
    return queries,{'document':design,'candidate_results':candidate_results,'selected_index':selected_index,
                    'reference_selected':selected_index==-1,'selected_design_lower':best}


def _run_world(job):
    spec,reg,*destinations=job
    progress=Path(destinations[0]) if destinations else None
    world=generate_world(spec,reg['master_seed']);records=[]
    if progress is not None:
        progress.mkdir(parents=True,exist_ok=True)
        json_write(progress/'generated_world.json',serializable_world(world))
    d=spec['k']*spec['a'];empty=sparse.csr_matrix((0,d))
    for rep in range(reg['sampling_repetitions']):
        obs=draw_counts(rng_stream(reg,2,spec['id'],rep,0),world['p_observation'],reg['observation_events_per_cell']*d)
        design_o=draw_counts(rng_stream(reg,1,spec['id'],rep,0),world['p_observation'],reg['design_observation_events_per_cell']*d)
        design_r=draw_counts(rng_stream(reg,1,spec['id'],rep,1),world['p_randomized'],reg['design_randomized_events_per_cell']*d)
        if progress is not None:
            json_write(progress/'active_inputs.json',{'rep':rep,'phase':'policy_design','observation':obs.tolist(),
                       'design_observation':design_o.tolist(),'design_randomized':design_r.tolist()})
        for sensitivity in reg['sensitivity']:
            radius=world['lambda_star'] if sensitivity=='true_boundary' else max(2.,1.25*world['lambda_star'])
            if sensitivity=='true_boundary' and radius>1:
                radius=float(np.nextafter(radius,np.inf))
            queries,selection=policy_queries(world,design_o,design_r,radius,reg)
            # Keep truth outside policy_queries; audit design and inference
            # coverage in distinct families, deduplicated across R budgets.
            selection['set_contains_truth']=evaluator_membership(selection['document'],world['target_mean'])
            for budget_index,per_cell in enumerate(reg['randomized_events_per_cell']):
                rand=draw_counts(rng_stream(reg,3,spec['id'],rep,budget_index),world['p_randomized'],per_cell*d)
                meta={'world_id':spec['id'],'rep':rep,'sensitivity':sensitivity,'r_events_per_cell':per_cell}
                if progress is not None:
                    json_write(progress/'active_inputs.json',{**meta,'radius':radius,'observation':obs.tolist(),
                        'randomized':rand.tolist(),'design_observation':design_o.tolist(),'design_randomized':design_r.tolist(),
                        'queries':queries,'design_selection':selection})
                lo,hi,_=observation_intervals(obs,radius,reg['alpha'])
                documents={'observation_consistency':create_view(lo,hi,empty,[],[],metadata={**meta,'source':'OBS'},highs_threads=1)}
                for construction,suffix in [('cell_projection','projected'),('group_score','group_score')]:
                    if construction=='cell_projection':
                        rlo,rhi,_=randomized_cell_projection(rand,world['rows'],reg['alpha'])
                    else:
                        rlo,rhi,_=randomized_intervals(rand,world['rows'],world['randomized_q'],reg['alpha'])
                    documents['randomized_'+suffix]=create_view(np.zeros(d),np.ones(d),world['rows'],
                        rlo-world['rho_declared'],rhi+world['rho_declared'],metadata={**meta,'source':'R moments','construction':construction},highs_threads=1)
                    documents['fusion_'+suffix]=materialize_counts(obs,rand,rows=world['rows'],randomized_q=world['randomized_q'],
                        radius=radius,alpha=reg['alpha'],transport_budget=world['rho_declared'],randomized_method=construction,
                        metadata=meta,highs_threads=1)
                methods={name:evaluate_document(document,queries,world,reg['threshold']) for name,document in documents.items()}
                records.append({**meta,'stratum':spec['stratum'],'unconfounded':spec['unconfounded'],'radius':radius,
                    'rho_nominal':world['rho_nominal'],'rho_declared':world['rho_declared'],
                    'observation_counts':obs.tolist(),'randomized_counts':rand.tolist(),
                    'design_observation_counts':design_o.tolist(),'design_randomized_counts':design_r.tolist(),
                    'queries':queries,'design_selection':selection,'methods':methods})
                if progress is not None:
                    with (progress/'completed_configurations.jsonl').open('a') as stream:
                        stream.write(json.dumps(records[-1],allow_nan=False)+'\n')
                        stream.flush()
    return {'world':serializable_world(world),'records':records}


def run_world(job):
    try:
        return _run_world(job)
    except Exception as error:
        if len(job)>2:
            progress=Path(job[2]);progress.mkdir(parents=True,exist_ok=True)
            json_write(progress/'failure.json',{'world_spec':job[0],'exception_type':type(error).__name__,
                       'message':str(error),'traceback':traceback.format_exc()})
        raise


def run(args):
    protocol=ROOT/'icde2027/configs/fresh_worlds_v1.json';reg=json.loads(protocol.read_text())
    manifest=start_run(args.out,protocol);(args.out/'worlds').mkdir()
    specs=world_specs(reg)
    json_write(args.out/'world_specifications.json',specs)
    jobs=[(spec,reg,str(args.out/'progress'/f'world_{spec["id"]:03d}')) for spec in specs]
    completed=0
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(run_world,jobs,chunksize=1):
                wid=result['world']['spec']['id']
                json_write(args.out/'worlds'/f'world_{wid:03d}.json',result)
                completed+=1
                print(f'completed fresh world {completed}/{len(specs)}; retained {len(result["records"])} configurations',flush=True)
    except Exception as error:
        manifest.update(status='failed',completed_world_files=completed,error=str(error),traceback=traceback.format_exc())
        json_write(args.out/'manifest.json',manifest)
        raise
    finish_run(args.out,manifest,worlds=completed,workers=args.workers,
               sampling='Exact multinomial sufficient counts; no raw-row throughput claim',
               configurations=completed*reg['sampling_repetitions']*len(reg['randomized_events_per_cell'])*len(reg['sensitivity']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    run(parser.parse_args())
