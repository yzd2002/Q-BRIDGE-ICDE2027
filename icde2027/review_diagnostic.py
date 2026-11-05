"""All-trial retrospective stronger-source analysis after independent review."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
from scipy import sparse

from .aggregate import randomized_intervals, policy_objective
from .confidence_view import create_view, materialize_counts, QueryEngine
from .experiment import load_worlds, json_write
from .source_intervals import observation_intervals, randomized_cell_projection
from .proof_check import CheckedView

ROOT=Path(__file__).resolve().parents[1]


def evaluate(document,c,truth,mean):
    candidate=document.get('view',document.get('candidate_view'))
    # This is an evaluator-only membership test for every trial, including a
    # refused materialization. It never feeds the known mean into inference.
    try:
        CheckedView({**candidate,'feasible_point':mean.tolist()})
        set_covers=True
    except ValueError:
        set_covers=False
    if document['status']!='ready':
        return {'status':'uncertain','reason':document['reason'],'lower_bound':None,
                'set_covers':set_covers,'view_status':document['status'],'certificate':None}
    view=document['view']
    engine=QueryEngine(view,reuse=False,highs_threads=1)
    answer=engine.query(c,.001)
    answer.update(view_status='ready',set_covers=set_covers,
                  covers_target=answer['lower_bound'] is None or answer['lower_bound']<=truth,
                  view_digest=engine.digest,view=view)
    return answer


def run(args):
    if args.out.exists():
        raise SystemExit('Refusing to overwrite an existing diagnostic')
    protocol=ROOT/'icde2027/configs/review_diagnostic_v1.json'
    registration=json.loads(protocol.read_text())
    input_path=ROOT/registration['input']
    trials=[json.loads(line) for line in input_path.read_text().splitlines()]
    if len(trials)!=400 or len({(t['config_id'],t['seed']) for t in trials})!=400:
        raise ValueError('registered original primary family is incomplete')
    worlds=load_worlds(ROOT/'icde2027/data/worlds.json')
    args.out.mkdir(parents=True)
    snapshot=args.out/'source_snapshot';snapshot.mkdir()
    source_hash={}
    for path in sorted((ROOT/'icde2027').glob('*.py')):
        shutil.copy2(path,snapshot/path.name)
        source_hash[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={'kind':'retrospective_review_diagnostic','status':'running','registration':registration,
              'registration_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
              'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
              'source_sha256':source_hash,'python':platform.python_version(),
              'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    json_write(args.out/'manifest.json',manifest)
    groups=defaultdict(list)
    with (args.out/'trials.jsonl').open('w') as output:
        for i,t in enumerate(trials):
            world=worlds[t['world']];obs=np.asarray(t['observation_counts']);rand=np.asarray(t['randomized_counts'])
            rows=world['moment_rows'];c=policy_objective(t['target_weights'],t['policy'],t['reference'])
            d=len(c);truth=t['true_contrast'];mean=world['target_mean'].ravel()
            metadata={'analysis':'retrospective_review_diagnostic','source_trial':[t['config_id'],t['seed']]}
            lo,hi,details=observation_intervals(obs,t['radius'],.05)
            documents={'observation_consistency':create_view(lo,hi,sparse.csr_matrix((0,d)),[],[],metadata={**metadata,'source':'observation','intervals':details},highs_threads=1)}
            rl,rh,_=randomized_intervals(rand,rows,world['randomized_q'],.05)
            documents['randomized_group_standalone']=create_view(np.zeros(d),np.ones(d),rows,rl,rh,metadata={**metadata,'source':'randomized_group','alpha':.05},highs_threads=1)
            rl,rh,_=randomized_cell_projection(rand,rows,.05)
            documents['randomized_projected_standalone']=create_view(np.zeros(d),np.ones(d),rows,rl,rh,metadata={**metadata,'source':'randomized_projected','alpha':.05},highs_threads=1)
            for method,key in [('group_score','fusion_consistency_group'),('cell_projection','fusion_consistency_projected')]:
                documents[key]=materialize_counts(obs,rand,rows=rows,randomized_q=world['randomized_q'],radius=t['radius'],
                                                  alpha=.05,randomized_method=method,metadata=metadata,highs_threads=1)
            results={name:evaluate(document,c,truth,mean) for name,document in documents.items()}
            fallback=float(np.minimum(c,0).sum())
            baseline_names=registration['methods'][:3]
            best=max(fallback if results[m]['lower_bound'] is None else results[m]['lower_bound'] for m in baseline_names)
            row={'config_id':t['config_id'],'seed':t['seed'],'true_contrast':truth,'deterministic_lower':fallback,
                 'original_fusion_lower':t['results']['fusion']['lower_bound'],
                 'methods':results,'rescue':{m:results[m]['lower_bound'] is not None and results[m]['lower_bound']>.001 and best<=.001
                                            for m in registration['methods'][3:]}}
            output.write(json.dumps(row,allow_nan=False)+'\n');output.flush();groups[t['config_id']].append(row)
            if (i+1)%100==0: print(f'completed {i+1}/400',flush=True)
    summary={}
    for key,rows in groups.items():
        summary[key]={'n':len(rows),'methods':{}}
        for method in registration['methods']:
            values=[r['methods'][method]['lower_bound'] for r in rows]
            all_values=[r['deterministic_lower'] if v is None else v for r,v in zip(rows,values)]
            summary[key]['methods'][method]={'mean_with_fallback':float(np.mean(all_values)),
                'sd_with_fallback':float(np.std(all_values,ddof=1)),
                'certifies':sum(v is not None and v>.001 for v in values),
                'no_bound':sum(v is None for v in values),
                'set_not_covering_all_trials':sum(r['methods'][method].get('set_covers') is False for r in rows),
                'false_assertions':sum(v is not None and v>r['true_contrast'] for r,v in zip(rows,values))}
        summary[key]['rescue']={m:sum(r['rescue'][m] for r in rows) for m in registration['methods'][3:]}
    json_write(args.out/'summary.json',summary)
    manifest.update(status='complete',trials=len(trials),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    json_write(args.out/'manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
