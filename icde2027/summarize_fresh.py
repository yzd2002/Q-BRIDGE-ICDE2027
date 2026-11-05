"""Registered world-unit analysis, never treats within-world queries as iid."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
from .experiment import json_write
from .fresh_worlds import world_specs


def decision(record,method,policy):
    return record['methods'][method]['queries'][policy]['status']=='certified'


def primary_analysis(worlds,reg):
    strata=defaultdict(list);world_records=[]
    seen_ids=set()
    for bundle in worlds:
        spec=bundle['world']['spec']
        if spec['id'] in seen_ids:
            raise ValueError('Duplicate world ID is not an independent world')
        seen_ids.add(spec['id'])
        if spec['unconfounded']:
            continue
        records=[r for r in bundle['records'] if r['sensitivity']=='conservative' and r['r_events_per_cell']==100]
        if len(records)!=5 or {r['rep'] for r in records}!=set(range(5)):
            raise ValueError('Incomplete registered primary sampling repetitions')
        diffs=[];extra=[];loss=[];allsource=[]
        for r in records:
            f=decision(r,'fusion_projected','metadata')
            o=decision(r,'observation_consistency','metadata')
            p=decision(r,'randomized_projected','metadata')
            g=decision(r,'randomized_group_score','metadata')
            diffs.append(int(f)-int(o or p))
            extra.append(int(f and not (o or p)));loss.append(int((o or p) and not f))
            allsource.append(int(f)-int(o or p or g))
        rate=float(np.mean(diffs));strata[spec['stratum']].append(rate)
        world_records.append({'world_id':spec['id'],'stratum':spec['stratum'],'paired_rate':rate,
            'sampling_differences':diffs,'extra':sum(extra),'lost':sum(loss),
            'descriptive_all_three_single_sources_rate':float(np.mean(allsource))})
    if len(strata)!=12 or len(world_records)!=48 or any(len(v)!=4 for v in strata.values()):
        raise ValueError('Primary analysis must contain all 48 registered worlds in 12 strata')
    estimate=float(np.mean([r['paired_rate'] for r in world_records]))
    contribution=np.array([np.var(v,ddof=1)/(12**2*4) for v in strata.values()])
    variance=float(contribution.sum());se=math.sqrt(variance)
    if variance>0:
        df=variance**2/float(np.sum(contribution**2/3))
        halfwidth=float(student_t.ppf(.975,df))*se
        approximate=[max(-1.,estimate-halfwidth),min(1.,estimate+halfwidth)]
    else:
        df=None;approximate=None
    deduction=math.sqrt(2*math.log(20)/48)
    lower=max(-1.,estimate-deduction)
    return {'independent_worlds':48,'sampling_repetitions_per_world':5,'estimate':estimate,
            'one_sided_95pct_hoeffding_lower':lower,'hoeffding_deduction':deduction,
            'effect_target':.05,'effect_target_supported':lower>.05,
            'approximate_stratified_t_95pct_interval':approximate,'approximate_t_df':df,
            'stratified_standard_error':se,'worlds':world_records,
            'interpretation':'Only the Hoeffding lower endpoint governs the registered decision; t interval is descriptive and approximate.'}


def validate_census(worlds,reg):
    expected={s['id']:s for s in world_specs(reg)}
    ids=[bundle['world']['spec']['id'] for bundle in worlds]
    if len(ids)!=len(expected) or len(set(ids))!=len(ids) or set(ids)!=set(expected):
        raise ValueError('World IDs do not equal the registered independent-world census')
    expected_keys={(rep,sensitivity,budget) for rep in range(reg['sampling_repetitions'])
                   for sensitivity in reg['sensitivity'] for budget in reg['randomized_events_per_cell']}
    for bundle in worlds:
        spec=bundle['world']['spec']
        if spec!=expected[spec['id']]:
            raise ValueError('World specification differs from registration')
        keys=[(r['rep'],r['sensitivity'],r['r_events_per_cell']) for r in bundle['records']]
        if len(keys)!=len(expected_keys) or set(keys)!=expected_keys:
            raise ValueError('World configurations differ from registered Cartesian product')
        if any(r['world_id']!=spec['id'] or r['stratum']!=spec['stratum'] or r['unconfounded']!=spec['unconfounded'] for r in bundle['records']):
            raise ValueError('Configuration bound to incorrect world')


def summarize(path):
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest['status']!='complete':
        raise ValueError('Refusing final analysis of an incomplete study')
    reg=json.loads((path/'protocol.json').read_text())
    worlds=[json.loads(p.read_text()) for p in sorted((path/'worlds').glob('world_*.json'))]
    if len(worlds)!=60:
        raise ValueError('All 60 generated worlds must be retained')
    validate_census(worlds,reg)
    primary=primary_analysis(worlds,reg)
    cells=defaultdict(list);coverage=defaultdict(lambda:defaultdict(int));selections=[]
    for bundle in worlds:
        spec=bundle['world']['spec']
        design_seen=set()
        if len(bundle['records'])!=20:
            raise ValueError('Missing registered configuration')
        for record in bundle['records']:
            design_key=(record['rep'],record['sensitivity'])
            if design_key not in design_seen:
                design_seen.add(design_key)
                selections.append({'world':spec['id'],'rep':record['rep'],'sensitivity':record['sensitivity'],
                                   'reference_selected':record['design_selection']['reference_selected'],
                                   'set_contains_truth':record['design_selection']['set_contains_truth'],
                                   'design_view_status':record['design_selection']['document']['status']})
            for method,result in record['methods'].items():
                counts=coverage[method];counts['views']+=1
                counts['whole_set_misses']+=not result['set_contains_truth']
                counts['refused_views']+=result['document']['status']!='ready'
                for policy,answer in result['queries'].items():
                    counts['queries']+=1;counts['exceeds_truth']+=answer['exceeds_truth'];counts['false_assertions']+=answer['false_assertion']
                    counts['uncertain_queries']+=answer['status']=='uncertain'
                    key=(spec['stratum'],record['r_events_per_cell'],record['sensitivity'],policy,method)
                    cells[key].append({'world':spec['id'],'rep':record['rep'],'certifies':answer['status']=='certified',
                                       'lower_bound':answer['lower_bound'],'true_contrast':answer['true_contrast']})
    conditions=[]
    for key,rows in sorted(cells.items()):
        byworld=defaultdict(list)
        for row in rows:
            byworld[row['world']].append(int(row['certifies']))
        world_rates=[float(np.mean(v)) for v in byworld.values()]
        bounds=[r['lower_bound'] for r in rows if r['lower_bound'] is not None]
        conditions.append({'stratum':key[0],'r_events_per_cell':key[1],'sensitivity':key[2],'policy':key[3],
            'method':key[4],'worlds':len(byworld),'sample_configurations':len(rows),'certifies':sum(r['certifies'] for r in rows),
            'mean_world_certification_rate':float(np.mean(world_rates)),
            'sd_between_world_certification_rates':float(np.std(world_rates,ddof=1)) if len(world_rates)>1 else None,
            'mean_lower_given_available':float(np.mean(bounds)) if bounds else None,
            'mean_true_contrast':float(np.mean([r['true_contrast'] for r in rows])),
            'no_bound':len(rows)-len(bounds)})
    summary={'primary':primary,'coverage_diagnostics':dict(coverage),'policy_selection':selections,
             'design_coverage':{'views':len(selections),'whole_set_misses':sum(not s['set_contains_truth'] for s in selections),
                                'refused_views':sum(s['design_view_status']!='ready' for s in selections)},'all_conditions':conditions,
             'coverage_scope':'Dependent configurations and queries; these raw counts are diagnostics, not iid binomial coverage estimates.'}
    json_write(path/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    result=summarize(parser.parse_args().run)
    print(json.dumps({k:v for k,v in result['primary'].items() if k!='worlds'},indent=2))
