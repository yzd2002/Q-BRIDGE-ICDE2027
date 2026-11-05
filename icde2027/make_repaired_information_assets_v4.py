"""Produce compact paper assets from independently audited repaired programs.

No statistical resampling, policy fitting, interval construction or service calls.
The redraw mode uses only the exported plotting data and Matplotlib.
"""
from collections import Counter, defaultdict
from fractions import Fraction as F
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT/'runs/icde2027/analysis/repaired_information_v4_results.json'
AUDIT = ROOT/'runs/icde2027/reviews/repaired_information_v4_result_audit.json'
PAPER = ROOT/'icde2027/paper'
METHODS = ('observation_consistency', 'randomized_projected', 'fusion_projected')
FAMILIES = ('metadata', 'randomized_heuristic', 'robust_design_selection')
ORDER = [('conservative',100),('conservative',1000),('true_boundary',100),('true_boundary',1000)]
RESULT_SHA = 'b86636ec9bafbb41d39a1b12b9bb3454d10aa19fec4aa2c2977032ad5ab54c17'
AUDIT_SHA = '791b1678d6dc583c3d4ae90c8174cbe2faada26624627c41d076cbe2c2af5efd'
CATEGORIES = ('finite_sample_certified','finite_relaxation_limit',
              'population_and_finite_relaxation_limit','true_effect_not_above_margin')


def require(ok, why):
    if not ok: raise ValueError(why)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''): h.update(part)
    return h.hexdigest()


def read(p): return json.loads(p.read_text())
def rational(p): return F(int(p[0]),int(p[1]))

def check_sources(pins):
    for rel,value in pins.items():
        p=Path(rel)
        require(not p.is_absolute() and '..' not in p.parts and (ROOT/p).resolve().is_relative_to(ROOT),'Unsafe evidence path')
        require(sha(ROOT/p)==value,'Changed evidence: '+rel)


def derive(result):
    records=result['queries']; views=result['views']
    require(result['status']=='complete' and len(records)==2880 and len(views)==960,'Incomplete analysis')
    expected={(w,r,s,b,f) for w in range(48) for r in range(5) for s,b in ORDER for f in FAMILIES}
    require({(x['world_id'],x['rep'],x['sensitivity'],x['r_events_per_cell'],x['family']) for x in records}==expected,
            'Incomplete or duplicated fixed query grid')
    bygroup=defaultdict(list)
    for r in records: bygroup[r['family'],r['sensitivity'],r['r_events_per_cell']].append(r)
    data=[]
    for family in FAMILIES:
        for label,budget in ORDER:
            rows=bygroup[family,label,budget]
            categories=dict(Counter(x['category'] for x in rows))
            counts={m:sum(x['finite'][m]['status']=='certified' for x in rows) for m in METHODS}
            extras=[x for x in rows if x['finite'][METHODS[2]]['status']=='certified' and
                    not any(x['finite'][m]['status']=='certified' for m in METHODS[:2])]
            losses=[x for x in rows if x['finite'][METHODS[2]]['status']!='certified' and
                    any(x['finite'][m]['status']=='certified' for m in METHODS[:2])]
            available=[x for x in rows if 'budget' in x]
            means={k:float(sum((rational(x['budget'][k]) for x in available),F())/len(available))
                   if available else None for k in ('achieved_gain','budget_cost','saved_net','optimal_gain')}
            group={'family':family,'sensitivity':label,'r_events_per_cell':budget,'queries':len(rows),
                   'worlds':len({x['world_id'] for x in rows}),'certified':counts,'categories':categories,
                   'fusion_extra_vs_standalone_union':len(extras),'fusion_lost_vs_standalone_union':len(losses),
                   'budget_available':len(available),'mean_budget':means,
                   'face':dict(Counter(x['face']['classification'] for x in rows if 'face' in x)),
                   'all_zero_objectives':sum(x['all_zero_objective'] for x in rows)}
            declared=next(g for g in result['summary'] if (g['family'],g['sensitivity'],g['r_events_per_cell'])==(family,label,budget))
            for k,v in group.items(): require(declared[k]==v,'Derived group disagrees: '+k)
            data.append(group)
    metadata=[g for g in data if g['family']=='metadata']
    losses=[r for r in records if r['family']=='metadata' and r['finite'][METHODS[2]]['status']!='certified' and
            any(r['finite'][m]['status']=='certified' for m in METHODS[:2])]
    loss_facts=[]
    for r in losses:
        optimal=rational(r['finite'][METHODS[2]]['optimum']); full=rational(r['budget']['full_box'])
        threshold=rational(r['threshold']); gain=rational(r['budget']['optimal_gain']); cost=rational(r['budget']['budget_cost'])
        loss_facts.append({'world_id':r['world_id'],'rep':r['rep'],'sensitivity':r['sensitivity'],
                          'r_events_per_cell':r['r_events_per_cell'],'face':r['face']['classification'],
                          'exact_optimum_fails':optimal<=threshold,'obs_box_passes':full>threshold,
                          'gain_below_budget_cost':gain<cost,'optimal_gain':float(gain),'budget_cost':float(cost),
                          'optimal_net':float(optimal-full),'saved_dual_equals_optimum':rational(r['finite'][METHODS[2]]['saved_dual'])==optimal})
    primary=bygroup['metadata','conservative',100]
    extra=[r for r in primary if r['finite'][METHODS[2]]['status']=='certified' and
           not any(r['finite'][m]['status']=='certified' for m in METHODS[:2])]
    containment=Counter((v['population_subset_finite']['contained'],v['finite_subset_population']['contained']) for v in views)
    oracle=[float(rational(r['radius'])) for r in records if r['sensitivity']=='true_boundary']
    robust_primary=bygroup['robust_design_selection','conservative',100]
    paired={(r['world_id'],r['rep'],r['r_events_per_cell'],r['sensitivity']):r['objective']
            for r in records if r['family']=='robust_design_selection'}
    robust_changes=sum(paired[w,rep,b,'conservative']!=paired[w,rep,b,'true_boundary']
                       for w in range(48) for rep in range(5) for b in (100,1000))
    return {'status':'complete','conditions':metadata,'all_groups':data,'metadata_losses':loss_facts,
            'robust_primary_zero_objectives':sum(r['all_zero_objective'] for r in robust_primary),
            'robust_radius_paired_changes':robust_changes,
            'primary_extra_worlds':len({r['world_id'] for r in extra}),
            'primary_extras_population_fusion_only':sum(r['population_comparison']['fusion_only'] for r in extra),
            'primary_extras_population_standalone':sum(r['population_comparison']['standalone_union_passes'] for r in extra),
            'primary_population_fusion_only_queries':sum(r['population_comparison']['fusion_only'] for r in primary),
            'primary_population_fusion_only_worlds':len({r['world_id'] for r in primary if r['population_comparison']['fusion_only']}),
            'oracle_radius_minmax':[min(oracle),max(oracle)],
            'strict_population_subset_finite_pairs':containment[True,False],
            'face_all':dict(Counter(r['face']['classification'] for r in records if 'face' in r)),
            'all_zero_objectives':sum(r['all_zero_objective'] for r in records),
            'old_categories_changed':sum(r['category_changed'] is True for r in records),
            'old_face_changed':sum(r['old_new'].get('face_category_changed') is True for r in records),
            'fusion_numerically_missed_certifications':sum('optimum' in r['finite'][METHODS[2]] and
                rational(r['finite'][METHODS[2]]['optimum'])>rational(r['threshold']) and
                r['finite'][METHODS[2]]['status']!='certified' for r in records),
            'unavailable_entries':len(result['unavailable'])}


def draw(data, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.2,'axes.labelsize':8.2,
                         'axes.titlesize':9.3,'pdf.fonttype':42,'ps.fonttype':42})
    conditions=data['conditions']; require(len(conditions)==4,'Four metadata conditions required')
    blue,orange,gray='#537EAE','#D78B25','#34424E'
    fig,axes=plt.subplots(1,2,figsize=(7.25,2.6),gridspec_kw={'width_ratios':[1,1.12]})
    labels=['$\\Lambda=2$\n100','$\\Lambda=2$\n1,000','Oracle $\\Lambda^*$\n100','Oracle $\\Lambda^*$\n1,000']
    width=.32; xs=list(range(4)); ax=axes[0]
    for shift,m,color,label in [(-width/2,METHODS[0],blue,'OBS, full budget'),(width/2,METHODS[2],orange,'Fusion, split budget')]:
        counts=[r['certified'][m] for r in conditions]
        ax.bar([i+shift for i in xs],[100*c/r['queries'] for c,r in zip(counts,conditions)],width,color=color,label=label)
        for i,n in enumerate(counts):
            lift=5 if m==METHODS[0] and n>100 else 1.5
            ax.text(i+shift,100*n/conditions[i]['queries']+lift,str(n),ha='center',va='bottom',fontsize=8)
    ax.set_ylim(0,111); ax.set_yticks([0,25,50,75,100]); ax.set_ylabel('Certified queries (%)')
    ax.set_title('(a) Same policies, different declarations',loc='left')
    ax.legend(loc='upper left',frameon=False,fontsize=7.5)
    require(all(r['certified'][METHODS[1]]==0 and r['queries']==240 for r in conditions),
            'Standalone R annotation needs revision for these data')
    ax.text(.99,.95,'R alone: 0 / 240',ha='right',va='top',transform=ax.transAxes,fontsize=7.5,color=gray)
    ax=axes[1]
    gains=[100*r['mean_budget']['achieved_gain'] for r in conditions]
    costs=[100*r['mean_budget']['budget_cost'] for r in conditions]
    nets=[100*r['mean_budget']['saved_net'] for r in conditions]
    ax.bar(xs,gains,.54,color=orange,label='Added-constraint gain (saved dual)')
    ax.bar(xs,[-x for x in costs],.54,color=blue,label='Confidence-budget cost')
    ax.scatter(xs,nets,s=26,marker='D',color=gray,zorder=4,label='Net bound difference')
    ax.axhline(0,color=gray,lw=.7)
    ax.set_ylim(-.26,2.35); ax.set_yticks([0,.5,1,1.5,2]); ax.set_ylabel('Mean bound difference (pp)')
    ax.set_title('(b) Added information must repay its cost',loc='left')
    ax.legend(loc='upper left',frameon=False,fontsize=7.2)
    for ax in axes:
        ax.set_xticks(xs,labels); ax.tick_params(axis='x',labelsize=8)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#DDE2E7',lw=.5);ax.set_axisbelow(True)
    fig.subplots_adjust(left=.075,right=.995,top=.87,bottom=.235,wspace=.35)
    for ext in ('pdf','png'): fig.savefig(prefix.with_suffix('.'+ext),dpi=240,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)


def prose(data):
    groups=data['conditions']; low,high,oracle,oracle_high=groups
    require(data['strict_population_subset_finite_pairs']==960 and data['unavailable_entries']==0,
            'Current prose requires all diagnosed programs available with strict containment')
    require(data['fusion_numerically_missed_certifications']==0,'Missed-certification prose needs revision')
    require(all(set(g['categories'])<=set(CATEGORIES) for g in data['all_groups']),
            'New diagnostic category needs explicit prose')
    losses=data['metadata_losses']; low_losses=[r for r in losses if r['r_events_per_cell']==100]
    high_losses=[r for r in losses if r['r_events_per_cell']==1000]
    require(len(low_losses)==2 and all(r['face']=='zero' and r['exact_optimum_fails'] and r['obs_box_passes'] for r in low_losses),
            'Loss-witness prose does not match actual low-budget records')
    require(len(high_losses)==1 and high_losses[0]['face']=='strict_positive' and
            high_losses[0]['gain_below_budget_cost'] and high_losses[0]['exact_optimum_fails'],
            'High-budget loss prose differs')
    return rf'''% Generated only after independent repaired-program result audit.
\textbf{{Finite width or population ambiguity?}}
Exact primal--dual certificates give finite and population optima for all three methods on all 2,880 query records. All 960 finite fusion views strictly contain their population counterparts. The latter replace sampling intervals by exact source quantities under the same declared relaxation; they do not assume full-cell transport or latent-model sharpness. Table~\ref{{tab:fresh}} partitions the conservative, lower-budget results. No finite optimum passes when the returned fusion answer fails. All 240 primary true effects exceed $\delta$: {low['categories']['population_and_finite_relaxation_limit']} remain blocked by population ambiguity and {low['categories']['finite_relaxation_limit']} by the current finite intervals. These categories do not estimate required sample sizes.

Of the {low['fusion_extra_vs_standalone_union']} additional finite certifications across {data['primary_extra_worlds']} worlds, {data['primary_extras_population_standalone']} queries have a sufficient population standalone source and {data['primary_extras_population_fusion_only']} pass only with population fusion. The latter regime spans {data['primary_population_fusion_only_worlds']} worlds ({data['primary_population_fusion_only_queries']} repeated queries). The robust design rule retains the reference in {data['robust_primary_zero_objectives']}/240 lower-budget cases. These are diagnoses of the saved worlds, not general frequencies of source complementarity.

\textbf{{A tighter declaration can make fusion unnecessary.}}
The oracle radii range from {data['oracle_radius_minmax'][0]:.4f} to {data['oracle_radius_minmax'][1]:.4f}. With the same metadata policies and count samples, the low-budget oracle setting certifies {oracle['certified'][METHODS[0]]}/240 from OBS and {oracle['certified'][METHODS[2]]}/240 from fusion, compared with {low['certified'][METHODS[0]]} and {low['certified'][METHODS[2]]} at $\Lambda=2$ (Fig.~\ref{{fig:sensitivity}}). The four settings add/lose {low['fusion_extra_vs_standalone_union']}/{low['fusion_lost_vs_standalone_union']}, {high['fusion_extra_vs_standalone_union']}/{high['fusion_lost_vs_standalone_union']}, {oracle['fusion_extra_vs_standalone_union']}/{oracle['fusion_lost_vs_standalone_union']} and {oracle_high['fusion_extra_vs_standalone_union']}/{oracle_high['fusion_lost_vs_standalone_union']} certifications relative to the standalone union. Truth supplies the oracle radius; the plot does not learn a defensible declaration from observed data.

\textbf{{Why the budget split can lose a certification.}}
Replacing $V_F$ by the saved exact dual $D_F$ in~\eqref{{eq:gain-cost}} gives the achieved-gain decomposition in Fig.~\ref{{fig:sensitivity}}b; $D_F$ need not be optimal. At the lower budget and oracle radius, mean achieved gain is {100*oracle['mean_budget']['achieved_gain']:.4f}~pp and budget cost {100*oracle['mean_budget']['budget_cost']:.4f}~pp, leaving {100*oracle['mean_budget']['saved_net']:.4f}~pp. These are bound differences, not policy effects. Proposition~\ref{{prop:box-face}} proves zero optimal gain for {oracle['face']['zero']}/240 oracle metadata queries versus {low['face']['zero']}/240 at $\Lambda=2$; at the larger budget the counts are {oracle_high['face']['zero']} and {high['face']['zero']}. The two low-budget certification losses have zero optimal gain: the half-budget box minimizer already satisfies every group constraint. The third loss, at the larger budget, has positive optimal gain smaller than its budget cost. Exact optima rule out missed optimization in all three.

Across all policies and settings, {data['face_all']['zero']} of 2,880 queries have zero optimal constraint gain, including {data['all_zero_objectives']} trivial zero objectives; the nonzero subset has {data['face_all']['zero']-data['all_zero_objectives']}/{2880-data['all_zero_objectives']} such cases. The randomized budgets are separate count streams. The robust policy changes in {data['robust_radius_paired_changes']} of 480 radius-paired records, so its comparison cannot isolate radius alone. Complete strata and all policy families remain in the artifact.
'''


def table(data):
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Policy & Cert. & Finite & Pop. & True $\le\delta$\\',r'\midrule']
    names=dict(zip(FAMILIES,('Metadata','R heuristic','Robust design')))
    for family in FAMILIES:
        g=next(g for g in data['all_groups'] if (g['family'],g['sensitivity'],g['r_events_per_cell'])==(family,'conservative',100))
        require(sum(g['categories'].values())==240,'Category census differs')
        lines.append(names[family]+' & '+' & '.join(str(g['categories'].get(k,0)) for k in CATEGORIES)+r'\\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])+'\n'


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--redraw-data',type=Path)
    parser.add_argument('--output-prefix',type=Path); args=parser.parse_args()
    if args.redraw_data:
        require(args.output_prefix is not None,'Redraw needs an explicit output prefix')
        draw(read(args.redraw_data),args.output_prefix); return
    require(AUDIT.is_file(),'Independent result audit required before assets')
    require(sha(AUDIT)==AUDIT_SHA,'Not the admitted complete independent result audit')
    audit=read(AUDIT)
    require(all(audit.get(k) is True for k in ('complete','passed','independent_review')),'Incomplete independent audit')
    require(audit['production_checker_imported_or_used'] is False and
            audit['production_solver_imported_or_used'] is False,'Audit independence differs')
    for key,count in (('queries_checked',2880),('pairs_checked',960),('programs_checked',3168),
                      ('certificates_checked',11640),('saved_dual_records_checked',8640),
                      ('old_classification_records_checked',720),('old_classification_not_computed',2160)):
        require(type(audit[key]) is int and audit[key]==count,'Incomplete audit: '+key)
    require(audit['input_results_sha256']==sha(RESULT)==RESULT_SHA,'Result identity differs')
    check_sources(audit['sources_sha256'])
    result=read(RESULT); check_sources(result['sources_sha256']); data=derive(result)
    groups={(g['family'],g['sensitivity'],g['r_events_per_cell']):g for g in audit['summary']}
    require(len(groups)==12,'Incomplete independently recomputed groups')
    for group in data['all_groups']:
        checked=groups[group['family'],group['sensitivity'],group['r_events_per_cell']]
        require(all(checked[k]==v for k,v in group.items()),'Independent summary differs')
    data['sources_sha256']={str(RESULT.relative_to(ROOT)):sha(RESULT),str(AUDIT.relative_to(ROOT)):sha(AUDIT)}
    data['generator_sha256']=sha(Path(__file__))
    data['scope']='Retrospective repaired-program analysis: 48 worlds, five count samples each. Oracle radius uses truth. No new confirmation.'
    text=prose(data); tex_table=table(data)
    (PAPER/'repaired_information_facts_v4.json').write_text(json.dumps(data,indent=2)+'\n')
    (PAPER/'repaired_information_results_v4.tex').write_text(text)
    (PAPER/'repaired_information_categories_v4.tex').write_text(tex_table)
    draw(data,PAPER/'figures/repaired_information_v4')
    print(json.dumps({'status':'complete','conditions':len(data['conditions']),'all_groups':len(data['all_groups']),
                      'result_sha256':RESULT_SHA,'losses':data['metadata_losses']}))

if __name__=='__main__': main()
