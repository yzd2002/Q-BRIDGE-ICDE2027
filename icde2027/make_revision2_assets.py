"""Original vector figures and evidence-linked tables for the revised paper."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT/'icde2027/paper';FIG=PAPER/'figures'
BLUE='#0072B2';GREEN='#008B72';ORANGE='#D58400';INK='#25313B';PURPLE='#7655A5'


def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',pad_inches=.025)
    fig.savefig(FIG/f'{name}.png',dpi=200,bbox_inches='tight',pad_inches=.025)
    plt.close(fig)


def worked_example():
    data=json.loads((ROOT/'runs/icde2027/exemplars/revision2_four_action_check.json').read_text())
    fig=plt.figure(figsize=(7.1,2.65));left=fig.add_axes([.015,.13,.43,.79]);right=fig.add_axes([.49,.13,.5,.79])
    for ax in [left,right]: ax.axis('off')
    left.text(0,1.06,'(a) One fixed view, four action means',weight='bold',fontsize=9,color=INK)
    cells=[[action,f'[{lo:.2f}, {hi:.2f}]',f'{co:+.2f}'] for action,lo,hi,co in zip('ABCD',data['view_lo'],data['view_hi'],data['queries'][0]['objective'])]
    table=left.table(cellText=cells,colLabels=['Action','Allowed mean','Q1 change'],cellLoc='center',bbox=[0,.27,1,.7],colWidths=[.2,.48,.32])
    table.auto_set_font_size(False);table.set_fontsize(8.5)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#D8E0E5');cell.set_linewidth(.5)
        if row==0:cell.set_facecolor('#EDF3F7');cell.set_text_props(weight='bold',color=INK)
    left.text(.02,.14,'Group means:  (A + C)/2 = 0.65',fontsize=8.5,color=BLUE)
    left.text(.30,.015,'(B + D)/2 = 0.70',fontsize=8.5,color=BLUE)
    right.text(0,1.06,'(b) New query, checked reuse or fallback',weight='bold',fontsize=9,color=INK)
    y=[.81,.475,.14];routes=['Solve once','Reuse','Solve after failed reuse']
    for index,(query,yy,route) in enumerate(zip(data['queries'],y,routes)):
        right.add_patch(FancyBboxPatch((0,yy-.145),1,.29,boxstyle='round,pad=.012,rounding_size=.015',
                        edgecolor='#CFDAD9',facecolor='#F1F7F5',lw=.6))
        right.text(.025,yy+.088,query['query'],weight='bold',fontsize=8.8,color=INK)
        objective=', '.join(f'{x:+.3f}'.rstrip('0').rstrip('.') for x in query['objective'])
        right.text(.14,yy+.088,f'c = ({objective})',fontsize=7.9,color=INK)
        candidates=query['reuse_candidate_bounds']
        if candidates:
            bound=max(candidates);relation='>' if bound>data['margin'] else '<'
            comparison=f'Cached: {100*bound:+.3f} pp {relation} {100*data["margin"]:+.3f} pp threshold'
        else:
            comparison='No saved candidate yet'
        right.text(.025,yy-.002,comparison,fontsize=7.9,color=INK)
        right.text(.025,yy-.091,route,fontsize=8.1,color=GREEN)
        right.text(.975,yy-.091,f'{100*query["returned_bound"]:+.3f} pp',ha='right',fontsize=9,weight='bold',color=GREEN)
    fig.text(.5,.012,'Illustrative program • pp = percentage points • all displayed bounds checked by the implementation',ha='center',fontsize=7.7,color=INK)
    save(fig,'worked_query_v2')


def flow_diagram():
    fig,ax=plt.subplots(figsize=(7.1,2.65));ax.set_xlim(0,10);ax.set_ylim(0,3.6);ax.axis('off')
    def box(x,y,w,h,title,body,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025,rounding_size=.06',facecolor='white',edgecolor=color,lw=1.05))
        ax.text(x+w/2,y+h-.20,title,ha='center',va='center',fontsize=8.8,weight='bold',color=color)
        ax.text(x+w/2,y+.29,body,ha='center',va='center',fontsize=7.7,color=INK,linespacing=1.25)
    def arrow(a,b,label=None):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','lw':.85,'color':'#56656E','shrinkA':2,'shrinkB':2})
        if label:
            vertical=abs(a[0]-b[0])<.01
            ax.text((a[0]+b[0])/2+(.13 if vertical else 0),(a[1]+b[1])/2+(0 if vertical else .10),label,
                    ha='left' if vertical else 'center',fontsize=7.4,color=INK)
    box(.05,2.0,1.75,1.2,'Input logs','Observational O\nRandomized R',BLUE)
    box(2.18,2.0,1.95,1.2,'Build view','Counts + intervals\nFeasibility check',BLUE)
    box(4.58,2.0,2.10,1.2,'Try reuse','New objective\nCheck saved terms',GREEN)
    ax.add_patch(FancyBboxPatch((7.21,.67),2.68,2.53,boxstyle='round,pad=.025,rounding_size=.06',facecolor='white',edgecolor=GREEN,lw=1.05))
    ax.text(8.55,2.98,'Return status',ha='center',fontsize=8.8,weight='bold',color=GREEN)
    ax.text(7.40,2.52,'Bound > margin',fontsize=8.0,color=INK)
    ax.text(7.40,2.27,'Certified + proof',fontsize=8.2,color=GREEN,weight='bold')
    ax.text(7.40,1.88,'Bound ≤ margin',fontsize=8.0,color=INK)
    ax.text(7.40,1.63,'Unresolved + proof',fontsize=8.2,color=INK,weight='bold')
    ax.text(7.40,1.22,'No verified bound',fontsize=8.0,color=INK)
    ax.text(7.40,.97,'Uncertain',fontsize=8.2,color=PURPLE,weight='bold')
    arrow((1.8,2.6),(2.18,2.6));arrow((4.13,2.6),(4.58,2.6));arrow((6.68,2.6),(7.21,2.6),'yes')
    box(4.58,.22,2.10,1.20,'Optimize','Component solves\nExact global check',PURPLE)
    arrow((5.63,2.0),(5.63,1.42),'not enough')
    arrow((6.68,.82),(7.21,.82))
    box(2.18,.22,1.95,1.20,'Saved duals','Bounded bank\nfor later queries',PURPLE)
    arrow((4.58,.82),(4.13,.82))
    ax.text(4.35,1.00,'save',ha='center',fontsize=6.7,color=INK)
    arrow((4.13,1.42),(4.58,2.13))
    ax.text(.05,.72,'Changed data or\nassumptions require\na new view.',fontsize=8,color=INK,va='center',linespacing=1.5)
    ax.text(.05,3.43,'Build once',fontsize=8,weight='bold',color=BLUE)
    ax.text(4.58,3.43,'Lower-bound execution per query',fontsize=8,weight='bold',color=GREEN)
    save(fig,'execution_flow_v2')


def evidence_tables():
    diagnostic=json.loads((ROOT/'runs/icde2027/stronger_baseline_diagnostic_v1/summary.json').read_text())
    names={'controlled_2x2_external':'2$\\times$2, external','controlled_2x2_learned':'2$\\times$2, learned',
           'kuairec_8x8_external':'8$\\times$8, external','kuairec_8x8_learned':'8$\\times$8, learned'}
    methods=['observation_consistency','randomized_group_standalone','randomized_projected_standalone','fusion_consistency_group','fusion_consistency_projected']
    lines=[r'\begin{tabular}{lrrrrrrr}',r'\toprule',r'Query family & OBS & R-group & R-proj. & F-group & F-proj. & Extra group & Extra proj.\\',r'\midrule']
    for key,label in names.items():
        d=diagnostic[key]
        cells=[str(d['methods'][m]['certifies']) for m in methods]
        extra=[str(d['rescue'][m]) for m in ['fusion_consistency_group','fusion_consistency_projected']]
        lines.append(label+' & '+' & '.join(cells+extra)+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (PAPER/'strong_baseline_table_v2.tex').write_text('\n'.join(lines)+'\n')
    fresh=json.loads((ROOT/'runs/icde2027/fresh_worlds_v1/summary.json').read_text())
    cov=fresh['coverage_diagnostics'];labels={'observation_consistency':'OBS','randomized_projected':'R projected','fusion_projected':'Fused projected','randomized_group_score':'R group score','fusion_group_score':'Fused group score'}
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Construction & Set misses & Query violations & Refusals\\',r'\midrule']
    for key,label in labels.items():
        d=cov[key];lines.append(f'{label} & {d["whole_set_misses"]}/{d["views"]} & {d["exceeds_truth"]}/{d["queries"]} & {d["refused_views"]}/{d["views"]}'+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(PAPER/'coverage_table_v2.tex').write_text('\n'.join(lines)+'\n')
    counts=Counter()
    for row in fresh['all_conditions']:
        if 'unconfounded' not in row['stratum'] and row['sensitivity']=='conservative':
            counts[(row['r_events_per_cell'],row['policy'],row['method'])]+=row['certifies']
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.85))
    fig.subplots_adjust(bottom=.27,top=.76,wspace=.24,left=.08,right=.99)
    policies=['metadata','randomized_heuristic','robust_design_selection'];policy_names=['Metadata\npolicy','R-trained\nheuristic','Robust design\nselection']
    comparisons=[('observation_consistency','OBS',BLUE),('randomized_projected','R projected',ORANGE),('fusion_projected','Fused projected',GREEN)]
    for ax,budget in zip(axes,[100,1000]):
        x=np.arange(3);width=.23
        if budget==100:
            ax.axvspan(-.42,.42,color='#EEF5F2',zorder=0)
        for j,(method,label,color) in enumerate(comparisons):
            values=[counts[(budget,p,method)] for p in policies]
            bars=ax.bar(x+(j-1)*width,values,width,color=color,label=label,edgecolor='white',linewidth=.5)
            for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+.35,str(value),ha='center',fontsize=8,color=INK)
        ax.set_xticks(x,policy_names,fontsize=7.5);ax.set_ylim(0,20)
        ax.set_ylabel('Certified queries / 240',fontsize=8.5)
        ax.set_title(f'R traffic = {100*budget/2000:g}% of O traffic',fontsize=9,weight='bold')
        ax.grid(axis='y',alpha=.23);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,fontsize=8)
    extra=counts[(100,'metadata','fusion_projected')]-counts[(100,'metadata','observation_consistency')]
    assert counts[(100,'metadata','randomized_projected')]==0, 'Recompute the standalone union from paired records before changing this annotation.'
    primary_fused=counts[(100,'metadata','fusion_projected')]
    primary_either=counts[(100,'metadata','observation_consistency')]
    fig.text(.5,.035,f'Primary (shaded): fusion {primary_fused}/240 vs either standalone {primary_either}/240; +{100*extra/240:.1f} pp; target not confirmed.',
             ha='center',fontsize=7.8,color=INK)
    save(fig,'fresh_certification_v2')
    payload={'source_paths':['runs/icde2027/stronger_baseline_diagnostic_v1/summary.json','runs/icde2027/fresh_worlds_v1/summary.json'],
             'fresh_counts':[{'r_per_cell':k[0],'policy':k[1],'method':k[2],'certified':v,'denominator':240} for k,v in sorted(counts.items())],
             'primary':{k:v for k,v in fresh['primary'].items() if k!='worlds'}}
    (PAPER/'figure_data_v2.json').write_text(json.dumps(payload,indent=2)+'\n')


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'grid.linewidth':.4})
    worked_example();flow_diagram();evidence_tables()


if __name__=='__main__':main()
