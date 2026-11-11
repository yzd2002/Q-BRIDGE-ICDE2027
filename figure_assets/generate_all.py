"""Original vector graphics from unchanged saved measurements; no experiments."""
from pathlib import Path
from fractions import Fraction
import json, hashlib, math, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, PathPatch, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from matplotlib.text import Text
HERE=Path(__file__).resolve().parent
INK='#203449'; MUTED='#5E6974'; BLUE='#396EA8'; CORAL='#CF8233'; PURPLE='#52677B'; TEAL='#168A86'; GRAY='#8A9299'; RULE='#D5DEE5'
BP='#EFF4FA'; CP='#FCF3E7'; PP='#F4F6F8'; TP='#EDF7F5'
ROUNDS=['#B6D6D4','#55A5A0','#176B69','#CF8233','#68727A'];MARKERS=['o','s','^','D','v']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'text.color':INK,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':INK,'axes.edgecolor':RULE,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'dejavusans','axes.linewidth':.6})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
PINS={'factorial_plot_data.json': '0c3ff752f4b73f393b46322fec56e223142da22777fd5e42d00af1bbdaefd38a', 'family_plot_data.json': 'c4abb91246c98f719ba4b5c4b27516321410affe9f6f5c4732aabd1f6f769a7f', 'family_certificate.json': 'de63e22c4dc7ceb3d17a60a5816d2c64f0fdcb875c0a02ca17181979c652be9b', 'query_example.json': 'ecd00a506185b94d286aa556de73132a40490974cc2ddcda044f48c92b3a25cb'}
def read(n):
 p=HERE/'inputs'/n
 assert sha(p)==PINS[n], ('Input differs from saved measurement',n)
 return json.loads(p.read_text())
def save(fig,out,name,records):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[]
 for t in fig.findobj(match=Text):
  if not t.get_visible() or not t.get_text().strip():continue
  b=t.get_window_extent(renderer)
  assert b.x0>=-.6 and b.y0>=-.6 and b.x1<=fig.bbox.x1+.6 and b.y1<=fig.bbox.y1+.6,(name,t.get_text(),b,fig.bbox)
  bounds.append({'text':t.get_text(),'pt':t.get_fontsize(),'bbox':list(b.bounds)})
 for ext in ['pdf','svg','png']:
  kwargs={'dpi':360} if ext=='png' else {}
  if ext=='pdf':kwargs['metadata']={'CreationDate':None,'ModDate':None,'Creator':'Q-BRIDGE vector figure from saved inputs'}
  if ext=='svg':kwargs['metadata']={'Date':None}
  fig.savefig(out/(name+'.'+ext),**kwargs)
 records['text_bounds']=bounds;records['size_inches']=list(fig.get_size_inches());records['min_font_pt']=min(b['pt'] for b in bounds)
 plt.close(fig);return records

def main_figure(out):
 data=read('query_example.json');family=read('family_certificate.json')['certificate'];gain=100*float(Fraction(family['capacity']))
 assert family['policy']==[['0/1','0/1','1/1','0/1']]
 assert family['policy_dual_lower_bound']==family['common_point_upper_bound']==family['capacity']
 W,H=516,242;fig=plt.figure(figsize=(W/72,H/72),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
 def text(x,y,s,fs=8,color=INK,bold=False,ha='left',**kw):return ax.text(x,y,s,fontsize=fs,color=color,fontweight='bold' if bold else 'normal',ha=ha,va='center',**kw)
 def line(x1,y1,x2,y2,c=RULE,lw=.7,**kw):ax.plot([x1,x2],[y1,y2],c=c,lw=lw,solid_capstyle='round',**kw)
 def box(x,y,w,h,fc='white',ec=RULE):ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=3',facecolor=fc,edgecolor=ec,lw=.65))
 def arrow(x1,y1,x2,y2,c=PURPLE):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=8,color=c,lw=1.05,shrinkA=0,shrinkB=0))
 text(0,9,'From uncertain outcomes to a verifiable policy decision',10.2,bold=True)
 text(0,25,'Four jackets A–D  ·  Success = high rating  ·  Reference = 25% per jacket',8,MUTED)
 # Source constraints. Explicit values replace tiny axes and ornamental icons.
 box(0,40,108,94,BP);text(8,52,'Observational logs',8.1,BLUE,True)
 text(8,66,'Success-rate bounds',7.7,MUTED)
 coords=[]
 for i,(lo,hi) in enumerate(zip(data['view_lo'],data['view_hi'])):
  y=81+14*i;text(10,y,'ABCD'[i],8.4,BLUE,True);text(98,y,f'{lo:.2f} – {hi:.2f}',8.4,ha='right')
  coords.append({'action':'ABCD'[i],'lo':lo,'hi':hi})
 box(0,144,108,75,CP);text(8,156,'Randomized data',8.1,CORAL,True)
 text(8,170,'Transferred averages',7.7,MUTED)
 text(54,187,r'$(m_A+m_C)/2=.65$',8.2,ha='center')
 text(54,205,r'$(m_B+m_D)/2=.70$',8.2,ha='center')
 line(109,88,115,88,PURPLE,1);line(109,180,115,180,PURPLE,1);line(115,88,115,180,PURPLE,1);arrow(115,130,122,130)
 # Actual feasible examples, never a false Cartesian-product picture.
 box(125,40,139,179,PP);text(134,52,'One shared view',8.7,INK,True)
 text(134,69,'Keep all rates that satisfy',7.8,MUTED);text(134,81,'both sources together.',7.8,MUTED)
 text(134,100,'Example allowed scenarios',7.6,INK,True)
 xs=[143,175,207,239]
 for x,label in zip(xs,'ABCD'):text(x,115,label,8.1,BLUE,True,ha='center')
 scenarios=[data['candidate_feasible_mean_vector'],data['fused_worst_case_witness']]
 for k,vals in enumerate(scenarios):
  exact=[Fraction(str(v)) for v in vals]
  assert all(Fraction(str(lo))<=v<=Fraction(str(hi)) for lo,v,hi in zip(data['view_lo'],exact,data['view_hi']))
  assert (exact[0]+exact[2])/2==Fraction('0.65') and (exact[1]+exact[3])/2==Fraction('0.70')
  if k==1:box(132,137,125,17,'#E1EAF3','none')
  for x,v in zip(xs,vals):text(x,129+k*17,f'{v:.2f}',8.3,ha='center')
 line(134,163,255,163)
 text(134,177,'Together these imply',7.7,MUTED)
 text(134,193,r'$m_C\geq 0.70$',9.8,TEAL,True)
 text(134,209,'Reference mean = 0.675',7.8,INK)
 line(265,130,274,130,PURPLE,1);line(274,84,274,178,PURPLE,1);arrow(274,84,283,84);arrow(274,178,283,178)
 # Fixed policies and whole-family optimization branch independently.
 box(286,40,230,87,'white');text(294,52,'Check a proposed policy',8.7,INK,True)
 text(294,67,'Lower bounds; target > 0.100 pp (IV-D)',7.6,MUTED)
 for i,q in enumerate(data['queries']):
  y=84+16*i;text(294,y,q['query'],8.1,bold=True)
  text(319,y,['Solve + check','Reuse + check','New solve + check'][i],8.0,TEAL if i==1 else INK)
  text(507,y,f'{100*q["returned_bound"]:+.3f} pp',8.6,TEAL,True,ha='right')
 box(286,139,230,80,TP,'#C3DDDA');text(294,152,'Find the best guaranteed gain',8.6,INK,True)
 text(344,170,'Always C guarantees',7.8,ha='center');text(462,170,'No larger guarantee',7.8,ha='center')
 text(344,190,f'{gain:.3f} pp',13.4,TEAL,True,ha='center');text(403,190,'=',13,TEAL,True,ha='center');text(462,190,f'{gain:.3f} pp',13.4,TEAL,True,ha='center')
 text(401,210,'Matching checked bounds prove optimality',7.6,TEAL,ha='center')
 text(3,231,'Constructed example · pp = percentage points',7.5,MUTED)
 text(513,231,'Every answer carries an independently checked proof',7.5,MUTED,ha='right')
 return save(fig,out,'qbridge_overview',{'intervals':coords,'queries':[{k:q[k] for k in ['query','returned_bound','executed_route']} for q in data['queries']],'family_bound_pp':gain,'feasible_scenarios':scenarios,'scenario_check':'Exact decimal fractions satisfy individual and group constraints; illustrative, not exhaustive.','view_forks_independently':True})

def base_axes(ax):
 for s in ['top','right','left']:ax.spines[s].set_visible(False)
 ax.spines['bottom'].set_color(RULE);ax.tick_params(axis='y',length=0,pad=5);ax.tick_params(axis='x',length=2.5,width=.5)
 ax.xaxis.set_minor_locator(NullLocator());ax.set_axisbelow(True)

def family_figure(out):
 d=read('family_plot_data.json');W,H=516,187;fig=plt.figure(figsize=(W/72,H/72),facecolor='white')
 # Lead with the scientific result; use a count-scaled bar plus direct labels.
 fig.text(.005,.96,'(a) Search beyond the initial policies',fontsize=9.5,weight='bold',va='top')
 fig.text(.005,.842,'107 unresolved views → 107 decisions',fontsize=8.2,color=MUTED)
 ax=fig.add_axes([.005,.48,.425,.22]);ax.set(xlim=(0,107),ylim=(0,1));ax.axis('off')
 vals=[p['count'] for p in d['recovery_categories']];assert vals==[0,86,21,0]
 for left,n,color,txtcol in [(0,86,TEAL,'white'),(86,21,CORAL,'white')]:
  ax.barh(.5,n,left=left,height=.85,color=color,edgecolor='white',linewidth=.8)
  ax.text(left+n/2,.5,str(n),fontsize=20,weight='bold',color=txtcol,ha='center',va='center')
 fig.text(.005,.765,'Same evidence; unchanged target',fontsize=8.2,color=MUTED)
 fig.text(.005,.405,'86 views: a different policy meets the target',fontsize=8,color=TEAL)
 fig.text(.005,.31,'21 views: no policy can guarantee the target',fontsize=8,color=CORAL)
 fig.add_artist(Line2D([.005,.43],[.255,.255],transform=fig.transFigure,color=RULE,lw=.7))
 fig.text(.005,.19,'Original policies: 0 / 107 qualify after exact solve',fontsize=7.6,color=MUTED)
 fig.text(.005,.095,'Incomplete recovery decompositions: 0',fontsize=7.6,color=MUTED)
 fig.text(.005,.012,'Target = worst-case gain > 0.1 percentage points',fontsize=7.6,color=MUTED)
 # The comparison preserves every x value; round lanes reduce overplotting.
 fig.text(.49,.96,'(b) Cost of a checked answer',fontsize=9.5,weight='bold',va='top')
 fig.text(.49,.842,'78 requests × 3 rounds = 234 paired timings',fontsize=8.2,color=MUTED)
 ax2=fig.add_axes([.56,.24,.425,.49]);ax2.set_xscale('log');ax2.set_xlim(d['geometry']['x_limits']);ax2.set_ylim(2.5,-.5)
 ax2.axhspan(.5,1.5,color='#F4F6F8',zorder=0);ax2.grid(axis='x',color=RULE,lw=.5)
 ax2.axvline(1,color=PURPLE,lw=.8,ls=(0,(3,2)));plotted=[]
 for rnd in [1,2,3]:
  ps=[p for p in d['points'] if p['round']==rnd]
  artist=ax2.scatter([p['plot_x'] for p in ps],[p['plot_y'] for p in ps],s=13,color=ROUNDS[rnd-1],marker=MARKERS[rnd-1],alpha=.95,edgecolors='white',lw=.28,zorder=3)
  for p,xy in zip(ps,artist.get_offsets()):assert float(xy[0])==p['plot_x'] and float(xy[1])==p['plot_y']
  plotted+=ps
 ax2.set_yticks([0,1,2],['8','64','256'],fontsize=8.6);ax2.set_ylabel('Context–action pairs',fontsize=7.6,labelpad=4)
 ax2.xaxis.set_major_locator(FixedLocator([.5,1,2]));ax2.xaxis.set_major_formatter(FixedFormatter(['0.5','1','2']));base_axes(ax2)
 fig.text(.56,.765,'← Partition faster',fontsize=7.5,color=TEAL)
 fig.text(.985,.765,'LP faster →',fontsize=7.5,color=MUTED,ha='right')
 ax2.set_xlabel('Partition / exact LP time (log scale)',fontsize=8.1,labelpad=4)
 handles=[Line2D([],[],marker=MARKERS[i],color=ROUNDS[i],linestyle='none',markersize=4,label=f'Round {i+1}') for i in range(3)]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.765,-.013),ncol=3,frameon=False,fontsize=7.7,handletextpad=.3,columnspacing=1.)
 assert len(plotted)==234
 return save(fig,out,'verified_family_v7_main',{'points':plotted,'recovery_categories':d['recovery_categories'],'failure_denominators':d['failure_denominators'],'all_points_exact':True,'count_bar_segments':[86,21],'zero_categories_displayed_as_text':True,'panel_order':['family_recovery','checked_cost']})

def factorial_figure(out):
 d=read('factorial_plot_data.json');W,H=516,193;fig=plt.figure(figsize=(W/72,H/72),facecolor='white')
 # Table-like row labels remove repetition; shared axes make costs comparable.
 groups=[('original_main','(a) Primary streams','3 inputs · 1,000 queries each'),('threshold_engine','(b) Geometry streams','54 inputs · 100 queries each'),('source_derived','(c) Generated-log streams','6 inputs · 100 queries each')]
 vals=[r['geometric_mean'] for r in d['records']];lo,hi=min(vals),max(vals);pad=.065*(math.log(hi)-math.log(lo));limits=(math.exp(math.log(lo)-pad),math.exp(math.log(hi)+pad));plotted=[]
 lefts=[.285,.525,.765];width=.215;bottom=.24;height=.49
 fig.text(.007,.94,'Feature tested',fontsize=8.1,weight='bold');fig.text(.177,.94,'Other',fontsize=8.1,weight='bold',ha='center')
 fig.text(.177,.878,'feature',fontsize=7.6,color=MUTED,ha='center')
 fig.text(.177,.824,'state',fontsize=7.6,color=MUTED,ha='center')
 # Data rows map the original y=0..3 to the same screen positions.
 def fy(y):return bottom+height*(3.55-y)/4.1
 for y in [0,1,2,3]:
  c=CORAL if y<2 else TEAL
  fig.text(.177,fy(y),'off' if y%2==0 else 'on',fontsize=8.3,ha='center',va='center',color=c)
 for mid,label,desc,c in [(.5,'Screening','defer candidates',CORAL),(2.5,'Retention','save exact terms',TEAL)]:
  fig.text(.007,fy(mid)+.022,label,fontsize=8.8,weight='bold',color=c,va='center')
  fig.text(.007,fy(mid)-.04,desc,fontsize=7.5,color=MUTED,va='center')
 for y in [-.52,1.5,3.52]:fig.add_artist(Line2D([.005,.98],[fy(y),fy(y)],transform=fig.transFigure,color=RULE,lw=.55,zorder=0))
 for left,(group,title,desc) in zip(lefts,groups):
  ax=fig.add_axes([left,bottom,width,height]);ax.set_xscale('log');ax.set_xlim(limits);ax.set_ylim(3.55,-.55)
  ax.set_facecolor('none');ax.grid(axis='x',color=RULE,lw=.4);ax.axvline(1,color=PURPLE,ls=(0,(3,2)),lw=.85)
  fig.text(left+width/2,.94,title,fontsize=8.2,weight='bold',ha='center')
  fig.text(left+width/2,.879,desc,fontsize=7.5,color=MUTED,ha='center')
  fig.text(left+.005,.785,'← Faster',fontsize=7.5,color=TEAL)
  fig.text(left+width-.005,.785,'Slower →',fontsize=7.5,color=MUTED,ha='right')
  ax.set_yticks([]);ax.xaxis.set_major_locator(FixedLocator([.8,1,1.2]));ax.xaxis.set_major_formatter(FixedFormatter(['0.8','1','1.2']));base_axes(ax)
  ax.spines['bottom'].set_visible(False)
  for row in d['ranges']:
   if row['group']!=group:continue
   y=row['row'];color=CORAL if y<2 else TEAL
   ax.hlines(y,row['observed_min'],row['observed_max'],color=color,lw=1.7,alpha=.48,zorder=2)
   for rnd,value in enumerate(row['round_values']):
    yy=y+(rnd-2)*.11
    artist=ax.scatter([value],[yy],color=color,marker=MARKERS[rnd],s=17,edgecolors='white',lw=.35,zorder=3)
    assert float(artist.get_offsets()[0][0])==value
    plotted.append({'group':group,'row':y,'round':rnd,'value':value})
 fig.text(.635,.128,'Query cost: feature on / off (log scale)',ha='center',fontsize=8.7,weight='bold')
 handles=[Line2D([],[],marker=MARKERS[i],color='#68757F',linestyle='none',markersize=3.7,label=f'R{i+1}') for i in range(5)]
 fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.0,-.012),ncol=5,frameon=False,fontsize=7.5,handletextpad=.25,columnspacing=.65)
 fig.text(.985,.016,'Points: geometric means · Lines: round ranges',ha='right',fontsize=7.6,color=MUTED)
 assert len(plotted)==60
 return save(fig,out,'optimized_v6_compact_factorial',{'points':plotted,'source_records':d['records'],'ranges':d['ranges'],'all_points_exact':True,'common_log_limits':list(limits),'color_meaning':'orange: screening; teal: retention','round_identity':'marker shape; R1–R5 denote rounds 1–5'})

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
 records={f.__name__:f(a.out) for f in [main_figure,family_figure,factorial_figure]}
 result={'status':'PASS','input_hashes':{p.name:sha(p) for p in (HERE/'inputs').glob('*.json')},'generator_sha256':sha(Path(__file__)),'new_experiments':0,'records':records,'outputs':{p.name:sha(p) for p in a.out.iterdir()}}
 (a.out/'GENERATION_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n');print('Generated all 3 vector figures with saved data.')
if __name__=='__main__':main()
