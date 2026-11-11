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
INK='#1D2B42'; MUTED='#56667B'; BLUE='#347BB5'; CORAL='#BC745E'; PURPLE='#6656AE'; TEAL='#128677'; GRAY='#8895A5'; RULE='#D7DEE8'
BP='#F0F6FC'; CP='#FCF3EE'; PP='#F3F0FA'; TP='#ECF7F3'
ROUNDS=[BLUE,CORAL,PURPLE,TEAL,GRAY];MARKERS=['o','s','^','D','v']
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
 W,H=516,254;fig=plt.figure(figsize=(W/72,H/72),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
 def text(x,y,s,fs=7.6,color=INK,bold=False,ha='left',**kw):return ax.text(x,y,s,fontsize=fs,color=color,fontweight='bold' if bold else 'normal',ha=ha,va='center',**kw)
 def line(x1,y1,x2,y2,c=RULE,lw=.7,**kw):ax.plot([x1,x2],[y1,y2],c=c,lw=lw,solid_capstyle='round',**kw)
 def box(x,y,w,h,fc='white',ec=RULE,r=4):ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',facecolor=fc,edgecolor=ec,lw=.65))
 def arrow(x1,y1,x2,y2,c=PURPLE):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=7,color=c,lw=1,shrinkA=0,shrinkB=0))
 def jacket(x,y,s=1,c=MUTED):
  pts=[(-3,-4),(-6,-2),(-8,4),(-5,5),(-4,1),(-4,9),(4,9),(4,1),(5,5),(8,4),(6,-2),(3,-4),(1,-2),(-1,-2),(-3,-4)]
  ax.add_patch(PathPatch(MPath([(x+s*a,y+s*b) for a,b in pts]),fill=False,edgecolor=c,lw=.65,joinstyle='round'))
  line(x,y-2*s,x,y+9*s,c,.5)
 text(4,9,'Q-BRIDGE',10.5,bold=True);line(72,2,72,16,GRAY,.6);text(81,9,'One shared view. Two policy questions.',9.1)
 text(4,27,'Four jackets A–D  ·  High rating = success  ·  Reference: 25% each',7.4,MUTED)
 # A: input evidence and the shared view.
 box(3,43,118,115,BP);text(11,54,'Observational logs',8.5,BLUE,True);text(11,67,'Bounds on each success rate',6.9,MUTED)
 x0,x1=23,111
 for v in [0,.5,1]:
  x=x0+(x1-x0)*v;line(x,81,x,134,'#DCE6EF',.5);text(x,143,f'{v:g}',6.9,MUTED,ha='center')
 coords=[]
 for i,(lo,hi) in enumerate(zip(data['view_lo'],data['view_hi'])):
  y=83+i*16;text(10,y,'ABCD'[i],7.6,bold=True);a=x0+(x1-x0)*lo;b=x0+(x1-x0)*hi
  line(a,y,b,y,BLUE,2.2);line(a,y-2,a,y+2,BLUE,.8);line(b,y-2,b,y+2,BLUE,.8)
  text((a+b)/2,y-6.5,f'{lo:.2f}–{hi:.2f}',6.8,BLUE,ha='center');coords.append({'action':'ABCD'[i],'lo':lo,'hi':hi,'line_x':[a,b]})
 text(67,153,'Success probability',6.9,MUTED,ha='center')
 box(3,167,118,72,CP);text(11,178,'Randomized experiments',7.6,CORAL,True);text(11,190,'Transferred group averages',6.8,MUTED)
 text(62,207,r'$(m_A+m_C)/2=0.65$',8.2,ha='center');text(62,225,r'$(m_B+m_D)/2=0.70$',8.2,ha='center')
 # Converge inputs, then fork view; all connectors outside panels.
 line(122,104,129,104,PURPLE,1);line(122,203,129,203,PURPLE,1);line(129,104,129,203,PURPLE,1);arrow(129,151,138,151)
 box(140,43,113,196,PP,'#CFC5E7');text(149,55,'(a) Joint confidence view',7.2,PURPLE,True)
 text(196.5,77,'All compatible',8.4,INK,True,ha='center');text(196.5,89,'success rates',8.4,INK,True,ha='center')
 a=Ellipse((185,139),68,76,angle=-12,fc='#D4E5F7',ec='none');b=Ellipse((208,139),68,76,angle=12,fc='#F1D9CF',ec='none');ax.add_patch(a);ax.add_patch(b)
 inter=Ellipse((208,139),68,76,angle=12,fc='#ADA0D1',ec='none');inter.set_clip_path(a);ax.add_patch(inter)
 for dx,dy in [(-6,-18),(8,-11),(-10,11),(9,19)]:ax.plot([196.5+dx],[139+dy],marker='o',ms=2.5,color='white')
 text(197,141,r'$\mathcal{C}$',20,'white',ha='center')
 text(196.5,187,'Individual + group',7.2,MUTED,ha='center');text(196.5,198,'constraints hold together',7.0,MUTED,ha='center')
 box(148,216,97,15,'#E6E0F4','none',3);text(196.5,223.5,'Fixed data + assumptions',6.5,PURPLE,ha='center')
 line(253,151,261,151,PURPLE,1);line(261,98,261,199,PURPLE,1);arrow(261,98,270,98);arrow(261,199,270,199)
 # B: fixed-policy queries.
 box(272,43,241,96,'white');text(280,55,'(b) Can this policy guarantee a gain?',8.1,INK,True)
 text(280,69,'Requested gain: 0.100 pp',7.0,MUTED);text(505,69,'Checked lower bound',6.8,MUTED,ha='right')
 for i,q in enumerate(data['queries']):
  y=85+i*18;c=[BLUE,TEAL,CORAL][i];fill=[BP,TP,CP][i]
  text(282,y,q['query'],8.2,INK,True);arrow(300,y,308,y,MUTED)
  box(312,y-7,110,14,fill,'none',3);text(367,y,['Solve + check','Reuse + check','New solve + check'][i],7.6,c,True,ha='center');arrow(427,y,438,y,MUTED)
  text(502,y,f'{100*q["returned_bound"]:+.3f} pp',9.1,TEAL,True,ha='right')
 text(367,132,'Q3: saved proof inconclusive → solve',6.6,MUTED,ha='center')
 # C: family capacity; no dependency on preceding Q3.
 box(272,149,241,90,TP,'#C7DED8');text(280,161,'(c) What is the best possible guarantee?',7.8,INK,True)
 text(280,174,'Disjoint groups: cap means → allocate → recover policy',6.7,MUTED)
 text(301,188,'Always C',7.3,TEAL,True,ha='center')
 for i in range(4):
  x=281+12*i;h=15 if i==2 else 0
  if h:ax.add_patch(Rectangle((x,211-h),7,h,fc=TEAL,ec='none'))
  else:line(x,211,x+7,211,GRAY,.8)
  text(x+3.5,218,'ABCD'[i],6.8,MUTED,ha='center')
 text(366,190,'Policy lower bound',6.9,ha='center');text(465,190,'Family upper bound',6.9,ha='center')
 text(366,208,f'{gain:.3f} pp',13.1,TEAL,True,ha='center');text(416,208,'=',14,TEAL,True,ha='center');text(465,208,f'{gain:.3f} pp',13.1,TEAL,True,ha='center')
 text(416,225,'Optimal robust guarantee',8.0,TEAL,True,ha='center')
 text(4,249,'Constructed example · pp = percentage points',6.9,MUTED)
 text(513,249,'One allowed vector: (0.60, 0.70, 0.70, 0.70)',6.9,MUTED,ha='right')
 return save(fig,out,'qbridge_overview',{'intervals':coords,'queries':[{k:q[k] for k in ['query','returned_bound','executed_route']} for q in data['queries']],'family_bound_pp':gain,'view_icon':'Schematic set intersection, not a quantitative projection','view_forks_independently':True})

def base_axes(ax):
 for s in ['top','right','left']:ax.spines[s].set_visible(False)
 ax.spines['bottom'].set_color(RULE);ax.tick_params(axis='y',length=0,pad=5);ax.tick_params(axis='x',length=2.5,width=.5)
 ax.xaxis.set_minor_locator(NullLocator());ax.set_axisbelow(True)

def family_figure(out):
 d=read('family_plot_data.json');fig=plt.figure(figsize=(516/72,181/72),facecolor='white')
 fig.text(.015,.95,'(a) Verified response cost',fontsize=9.1,weight='bold',va='top');fig.text(.535,.95,'(b) Search beyond the original candidates',fontsize=8.7,weight='bold',va='top')
 fig.text(.015,.853,'234 matched timing pairs · 78 fixed requests',fontsize=7.4,color=MUTED);fig.text(.535,.853,'107 unresolved views · same data and threshold',fontsize=7.4,color=MUTED)
 ax=fig.add_axes([.095,.27,.39,.48]);ax.set_xscale('log');ax.set_xlim(d['geometry']['x_limits']);ax.set_ylim(2.5,-.5)
 ax.axvspan(ax.get_xlim()[0],1,color=TP,zorder=0);ax.axvline(1,color=GRAY,lw=.8,ls=(0,(3,2)));ax.grid(axis='x',color=RULE,lw=.4)
 plotted=[]
 for r in [1,2,3]:
  ps=[p for p in d['points'] if p['round']==r];artist=ax.scatter([p['plot_x'] for p in ps],[p['plot_y'] for p in ps],s=10.5,color=ROUNDS[r-1],marker=MARKERS[r-1],alpha=.85,edgecolors='white',lw=.2,zorder=3)
  for p,xy in zip(ps,artist.get_offsets()):assert float(xy[0])==p['plot_x'] and float(xy[1])==p['plot_y']
  plotted+=ps
 ax.set_yticks([0,1,2],[r'$d=8$',r'$d=64$',r'$d=256$']);ax.xaxis.set_major_locator(FixedLocator([.5,1,2]));ax.xaxis.set_major_formatter(FixedFormatter(['0.5','1','2']));base_axes(ax)
 ax.set_xlabel('Partition / exact LP time (log scale)',fontsize=7.7,labelpad=4)
 handles=[Line2D([],[],marker=MARKERS[i],color=ROUNDS[i],linestyle='none',markersize=3.6,label=f'Round {i+1}') for i in range(3)]
 fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.075,.02),ncol=3,frameon=False,fontsize=7.3,handletextpad=.25,columnspacing=.8)
 # Right panel: exact counts, including both zero categories.
 ax2=fig.add_axes([.815,.27,.168,.48]);vals=[p['count'] for p in d['recovery_categories']];assert vals==[0,86,21,0]
 ax2.barh(range(4),vals,height=.56,color=[BLUE,TEAL,PURPLE,GRAY]);ax2.set_xlim(0,111);ax2.set_ylim(3.6,-.6)
 labels=['Original candidates qualify','New qualifying policy',r'Family bound $\leq\delta$','Decomposition unknown']
 ax2.set_yticks(range(4),labels,fontsize=7.4);ax2.xaxis.set_major_locator(FixedLocator([0,50,100]));base_axes(ax2)
 for i,v in enumerate(vals):ax2.text(v+3,i,str(v),ha='left',va='center',fontsize=8.8,color=TEAL if i==1 else INK,weight='bold')
 ax2.set_xlabel('Views',fontsize=7.7,labelpad=4)
 fig.text(.535,.085,'A qualifying policy in 86 views; stopping justified in 21.',fontsize=7.1,color=TEAL)
 fig.text(.015,.01,'Below 1: partition is faster. Each point is one checked query in one timing round.',fontsize=7.1,color=MUTED)
 assert len(plotted)==234
 return save(fig,out,'verified_family_v7_main',{'points':plotted,'recovery_categories':d['recovery_categories'],'failure_denominators':d['failure_denominators'],'all_points_exact':True})

def factorial_figure(out):
 d=read('factorial_plot_data.json');fig,axes=plt.subplots(1,3,figsize=(516/72,176/72),sharex=True,sharey=True);fig.subplots_adjust(left=.195,right=.99,bottom=.27,top=.72,wspace=.15)
 groups=[('original_main','(a) Original main',3),('threshold_engine','(b) Engine streams',54),('source_derived','(c) Source-derived',6)]
 vals=[r['geometric_mean'] for r in d['records']];lo,hi=min(vals),max(vals);pad=.065*(math.log(hi)-math.log(lo));limits=(math.exp(math.log(lo)-pad),math.exp(math.log(hi)+pad));plotted=[]
 for ax,(g,title,n) in zip(axes,groups):
  ax.set_xscale('log');ax.set_xlim(limits);ax.set_ylim(3.6,-.6);ax.axvspan(limits[0],1,color=TP,zorder=0);ax.axvline(1,color=GRAY,ls=(0,(3,2)),lw=.8)
  ax.grid(axis='x',color=RULE,lw=.4);ax.axhline(1.5,color=RULE,lw=.6)
  ax.set_title(f'{title}\n{n} fixed inputs',fontsize=8.2,pad=7);ax.set_yticks(range(4),['Screening | P=0','Screening | P=1','Retention | S=0','Retention | S=1'],fontsize=7.7)
  ax.xaxis.set_major_locator(FixedLocator([.8,1,1.2]));ax.xaxis.set_major_formatter(FixedFormatter(['0.8','1','1.2']));base_axes(ax)
  for row in d['ranges']:
   if row['group']!=g:continue
   ax.hlines(row['row'],row['observed_min'],row['observed_max'],color=GRAY,lw=1.0,zorder=2)
   for r,value in enumerate(row['round_values']):
    y=row['row']+(r-2)*.13;ar=ax.scatter([value],[y],color=ROUNDS[r],marker=MARKERS[r],s=16,edgecolors='white',lw=.3,zorder=3);assert float(ar.get_offsets()[0][0])==value
    plotted.append({'group':g,'row':row['row'],'round':r,'value':value})
  for k,t in enumerate(ax.get_yticklabels()):
   if k>=2:t.set_color(TEAL);t.set_weight('bold')
 handles=[Line2D([],[],marker=MARKERS[i],color=ROUNDS[i],linestyle='none',markersize=3.5,label=f'Round {i+1}') for i in range(5)]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.59,.995),ncol=5,frameon=False,fontsize=7.3,handletextpad=.3,columnspacing=1.2)
 fig.text(.59,.105,'Cost ratio: factor enabled / disabled (log scale)',ha='center',fontsize=7.9)
 fig.text(.015,.018,'Below 1: faster. Lines span five timing rounds; dots are equal-input geometric means.',fontsize=7.2,color=MUTED)
 assert len(plotted)==60
 return save(fig,out,'optimized_v6_compact_factorial',{'points':plotted,'source_records':d['records'],'ranges':d['ranges'],'all_points_exact':True,'common_log_limits':list(limits)})

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
 records={f.__name__:f(a.out) for f in [main_figure,family_figure,factorial_figure]}
 result={'status':'PASS','input_hashes':{p.name:sha(p) for p in (HERE/'inputs').glob('*.json')},'generator_sha256':sha(Path(__file__)),'new_experiments':0,'records':records,'outputs':{p.name:sha(p) for p in a.out.iterdir()}}
 (a.out/'GENERATION_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n');print('Generated all 3 vector figures with saved data.')
if __name__=='__main__':main()
