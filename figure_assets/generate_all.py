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
INK='#202020'; MUTED='#555555'; BLUE='#2E75B6'; CORAL='#E58A2B'; PURPLE='#41566F'; TEAL='#2E75B6'; GRAY='#8A9299'; RULE='#D5D9DD'
BP='#F1F6FB'; CP='#FFF5E8'; PP='#F4F6F8'; TP='#F1F6FB'
ROUNDS=['#A9C9E7','#609ACE','#2668A5','#E58A2B','#68727A'];MARKERS=['o','s','^','D','v']
plt.rcParams.update({'font.family':'STIXGeneral','font.size':8,'text.color':INK,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':INK,'axes.edgecolor':RULE,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'stix','axes.linewidth':.6})
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
 W,H=516,254;fig=plt.figure(figsize=(W/72,224/72),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
 def text(x,y,s,fs=7.6,color=INK,bold=False,ha='left',**kw):return ax.text(x,y,s,fontsize=fs,color=color,fontweight='bold' if bold else 'normal',ha=ha,va='center',**kw)
 def line(x1,y1,x2,y2,c=RULE,lw=.7,**kw):ax.plot([x1,x2],[y1,y2],c=c,lw=lw,solid_capstyle='round',**kw)
 def box(x,y,w,h,fc='white',ec=RULE,r=0):ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size=0',facecolor=fc,edgecolor=ec,lw=.65))
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
 box(140,43,113,196,PP,'#AEB9C5');text(149,55,'(a) Joint confidence view',7.2,PURPLE,True)
 text(196.5,77,'All compatible',8.4,INK,True,ha='center');text(196.5,89,'success rates',8.4,INK,True,ha='center')
 a=Ellipse((185,139),68,76,angle=-12,fc='#D7E7F6',ec='none');b=Ellipse((208,139),68,76,angle=12,fc='#F8DFC0',ec='none');ax.add_patch(a);ax.add_patch(b)
 inter=Ellipse((208,139),68,76,angle=12,fc='#829DB5',ec='none');inter.set_clip_path(a);ax.add_patch(inter)
 for dx,dy in [(-6,-18),(8,-11),(-10,11),(9,19)]:ax.plot([196.5+dx],[139+dy],marker='o',ms=2.5,color='white')
 text(197,141,r'$\mathcal{C}$',20,'white',ha='center')
 text(196.5,187,'Individual + group',7.2,MUTED,ha='center');text(196.5,198,'constraints hold together',7.0,MUTED,ha='center')
 box(148,216,97,15,'#E7EDF3','none',3);text(196.5,223.5,'Fixed data + assumptions',6.5,PURPLE,ha='center')
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
 box(272,149,241,90,TP,'#B9D0E5');text(280,161,'(c) What is the best possible guarantee?',7.8,INK,True)
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
 d=read('family_plot_data.json');W,H=516,181;fig=plt.figure(figsize=(W/72,H/72),facecolor='white')
 fig.text(.015,.965,'(a) Cost of an independently checked answer',fontsize=10,weight='bold',va='top')
 fig.text(.565,.965,'(b) What full-family search adds',fontsize=10,weight='bold',va='top')
 fig.text(.015,.862,'78 requests × 3 timing rounds = 234 paired measurements',fontsize=7.8,color=MUTED)
 fig.text(.565,.862,'107 views unresolved by the initial checks',fontsize=8.1,color=MUTED)
 ax=fig.add_axes([.09,.285,.43,.49]);ax.set_xscale('log');ax.set_xlim(d['geometry']['x_limits']);ax.set_ylim(2.5,-.5)
 ax.axhspan(.5,1.5,color='#F6F7F9',zorder=0);ax.grid(axis='x',color=RULE,lw=.45)
 ax.axvline(1,color='#555555',lw=.85,ls=(0,(3,2)));plotted=[]
 for rnd in [1,2,3]:
  ps=[p for p in d['points'] if p['round']==rnd]
  artist=ax.scatter([p['plot_x'] for p in ps],[p['plot_y'] for p in ps],s=16,color=ROUNDS[rnd-1],marker=MARKERS[rnd-1],alpha=.93,edgecolors='white',lw=.25,zorder=3)
  for p,xy in zip(ps,artist.get_offsets()):assert float(xy[0])==p['plot_x'] and float(xy[1])==p['plot_y']
  plotted+=ps
 ax.set_yticks([0,1,2],[r'$d=8$',r'$d=64$',r'$d=256$'],fontsize=8.7)
 ax.xaxis.set_major_locator(FixedLocator([.5,1,2]));ax.xaxis.set_major_formatter(FixedFormatter(['0.5','1','2']));base_axes(ax)
 ax.set_xlabel('Partition / exact LP response time',fontsize=9,labelpad=4)
 handles=[Line2D([],[],marker=MARKERS[i],color=ROUNDS[i],linestyle='none',markersize=4.1,label=f'Round {i+1}') for i in range(3)]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.297,.025),ncol=3,frameon=False,fontsize=8,handletextpad=.35,columnspacing=1.25)
 fig.text(.015,.008,'Below 1: partition is faster.  d = context–action pairs.',fontsize=7.8,color=MUTED)
 # A count-preserving stacked bar exposes the complete partition of 107 views.
 vals=[p['count'] for p in d['recovery_categories']];assert vals==[0,86,21,0]
 ax2=fig.add_axes([.565,.365,.42,.30]);ax2.set(xlim=(0,107),ylim=(0,1));ax2.axis('off')
 ax2.barh(.52,86,height=.45,color=BLUE,edgecolor=INK,lw=.55)
 ax2.barh(.52,21,left=86,height=.45,color=CORAL,edgecolor=INK,lw=.55)
 ax2.text(43,.52,'86',fontsize=17.5,color='white',weight='bold',ha='center',va='center')
 ax2.text(96.5,.52,'21',fontsize=17.5,color=INK,weight='bold',ha='center',va='center')
 ax2.text(43,.02,'New qualifying policy',fontsize=8.6,color=BLUE,weight='bold',ha='center',va='center')
 ax2.text(96.5,-.09,'No such\nguarantee',fontsize=7.9,color='#A85D16',ha='center',va='center',linespacing=1.05)
 fig.text(.565,.735,'Guaranteed gain above the requested threshold',fontsize=8.1,style='italic',color=MUTED)
 for y,label,value in [(.20,'Original candidates qualify',0),(.09,'Decomposition unknown',0)]:
  fig.text(.575,y,label,fontsize=8.1,color=MUTED)
  fig.text(.975,y,str(value),ha='right',fontsize=10,weight='bold')
 fig.add_artist(Line2D([.565,.985],[.28,.28],transform=fig.transFigure,color=RULE,lw=.6))
 assert len(plotted)==234
 return save(fig,out,'verified_family_v7_main',{'points':plotted,'recovery_categories':d['recovery_categories'],'failure_denominators':d['failure_denominators'],'all_points_exact':True,'count_bar_segments':[86,21],'zero_categories_displayed_as_text':True})

def factorial_figure(out):
 d=read('factorial_plot_data.json');W,H=516,180;fig,axes=plt.subplots(1,3,figsize=(W/72,H/72),sharex=True,sharey=True)
 fig.subplots_adjust(left=.21,right=.985,bottom=.24,top=.72,wspace=.20)
 groups=[('original_main','(a) Repeated-query streams',3),('threshold_engine','(b) Geometry streams',54),('source_derived','(c) Generated-log streams',6)]
 vals=[r['geometric_mean'] for r in d['records']];lo,hi=min(vals),max(vals);pad=.065*(math.log(hi)-math.log(lo));limits=(math.exp(math.log(lo)-pad),math.exp(math.log(hi)+pad));plotted=[]
 labels=['Screening\nretention off','Screening\nretention on','Retention\nscreening off','Retention\nscreening on']
 for ax,(group,title,n) in zip(axes,groups):
  ax.set_xscale('log');ax.set_xlim(limits);ax.set_ylim(3.55,-.55)
  ax.grid(axis='x',color=RULE,lw=.45);ax.axvline(1,color='#555555',ls=(0,(3,2)),lw=.8);ax.axhline(1.5,color='#B8BEC4',lw=.65)
  ax.set_title(f'{title}\n{n} fixed inputs',fontsize=8.9,pad=9)
  ax.set_yticks(range(4),labels,fontsize=8.5,linespacing=1.0)
  ax.xaxis.set_major_locator(FixedLocator([.8,1,1.2]));ax.xaxis.set_major_formatter(FixedFormatter(['0.8','1','1.2']));base_axes(ax)
  for row in d['ranges']:
   if row['group']!=group:continue
   y=row['row'];color=CORAL if y<2 else BLUE
   ax.hlines(y,row['observed_min'],row['observed_max'],color=color,lw=1.3,alpha=.65,zorder=2)
   for rnd,value in enumerate(row['round_values']):
    yy=y+(rnd-2)*.105
    artist=ax.scatter([value],[yy],color=color,marker=MARKERS[rnd],s=24,edgecolors='white',lw=.4,zorder=3)
    assert float(artist.get_offsets()[0][0])==value
    plotted.append({'group':group,'row':y,'round':rnd,'value':value})
  for k,label in enumerate(ax.get_yticklabels()):label.set_color('#B56B20' if k<2 else BLUE)
 handles=[Line2D([],[],marker=MARKERS[i],color='#626B73',linestyle='none',markersize=4.0,label=f'Round {i+1}') for i in range(5)]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.595,1.0),ncol=5,frameon=False,fontsize=8.2,handletextpad=.35,columnspacing=1.25)
 fig.text(.59,.09,'Query cost with the named feature on / off',ha='center',fontsize=9.2)
 fig.text(.015,.01,'Below 1: faster.  Points: geometric means. Lines: observed ranges over five rounds.',fontsize=7.9,color=MUTED)
 assert len(plotted)==60
 return save(fig,out,'optimized_v6_compact_factorial',{'points':plotted,'source_records':d['records'],'ranges':d['ranges'],'all_points_exact':True,'common_log_limits':list(limits),'color_meaning':'orange: screening; blue: retention','round_identity':'marker shape'})

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
 records={f.__name__:f(a.out) for f in [main_figure,family_figure,factorial_figure]}
 result={'status':'PASS','input_hashes':{p.name:sha(p) for p in (HERE/'inputs').glob('*.json')},'generator_sha256':sha(Path(__file__)),'new_experiments':0,'records':records,'outputs':{p.name:sha(p) for p in a.out.iterdir()}}
 (a.out/'GENERATION_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n');print('Generated all 3 vector figures with saved data.')
if __name__=='__main__':main()
