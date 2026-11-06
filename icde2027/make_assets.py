"""Publication figures and tables, generated only from audited result records."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/"icde2027/paper"
FIG=PAPER/"figures"
COLORS={"observation":"#3B6DA8","moment":"#C16A21","fusion":"#16836B","truth":"#333333"}
LABELS={"observation":"Observational only","moment":"Selected moments only","fusion":"Q-BRIDGE"}
NAMES={"controlled_2x2":"Controlled 2×2","kuairec_8x8":"KuaiRec-derived 8×8"}


def save(fig,name):
    fig.savefig(FIG/(name+".pdf"),bbox_inches="tight",pad_inches=.035)
    fig.savefig(FIG/(name+".png"),dpi=180,bbox_inches="tight",pad_inches=.035)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    s=json.loads((ROOT/"runs/icde2027/analysis/summary.json").read_text())
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9,
        "axes.labelsize":8,"legend.fontsize":7,"xtick.labelsize":7,"ytick.labelsize":7,
        "pdf.fonttype":42,"ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,
        "axes.linewidth":.6,"lines.linewidth":1.6,"grid.linewidth":.4})

    fig,axs=plt.subplots(1,2,figsize=(7.0,2.25),layout="constrained")
    for ax,(world,pop) in zip(axs,s["population"].items()):
        x=np.array([p["factor"] for p in pop["curve"]])
        for method in ("observation","moment","fusion"):
            y=np.array([p[method] for p in pop["curve"]])
            ax.plot(x,y,color=COLORS[method],label=LABELS[method],linestyle={"observation":"--","moment":":","fusion":"-"}[method])
        ax.axhline(pop["true_contrast"],color=COLORS["truth"],lw=.9,label="True contrast")
        obs=np.array([p["observation"] for p in pop["curve"]]);fused=np.array([p["fusion"] for p in pop["curve"]])
        ax.fill_between(x,0,fused,where=(obs<=0)&(fused>0),color=COLORS["fusion"],alpha=.15)
        ax.axhline(0,color="#555555",lw=.6)
        at=2/pop["lambda_star"]
        ax.axvline(at,color="#777777",lw=.8,ls="-.")
        ax.text(at+.02,.12,r"$\Lambda=2$",fontsize=7,transform=ax.get_xaxis_transform())
        ax.set(xlabel=r"Declared / true radius ($\Lambda/\Lambda^\star$)",ylabel="Population lower bound",title=NAMES[world])
        ax.grid(axis="y",alpha=.25)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncol=4,frameon=False)
    save(fig,"population_radius")

    fig,axs=plt.subplots(1,2,figsize=(7.0,2.25),layout="constrained")
    for ax,world in zip(axs,NAMES):
        groups=[g for g in s["stages"]["secondary"] if g["config"]["world"]==world and "budget" in g["id"]]
        ns=sorted({g["config"]["n_observation"] for g in groups})
        ratios=[.01,.05,.1,.2,1.]
        data=np.zeros((len(ns),len(ratios)))
        for g in groups:
            n=g["config"]["n_observation"];r=g["config"]["n_randomized"]/n
            data[ns.index(n),ratios.index(r)]=100*g["rescue"]["rate"]
        im=ax.imshow(data,vmin=0,vmax=100,cmap="YlGnBu",aspect="auto")
        for (i,j),value in np.ndenumerate(data):
            ax.text(j,i,f"{value:.0f}",ha="center",va="center",fontsize=9,color="white" if value>55 else "#222222")
        ax.set_xticks(range(5),["1%","5%","10%","20%","100%"])
        ax.set_yticks(range(4),[f"{n//1000}k" for n in ns])
        ax.set(xlabel="Randomized / observational traffic",ylabel="Observational events",title=NAMES[world])
    colorbar=fig.colorbar(im,ax=axs,shrink=.9,pad=.02)
    colorbar.set_label("Decision rescue (%)")
    save(fig,"budget_grid")

    fig,axs=plt.subplots(1,2,figsize=(7.0,2.25),layout="constrained")
    for method,label,color,marker in [("bincount_stream","Chunked count aggregation",COLORS['fusion'],'o'),("pandas_groupby","pandas groupby",COLORS['observation'],'s')]:
        rows=[r for r in s["systems"]["aggregation"] if r["method"]==method]
        x=np.array([r["n"] for r in rows]); vals=np.array([r["warm_seconds"] for r in rows])*1000
        mid=np.median(vals,axis=1)
        axs[0].errorbar(x,mid,yerr=[mid-vals.min(1),vals.max(1)-mid],label=label,color=color,marker=marker,ms=3,capsize=2)
    for method,label,color,marker in [("knapsack","Disjoint-row plan + verifier",COLORS['fusion'],'o'),("highs","HiGHS + same verifier",COLORS['observation'],'s')]:
        rows=[r for r in s["systems"]["solvers"] if r["backend"]==method]
        x=np.array([r["cells"] for r in rows]);vals=np.array([[v["seconds"] for v in r["warm"]] for r in rows])*1000
        mid=np.median(vals,axis=1)
        axs[1].errorbar(x,mid,yerr=[mid-vals.min(1),vals.max(1)-mid],label=label,color=color,marker=marker,ms=3,capsize=2)
    for ax in axs:
        ax.set_xscale("log");ax.set_yscale("log");ax.grid(alpha=.25,which="major");ax.legend(frameon=False,loc="upper left",fontsize=6.5)
    axs[0].set(xlabel="Input events",ylabel="Aggregation time (ms)",title="Actual row scans; 64 cells")
    axs[1].set(xlabel="Context–action cells",ylabel="Solve + verification time (ms)",title="Same intervals; four cells per moment")
    save(fig,"execution")

    fig,ax=plt.subplots(figsize=(7.0,1.35))
    ax.set_xlim(0,10);ax.set_ylim(0,2);ax.axis("off")
    blocks=[(.05,2.2,"Event tables","Observational source O\nRandomized source R",COLORS['observation']),
            (2.65,2.2,"Sufficient counts","Group by context,\naction and outcome",COLORS['observation']),
            (5.25,2.2,"Mean constraints","OBS cell intervals\n+ R group intervals",COLORS['moment']),
            (7.85,2.1,"Verified bound","Disjoint plan / sparse LP\nWeak-dual verification",COLORS['fusion'])]
    for x,width,title,body,color in blocks:
        ax.add_patch(FancyBboxPatch((x,.48),width,1.36,boxstyle="round,pad=.025",facecolor="white",edgecolor=color,lw=1.2))
        ax.text(x+width/2,1.55,title,ha="center",va="center",fontsize=8,fontweight="bold",color=color)
        ax.text(x+width/2,1.05,body,ha="center",va="center",fontsize=7.3,linespacing=1.6)
    for x in [2.3,4.9,7.5]:
        ax.annotate("",xy=(x+.3,1.15),xytext=(x,1.15),arrowprops={"arrowstyle":"->","color":"#555555","lw":1})
    ax.text(5,.1,"Declared contract: target population, policy pair, sensitivity radius, randomized design, transported groups and tolerance",ha="center",fontsize=7)
    save(fig,"overview")

    # Tables share the exact JSON source used by the plots and numeric audit.
    rows=[]
    for g in s['stages']['primary']:
        label=('2$\\times$2' if g['config']['world']=='controlled_2x2' else '8$\\times$8')+' / '+g['config']['policy']
        m=g['methods'];r=g['rescue'];f=m['fusion']
        rows.append(f"{label} & {g['truth_mean']:.4f} & {m['observation_standalone']['mean_with_fallback']:.4f} & {m['moment_standalone']['mean_with_fallback']:.4f} & ${f['mean_with_fallback']:.4f} \\pm {f['sd_with_fallback']:.4f}$ & {r['successes']}/100 [{100*r['ci95'][0]:.1f}, {100*r['ci95'][1]:.1f}] \\\\")
    (PAPER/'primary_rows.tex').write_text('\n'.join(rows)+'\n')
    (PAPER/'population_rows.tex').write_text('\n'.join(
        ('2$\\times$2' if w=='controlled_2x2' else '8$\\times$8')+f" & {p['lambda_star']:.4f} & {p['true_contrast']:.4f} & "+' & '.join(f"{p['at_radius_two'][m]:.4f}" for m in ('observation','moment','fusion'))+' \\\\'
        for w,p in s['population'].items())+'\n')
    for key,columns,header in [
        ('primary','lrrrrl',r'World / policy pair & True $\Delta$ (mean) & OBS only & Moments only & Fusion mean $\pm$ SD & Rescue [95\% CI, \%]'),
        ('population','lrrrrr',r'World & $\Lambda^\star$ & True $\Delta$ & OBS & Moments & Fusion')]:
        contents='\\begin{tabular}{@{}'+columns+'@{}}\n\\toprule\n'+header+r'\\'+'\n\\midrule\n'
        contents+=(PAPER/(key+'_rows.tex')).read_text()
        contents+='\\bottomrule\n\\end{tabular}\n'
        (PAPER/(key+'_table.tex')).write_text(contents)
    print('Generated figures and table rows in',PAPER)


if __name__=='__main__':
    main()
