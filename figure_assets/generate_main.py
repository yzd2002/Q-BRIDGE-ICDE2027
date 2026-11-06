"""Draw an original vector overview from existing, hash-bound teaching records.

This is a presentation-only transformation. It runs no experiment or optimizer.
Coordinates are publication points; the SVG retains editable text.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE
SOURCE = HERE / 'inputs/query_example.json'
FAMILY = HERE / 'inputs/family_certificate.json'
PINS = {
    SOURCE: 'ecd00a506185b94d286aa556de73132a40490974cc2ddcda044f48c92b3a25cb',
    FAMILY: 'de63e22c4dc7ceb3d17a60a5816d2c64f0fdcb875c0a02ca17181979c652be9b',
}
INK, MUTED = '#17334A', '#526577'
BLUE, TEAL, AMBER = '#26709B', '#087F78', '#A96819'
RULE, PALE, GREEN_PALE, AMBER_PALE = '#D8E2E9', '#F3F7FA', '#EDF7F4', '#FBF4E9'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    for path, expected in PINS.items():
        if sha(path) != expected:
            raise ValueError('Changed teaching source: ' + str(path))
    data, family = json.loads(SOURCE.read_text()), json.loads(FAMILY.read_text())
    cert = family['certificate']
    assert family['independent_receiver_receipt']['verified']
    assert cert['policy'] == [['0/1', '0/1', '1/1', '0/1']]
    assert cert['policy_dual_lower_bound'] == cert['common_point_upper_bound'] == cert['capacity']
    gain = float(100 * Fraction(cert['capacity']))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.2,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'mathtext.fontset': 'dejavusans',
                         'axes.unicode_minus': True})
    W, H = 516, 230
    fig = plt.figure(figsize=(W/72, H/72), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(276, 0))
    ax.axis('off')
    texts = []

    def text(x, y, s, size=8.2, color=INK, weight='normal', ha='left', **kw):
        t = ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                    ha=ha, va='center', **kw)
        texts.append(t)
        return t

    def line(x1, y1, x2, y2, color=RULE, width=.7, style='-'):
        ax.plot([x1, x2], [y1, y2], color=color, lw=width, ls=style,
                solid_capstyle='round', zorder=1)

    def box(x, y, w, h, face='white', edge=RULE, lw=.65, radius=3):
        patch = FancyBboxPatch((x, y), w, h,
            boxstyle=f'round,pad=0,rounding_size={radius}',
            facecolor=face, edgecolor=edge, lw=lw, zorder=0)
        ax.add_patch(patch)

    def arrow(x1, y1, x2, y2, color=MUTED):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle='-|>',
                     mutation_scale=7, lw=.9, color=color,
                     shrinkA=0, shrinkB=0, zorder=2))

    # A: observed intervals and transferred group means jointly define one view.
    text(4, 12, '(a) One confidence view', 10, weight='bold')
    box(4, 28, 151, 226, face=PALE, edge=RULE)
    text(13, 43, 'Observational intervals', 8.3, BLUE, 'bold')
    x0, x1 = 32, 142
    for v in (0, .5, 1):
        xx=x0+(x1-x0)*v
        line(xx, 59, xx, 137, '#DFE7ED', .5)
        text(xx, 145, f'{v:g}', 7.5, MUTED, ha='center')
    for i, (lo, hi) in enumerate(zip(data['view_lo'], data['view_hi'], strict=True)):
        y = 65 + 22*i
        text(15, y, 'ABCD'[i], 9, weight='bold')
        a, b = x0+(x1-x0)*lo, x0+(x1-x0)*hi
        line(a,y,b,y, BLUE,2.5)
        line(a,y-3,a,y+3, BLUE,1)
        line(b,y-3,b,y+3, BLUE,1)
        text(a,y-9,f'{lo:.2f}',7.5,BLUE,ha='center')
        text(b,y-9,f'{hi:.2f}',7.5,BLUE,ha='center')
    text(87,160,'Success probability',7.5,MUTED,ha='center')
    line(13, 175, 145, 175)
    text(13,190,'Randomized group means',8.1,BLUE,'bold')
    text(79,210,r'$(m_A+m_C)/2 = 0.65$',9,ha='center')
    text(79,228,r'$(m_B+m_D)/2 = 0.70$',9,ha='center')
    text(79,245,'Fixed snapshot + assumptions',7.3,MUTED,ha='center')

    # A shared branch, rather than implying that the family depends on Q3.
    line(155,140,172,140,MUTED,.9)
    line(172,81,172,214,MUTED,.9)
    arrow(172,81,186,81)
    arrow(172,214,186,214)

    # B: all displayed endpoints are read from the saved four-action example.
    text(190,12,'(b) Can this policy guarantee a gain?',9.6,weight='bold')
    text(190,31,r'Requested gain: $\delta=0.100$ percentage points (pp)',7.8,MUTED)
    text(509,47,'Verified lower bound',7.3,MUTED,ha='right')
    for i,q in enumerate(data['queries']):
        y = 66 + 34*i
        text(191,y,q['query'],9.0,weight='bold')
        candidates=q['reuse_candidate_bounds']
        if not candidates:
            evidence='No saved proof'
            action='Solve + check'
            color,fill=BLUE,PALE
        else:
            cached=100*max(candidates)
            accepted=max(candidates)>data['margin']
            evidence=f'{cached:+.3f} pp '+('> ' if accepted else '< ')+r'$\delta$'
            action='Reuse + check' if accepted else 'Solve + check'
            color,fill=(TEAL,GREEN_PALE) if accepted else (AMBER,AMBER_PALE)
        text(215,y,evidence,7.8,color)
        arrow(307,y,320,y,color)
        if i==0:
            box(325,y-15,91,30,fill,RULE)
            text(370.5,y-5,action,8.0,color,'bold',ha='center')
            text(370.5,y+8,'save constraint weights',7.1,color,ha='center')
        else:
            box(325,y-11,91,22,fill,RULE)
            text(370.5,y,action,8.0,color,'bold',ha='center')
        arrow(420,y,443,y,TEAL)
        text(510,y,f'{100*q["returned_bound"]:+.3f} pp',9.2,TEAL,'bold',ha='right')
    text(215,154,'Insufficient reuse triggers a solve, not policy rejection.',7.5,MUTED)
    line(190,166,512,166)

    # C: give whole-family optimality equal visual prominence.
    text(190,182,'(c) What is the best possible guarantee?',9.6,weight='bold')
    text(190,199,'Disjoint groups: cap means  →  allocate  →  recover a policy',7.5,MUTED)
    text(227,216,'Choose C',8.0,BLUE,'bold',ha='center')
    for i in range(4):
        x=199+17*i
        height=18 if i==2 else 0
        if height:
            ax.add_patch(Rectangle((x,249-height),9,height,facecolor=BLUE,edgecolor='none'))
            text(x+4.5,249-height-5,'1',7.3,BLUE,ha='center')
        else:
            line(x,249,x+9,249,MUTED,1.1)
        text(x+4.5,257,'ABCD'[i],7.5,MUTED,ha='center')
    for x,title in [(279,'Policy lower bound'),(402,'Family upper bound')]:
        box(x,211,109,45,GREEN_PALE,'#C7DFD9')
        text(x+54.5,223,title,7.7,INK,ha='center')
        text(x+54.5,242,f'{gain:.3f} pp',12,TEAL,'bold',ha='center')
    text(395,241,'=',12,TEAL,'bold',ha='center')
    arrow(263,241,276,241,BLUE)
    text(333,268,'Optimal robust guarantee',7.3,TEAL,'bold',ha='center')
    text(456.5,268,r'Feasible $m=(.6,.7,.7,.7)$',7.1,MUTED,ha='center')
    text(4,269,'Illustration · reference: 25% per action',7.1,MUTED)

    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bounds=[]
    for t in texts:
        b=t.get_window_extent(renderer)
        if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:
            raise ValueError('Text outside figure canvas: '+t.get_text())
        bounds.append({'text':t.get_text(),'font_pt':t.get_fontsize(),
                       'bbox_px':list(map(float,b.bounds))})
    outputs=[]
    for ext in ('pdf','svg','png'):
        dest=args.out/f'qbridge_overview.{ext}'
        metadata={'Creator':'Q-BRIDGE vector figure renderer', 'Date':None} if ext=='svg' else None
        if ext=='pdf': metadata={'Creator':'Q-BRIDGE vector figure renderer','CreationDate':None,'ModDate':None}
        fig.savefig(dest,dpi=240,metadata=metadata)
        outputs.append({'path':dest.name,'sha256':sha(dest),'bytes':dest.stat().st_size})
    plt.close(fig)
    manifest={'scope':'presentation-only; constructed example, not empirical evidence',
        'inputs':{str(k.relative_to(ROOT)):v for k,v in PINS.items()},
        'script_sha256':sha(Path(__file__)), 'width_pt':W,'height_pt':H,
        'data':{'queries':[{k:q[k] for k in ('query','objective','reuse_candidate_bounds','returned_bound')} for q in data['queries']],
                'policy':cert['policy'],'family_capacity_exact':cert['capacity']},
        'outputs':outputs,'text_bounds':bounds,'visual_review':'pending'}
    (args.out/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'outputs':outputs,'width_pt':W,'height_pt':H}))


if __name__=='__main__':
    main()
