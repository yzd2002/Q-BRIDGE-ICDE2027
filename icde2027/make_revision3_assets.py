"""Original vector illustration of the final three-valued query contract."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PAPER = Path(__file__).resolve().parent / 'paper'
FIGURES = PAPER / 'figures'
INK, BLUE, GREEN, ORANGE, PURPLE = '#25313B', '#0072B2', '#008B72', '#B36B00', '#7655A5'


def flow():
    fig, ax = plt.subplots(figsize=(7.1, 2.65))
    ax.set_xlim(-.05, 12.08); ax.set_ylim(-.05, 3.64); ax.axis('off')

    def box(x, y, width, height, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                     boxstyle='round,pad=.025,rounding_size=.055',
                     facecolor='white', edgecolor=color, lw=.9))
        ax.text(x+width/2, y+height-.22, title, ha='center', va='center',
                fontsize=8.1, weight='bold', color=color)
        ax.text(x+width/2, y+.34, body, ha='center', va='center',
                fontsize=7.4, linespacing=1.35, color=INK)

    def arrow(start, end, label=None, offset=.12):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops={'arrowstyle': '-|>', 'lw': .8, 'color': '#66727B',
                                'shrinkA': 2, 'shrinkB': 2})
        if label:
            vertical = abs(start[0]-end[0]) < .02
            ax.text((start[0]+end[0])/2 + (offset if vertical else 0),
                    (start[1]+end[1])/2 + (0 if vertical else offset), label,
                    ha='left' if vertical else 'center', va='center', fontsize=6.7, color=INK)

    ax.text(.04, 3.43, 'Build once: fixed source declaration', color=BLUE, fontsize=7.7, weight='bold')
    ax.text(5.48, 3.43, 'Query: new objective and margin', color=GREEN, fontsize=7.7, weight='bold')
    box(.05, 2.05, 1.95, 1.12, 'Evidence', 'O / R counts or\nuser-block scores', BLUE)
    box(2.50, 2.05, 2.47, 1.12, 'Confidence view', 'Intervals + moments\nExact feasible witness', BLUE)
    box(5.48, 2.05, 2.55, 1.12, 'Try reuse', 'Initial / saved evidence\nLower L and upper U', GREEN)
    box(5.48, .31, 2.55, 1.12, 'Compute', 'Solve proposals\nExact global checks', PURPLE)
    box(2.50, .31, 2.47, 1.12, 'Save candidates', 'Checked duals / points\nfor later queries', PURPLE)

    arrow((2., 2.61), (2.50, 2.61))
    arrow((4.97, 2.61), (5.48, 2.61))
    arrow((8.03, 2.61), (8.55, 2.61), 'decided', .16)
    arrow((6.75, 2.05), (6.75, 1.43), 'not yet')
    arrow((8.03, .87), (8.55, .87))
    arrow((5.48, .87), (4.97, .87), 'save', .16)
    arrow((4.97, 1.40), (5.48, 2.13))
    ax.text(2.50, 1.77, 'No verified view: UNKNOWN', fontsize=6.7, color=PURPLE)
    ax.text(.05, .90, 'Changed data or\nsource assumptions\nrequire a new view.',
            fontsize=7.4, color=INK, va='center', linespacing=1.4)

    ax.add_patch(FancyBboxPatch((8.55, .31), 3.43, 2.86,
                 boxstyle='round,pad=.025,rounding_size=.055',
                 facecolor='#F8FAFA', edgecolor='#ACBABE', lw=.9))
    x = 8.74
    ax.text(x, 2.93, 'Threshold verdict', color=INK, fontsize=8.0, weight='bold')
    for yy, heading, body, color in [
        (2.51, r'$L>\delta$  →  TRUE', 'All allowed means pass', GREEN),
        (1.67, r'$U\leq\delta$  →  FALSE', 'A feasible mean fails', ORANGE),
        (.83, 'Else  →  UNKNOWN', 'Not yet established', PURPLE),
    ]:
        ax.text(x, yy, heading, fontsize=8.2, color=color, weight='bold')
        ax.text(x, yy-.28, body, fontsize=7.2, color=INK)
    for yy in (2.02, 1.18):
        ax.plot([8.74, 11.78], [yy, yy], color='#D7DFE1', lw=.6)
    fig.text(.5, .012, 'FALSE concerns the worst allowed mean vector; it does not establish a negative true policy effect.',
             ha='center', fontsize=7.1, color=INK)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'png'):
        fig.savefig(FIGURES/f'execution_flow_v3.{suffix}', dpi=200, bbox_inches='tight', pad_inches=.025)
    plt.close(fig)
    (PAPER/'execution_flow_v3_contract.json').write_text(json.dumps({
        'kind': 'original explanatory diagram, not an empirical result',
        'true': 'checked lower > requested margin',
        'false': 'checked globally feasible primal upper <= requested margin',
        'unknown': 'neither side has been established, including explicit numerical no-bound/refusal',
        'scope': 'the robust minimum over the caller-trusted represented confidence view',
        'false_does_not_imply': 'the true policy effect is nonpositive or below the margin',
    }, indent=2)+'\n')


if __name__ == '__main__':
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    flow()
