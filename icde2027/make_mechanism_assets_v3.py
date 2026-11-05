"""Render the audited, retrospective information diagnostic; never rerun inference."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'icde2027/paper'
ANALYSIS = ROOT / 'runs/icde2027/analysis'
AUDIT = ROOT / 'runs/icde2027/reviews/fresh_mechanism_independent_review.json'
CATEGORIES = (
    'finite_sample_certified',
    'finite_relaxation_limit',
    'population_and_finite_relaxation_limit',
    'true_effect_not_above_margin',
)
COLORS = ('#008B72', '#E5A349', '#537EAE', '#B8BFC5')
LABELS = ('Certified from the sample', 'Finite interval width',
          'Population constraints still insufficient', 'True effect at most the margin')
POLICIES = ('metadata', 'randomized_heuristic', 'robust_design_selection')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_verified():
    audit = json.loads(AUDIT.read_text())
    if audit['status'] != 'independently_verified_with_scoped_conclusions':
        raise RuntimeError('Independent scientific audit is not complete')
    finite_audit = audit['finite_gap_independent_audit']
    if finite_audit['status'] != 'independently_verified':
        raise RuntimeError('Exact finite optimum and containment audit is required')
    bindings = audit['inputs_sha256'] | finite_audit['source_files_sha256']
    for relative, digest in bindings.items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError(f'Audited input changed: {relative}')
    population = json.loads((ANALYSIS/'fresh_mechanism_results.json').read_text())
    finite = json.loads((ANALYSIS/'fresh_mechanism_finite_gap_results.json').read_text())
    if population['status'] != 'complete' or finite['status'] != 'complete':
        raise RuntimeError('Both retrospective analyses must be complete')
    if not finite['summary']['all_exact_primal_dual_equal'] or finite['summary']['anomalies']:
        raise RuntimeError('Exact optimum evidence is incomplete')
    if finite['summary']['population_not_subset_finite_views'] or finite['summary']['unknown_queries']:
        raise RuntimeError('Cannot classify sample limitations without containment')
    rows = finite['queries']
    counts = Counter((r['policy'], r['updated_category']) for r in rows)
    independent = {(r['policy'], r['category']): r['count'] for r in finite_audit['partition']}
    if dict(counts) != independent or len(rows) != 720:
        raise RuntimeError('Rendered partition differs from independent reconstruction')
    if any(r['updated_category'] not in CATEGORIES for r in rows):
        raise RuntimeError('An unrendered category must not be silently omitted')
    sources = {p: sha(ROOT/p) for p in (
        'runs/icde2027/analysis/fresh_mechanism_results.json',
        'runs/icde2027/analysis/fresh_mechanism_finite_gap_results.json',
        'runs/icde2027/reviews/fresh_mechanism_independent_review.json',
        'runs/icde2027/analysis/fresh_mechanism_replay.json',
        'runs/icde2027/analysis/fresh_mechanism_finite_gap_replay.json',
    )}
    return population, finite, rows, counts, sources


def main():
    population, finite, rows, counts, sources = read_verified()
    policy_counts = [{'policy': p, 'n': sum(counts[p, c] for c in CATEGORIES),
                      'counts': {c: counts[p, c] for c in CATEGORIES}} for p in POLICIES]
    groups = [(s, drift) for s in ('2x4', '8x8', '16x16') for drift in (0., .02)]
    stratum_counts = []
    for size, drift in groups:
        selected = [r for r in rows if r['policy'] == 'metadata' and
                    r['size'] == size and r['nominal_drift'] == drift]
        cc = Counter(r['updated_category'] for r in selected)
        if len(selected) != 40:
            raise RuntimeError('The complete two-randomization stratum must contain 40 queries')
        stratum_counts.append({'size': size, 'nominal_drift': drift, 'n': len(selected),
                              'counts': {c: cc[c] for c in CATEGORIES}})
    complete_strata = []
    for p in POLICIES:
        for size, drift in groups:
            for randomization in ('uniform', 'skew'):
                selected = [r for r in rows if (r['policy'], r['size'], r['nominal_drift'], r['randomization'])
                            == (p, size, drift, randomization)]
                cc = Counter(r['updated_category'] for r in selected)
                if len(selected) != 20:
                    raise RuntimeError('Incomplete full stratum')
                complete_strata.append({'policy': p, 'size': size, 'nominal_drift': drift,
                    'randomization': randomization, 'n': len(selected),
                    'counts': {c: cc[c] for c in CATEGORIES}})
    extra = [r for r in population['queries'] if r['policy'] == 'metadata' and r['finite_extra']]
    regimes = Counter('fusion_only' if r['population_fusion_only'] else 'standalone_sufficient' for r in extra)
    facts = {
        'schema': 'mechanism-figure-v3', 'sources_sha256': sources,
        'scope': 'retrospective; same declared relaxation, conservative radius, low randomized budget',
        'threshold': .001, 'independent_worlds': 48, 'count_repetitions_per_world': 5,
        'policy_counts': policy_counts, 'metadata_size_drift_counts': stratum_counts,
        'complete_joint_strata': complete_strata, 'finite_extra_population_regimes': dict(regimes),
        'finite_extra_worlds': sorted({r['world_id'] for r in extra}),
        'finite_gap_maximum_metadata': finite['summary']['policies'][0]['max_optimum_minus_saved_lower'],
        'interpretation': 'The partition diagnoses the stated relaxation, not all causal models; strata are descriptive, not factorial interventions.',
        'numerical_admission': 'Exact finite optima and population-subset-finite certificates rule out a loose computational bound in all 720 queries.',
    }
    (PAPER/'mechanism_figure_data_v3.json').write_text(json.dumps(facts, indent=2)+'\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.12, 2.85), gridspec_kw={'width_ratios': [1, 1.07]})
    fig.subplots_adjust(left=.144, right=.987, bottom=.19, top=.72, wspace=.57)
    for ax, values, names, limit in (
        (axes[0], policy_counts, ['Metadata', 'R-trained\nheuristic', 'Robust\nselection'], 240),
        (axes[1], stratum_counts,
         [f"{x['size'].replace('x', '×')} / {x['nominal_drift']:g}" for x in stratum_counts], 40),
    ):
        left = [0]*len(values)
        for cat, color in zip(CATEGORIES, COLORS):
            widths = [v['counts'][cat] for v in values]
            ax.barh(range(len(values)), widths, left=left, height=.63,
                    color=color, edgecolor='white', linewidth=.6, zorder=3)
            for j, width in enumerate(widths):
                if width/limit >= .045:
                    ax.text(left[j]+width/2, j, str(width), ha='center', va='center',
                            fontsize=7.0, color='white' if color in (COLORS[0], COLORS[2]) else '#25313B')
                left[j] += width
        ax.set_yticks(range(len(names)), names, fontsize=7.7)
        ax.invert_yaxis(); ax.set_xlim(0, limit)
        ax.set_xticks([0, limit//2, limit]); ax.set_xlabel('Query instances', fontsize=7.8)
        ax.tick_params(axis='both', length=0, pad=4)
        ax.grid(axis='x', color='#DCE1E5', linewidth=.6, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_title('(a) Three policy families', loc='left', fontsize=8.4, weight='bold', pad=9)
    axes[1].set_title('(b) Metadata: size / drift', loc='left', fontsize=8.4, weight='bold', pad=9)
    fig.legend([Patch(facecolor=c) for c in COLORS], LABELS, loc='upper center',
               bbox_to_anchor=(.51, 1.015), ncol=2, frameon=False, fontsize=7.4,
               columnspacing=1.2, handlelength=1.2)
    fig.text(.54, .025, 'Lower randomized budget; fixed declarations. Repetitions within a world are dependent.',
             ha='center', fontsize=7.0, color='#3C4855')
    for ext in ('pdf', 'png'):
        fig.savefig(PAPER/f'figures/fresh_mechanism_v3.{ext}', dpi=220, bbox_inches='tight', pad_inches=.025)
    plt.close(fig)
    primary = policy_counts[0]['counts']
    robust = policy_counts[2]['counts']
    text = r'''% Generated from complete population AND exact finite-optimum evidence with independent audit.
\textbf{What prevents certification?} A retrospective diagnostic replaces sampling intervals by their population counterparts, keeping all saved queries, $\Lambda$, $\rho$, $T$, conditional-weight and joint-event restrictions. Exact matching primal--dual bounds establish both optima for all 720 queries; separate proofs show that all 240 finite fusion views strictly contain their population counterparts. No full-cell transport or stronger latent model is introduced.

All 240 primary true effects exceed $\delta$. Figure~\ref{fig:fresh} partitions them into CERTIFIED certified queries, POPULATION whose population constraints permit failure, and FINITE with $V_{\rm fin}\le\delta<V_{\rm pop}$. There is no computationally missed certification. The last group is limited by the current confidence construction; the diagnostic does not estimate needed sample sizes. In the $8\times8$ strata, 25 of 80 queries have this finite-width limitation; all 80 at $16\times16$ still fail under population constraints. Strata are descriptive, not matched interventions.

The six additional finite certifications occur in two zero-drift $2\times4$ worlds: four where a population standalone source suffices, and two where only fusion suffices. The population fusion-only regime spans seven worlds (35 repeated queries); only two of these 35 queries certify from finite data. This cross-classification is not a causal variance/identification decomposition. The robust learner selects the reference in REFERENCE/240 cases. The diagnosis separates situations where narrower intervals could help from those still blocked by the declared population relaxation; it does not establish impossibility for every causal method.
'''
    text = text.replace('CERTIFIED', str(primary['finite_sample_certified'])).replace(
        'POPULATION', str(primary['population_and_finite_relaxation_limit'])).replace(
        'FINITE', str(primary['finite_relaxation_limit'])).replace(
        'REFERENCE', str(robust['true_effect_not_above_margin']))
    (PAPER/'fresh_mechanism_results_v3.tex').write_text(text)
    print(json.dumps({'status': 'complete', 'queries': len(rows), 'source_files': len(sources),
                      'full_strata': len(complete_strata), 'primary': primary, 'extra_regimes': dict(regimes)}))


if __name__ == '__main__':
    main()
