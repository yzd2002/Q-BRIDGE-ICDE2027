"""Publish only the complete, independently audited matched-primary records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/icde2027/primary_aligned_v1'
AUDIT = ROOT/'runs/icde2027/audit/primary_aligned_result_audit.json'
PAPER = ROOT/'icde2027/paper'
METHODS = ('original_prepared', 'original_hot', 'threshold_hot', 'threshold_hot_auto', 'both_histories')
LABELS = ('Prepared', 'Hot LP', 'Threshold Hot', 'Threshold Auto', 'Both Histories')
COLORS = ('#008B72', '#707A84', '#0072B2', '#CC7B00', '#875FA5')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = json.loads(AUDIT.read_text())
    if not all(audit.get(k) is True for k in ('complete', 'passed', 'independent_review')):
        raise RuntimeError('A completed independent result audit is required')
    for relative, digest in audit['evidence_sha256'].items():
        if sha(ROOT/relative) != digest:
            raise RuntimeError(f'Audited evidence changed: {relative}')
    manifest = json.loads((RUN/'manifest.json').read_text())
    summary = json.loads((RUN/'summary.json').read_text())
    if manifest['status'] != 'complete' or summary['complete'] is not True or summary['responses'] != 15000:
        raise RuntimeError('The entire five-plan, three-stream experiment is required')
    conditions = json.loads((RUN/'completed_conditions.json').read_text())
    if len(conditions) != 3:
        raise RuntimeError('Not all original streams are present')
    sources = {str(p.relative_to(ROOT)): sha(p) for p in
               (RUN/'manifest.json', RUN/'protocol.json', RUN/'summary.json',
                RUN/'completed_conditions.json', AUDIT)}
    cost_path = ROOT/'runs/icde2027/audit/primary_aligned_cost_mechanism.json'
    cost_audit = json.loads(cost_path.read_text())
    if cost_audit['source_audit_sha256'] != sha(AUDIT):
        raise RuntimeError('Cost interpretation is bound to a different result audit')
    for relative, digest in cost_audit['evidence_sha256'].items():
        if sha(ROOT/relative) != digest:
            raise RuntimeError(f'Cost evidence changed: {relative}')
    sources[str(cost_path.relative_to(ROOT))] = sha(cost_path)
    rows = []
    for condition in conditions:
        directory = RUN/condition['name']
        for method in METHODS:
            path = directory/f'{method}_summary.json'
            sources[str(path.relative_to(ROOT))] = sha(path)
            data = json.loads(path.read_text())
            if data['queries'] != 1000:
                raise RuntimeError('A method stream was shortened')
            rows.append({'repeat': condition['repeat'], 'method': method,
                'producer_seconds': data['producer_seconds'],
                'receiver_seconds': data['persistent_replay']['total_consumer_seconds'],
                'joint_seconds': data['joint_seconds'], 'prefixes': data['prefixes'],
                'joint_prefixes': data['joint_prefixes'], 'decisions': data['decisions'],
                'routes': data['routes'], 'uncompressed_bytes': data['uncompressed_bytes'],
                'compressed_bytes': data['compressed_bytes']})
    costs = []
    for method, label in zip(METHODS, LABELS):
        selected = [r for r in rows if r['method'] == method]
        costs.append({'method': method, 'label': label,
            **{k: median(r[k] for r in selected) for k in
               ('producer_seconds', 'receiver_seconds', 'joint_seconds')},
            'mean_uncompressed_service_bytes_per_query': sum(r['uncompressed_bytes'] for r in selected)/3000,
            'mean_gzip_service_bytes_per_query': sum(r['compressed_bytes'] for r in selected)/3000})
    contrasts = {}
    for baseline in ('threshold_hot', 'threshold_hot_auto'):
        ratios = [next(r for r in rows if r['method'] == 'original_prepared' and r['repeat'] == rep)['joint_seconds']/
                  next(r for r in rows if r['method'] == baseline and r['repeat'] == rep)['joint_seconds'] for rep in range(3)]
        if ratios != summary['primary_contrasts'][baseline]['joint_ratios']:
            raise RuntimeError('Primary ratios differ from full audited result')
        contrasts[baseline] = {'joint_ratios': ratios, 'median': median(ratios),
                               'all_at_most_point8': all(r <= .8 for r in ratios)}
    target = all(r['all_at_most_point8'] for r in contrasts.values())
    if target != audit['engineering_target_supported'] or target != summary['engineering_target_supported']:
        raise RuntimeError('Do not change the registered engineering criterion')
    facts = {'schema': 'primary-aligned-figure-v3', 'sources_sha256': sources,
             'scope': 'Matched repeat on three inspected constructed streams, not a fresh application workload.',
             'rows': rows, 'median_costs': costs, 'primary_contrasts': contrasts,
             'mean_cost_mechanism': cost_audit['means_per_1000_queries'],
             'engineering_target_supported': target,
             'timing_exclusions': ['compressed-file I/O', 'network', 'source-table ingestion', 'native diagnostic export'],
             'uncertainty_display': 'min/max of three paired stream-specific measurements; not a confidence interval'}
    (PAPER/'primary_aligned_figure_data_v3.json').write_text(json.dumps(facts, indent=2)+'\n')
    lines = [r'\begingroup\fontsize{8}{9.5}\selectfont\setlength{\tabcolsep}{3.2pt}',
             r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrr@{}}',
             r'\toprule', r'Plan & Producer & Receiver & Joint\\', r'\midrule']
    for r in costs:
        lines.append(f"{r['label']} & {r['producer_seconds']:.2f} & {r['receiver_seconds']:.2f} & {r['joint_seconds']:.2f}\\\\")
    lines += [r'\bottomrule', r'\end{tabular*}', r'\endgroup']
    (PAPER/'primary_aligned_table_v3.tex').write_text('\n'.join(lines)+'\n')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.12, 2.7), gridspec_kw={'width_ratios': [1.1, 1]})
    fig.subplots_adjust(left=.085, right=.98, bottom=.20, top=.79, wspace=.31)
    q = [1, 10, 100, 1000]
    for method, label, color, marker, style in zip(METHODS, LABELS, COLORS, ['o','s','^','v','D'], ['-','--','-.',':','--']):
        values = [[r['joint_prefixes'][str(n)] for r in rows if r['method'] == method] for n in q]
        centers = [median(v) for v in values]
        axes[0].plot(q, centers, label=label, color=color, marker=marker, markersize=3.4,
                     linewidth=1.15, linestyle=style)
        axes[0].fill_between(q, [min(v) for v in values], [max(v) for v in values],
                             color=color, alpha=.09, linewidth=0)
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xticks(q); axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[0].set_ylabel('Cumulative joint time (s)', fontsize=7.8)
    axes[0].set_xlabel('Queries served, including first use', fontsize=7.8)
    axes[0].set_title('(a) All five serving plans', loc='left', weight='bold', fontsize=8.2, pad=8)
    for i, (baseline, color, label) in enumerate([
        ('threshold_hot', COLORS[2], 'Prepared / Threshold Hot'),
        ('threshold_hot_auto', COLORS[3], 'Prepared / Threshold Auto')]):
        ratios = contrasts[baseline]['joint_ratios']
        positions = [j+(i-.5)*.17 for j in range(3)]
        axes[1].scatter(positions, ratios, color=color, marker='o' if i == 0 else 's', s=23, label=label, zorder=3)
        for xpos, value in zip(positions, ratios):
            axes[1].annotate(f'{value:.3f}', (xpos, value), xytext=(0, -12 if i == 0 else 7),
                             textcoords='offset points', ha='center', fontsize=7.0, color=color)
    axes[1].axhline(.8, color='#7F4640', linestyle='--', linewidth=.9)
    axes[1].text(-.37, .802, 'Registered limit', ha='left', va='bottom', color='#7F4640', fontsize=7.0)
    axes[1].set_xlim(-.4,2.4); axes[1].set_ylim(.71,.832)
    axes[1].set_xticks([0,1,2], ['Stream 1','Stream 2','Stream 3'])
    axes[1].set_yticks([.72,.76,.80]); axes[1].set_ylabel('Prepared / control joint time', fontsize=7.8)
    axes[1].set_title('(b) Each 1,000-query comparison', loc='left', weight='bold', fontsize=8.2, pad=8)
    axes[1].set_xlabel('All six ratios must be at most 0.8', fontsize=7.8)
    axes[1].legend(loc='upper left', frameon=False, fontsize=6.8,
                   bbox_to_anchor=(0, -.31), borderaxespad=0)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', color='#E2E7EA', linewidth=.5, zorder=0)
        ax.tick_params(labelsize=7.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=5,
               frameon=False, fontsize=7.0, handlelength=2.4, columnspacing=.9,
               bbox_to_anchor=(.5, 1.015))
    for ext in ('pdf', 'png'):
        fig.savefig(PAPER/f'figures/primary_aligned_v3.{ext}', dpi=220, bbox_inches='tight', pad_inches=.025)
    plt.close(fig)
    hot = ', '.join(f'{v:.3f}' for v in contrasts['threshold_hot']['joint_ratios'])
    auto = ', '.join(f'{v:.3f}' for v in contrasts['threshold_hot_auto']['joint_ratios'])
    prose = r'''% Generated only after the full independent 15,000-response audit.
\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figures/primary_aligned_v3.pdf}
\caption{Matched execution on the original 512-cell, 1,000-query streams. (a) Joint producer and persistent-receiver time, including initialization; lines are medians and bands are minima/maxima over three paired streams, not confidence intervals. Both axes are logarithmic. (b) All six primary joint-cost ratios; one exceeds the registered $.8$ limit. Every plan answers every query true in this constructed workload.}
\label{fig:primary-aligned}
\end{figure*}
\begin{table}[t]
\centering
\caption{Matched 1,000-query costs in seconds: medians of three paired streams. Joint is added within each stream before taking its median; displayed component medians need not sum to it.}
\label{tab:primary-aligned}\input{primary_aligned_table_v3}
\end{table}

\textbf{Matched threshold-aware result.} Prepared's joint-cost ratios are HOTRATIOS against Threshold Hot and AUTORATIOS against Auto (Fig.~\ref{fig:primary-aligned}, Table~\ref{tab:primary-aligned}). All six comparisons favor Prepared on these sequences; the requirement that every ratio be at most $.8$ still fails. All five plans return 1,000 true answers on each sequence, with no lost decided answer or contradiction. An independent rational implementation validates all 15,000 lower proofs and their exact query/view bindings; no upper proof is returned here. This positive-query workload therefore supports a local cost comparison, not a test of negative decisions or near-threshold ambiguity. The broader threshold grid below supplies those cases. The three sequences do not support a significance claim.

\textbf{Same candidate, different checking work.} Prepared and all three threshold variants solve once per stream, then reuse one dual 999 times. Prepared ranks floating proposals before exact evaluation; the threshold variants first exactly evaluate the zero-dual box bound, then check the last dual, recording 2,000 exact candidate checks. Mean producer verification is PREPVERIFY\,s for Prepared versus HOTVERIFY/AUTOVERIFY\,s for the two controls; receiver verification is similar. Every threshold plan also pays for one rejected primal proposal, without losing a positive answer. These measured implementation paths explain where work occurs; no controlled ablation assigns the entire saving to one operation. In particular, the eager box check is not required by two-sided semantics. The comparison demonstrates local implementation savings, not an inherent advantage over threshold-aware execution or a benefit from a larger history.
'''
    prose = prose.replace('HOTRATIOS',hot).replace('AUTORATIOS',auto)
    for token, method in [('PREPVERIFY','original_prepared'), ('HOTVERIFY','threshold_hot'),
                          ('AUTOVERIFY','threshold_hot_auto')]:
        prose = prose.replace(token, f"{cost_audit['means_per_1000_queries'][method]['phase_seconds']['verify_seconds']:.2f}")
    (PAPER/'primary_aligned_results_v3.tex').write_text(prose)
    print(json.dumps({'complete': True, 'methods':len(costs), 'streams':len(conditions),
                      'responses':15000, 'ratios':contrasts, 'registered_target':target}))


if __name__ == '__main__':
    main()
