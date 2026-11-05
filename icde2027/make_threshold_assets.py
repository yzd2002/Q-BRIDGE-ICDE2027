"""Make figures and tables only from the complete registered threshold study.

Three timing repetitions are descriptive, never confidence intervals. This
module reconciles measurements; the independent Fraction audit is separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('full_hot', 'threshold_hot', 'primal_pool', 'dual_pool', 'both_pools', 'threshold_hot_auto')
BASELINES = ('threshold_hot', 'threshold_hot_auto')
STRUCTURES = ('disjoint', 'mixed', 'connected')
STREAMS = ('nearby_nonproportional', 'unrelated')
LABELS = dict(zip(METHODS, ('Full Hot', 'Threshold Hot', 'Primal History', 'Dual History',
                            'Both Histories', 'Threshold Hot Auto')))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def close(a, b):
    return math.isclose(a, b, rel_tol=2e-10, abs_tol=2e-9)


def geometric(values):
    values = list(values)
    require(bool(values) and all(math.isfinite(v) and v > 0 for v in values), 'Invalid ratio')
    return math.exp(sum(math.log(v) for v in values) / len(values))


def spread(values):
    values = list(values)
    require(bool(values) and all(math.isfinite(v) for v in values), 'Invalid repetition')
    return dict(median=median(values), minimum=min(values), maximum=max(values), repeats=values)


def make_facts(summary_path, run):
    data = read(summary_path); manifest = read(run / 'manifest.json'); protocol = read(run / 'protocol.json')
    require(manifest['status'] == 'complete' and manifest['completed_conditions'] == 60, 'Study incomplete')
    require(data['manifest'] == manifest and data['protocol'] == protocol, 'Summary provenance differs')
    require(sha(run / 'protocol.json') == manifest['protocol_sha256'], 'Protocol changed')
    require([m['name'] for m in protocol['methods']] == list(METHODS), 'Plan identities differ')
    for name, expected in manifest['source_sha256'].items():
        require(sha(run / 'source_snapshot' / name) == expected, 'Snapshot changed: ' + name)
    expected = {('engine', f'd{d}_{s}_{q}_r{r}') for d, s, q, r in
                itertools.product((64, 512, 2048), STRUCTURES, STREAMS, range(3))}
    expected.update(('end_to_end', f'{design}_r{r}') for design, r in itertools.product(('uniform', 'skew'), range(3)))
    conditions = data['conditions']
    require(len(conditions) == 60 and {(c['stage'], c['name']) for c in conditions} == expected,
            'Missing, duplicate or unexpected condition')
    indexed = read(run / 'completed_conditions.json')
    require(len(indexed) == 60 and {(c['stage'], c['name']) for c in indexed} == expected, 'Index differs')
    valid = missing = 0
    totals = {m: Counter() for m in METHODS}; routes = {m: Counter() for m in METHODS}
    max_payload = {m: 0 for m in METHODS}; uncompressed = {m: 0 for m in METHODS}
    engine = {}; source = {}
    for condition in conditions:
        loc = run / condition['stage'] / condition['name']
        require({k: v for k, v in condition.items() if k != 'methods'} == read(loc / 'condition.json'),
                'Condition record differs')
        require(sha(loc / 'queries.npy') == condition['queries_sha256'], 'Queries changed')
        for method in METHODS:
            result = condition['methods'][method]
            require(result == read(loc / f'{method}_summary.json'), 'Measurement summary changed')
            require(result['queries'] == 100 and len(result['per_query']) == 100, 'Incomplete query trace')
            require(sha(loc / result['proof_file']) == result['proof_file_sha256'], 'Proof stream changed')
            producer = sum(result['input_stage_seconds'].values()) + sum(
                q['query_seconds'] + q['serialization_seconds'] for q in result['per_query'])
            receiver = result['persistent_replay']
            receiving = sum(receiver[k] for k in ('checker_initialization_seconds', 'decode_seconds', 'verify_seconds'))
            require(close(producer, result['producer_seconds']) and
                    close(receiving, receiver['total_consumer_seconds']) and
                    close(producer + receiving, result['joint_seconds']), 'Cost arithmetic differs')
            require(receiver['valid_proofs'] + receiver['no_bound_responses'] == 100, 'Proof census incomplete')
            valid += receiver['valid_proofs']; missing += receiver['no_bound_responses']
            for q in (1, 10, 100):
                require(close(result['joint_prefixes'][str(q)], result['prefixes'][str(q)] + receiver['prefixes'][str(q)]),
                        'Prefix arithmetic differs')
            counts = Counter('true' if r['decision'] is True else 'false' if r['decision'] is False else 'unknown'
                             for r in result['per_query'])
            require(dict(counts) == result['decisions'] == receiver['decisions'], 'Decision census differs')
            totals[method].update(counts); routes[method].update(result['routes'])
            uncompressed[method] += result['uncompressed_bytes']
            memory = result['final_memory']
            if memory is not None:
                max_payload[method] = max(max_payload[method], memory['retained_exact_state_payload_bytes'])
        key = ((condition['dimension'], condition['structure'], condition['stream'])
               if condition['stage'] == 'engine' else (condition['stream'],))
        target = engine if condition['stage'] == 'engine' else source
        target.setdefault(key, []).append(condition)
    require(valid + missing == 36000 and data['proof_totals'] == {'valid': valid, 'missing': missing}, 'Response total differs')
    comparisons = {}; methods_table = []
    for method in METHODS:
        cell = {'method': method, 'label': LABELS[method], 'engine': {}, 'source_to_answer': {},
                'decisions': dict(totals[method]), 'routes': dict(routes[method]),
                'max_exact_state_python_bytes': max_payload[method], 'uncompressed_response_bytes': uncompressed[method]}
        for baseline in BASELINES:
            repeated = []
            for rep in range(3):
                rows = [c for c in conditions if c['stage'] == 'engine' and c['repeat'] == rep]
                require(len(rows) == 18, 'Incomplete engine repetition')
                repeated.append(geometric(c['methods'][method]['joint_seconds'] /
                                          c['methods'][baseline]['joint_seconds'] for c in rows))
            cell['engine'][baseline] = spread(repeated)
            if method == 'both_pools':
                loss = sum(c['comparisons'][baseline][method]['lost_decided_answers'] for c in conditions)
                paired = [(a['decision'], b['decision']) for c in conditions for a, b in
                          zip(c['methods'][method]['per_query'], c['methods'][baseline]['per_query'])]
                gain = sum(a is not None and b is None for a, b in paired)
                require(loss == sum(a is None and b is not None for a, b in paired),
                        'Paired answer-loss census differs')
                require(not any(a is not None and b is not None and a != b for a, b in paired),
                        'Contradictory threshold answers')
                support = all(v <= protocol['primary']['maximum_ratio'] for v in repeated) and loss == 0
                old = data['primary'][baseline]
                require(all(close(a, b) for a, b in zip(repeated, old['joint_geometric_ratios_by_repeat'])) and
                        loss == old['lost_decided_answers_all_conditions'] and support == old['engineering_target_supported'],
                        'Primary comparison differs')
                comparisons[baseline] = {'ratios': repeated, 'median': median(repeated),
                                         'lost_decided_answers': loss, 'gained_decided_answers': gain,
                                         'engineering_target_supported': support}
        for design in ('uniform', 'skew'):
            ordered = sorted(source[design,], key=lambda c: c['repeat'])
            require(len(ordered) == 3, 'Incomplete source repetition')
            cell['source_to_answer'][design] = {
                'joint_seconds': spread(c['methods'][method]['joint_seconds'] for c in ordered),
                'producer_seconds': spread(c['methods'][method]['producer_seconds'] for c in ordered),
                'ratios': {b: spread(c['methods'][method]['joint_seconds'] / c['methods'][b]['joint_seconds']
                                    for c in ordered) for b in BASELINES}}
        methods_table.append(cell)
    grid = []
    for (d, structure, stream), group in sorted(engine.items()):
        ordered = sorted(group, key=lambda c: c['repeat']); require(len(ordered) == 3, 'Incomplete grid repeat')
        grid.append({'dimension': d, 'structure': structure, 'stream': stream,
                     'comparisons': {b: spread(c['methods']['both_pools']['joint_seconds'] /
                                              c['methods'][b]['joint_seconds'] for c in ordered) for b in BASELINES}})
    return {'summary_sha256': sha(summary_path), 'manifest_sha256': sha(run / 'manifest.json'),
            'protocol_sha256': manifest['protocol_sha256'], 'complete_conditions': 60,
            'responses': 36000, 'proof_totals': {'valid': valid, 'missing': missing},
            'primary': comparisons, 'both_comparisons_supported': all(v['engineering_target_supported'] for v in comparisons.values()),
            'grid': grid, 'methods': methods_table,
            'scope': 'Measurement reconciliation; independent numerical audit is separate. Median and min/max over three paired repeats, not confidence intervals.'}


def plot(facts, paper):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42,
                         'text.color': '#25313B', 'axes.labelcolor': '#25313B', 'savefig.dpi': 220})
    cells = {(g['dimension'], g['structure'], g['stream']): g for g in facts['grid']}
    values = np.array([[cells[d, s, q]['comparisons'][b]['median'] for s, q in itertools.product(STRUCTURES, STREAMS)]
                       for b, d in itertools.product(BASELINES, (64, 512, 2048))])
    logs = np.log(values); extent = max(.1, float(np.max(np.abs(logs))))
    colors = LinearSegmentedColormap.from_list('threshold_ratio', ['#008B72', '#F7F8F7', '#D58400'])
    fig, ax = plt.subplots(figsize=(7.15, 3.25)); fig.subplots_adjust(left=.205, right=.86, top=.79, bottom=.22)
    im = ax.imshow(logs, aspect='auto', cmap=colors, norm=TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent))
    for r in range(6):
        for c in range(6):
            ax.text(c, r, f'{values[r,c]:.2f}', ha='center', va='center', fontsize=8.2,
                    color='white' if abs(logs[r,c]) > .7 * extent else '#25313B')
    ax.set_xticks(range(6), ['Nearby', 'Unrelated'] * 3, fontsize=7.6)
    ax.set_yticks(range(6), [f'{"Dual" if b == BASELINES[0] else "Auto"}, {d:,} cells'
                            for b, d in itertools.product(BASELINES, (64, 512, 2048))], fontsize=8)
    for pos in (.5, 1.5, 2.5, 3.5, 4.5):
        ax.axhline(pos, color='white', lw=2 if pos == 2.5 else .7)
        ax.axvline(pos, color='white', lw=2 if pos in (1.5, 3.5) else .7)
    for center, title in zip((.5, 2.5, 4.5), ('Disjoint', 'Mixed', 'Connected')):
        ax.text(center, -.8, title, ha='center', va='bottom', fontweight='bold', fontsize=9)
    r, c = np.unravel_index(np.argmax(values), values.shape)
    ax.add_patch(Rectangle((c-.49, r-.49), .98, .98, fill=False, edgecolor='#25313B', lw=1))
    ax.tick_params(length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(im, cax=fig.add_axes([.89, .23, .018, .55]))
    ticks = [-extent, 0, extent]
    bar.set_ticks(ticks, labels=[f'{math.exp(x):.2g}' for x in ticks])
    bar.set_label('Both Histories / threshold control', fontsize=7.8)
    fig.suptitle('Complete threshold comparison at Q = 100', y=.97, fontsize=11, fontweight='bold')
    fig.text(.205, .055, 'Joint producer + receiver: green < 1, orange > 1; outlined cell is slowest', fontsize=8)
    target = paper / 'figures'; target.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'png'):
        fig.savefig(target / f'threshold_boundary_v2.{suffix}')
    plt.close(fig)


def table(facts):
    lines = [r'\begingroup\fontsize{8}{9.4}\selectfont\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}', r'\toprule',
             r'Plan & Engine / Dual & Engine / Auto & Uniform (s) & Skew (s) & Decided / 6,000\\', r'\midrule']
    for row in facts['methods']:
        ratio = [f"{row['engine'][b]['median']:.3f} [{row['engine'][b]['minimum']:.3f}, {row['engine'][b]['maximum']:.3f}]"
                 for b in BASELINES]
        times = [f"{row['source_to_answer'][d]['joint_seconds']['median']:.3f}" for d in ('uniform', 'skew')]
        decided = sum(row['decisions'].get(k, 0) for k in ('true', 'false'))
        lines.append(' & '.join([row['label'], *ratio, *times, f'{decided:,}']) + r'\\')
    return '\n'.join([*lines, r'\bottomrule', r'\end{tabular*}', r'\endgroup', ''])


def prose(facts):
    verdict = ('The registered broad engineering target is supported.' if facts['both_comparisons_supported'] else
               'The registered broad engineering target is not supported.')
    row = next(r for r in facts['methods'] if r['method'] == 'both_pools')
    sentences = []
    for b, label in zip(BASELINES, ('Threshold Hot', 'Threshold Hot Auto')):
        vals = ', '.join(f'{v:.3f}' for v in facts['primary'][b]['ratios'])
        sentences.append(f'Against {label}, the three engine-wide geometric-mean joint-cost ratios are {vals}.')
    history = row['routes'].get('historical_primal', 0) + row['routes'].get('historical_dual', 0)
    last = row['routes'].get('last_primal', 0) + row['routes'].get('last_dual', 0)
    counts = row['decisions']
    availability = ' '.join(
        f"Compared with {label}, it resolves {facts['primary'][b]['gained_decided_answers']} "
        f"previously unknown answers but leaves {facts['primary'][b]['lost_decided_answers']} "
        'baseline-decided answers unknown.'
        for b, label in zip(BASELINES, ('Threshold Hot', 'Threshold Hot Auto')))
    return rf'''% Generated from the entire threshold follow-up. Independent numerical audit is separate.
\textbf{{Threshold-aware comparison.}} All 54 engine and six reused-source conditions completed. {' '.join(sentences)} {verdict} Table~\ref{{tab:threshold-plans}} includes all six plans; Fig.~\ref{{fig:threshold-boundary}} covers every engine cell. All ratios charge first use and the persistent receiver. Source-to-answer costs include aggregation and materialization on the unchanged encoded inputs.

\begin{{table*}}[t]
\centering
\caption{{Six plans at $Q=100$. Engine columns give median [min, max] of three repeat-wise geometric-mean joint-cost ratios over all 18 cells. Dual/Auto denote the Threshold Hot controls. Uniform/Skew give median source-to-answer joint seconds. Decided counts combine true and false robust-threshold answers; they do not count beneficial policies.}}
\label{{tab:threshold-plans}}\input{{threshold_plans_table}}
\end{{table*}}

\begin{{figure*}}[t]
\centering\includegraphics[width=\textwidth]{{figures/threshold_boundary_v2.pdf}}
\caption{{Both Histories / Threshold Hot (Dual) or Threshold Hot Auto (Auto) in every engine cell. Entries are medians of three paired joint-cost ratios on one logarithmic color scale. The target applies to full-grid geometric means.}}
\label{{fig:threshold-boundary}}
\end{{figure*}}

\textbf{{What the histories contribute.}} Both Histories returns {counts.get('true',0):,} true, {counts.get('false',0):,} false and {counts.get('unknown',0):,} unknown answers. Earlier history serves {history:,}/6,000 queries; the common last result serves {last:,}/6,000. Initial-witness and box paths are recorded separately. {availability} History changes answer availability slightly without the registered cost reduction. False means a checked counterexample within the view, as defined in Section~\ref{{sec:threshold}}. All 36,000 responses, including unknowns and {facts['proof_totals']['missing']:,} no-bound refusals, remain in the analysis.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=ROOT / 'runs/icde2027/threshold_workload_summary_v2/summary.json')
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/icde2027/threshold_workload_v2')
    parser.add_argument('--paper', type=Path, default=ROOT / 'icde2027/paper')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(); facts = make_facts(args.summary, args.run)
    if args.check_only:
        print('Complete threshold measurements reconciled; no figures written.'); return
    plot(facts, args.paper)
    (args.paper / 'threshold_plans_table.tex').write_text(table(facts))
    (args.paper / 'threshold_results_v2.tex').write_text(prose(facts))
    (args.paper / 'threshold_figure_data.json').write_text(json.dumps(facts, indent=2, allow_nan=False) + '\n')
    print('Generated threshold figure, six-plan table, measured prose and provenance.')


if __name__ == '__main__':
    main()
