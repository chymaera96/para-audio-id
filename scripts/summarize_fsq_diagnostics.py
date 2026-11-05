#!/usr/bin/env python3
"""Summarize a complete FSQ diagnostic run, including paired-query uncertainty."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def paired_interval(candidate, baseline, repetitions=4000):
    candidate, baseline = np.asarray(candidate), np.asarray(baseline)
    if candidate.shape != baseline.shape or candidate.ndim != 1 or not len(candidate):
        raise ValueError('Paired observations must be nonempty equal-length vectors')
    delta = candidate.astype(float) - baseline.astype(float)
    indices = np.random.default_rng(1337).integers(0, len(delta), size=(repetitions, len(delta)))
    low, high = np.quantile(delta[indices].mean(1), [.025, .975])
    return {'difference_pp': float(100 * delta.mean()),
            'ci95_low_pp': float(100 * low), 'ci95_high_pp': float(100 * high)}


def read_queries(path):
    rows = list(csv.DictReader(path.open()))
    result = {}
    for row in rows:
        key = row['stage'], int(row['bins']), row['condition']
        result.setdefault(key, []).append(row)
    for key, values in result.items():
        values.sort(key=lambda row: int(row['query_index']))
    return result


def summarize(directory):
    results = json.loads((directory / 'results.json').read_text())
    models = results['models']
    query_rows = {name: read_queries(directory / f'{name}-per-query.csv') for name in models}
    conditions = ['clean', 'noise_0db', 'noise_5db', 'rir', 'noise_rir_0db',
                  'offset_-80ms', 'offset_+80ms', 'rir_offset_-40ms',
                  'rir_offset_+40ms', 'context_left_1040ms', 'other_excerpt_same_recording']
    lines = ['# Comparative FSQ analysis', '',
             f"Protocol: `{results.get('protocol')}`. GPU: {results['gpu']}.", '',
             'All values below are fixed-readout diagnostic results, not PSI performance. '
             'All tc3--tc10 checkpoints are at 32K; tc2 includes the actual 28K PSI tokenizer '
             'and a 32K snapshot. tc1 at 15K is an unequal-budget reference.', '',
             '## Post-quantization Top-1: eight temporal bins', '',
             '| Model | Frames/s | Groups | Tokens/2s | Noise 0dB | Noise 5dB | RIR | Noise+RIR 0dB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, model in models.items():
        a = model['retrieval']['post']['bins_8']
        values = [100*a[c]['full']['top1'] for c in conditions[1:5]]
        lines.append(f"| {name} | {model['frame_rate']:g} | {model['groups']} | "
                     f"{round(2*model['frame_rate']*model['groups'])} | "
                     + ' | '.join(f'{v:.1f}' for v in values) + ' |')
    lines += ['', '## Readout sensitivity: global mean (Top-1)', '',
              '| Model | Noise 0dB | RIR | Noise+RIR 0dB | Other excerpt |',
              '|---|---:|---:|---:|---:|']
    for name, model in models.items():
        a = model['retrieval']['post']['bins_1']
        lines.append('| '+name+' | '+' | '.join(f"{100*a[c]['full']['top1']:.1f}" for c in
                     ['noise_0db', 'rir', 'noise_rir_0db', 'other_excerpt_same_recording'])+' |')
    lines += ['', '## Hard negatives and false matches: noise+RIR, eight bins', '',
              'The 0.1% FPR target is calibrated on half the query identities; actual test FPR is reported alongside recall. '
              'Models need not achieve identical held-out FPR. Do not interpret differences in recall alone as an equal-FPR result.', '',
              '| Model | Median hardest-negative margin | Recall at threshold % | Actual test FPR % | Top-1: smallest/500/full gallery % |',
              '|---|---:|---:|---:|---|']
    for name, model in models.items():
        a = model['retrieval']['post']['bins_8']['noise_rir_0db']
        m = a['full']
        sizes = sorted((int(k) for k in a if k != 'full'))
        scores = '/'.join(f"{100*a[str(k)]['top1']:.1f}" for k in sizes)
        lines.append(f"| {name} | {m['median_margin']:.4f} | {100*m['test_tpr']:.1f} | "
                     f"{100*m['test_fpr']:.3f} | {scores} |")
    comparisons = [('tc2-32k', 'tc2-28k'), ('tc4-32k', 'tc3-32k'),
                   ('tc6-32k', 'tc4-32k'), ('tc7-32k', 'tc3-32k'),
                   ('tc7-32k', 'tc4-32k'), ('tc8-32k', 'tc7-32k'),
                   ('tc8-32k', 'tc4-32k'), ('tc9-32k', 'tc4-32k'),
                   ('tc10-32k', 'tc8-32k'), ('tc10-32k', 'tc9-32k'),
                   ('tc10-32k', 'tc4-32k')]
    paired = []
    lines += ['', '## Paired comparisons (percentage-point differences)', '',
              '95% percentile bootstrap intervals resample query identities. They describe this query sample '
              'and fixed gallery, not variation across training seeds or a larger catalogue. '
              'Intervals are not adjusted for multiple comparisons. Candidate minus baseline.', '',
              '| Candidate vs baseline | Noise 0dB | RIR | Noise+RIR |',
              '|---|---|---|---|']
    for candidate, baseline in comparisons:
        if candidate not in models or baseline not in models:
            continue
        cells = []
        for c in ['noise_0db', 'rir', 'noise_rir_0db']:
            key = ('post', 8, c)
            a, b = query_rows[candidate][key], query_rows[baseline][key]
            if [r['track_id'] for r in a] != [r['track_id'] for r in b]:
                raise ValueError('Cannot pair different query identities')
            interval = paired_interval([int(r['rank']) == 1 for r in a],
                                       [int(r['rank']) == 1 for r in b])
            paired.append({'candidate': candidate, 'baseline': baseline, 'condition': c, **interval})
            cells.append(f"{interval['difference_pp']:+.1f} [{interval['ci95_low_pp']:+.1f}, {interval['ci95_high_pp']:+.1f}]")
        lines.append('| '+candidate+' vs '+baseline+' | '+' | '.join(cells)+' |')
    lines += ['', '## Quantization damage: post minus pre Top-1 (pp)', '',
              '| Model | Noise 0dB | RIR | Noise+RIR |', '|---|---:|---:|---:|']
    for name, model in models.items():
        a, b = model['retrieval']['post']['bins_8'], model['retrieval']['pre']['bins_8']
        lines.append('| '+name+' | '+' | '.join(f"{100*(a[c]['full']['top1']-b[c]['full']['top1']):+.1f}"
                     for c in ['noise_0db', 'rir', 'noise_rir_0db'])+' |')
    lines += ['', '## Offset/context stability: post-quantization Top-1 %', '',
              '| Model | -80ms | +80ms | RIR -40ms | RIR +40ms | Aligned extra context | Other excerpt |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, model in models.items():
        a = model['retrieval']['post']['bins_8']
        lines.append('| '+name+' | '+' | '.join(f"{100*a[c]['full']['top1']:.1f}" for c in conditions[5:])+' |')
    lines += ['', '## Redundancy and diversity', '',
              'Effective symbols are marginal per-group values, not total distinguishable recording count. '
              'MI excess subtracts one shuffled finite-sample control; it is not an unbiased capacity estimate.', '',
              '| Model | Mean effective symbols/group | Mean excess inter-group MI (bits) | Coordinate correlation at 80ms |',
              '|---|---:|---:|---:|']
    for name, model in models.items():
        d = model['redundancy']
        mi = d['group_mutual_information']
        values = np.mean([x['excess_mi_bits'] for x in mi]) if mi else 0
        lines.append(f"| {name} | {np.mean(d['effective_symbols_per_group']):.1f} | {values:.3f} | "
                     f"{d['coordinate_correlation_by_lag_ms']['80']:.3f} |")
    lines += ['', '## Latency and kernel equivalence', '',
              'FP32 ordinary-kernel timings exclude audio-file decoding. Host waveform timings include host-to-device transfer. '
              'Full timings and deterministic baselines are in REPORT.md/results.json.', '',
              '| Model | Host waveform B=1 ms | Host waveform B=16 ms | Device waveform B=160 ms | Symbol mismatches % |',
              '|---|---:|---:|---:|---:|']
    for name, model in models.items():
        def ms(batch, host):
            return next(r['median_batch_ms'] for r in model['latency'] if not r['deterministic']
                        and r['batch'] == batch and r['includes'].startswith('host' if host else 'mel'))
        eq = model['kernel_equivalence'].values()
        rate = 100*sum(r['mismatch_count'] for r in eq)/sum(r['symbol_count'] for r in eq)
        lines.append(f'| {name} | {ms(1, True):.2f} | {ms(16, True):.2f} | {ms(160, False):.2f} | {rate:.4f} |')
    (directory / 'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    (directory / 'paired-comparisons.json').write_text(json.dumps(paired, indent=2))
    # Flatten every measured condition, stage, readout and gallery size, not just
    # the headline subset above; this supports independent re-analysis.
    flat = []
    for name, model in models.items():
        for stage, readouts in model['retrieval'].items():
            for bins, conditions_by_name in readouts.items():
                for c, sizes in conditions_by_name.items():
                    for size, values in sizes.items():
                        if size != 'full':
                            flat.append({'model': name, 'stage': stage, 'readout': bins,
                                         'condition': c, 'gallery_size': int(size), **values})
    with (directory / 'all-metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(f'Wrote {directory / "COMPARISON.md"}; {len(flat)} metric rows; {len(paired)} paired comparisons')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)
