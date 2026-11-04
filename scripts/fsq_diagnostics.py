#!/usr/bin/env python3
"""Read-only FSQ diagnostics, separate from training and PSI evaluation.

Uses held-out tokenizer identities, a fixed clean-excerpt gallery and a fixed
non-learned cosine readout. Scores diagnose representation properties, NOT PSI
accuracy. No FAISS, sequence matching, learned readout, or training is involved.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from para_audio_id.audio import load_audio
from para_audio_id.audio_lm.fsq import FrozenFSQTokenizer
from para_audio_id.audio_lm.fsq_data import PairedCrops, assets, read_source
from para_audio_id.audio_lm.noise import stable_uint64
from para_audio_id.audio_lm.rir import convolve_full_wet


@contextmanager
def kernels(deterministic):
    previous = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=deterministic):
            yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_inputs(cfg, split, queries, gallery_size, workers):
    """Prepare identical audio ONCE for every checkpoint; never edit the split."""
    if queries < 4 or queries > len(split['monitor']) or gallery_size < queries:
        raise ValueError('Need 4 <= queries <= monitor count and gallery >= queries')
    monitor = split['monitor']
    key_by_id = dict(zip((r['track_id'] for r in monitor), split['monitor_crop_keys']))
    forbidden = {r['track_id'] for r in split['training'] + monitor}
    others = [json.loads(s) for s in Path(cfg['data']['catalogue']).open() if s.strip()]
    others = sorted((r for r in others if r['track_id'] not in forbidden),
                    key=lambda r: stable_uint64(cfg['seed'], r['track_id'], 'fsq-diagnostic'))
    candidates = monitor + others
    if len(candidates) < gallery_size:
        raise ValueError('Insufficient disjoint gallery identities')
    audit = []

    def read(record):
        key = key_by_id.get(record['track_id'], stable_uint64(cfg['seed'], record['track_id'], 'gallery'))
        try:
            context, start, _ = read_source(cfg, record, key)
            return record, key, context, start
        except (ValueError, RuntimeError, OSError) as exc:
            return record, str(exc)

    selected = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Small chunks avoid decoding the entire 100K catalogue speculatively.
        for begin in range(0, len(candidates), 64):
            for result in pool.map(read, candidates[begin:begin + 64]):
                if len(result) == 2:
                    record, error = result
                    if record['track_id'] in {r['track_id'] for r in monitor[:queries]}:
                        raise RuntimeError(f'Fixed query failed: {record["track_id"]}: {error}')
                    audit.append({'track_id': record['track_id'], 'error': error})
                else:
                    selected.append(result)
            if len(selected) >= gallery_size:
                break
    selected = selected[:gallery_size]
    if len(selected) != gallery_size:
        raise RuntimeError('Not enough readable gallery crops')
    assert [r[0]['track_id'] for r in selected[:queries]] == [r['track_id'] for r in monitor[:queries]]
    sr = cfg['data']['sample_rate']
    samples = round(sr * cfg['data']['crop_duration'])
    references = np.stack([r[2][-samples:] for r in selected])
    noise, rir = assets(cfg)
    zero_cfg = deepcopy(cfg)
    zero_cfg['data'].update(snr_bins=[[0, 0]] * 4, exact_zero_fraction_in_first_bin=0)
    five_cfg = deepcopy(zero_cfg)
    five_cfg['data']['snr_bins'] = [[5, 5]] * 4
    paired0 = PairedCrops(zero_cfg, monitor, noise, rir, monitor=True)
    paired5 = PairedCrops(five_cfg, monitor, noise, rir, monitor=True)
    offsets = [-80, -40, -20, -10, 10, 20, 40, 80]

    def query_inputs(row):
        record, key, context, start = row
        clean = context[-samples:]
        values = {'clean': clean,
                  'noise_0db': paired0.pair(record, key, 1, 0)['degraded'].numpy(),
                  'noise_5db': paired5.pair(record, key, 1, 0)['degraded'].numpy(),
                  'rir': paired0.pair(record, key, 2, 0)['degraded'].numpy(),
                  'noise_rir_0db': paired0.pair(record, key, 3, 0)['degraded'].numpy(),
                  'context_left_1s': context[-(samples + sr):]}
        path = Path(cfg['data']['audio_root']) / record['path']
        actual = {}
        for ms in offsets:
            shifted = min(max(0, start + round(ms * sr / 1000)),
                          max(0, round(record['duration'] * sr) - samples))
            actual[str(ms)] = (shifted - start) / sr * 1000
            values[f'offset_{ms:+d}ms'] = load_audio(path, sample_rate=sr,
                                                    start=shifted / sr, duration=samples / sr, pad=True)
        # Separate, deliberately non-overlapping crop: secondary diagnostic only.
        other = 0 if start >= samples else max(0, round(record['duration'] * sr) - samples)
        if abs(other - start) < samples:
            raise ValueError(f'No non-overlapping diagnostic crop for {record["track_id"]}')
        values['other_excerpt_same_recording'] = load_audio(
            path, sample_rate=sr, start=other / sr, duration=samples / sr, pad=True)
        for ms in [-40, 40]:
            shifted = min(max(0, start + round(ms * sr / 1000)),
                          max(0, round(record['duration'] * sr) - samples))
            past = round(cfg['data']['past_context_duration'] * sr)
            available = min(past, shifted)
            shifted_context = load_audio(path, sample_rate=sr, start=(shifted - available) / sr,
                                         duration=(available + samples) / sr, pad=True)
            shifted_context = np.pad(shifted_context, (past - available, 0))
            response, _ = rir.load_validation(key)
            values[f'rir_offset_{ms:+d}ms'] = convolve_full_wet(
                shifted_context, response, past_context_samples=past, output_samples=samples)
        return values, actual

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pairs = list(tqdm(pool.map(query_inputs, selected[:queries]), total=queries, desc='Diagnostic audio'))
    query_audio = {name: np.stack([p[0][name] for p in pairs]) for name in pairs[0][0]}
    manifest = [{'track_id': r['track_id'], 'path': r['path'], 'key': key,
                 'start_sample': start, 'query': i < queries}
                for i, (r, key, _, start) in enumerate(selected)]
    return references, query_audio, {'gallery': manifest, 'skipped_gallery': audit,
                                    'actual_offsets_ms': [p[1] for p in pairs],
                                    'data_config': cfg['data'],
                                    'asset_manifests': {'noise': noise.manifest(), 'rir': rir.manifest()}}


@torch.inference_mode()
def extract(tokenizer, waveforms, batch_size, keep_seconds=2):
    """Bounded pre-round coordinates versus hard coordinates; IDs stay categorical."""
    result = {'pre': [], 'post': [], 'ids': []}
    q = tokenizer.model.quantizer
    frames = round(keep_seconds * tokenizer.model.frame_rate)
    for begin in range(0, len(waveforms), batch_size):
        waveform = torch.from_numpy(waveforms[begin:begin + batch_size]).to(tokenizer.device)
        mel = tokenizer.model.features(waveform)
        z = tokenizer.model.encoder(mel).transpose(1, 2).float()
        post, ids = q(z)
        shaped = z.reshape(*z.shape[:-1], q.groups, q.dimensions)
        levels = q.levels.float()
        half = (levels - 1) * (1 + 1e-3) / 2
        offset = torch.where(q.levels % 2 == 0, .5, 0.)
        shift = torch.atanh(offset / half)
        pre = ((shaped + shift).tanh() * half - offset) / (q.levels // 2)
        result['pre'].append(pre.flatten(-2)[:, -frames:].cpu().numpy())
        result['post'].append(post[:, -frames:].cpu().numpy())
        result['ids'].append(ids[:, -frames:].cpu().numpy())
    return {k: np.concatenate(v) for k, v in result.items()}


def readout(x, mean, std, bins):
    # Fixed temporal bins, no shift search or dynamic programming. Global mean
    # (bins=1) is also reported so rankings do not depend on one pooling choice.
    standardized = (x - mean) / std
    vector = np.concatenate([a.mean(axis=1) for a in np.array_split(standardized, bins, axis=1)], axis=1)
    return vector / np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-8)


def retrieval(gallery, queries, size):
    scores = queries @ gallery[:size].T
    target = np.arange(len(queries))
    positive = scores[target, target]
    negative = scores.copy()
    negative[target, target] = -np.inf
    # Count ties pessimistically; np.argmax would reward gallery order.
    ranks = 1 + (negative >= positive[:, None]).sum(axis=1)
    margin = positive - negative.max(axis=1)
    split = len(queries) // 2
    cal = negative[:split][np.isfinite(negative[:split])]
    threshold = float(np.quantile(cal, .999, method='higher'))
    test_neg = negative[split:][np.isfinite(negative[split:])]
    return {'top1': float((ranks == 1).mean()), 'median_rank': float(np.median(ranks)),
            'median_margin': float(np.median(margin)), 'positive_mean': float(positive.mean()),
            'negative_mean': float(negative[np.isfinite(negative)].mean()),
            'calibrated_fpr_target': .001, 'threshold': threshold,
            'test_fpr': float((test_neg >= threshold).mean()),
            'test_tpr': float((positive[split:] >= threshold).mean()),
            'calibration_pairs': len(cal), 'test_negative_pairs': len(test_neg),
            'test_positive_count': len(positive[split:])}, ranks, margin, negative.argmax(axis=1)


def mutual_information(a, b):
    counts = np.bincount(a * 1024 + b, minlength=1024 ** 2).reshape(1024, 1024)
    i, j = np.nonzero(counts)
    joint = counts[i, j] / len(a)
    return float(np.sum(joint * np.log2(joint / ((counts.sum(1)[i] / len(a)) *
                                                (counts.sum(0)[j] / len(a))))))


def redundancy(features, groups):
    ids = features['ids'].reshape(-1, groups)
    entropy, used = [], []
    for g in range(groups):
        counts = np.bincount(ids[:, g], minlength=1024)
        p = counts[counts > 0] / len(ids)
        entropy.append(float(-(p * np.log2(p)).sum()))
        used.append(int((counts > 0).sum()))
    post = features['post']
    a, b = post[:, :-1].reshape(-1, post.shape[-1]), post[:, 1:].reshape(-1, post.shape[-1])
    a, b = a - a.mean(0), b - b.mean(0)
    correlation = (a * b).mean(0) / np.maximum(a.std(0) * b.std(0), 1e-8)
    rng = np.random.default_rng(1337)
    mi = []
    for i in range(groups):
        for j in range(i):
            observed = mutual_information(ids[:, i], ids[:, j])
            shuffled = mutual_information(ids[:, i], rng.permutation(ids[:, j]))
            mi.append({'groups': [j, i], 'mi_bits': observed, 'permutation_mi_bits': shuffled,
                       'excess_mi_bits': observed - shuffled})
    return {'entropy_bits_per_group': entropy, 'effective_symbols_per_group': [2 ** h for h in entropy],
            'used_symbols_per_group': used, 'adjacent_symbol_repeat_fraction': float(
                (features['ids'][:, 1:] == features['ids'][:, :-1]).mean()),
            'adjacent_coordinate_correlation_mean': float(correlation.mean()),
            'group_mutual_information': mi,
            'note': 'MI is a finite-sample plug-in estimate; shuffled control exposes occupancy bias.'}


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def latency(tokenizer, waveforms, repeats):
    results = []
    for deterministic in [False, True]:
        with kernels(deterministic):
            for batch in [1, 16, 160]:
                w = np.tile(waveforms, (math.ceil(batch / len(waveforms)), 1))[:batch]
                x = torch.from_numpy(w).to(tokenizer.device)
                tokenizer.tokenize(x)
                durations = []
                for _ in range(repeats if not deterministic else 1):
                    synchronize(tokenizer.device)
                    started = time.perf_counter()
                    tokenizer.tokenize(x)
                    synchronize(tokenizer.device)
                    durations.append((time.perf_counter() - started) * 1000)
                results.append({'deterministic': deterministic, 'batch': batch,
                                'median_batch_ms': float(np.median(durations)),
                                'amortized_ms_per_crop': float(np.median(durations) / batch),
                                'samples': len(durations), 'includes': 'mel + encoder + FSQ + serialization; device-resident waveform'})
    with kernels(False):
        for batch in [1, 16]:
            w = np.tile(waveforms, (math.ceil(batch / len(waveforms)), 1))[:batch]
            durations = []
            for _ in range(repeats):
                synchronize(tokenizer.device)
                started = time.perf_counter()
                tokenizer.tokenize(torch.from_numpy(w).to(tokenizer.device))
                synchronize(tokenizer.device)
                durations.append((time.perf_counter() - started) * 1000)
            results.append({'deterministic': False, 'batch': batch,
                            'median_batch_ms': float(np.median(durations)),
                            'amortized_ms_per_crop': float(np.median(durations) / batch),
                            'samples': repeats, 'includes': 'host waveform transfer + mel + encoder + FSQ + serialization; excludes decoding audio file'})
    return results


def save_report(out, results, manifest):
    (out / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    lines = ['# FSQ diagnostics', '', 'Fixed readout diagnostics, **not PSI accuracy**.', '',
             f"Queries: {len(manifest['actual_offsets_ms'])}; gallery: {len(manifest['gallery'])}.", '',
             'Query identities are held out from tokenizer training. Clean queries reproduce gallery excerpts by design; '
             'degraded/offset queries test robustness. Other excerpts are secondary, not required positives in feature space.', '',
             'Pre = bounded coordinates before rounding; post = hard FSQ coordinates. '
             'Cosine readouts use gallery-only channel standardization and 1 or 8 fixed temporal bins. '
             'No sequence matching, shift optimization, learned classifier or external ANN index.', '',
             '| Checkpoint | Stage | Clean | Noise 0dB | RIR | Noise+RIR 0dB | Offset +40ms | Other excerpt |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for name, result in results['models'].items():
        for stage in ['pre', 'post']:
            entries = result['retrieval'][stage]['bins_8']
            vals = [entries[c]['full']['top1'] * 100 for c in
                    ['clean', 'noise_0db', 'rir', 'noise_rir_0db', 'offset_+40ms', 'other_excerpt_same_recording']]
            lines.append('| ' + name + ' | ' + stage + ' | ' + ' | '.join(f'{v:.1f}' for v in vals) + ' |')
    lines += ['', '## Latency and kernel equivalence', '',
              'Milliseconds per complete batch on device-resident waveforms; includes mel, encoder, FSQ and serialization. '
              'Audio-file decoding is excluded. Ordinary kernels remain FP32, with the recorded TF32 policy.', '',
              '| Checkpoint | Ordinary B=1 ms | Ordinary B=16 ms | Deterministic B=16 ms | Audited symbol mismatches |',
              '|---|---:|---:|---:|---:|']
    for name, result in results['models'].items():
        def timing(batch, deterministic):
            return next(x['median_batch_ms'] for x in result['latency']
                        if x['batch'] == batch and x['deterministic'] == deterministic
                        and x['includes'].startswith('mel'))
        mismatch = sum(x['mismatch_count'] for x in result['kernel_equivalence'].values())
        total = sum(x['symbol_count'] for x in result['kernel_equivalence'].values())
        lines.append(f'| {name} | {timing(1, False):.2f} | {timing(16, False):.2f} | '
                     f'{timing(16, True):.2f} | {mismatch}/{total} ({100*mismatch/total:.4f}%) |')
    lines += ['', '**Nonzero mismatches mean the modes are not bit-identical; '
              'do not silently change an existing frozen-tokenizer PSI run.**', '',
              '## Interpretation limits', '',
              '- Same-excerpt retrieval diagnoses discrimination under perturbation; it does not establish catalogue memorisation.',
              '- Compare both readouts. Temporal pooling and coordinate geometry may favour some tokenizers over others.',
              '- Thresholds use the first half of query identities for calibration and the second half for testing. '
              'The gallery identities are shared; this is a modest pilot, not an operational false-match guarantee.',
              '- Context queries add one second of preceding audio, then keep only the last two seconds of latent frames.',
              '- Latency is measured on the current allocated GPU with synchronization; deterministic timings use one measured repetition.',
              '- Full dataset manifest, per-query scores, code redundancy, rounding damage, latency and equivalence are in companion files.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', action='append', required=True, help='NAME=/absolute/path.ckpt')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--queries', type=int, default=100)
    p.add_argument('--gallery', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--latency-repeats', type=int, default=5)
    p.add_argument('--equivalence-queries', type=int, default=100)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(1337)
    np.random.seed(1337)
    args.output.mkdir(parents=True, exist_ok=False)
    specs = dict(entry.split('=', 1) for entry in args.checkpoint)
    if len(specs) != len(args.checkpoint):
        raise ValueError('Checkpoint labels must be unique')
    first = torch.load(next(iter(specs.values())), map_location='cpu', weights_only=False)
    cfg, split = first['config'], first['split']
    split_hash = digest_json(split)
    del first
    references, inputs, manifest = prepare_inputs(cfg, split, args.queries, args.gallery, args.workers)
    manifest.update(seed=1337, split_sha256=split_hash, conditions=list(inputs),
                    fpr_calibration_query_indices=list(range(args.queries // 2)),
                    fpr_test_query_indices=list(range(args.queries // 2, args.queries)))
    (args.output / 'dataset-manifest.json').write_text(json.dumps(manifest, indent=2))
    results = {'command': sys.argv, 'git_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name() if args.device.startswith('cuda') else 'cpu',
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'models': {}}
    for name, path in specs.items():
        print(f'Extracting {name}', flush=True)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if digest_json(payload['split']) != split_hash:
            raise ValueError(f'{name}: different tokenizer split; cannot compare paired queries')
        if payload['config']['data'] != cfg['data'] or payload['config']['seed'] != cfg['seed']:
            raise ValueError(f'{name}: different audio/degradation setup')
        step = payload['step']
        del payload
        tokenizer = FrozenFSQTokenizer(path, device=args.device)
        result = {'checkpoint': str(Path(path).resolve()), 'checkpoint_sha256': tokenizer.spec.revision,
                  'step': step, 'frame_rate': tokenizer.model.frame_rate,
                  'groups': tokenizer.selected_codebooks, 'model_config': tokenizer.model.cfg,
                  'retrieval': {}, 'stability': {}, 'kernel_equivalence': {}}
        with kernels(False):
            gallery = extract(tokenizer, references, args.batch_size)
            queries = {c: extract(tokenizer, w, args.batch_size) for c, w in tqdm(inputs.items(), desc=name)}
        result['redundancy'] = redundancy(gallery, tokenizer.selected_codebooks)
        details = []
        for stage in ['pre', 'post']:
            mean = gallery[stage].mean(axis=(0, 1))
            std = np.maximum(gallery[stage].std(axis=(0, 1)), 1e-4)
            result['retrieval'][stage] = {}
            for bins in [1, 8]:
                ref = readout(gallery[stage], mean, std, bins)
                result['retrieval'][stage][f'bins_{bins}'] = {}
                for condition, values in queries.items():
                    q = readout(values[stage], mean, std, bins)
                    sizes = sorted(set([args.queries, min(500, args.gallery), args.gallery]))
                    by_size = {}
                    for size in sizes:
                        metrics, ranks, margins, wrong = retrieval(ref, q, size)
                        by_size[str(size)] = metrics
                        if size == args.gallery:
                            by_size['full'] = metrics
                            details.extend({'query_index': i, 'track_id': manifest['gallery'][i]['track_id'],
                                            'stage': stage, 'bins': bins, 'condition': condition,
                                            'rank': int(rank), 'margin': float(margin),
                                            'hardest_wrong_track': manifest['gallery'][int(wrong[i])]['track_id']}
                                           for i, (rank, margin) in enumerate(zip(ranks, margins)))
                    result['retrieval'][stage][f'bins_{bins}'][condition] = by_size
        clean = queries['clean']
        for condition, values in queries.items():
            result['stability'][condition] = {
                'symbol_agreement': float((clean['ids'] == values['ids']).mean()),
                'coordinate_agreement': float((clean['post'] == values['post']).mean()),
                'pre_l1': float(np.abs(clean['pre'] - values['pre']).mean()),
                'post_l1': float(np.abs(clean['post'] - values['post']).mean())}
        result['rounding_damage'] = {
            c: {'coordinate_rounding_mse': float(np.square(v['pre'] - v['post']).mean())}
            for c, v in queries.items()}
        # All fixed degraded crops and representative offsets/context. No claim
        # about equivalence beyond this explicitly enumerated audit set.
        for c in ['clean', 'noise_0db', 'noise_5db', 'rir', 'noise_rir_0db',
                  'offset_+40ms', 'context_left_1s']:
            n = min(args.equivalence_queries, args.queries)
            with kernels(True):
                deterministic = extract(tokenizer, inputs[c][:n], args.batch_size)
            equal = deterministic['ids'] == queries[c]['ids'][:n]
            result['kernel_equivalence'][c] = {'queries': n, 'symbol_count': int(equal.size),
                                               'mismatch_count': int((~equal).sum()),
                                               'query_mismatch_count': int((~equal).any(axis=(1, 2)).sum())}
        result['latency'] = latency(tokenizer, references[:16], args.latency_repeats)
        with (args.output / f'{name}-per-query.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(details[0]))
            writer.writeheader()
            writer.writerows(details)
        results['models'][name] = result
        save_report(args.output, results, manifest)
        print(f'Completed {name}: ' + json.dumps(result['latency']), flush=True)
        del tokenizer, gallery, queries
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    print(f'Diagnostics complete: {args.output / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
