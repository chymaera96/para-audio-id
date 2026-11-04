import json
from copy import deepcopy

import numpy as np
import soundfile as sf
import torch
import yaml

from scripts.fsq_diagnostics import (extract, kernels, prepare_inputs, readout,
                                     redundancy, retrieval)
from para_audio_id.audio_lm.fsq import CNNFSQ, FrozenFSQTokenizer
from para_audio_id.audio_lm.fsq_data import source_split


def test_retrieval_ranks_ties_and_calibration():
    ref = np.eye(6, dtype=np.float32)
    metrics, ranks, margins, wrong = retrieval(ref, ref[:4], 6)
    assert metrics['top1'] == 1
    assert np.all(ranks == 1)
    assert np.all(margins == 1)
    assert np.all(wrong != np.arange(4))
    assert metrics['calibration_pairs'] == 10
    assert metrics['test_negative_pairs'] == 10
    # Constant representations must NOT pass because targets come first.
    m, r, _, _ = retrieval(np.ones((6, 2)), np.ones((4, 2)), 6)
    assert m['top1'] == 0 and np.all(r == 6)


def test_fixed_readout_is_scale_normalized_and_order_sensitive():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 50, 5)).astype(np.float32)
    mean, std = x.mean((0, 1)), x.std((0, 1))
    assert readout(x, mean, std, 8).shape == (8, 40)
    assert np.allclose(np.linalg.norm(readout(x, mean, std, 8), axis=1), 1)
    assert np.allclose(readout(x, mean, std, 1), readout(x[:, ::-1], mean, std, 1))
    assert not np.allclose(readout(x, mean, std, 8), readout(x[:, ::-1], mean, std, 8))


def test_extraction_matches_frozen_tokens_and_restores_kernel_flags(tmp_path):
    cfg = yaml.safe_load(open('configs/fsq_tc6.yaml'))
    cfg['model']['channels'] = [16, 24, 32]
    model = CNNFSQ(cfg['model'])
    path = tmp_path / 'model.ckpt'
    torch.save({'format': 'cnn_fsq_tokenizer_v1', 'config': cfg,
                'model': model.state_dict()}, path)
    tokenizer = FrozenFSQTokenizer(path, device='cpu')
    waves = np.random.default_rng(1).normal(0, .1, size=(4, 48000)).astype(np.float32)
    previous = torch.are_deterministic_algorithms_enabled()
    with kernels(False):
        features = extract(tokenizer, waves, 2)
    assert torch.are_deterministic_algorithms_enabled() == previous
    assert features['pre'].shape == features['post'].shape == (4, 100, 10)
    expected = tokenizer.tokenize(torch.from_numpy(waves)).numpy().reshape(4, 100, 2)
    expected -= np.arange(2)[None, None, :] * 1024
    assert np.array_equal(features['ids'], expected)
    assert np.max(np.abs(features['pre'] - features['post'])) <= .251
    longer = np.pad(waves, ((0, 0), (24000, 0)))
    assert extract(tokenizer, longer, 2)['post'].shape == features['post'].shape
    stats = redundancy(features, 2)
    assert len(stats['entropy_bits_per_group']) == 2
    assert len(stats['group_mutual_information']) == 1
    json.dumps(stats, allow_nan=False)


def test_audio_preparation_has_disjoint_gallery_and_paired_conditions(tmp_path, monkeypatch):
    from scripts import fsq_diagnostics as diagnostic
    cfg = yaml.safe_load(open('configs/fsq_tokenizer.yaml'))
    cfg = deepcopy(cfg)
    cfg['data'].update(audio_root=str(tmp_path), catalogue=str(tmp_path / 'catalogue.jsonl'),
                       split_path=str(tmp_path / 'split.json'), training_tracks=2, monitor_tracks=4)
    rows = []
    sr = 24000
    for i in range(8):
        wave = .2 * np.sin(np.arange(sr * 6) * 2 * np.pi * (200 + 17 * i) / sr)
        sf.write(tmp_path / f'{i}.wav', wave, sr)
        rows.append(dict(track_id=str(i), path=f'{i}.wav', duration=6, code=f'{i:05d}'))
    (tmp_path / 'catalogue.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    split = source_split(cfg)

    class Noise:
        def manifest(self):
            return {'test': 'noise'}

        def load_validation(self, key, *, samples):
            return np.random.default_rng(key).normal(0, .1, samples).astype(np.float32)

    class IR:
        def manifest(self):
            return {'test': 'ir'}

        def load_validation(self, key):
            return np.exp(-np.arange(2000) / 100).astype(np.float32), 'test-ir'

    monkeypatch.setattr(diagnostic, 'assets', lambda _: (Noise(), IR()))
    ref, query, manifest = prepare_inputs(cfg, split, 4, 6, 1)
    assert ref.shape == (6, 48000)
    assert np.array_equal(query['clean'], ref[:4])
    assert query['context_left_1s'].shape == (4, 72000)
    assert query['rir_offset_+40ms'].shape == (4, 48000)
    assert not np.array_equal(query['noise_0db'], query['clean'])
    assert not {r['track_id'] for r in manifest['gallery']} & {r['track_id'] for r in split['training']}
    assert len(set(r['track_id'] for r in manifest['gallery'])) == 6
    assert set(manifest['actual_offsets_ms'][0]) == {'-80', '-40', '-20', '-10', '10', '20', '40', '80'}
