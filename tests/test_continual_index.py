from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from para_audio_id.catalogue import CatalogueRecord
from para_audio_id.audio_lm.continual import (
    CONDITIONS, AdditionBatchSampler, AdditionCollator, choose_groups,
    evaluate_tokens, full_schedule,
)
from para_audio_id.audio_lm.noise import resolved_augmentation_schedule
from para_audio_id.audio_lm.random_crops import OnlineTrackDataset


def test_addition_sampling_balances_pools_pairs_bc_and_resumes_exactly():
    samplers = {arm: list(AdditionBatchSampler(1000, 25000 if arm == 'B' else 1000,
                        arm=arm, steps=3)) for arm in 'ABC'}
    for arm, batches in samplers.items():
        assert len(batches) == 6
        for batch in batches:
            assert len(batch) == 40
            assert len({r[0] for r in batch}) == 40
            assert sum(r[0] < 1000 for r in batch) == (40 if arm == 'A' else 20)
        resumed = list(AdditionBatchSampler(1000, 25000 if arm == 'B' else 1000,
                                           arm=arm, steps=3, start=1))
        assert resumed == batches[2:]
    assert [b[:20] for b in samplers['B']] == [b[:20] for b in samplers['C']]


def test_group_selection_excludes_old_and_is_deterministic():
    records = [CatalogueRecord(f'{i}.wav', str(i), f'{i:05d}', 10.0) for i in range(20)]
    kwargs = dict(new_count=5, buffer_count=3, seed=7,
                  validator=lambda r: 'bad' if r.track_id == '9' else None)
    one = choose_groups(records, [str(i) for i in range(8)], **kwargs)
    two = choose_groups(records, [str(i) for i in range(8)], **kwargs)
    assert one == two
    old, new, buffer, _ = one
    assert len(new) == 5 and len(buffer) == 3
    assert not {r.track_id for r in old} & {r.track_id for r in new}
    assert set(buffer) <= {r.track_id for r in old}
    assert '9' not in {r.track_id for r in new}


def test_full_schedule_does_not_restart_curriculum():
    schedule = full_schedule({'schedule_config': dict(name='noise-rir',
        curriculum='tc12_noise_rir_curriculum_v1', clean_until_step=10000,
        degradation_ramp_until_step=30000, combined_ramp_until_step=60000,
        snr_bin_probabilities=[.4, .3, .2, .1])})
    for step in [0, 1, 100, 10000]:
        state = resolved_augmentation_schedule(step, schedule)
        assert state.rir_severity_quantile == 1
        assert state.clean_probability == pytest.approx(.1)
        assert state.noise_rir_probability == .25


def test_collator_keeps_replacements_within_old_or_new_pool(tmp_path):
    sr = 8000
    signal = .2 * np.sin(np.arange(sr * 6) * 2 * np.pi * 330 / sr).astype(np.float32)
    for name in ['new-good', 'old-good']:
        sf.write(tmp_path / f'{name}.wav', signal, sr)
    new = [CatalogueRecord('missing.wav', 'new-bad', '00001', 6.0),
           CatalogueRecord('new-good.wav', 'new-good', '00002', 6.0)]
    old = [CatalogueRecord('missing.wav', 'old-bad', '00003', 6.0),
           CatalogueRecord('old-good.wav', 'old-good', '00004', 6.0)]
    dataset = OnlineTrackDataset(new + old)
    # Clean-only schedule isolates identity replacement from IR/noise fixtures.
    collator = AdditionCollator(new, old, audio_root=tmp_path, noise_assets=object(),
        rir_assets=None, schedule=dict(name='noise-rir', curriculum='tc12_noise_rir_curriculum_v1',
            clean_until_step=10, degradation_ramp_until_step=20, combined_ramp_until_step=30,
            snr_bin_probabilities=[.4, .3, .2, .1]), sample_rate=sr,
        crop_duration=2, past_context_duration=2, seed=7, reserved_starts={})
    result = collator([dataset[(0, 0, 0, 0)], dataset[(2, 0, 0, 1)]])
    assert [r['track_id'] for r in result['metadata']] == ['new-good'] * 2 + ['old-good'] * 2
    assert result['waveforms'].shape == (4, sr * 2)
    assert len(result['replacements']) == 2
    assert result['categories'] == {'clean': 2}
    assert Counter(r['pair_role'] for r in result['metadata']) == {'anchor': 2, 'secondary': 2}


def test_validation_is_top1_unconstrained_and_keeps_cohorts_separate(monkeypatch):
    import para_audio_id.audio_lm.continual as module

    def fake_generate(model, prompts, vocabulary, *, width, score_eos):
        assert width == 10 and score_eos
        return [[SimpleNamespace(code='00001')] for _ in prompts]

    monkeypatch.setattr(module, 'batched_beam_generate', fake_generate)
    monkeypatch.setattr(module, 'prompts_from_audio_tokens', lambda tokens, _: tokens)
    cache = dict(tokens=torch.zeros(6, 200, dtype=torch.int16), rows=[
        dict(group=group, condition=c, code='00001' if group == 'old' else '00002')
        for group in ['old', 'new'] for c in CONDITIONS])
    model = SimpleNamespace(eval=lambda: None)
    metrics, details = evaluate_tokens(model, None, cache, 'cpu', batch_size=2)
    assert len(metrics) == 6 and len(details) == 6
    assert all(metrics[f'val/old/{c}'] == 1 for c in CONDITIONS)
    assert all(metrics[f'val/new/{c}'] == 0 for c in CONDITIONS)


def test_checkpoint_metadata_and_optimizer_roundtrip(tmp_path, monkeypatch):
    from continual_index import write_checkpoint
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-5)
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    source = dict(architecture='audio_lm_v1', tokenizer_spec={}, tokenizer_fingerprint='fixed',
                  model_config={}, query_spec={}, batch_spec={},
                  resolved_training_profile={'variant': 'cnn-fsq-25k-v2'})
    state = dict(catalogue_ids=['old', 'new'], probe_ids=['old'], mapping_sha256='mapping',
                 source_global_step=225000, update_step=1)
    cfg = {'model': {}, 'train': {'learning_rate_schedule': {'policy': 'continual_constant_v1'}},
           'resolved_training_profile': source['resolved_training_profile']}
    scaler = SimpleNamespace(state_dict=lambda: {})
    vocab = SimpleNamespace(to_dict=lambda: {})
    cfg['train']['schedule'] = {}
    target = tmp_path / 'last.ckpt'
    write_checkpoint(target, model, vocab, cfg, source, state, optimizer, scaler)
    saved = torch.load(target, weights_only=False)
    assert saved['global_step'] == 225001
    assert saved['continual_state']['update_step'] == 1
    assert saved['hyper_parameters']['resolved_training_profile'] == saved['resolved_training_profile']
    assert saved['resolved_training_profile']['database_size'] == 2
    assert cfg['resolved_training_profile'] == source['resolved_training_profile']
    assert all(v['step'] == 1 for v in saved['optimizer_state']['state'].values())
