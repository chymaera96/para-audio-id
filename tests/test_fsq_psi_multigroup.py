from dataclasses import asdict
from copy import deepcopy
import hashlib
import json

import numpy  # noqa: F401 -- local runtime initialization order
import pytest
import torch
import yaml

from para_audio_id.audio_lm.fsq import CNNFSQ, FrozenFSQTokenizer, fsq_execution_policy
from para_audio_id.audio_lm.tokenizer import build_tokenizer


def make_tokenizer(tmp_path):
    cfg = yaml.safe_load(open('configs/fsq_tc8.yaml'))
    cfg['model']['channels'] = [16, 24, 32]
    cfg['model']['decoder_channels'] = [24, 32, 48]
    model = CNNFSQ(cfg['model'])
    path = tmp_path / 'frozen.ckpt'
    torch.save({'format': 'cnn_fsq_tokenizer_v1', 'config': cfg,
                'model': model.state_dict(), 'step': 32000}, path)
    return path


def test_fsq_fast_policy_is_versioned_and_legacy_hash_is_unchanged(tmp_path):
    path = make_tokenizer(tmp_path)
    legacy = FrozenFSQTokenizer(path, device='cpu')
    old_dict = asdict(legacy.spec)
    old_dict.pop('execution_policy')
    assert legacy.spec.to_dict() == old_dict
    old_hash = hashlib.sha256(json.dumps(old_dict, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert legacy.spec.fingerprint == old_hash
    fast = build_tokenizer({'kind': 'fsq', 'checkpoint': str(path),
                            'device': 'cpu', 'execution_policy': 'fast_fp32_v1'})
    assert fast.spec.execution_policy == 'fast_fp32_v1'
    assert fast.spec.fingerprint != legacy.spec.fingerprint
    assert fast.spec.revision == legacy.spec.revision
    wave = torch.randn(2, 48000)
    tokens = fast.tokenize(wave)
    assert tokens.shape == (2, 200)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert torch.equal(tokens, fast.tokenize(wave))


def test_fast_policy_restores_surrounding_psi_flags_even_on_error():
    previous = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = torch.backends.cudnn.enabled
    matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.raises(RuntimeError, match='intentional'):
            with fsq_execution_policy('fast_fp32_v1'):
                assert not torch.are_deterministic_algorithms_enabled()
                assert torch.backends.cudnn.enabled
                assert not torch.backends.cudnn.deterministic
                assert not torch.backends.cuda.matmul.allow_tf32
                raise RuntimeError('intentional')
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.enabled == cudnn
        assert torch.backends.cuda.matmul.allow_tf32 == matmul
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn)


def test_tc21_profile_tokens_objective_resume_and_evaluation(tmp_path):
    from para_audio_id.audio_lm.profiles import resolve_training_config, evaluation_checkpoint_profile
    from para_audio_id.audio_lm.training import validate_tc18_query_configuration, validate_tc18_batch_configuration
    from para_audio_id.audio_lm.evaluation import _checkpoint_tokenizer
    from para_audio_id.audio_lm.dataset import collate_causal_documents

    path = make_tokenizer(tmp_path)
    base = yaml.safe_load(open('configs/fsq_psi_tc21.yaml'))
    base['tokenizer']['checkpoint'] = str(path)
    base['tokenizer']['device'] = 'cpu'
    resolved = resolve_training_config(base)
    assert resolved['model']['num_layers'] == 12
    assert resolved['model']['hidden_size'] == 768
    assert resolved['train']['id_digit_weight'] == 16
    assert resolved['train']['max_steps'] == 225000
    assert resolved['train']['distillation']['maximum_weight'] == 0
    assert resolved['resolved_training_profile']['variant'] == 'cnn-fsq-25k-v2'
    frozen = build_tokenizer(resolved['tokenizer'])
    query = validate_tc18_query_configuration(resolved, frozen.spec, frozen.vocabulary)
    assert query['audio_targets'] == 200
    assert query['document_tokens'] == 208
    assert query['frames'] == 25
    assert query['selected_codebooks'] == 8
    assert query['id_digit_weight'] == 16
    batch_spec = validate_tc18_batch_configuration(resolved, query)
    assert batch_spec['tracks_per_optimizer_step'] == 80
    waveform = torch.randn(2, 48000)
    tokens = frozen.tokenize(waveform)
    batch = collate_causal_documents([
        {'audio_tokens': t, 'code': '12345', 'track_id': 'example',
         'document_index': -1, 'view_type': view}
        for t, view in zip(tokens, ['clean', 'noise'])], frozen.vocabulary, 512)
    assert int(batch['audio_target_mask'].sum()) == 400
    assert int(batch['id_target_mask'].sum()) == 10
    checkpoint = {'resolved_training_profile': resolved['resolved_training_profile'],
                  'tokenizer_spec': frozen.spec.to_dict(), 'tokenizer_fingerprint': frozen.spec.fingerprint,
                  'query_spec': query, 'hyper_parameters': resolved,
                  'training_track_ids': list(range(25000))}
    assert evaluation_checkpoint_profile(checkpoint) == resolved['resolved_training_profile']
    saved = tmp_path / 'psi.ckpt'
    torch.save(checkpoint, saved)
    resumed = resolve_training_config(base, checkpoint=saved)
    assert resumed['resolved_training_profile'] == resolved['resolved_training_profile']
    loaded = _checkpoint_tokenizer(checkpoint, 'cpu')
    assert loaded.execution_policy == 'fast_fp32_v1'
    assert loaded.spec.fingerprint == frozen.spec.fingerprint
    assert torch.equal(tokens, loaded.tokenize(waveform))
    bad = deepcopy(base)
    bad['tokenizer']['selected_codebooks'] = 4
    with pytest.raises(ValueError, match='group count'):
        resolve_training_config(bad)
