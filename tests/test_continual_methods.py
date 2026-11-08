import numpy as np
import pytest
import torch

from para_audio_id.audio_lm.continual import AdditionBatchSampler
from para_audio_id.audio_lm.continual_methods import (
    configure_write_parameters, constrain_adam_displacement, exported_state,
    flat_parameters, projection_multipliers,
)
from para_audio_id.audio_lm.continual_replay import CachedAdditionCollator


def tiny_model():
    from para_audio_id.audio_lm.model import AudioCausalLM
    from para_audio_id.audio_lm.vocabulary import AudioLMVocabulary
    return AudioCausalLM(dict(model=dict(architecture='gpt2', hidden_size=16,
        num_layers=6, num_attention_heads=2, max_position_embeddings=32,
        resid_pdrop=0., embd_pdrop=0., attn_pdrop=0., tie_word_embeddings=True)),
        AudioLMVocabulary(num_codebooks=1, codebook_size=16))


def test_localized_scope_only_last_four_mlps_change():
    model = tiny_model()
    configure_write_parameters(model, 'F')
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all('.mlp.' in n for n in names)
    assert all(any(f'.h.{i}.' in n for i in [2, 3, 4, 5]) for n in names)
    before = {n: p.clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    model(torch.tensor([[1, 2, 3, 4]])).square().sum().backward()
    optimizer.step()
    changed = {n for n, p in model.named_parameters() if not torch.equal(before[n], p)}
    assert changed and changed <= set(names)


def test_lora_initial_identity_merged_export_and_original_base_resume():
    model = tiny_model().eval()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    inputs = torch.tensor([[1, 2, 3, 4]])
    initial = model(inputs).detach()
    parameters = configure_write_parameters(model, 'I', rank=2)
    torch.testing.assert_close(model(inputs), initial, rtol=0, atol=0)
    assert all('.parametrizations.weight.0.' in n for n, p in model.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=.01)
    model(inputs).square().sum().backward()
    optimizer.step()
    output = model(inputs).detach()
    merged, adapters = exported_state(model)
    assert adapters and not any('.parametrizations.' in k for k in merged)
    plain = tiny_model().eval()
    plain.load_state_dict(merged, strict=True)
    torch.testing.assert_close(plain(inputs), output)
    resumed = tiny_model().eval()
    resumed.load_state_dict(original)
    configure_write_parameters(resumed, 'I', rank=2, saved_adapters=adapters)
    torch.testing.assert_close(resumed(inputs), output)
    # Frozen original weights remain exactly unchanged despite adapter learning.
    for name, module in model.named_modules():
        if hasattr(module, 'parametrizations'):
            torch.testing.assert_close(module.parametrizations.weight.original,
                                       original[name + '.weight'], rtol=0, atol=0)


@pytest.mark.parametrize('gram,products', [
    (np.eye(3), np.array([1., -2., 3.])),
    (np.ones((3, 3)), np.ones(3)),
    (np.array([[1., -1.], [-1., 1.]]), np.array([1., -1.])),
    (np.eye(2), np.array([-1., -1.])),
])
def test_halfspace_qp_feasible_and_kkt(gram, products):
    multipliers = projection_multipliers(gram, products)
    assert multipliers is not None
    assert np.min(multipliers) >= -1e-9
    residual = products - gram @ multipliers
    assert residual.max() <= 1e-9
    np.testing.assert_allclose(multipliers * residual, 0, atol=1e-8)


def test_projection_uses_actual_adam_delta_not_raw_gradient():
    p = torch.nn.Parameter(torch.tensor([1., 2., 3.]))
    optimizer = torch.optim.AdamW([p], lr=.1, betas=(.9, .95), weight_decay=0.)
    # Manufacture prior momentum, so the proposed update need not follow the
    # current gradient's half-space sign.
    p.grad = torch.tensor([-10., 2., -1.])
    optimizer.step()
    before = flat_parameters([p])
    p.grad = torch.tensor([.01, .01, .01])
    optimizer.step()
    references = [torch.tensor([1., 0., 0.]), torch.tensor([0., -1., 0.])]
    report = constrain_adam_displacement([p], before, references)
    actual = p.detach() - before
    assert max(float(g @ actual) for g in references) <= 1e-7
    assert report['projection_correction_ratio'] > 0
    assert report['projection_update_norm'] > 0
    assert float(optimizer.state[p]['step']) == 2


def test_phased_sampler_balances_total_exposure_and_resumes_at_phase_boundary():
    batches = list(AdditionBatchSampler(1000, 25000, arm='H', steps=10000))
    new = sum(sum(i < 1000 for i, *_ in batch) for batch in batches)
    assert new == 400000
    assert sum(len(batch) for batch in batches) - new == 400000
    assert all(i < 1000 for i, *_ in batches[1999])
    assert all(i >= 1000 for i, *_ in batches[2000])
    resumed = list(AdditionBatchSampler(1000, 25000, arm='H', steps=10000, start=1000))
    assert resumed == batches[2000:]


def test_old_only_collation_does_not_decode_or_invent_new_draws(monkeypatch):
    from collections import defaultdict
    from types import SimpleNamespace
    from para_audio_id.audio_lm import continual_replay as module

    class NeverDecode:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, examples):
            raise AssertionError('Old-only phase must not decode audio')

    monkeypatch.setattr(module, 'AdditionCollator', NeverDecode)
    cache = SimpleNamespace(pairs=defaultdict(lambda: [0]), tokens=torch.ones(2, 200),
        teacher=torch.zeros(2, 6, 12), eligible=torch.tensor([True, True]),
        rows=[dict(track_id='old', category='clean'), dict(track_id='old', category='noise')])
    collator = CachedAdditionCollator([SimpleNamespace(track_id='new')], [], cache,
                                     seed=1, sample_rate=8, crop_duration=2)
    result = collator([dict(record=SimpleNamespace(track_id='old'), optimizer_step=2,
                           batch_idx=2, pair_slot=0)])
    assert result['waveforms'].shape == (0, 16)
    assert result['cached_tokens'].shape == (2, 200)
    assert all(r['track_id'] == 'old' for r in result['metadata'])
