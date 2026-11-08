from types import SimpleNamespace

import pytest
import torch

from para_audio_id.audio_lm.continual import AdditionBatchSampler
from para_audio_id.audio_lm.continual_replay import (
    CachedAdditionCollator, ReplayCache, grouped_log_probs, identifier_positions,
    preservation_loss,
)
from para_audio_id.audio_lm.dataset import collate_causal_documents
from para_audio_id.audio_lm.vocabulary import AudioLMVocabulary


def documents():
    vocab = AudioLMVocabulary(num_codebooks=1, codebook_size=16)
    docs = collate_causal_documents([
        dict(audio_tokens=torch.arange(4), code=code, track_id=str(i),
             document_index=-1, view_type='random_clean')
        for i, code in enumerate(['12345', '54321'])], vocab, 512)
    return vocab, docs


def test_grouped_distribution_preserves_normalizer_and_other_mass():
    vocab, _ = documents()
    logits = torch.randn(3, vocab.size)
    p = grouped_log_probs(logits, vocab).exp()
    full = logits.softmax(-1)
    ids = list(vocab.digit_token_ids) + [vocab.eos_token_id]
    torch.testing.assert_close(p[:, :11], full[:, ids])
    torch.testing.assert_close(p[:, -1], 1 - full[:, ids].sum(-1))
    torch.testing.assert_close(p.sum(-1), torch.ones(3))


def test_preservation_zero_for_identical_teacher_and_excludes_new_and_wrong_old():
    vocab, docs = documents()
    logits = torch.randn(2, docs['input_ids'].shape[1], vocab.size, requires_grad=True)
    mask = identifier_positions(docs, vocab)
    target = grouped_log_probs(logits[:, :-1][mask], vocab).reshape(2, 6, 12).detach()
    eligible = torch.tensor([True, False])
    same = preservation_loss(logits, docs, vocab, target, eligible)
    assert same.item() == pytest.approx(0, abs=1e-6)
    modified = target.clone()
    modified[:, :, 0] += 2
    loss = preservation_loss(logits, docs, vocab, modified, eligible)
    assert loss.item() > 0
    loss.backward()
    assert logits.grad[0].abs().sum() > 0
    assert logits.grad[1].abs().sum() == 0
    assert logits.grad[:, :-1][~mask].abs().sum() == 0
    assert torch.isfinite(logits.grad).all()


def test_de_have_identical_draws_and_resume():
    d = list(AdditionBatchSampler(1000, 25000, arm='D', steps=4))
    e = list(AdditionBatchSampler(1000, 25000, arm='E', steps=4))
    assert d == e
    assert list(AdditionBatchSampler(1000, 25000, arm='E', steps=4, start=2)) == e[4:]
    for batch in d:
        assert sum(i < 1000 for i, *_ in batch) == 20


def test_cached_collator_only_decodes_new_and_pairs_old_views(monkeypatch):
    from para_audio_id.audio_lm import continual_replay as module

    class Fresh:
        def __init__(self, new, old, **kwargs):
            assert not old

        def __call__(self, examples):
            assert [e['record'].track_id for e in examples] == ['new']
            return dict(waveforms=torch.ones(2, 4), metadata=[dict(track_id='new')] * 2,
                        categories={'clean': 1}, failures=[], replacements=[])

    monkeypatch.setattr(module, 'AdditionCollator', Fresh)
    cache = SimpleNamespace(pairs={'old': [0, 2]}, tokens=torch.arange(16).reshape(4, 4),
        teacher=torch.zeros(4, 6, 12), eligible=torch.tensor([True, False, False, True]),
        rows=[dict(track_id='old', pair_role=role, category=cat)
              for role, cat in [('anchor', 'clean'), ('secondary', 'noise_rir')] * 2])
    collator = CachedAdditionCollator([SimpleNamespace(track_id='new')], [], cache, seed=7)
    examples = [dict(record=SimpleNamespace(track_id=x), optimizer_step=3,
                     batch_idx=6, pair_slot=i) for i, x in enumerate(['new', 'old'])]
    one, two = collator(examples), collator(examples)
    torch.testing.assert_close(one['cached_tokens'], two['cached_tokens'])
    assert [r['track_id'] for r in one['metadata']] == ['new', 'new', 'old', 'old']
    assert one['categories'] == {'clean': 1, 'noise_rir': 1}
    assert one['teacher'].shape == (2, 6, 12)


def test_cache_checks_hash_and_pilot_contract(tmp_path):
    from para_audio_id.audio_lm.continual import digest, file_digest, write_json
    from para_audio_id.audio_lm.continual_replay import REPLAY_PROTOCOL
    path = tmp_path / 'shard-00000.pt'
    torch.save(dict(tokens=torch.ones(2, 4, dtype=torch.int16), teacher=torch.zeros(2, 6, 12),
                    eligible=torch.tensor([True, False]), rows=[{'track_id': 'old'}] * 2), path)
    pilot = {'source': 'fixed'}
    write_json(tmp_path / 'manifest.json', dict(protocol=REPLAY_PROTOCOL,
        contract={'pilot_sha256': digest(pilot)},
        shards=[dict(name=path.name, sha256=file_digest(path))]))
    assert ReplayCache(tmp_path, pilot).pairs['old'] == [0]
    with pytest.raises(ValueError, match='different pilot'):
        ReplayCache(tmp_path, {'source': 'other'})
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Changed replay shard'):
        ReplayCache(tmp_path, pilot)
