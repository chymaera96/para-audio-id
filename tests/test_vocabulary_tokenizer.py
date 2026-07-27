import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from para_audio_id.audio_lm.tokenization import (
    CatalogueDocument,
    _load_view_document,
    load_training_track_ids,
)
from para_audio_id.audio_lm.tokenizer import (
    MuQRVQTokenizer,
    flatten_time_major,
    unflatten_time_major,
)
from para_audio_id.audio_lm.vocabulary import AudioLMVocabulary
from para_audio_id.catalogue import CatalogueRecord


def test_vocabulary_ranges_and_codebook_separation():
    vocabulary = AudioLMVocabulary(2, 1024)
    vocabulary.validate()
    assert vocabulary.audio_token_id(0, 17) == 17
    assert vocabulary.audio_token_id(1, 17) == 1041
    assert vocabulary.bos_token_id == 2048
    assert vocabulary.id_token_id == 2049
    assert vocabulary.eos_token_id == 2060
    assert vocabulary.size == 2061
    assert vocabulary.decode_code(vocabulary.encode_code("00109")) == "00109"


def test_time_major_serialization_round_trip():
    vocabulary = AudioLMVocabulary(2, 1024)
    raw = torch.tensor([[[1, 2, 3], [1, 2, 3]]])
    tokens = flatten_time_major(raw, vocabulary)
    assert tokens.tolist() == [[1, 1025, 2, 1026, 3, 1027]]
    assert torch.equal(unflatten_time_major(tokens, 3, vocabulary), raw)


class DummyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = ["melspec_2048"]
        self.rvq = SimpleNamespace(n_codebooks=8, codebook_size=1024)

    def preprocessing(self, waveform, features):
        return {"melspec_2048": waveform.unsqueeze(1)}

    def normalize(self, features):
        return features

    def rearrange(self, features):
        return features

    def get_rvq_codes(self, mel, raw):
        batch = mel.shape[0]
        base = torch.arange(24, device=mel.device).reshape(1, 8, 3)
        return base.expand(batch, -1, -1)

    def get_targets(self, waveform):
        codes = self.get_rvq_codes(waveform, None)
        return {
            "melspec_2048": torch.cat(
                [codes[:, index, :] for index in range(8)], dim=-1
            )
        }, waveform


class DummyMuQ(nn.Module):
    config = SimpleNamespace(use_rvq_target=True)

    def __init__(self):
        super().__init__()
        self.model = DummyInner()

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls()


def test_muq_probe_verifies_layout_and_determinism(monkeypatch):
    monkeypatch.setitem(sys.modules, "muq", SimpleNamespace(MuQ=DummyMuQ))
    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info",
        lambda self, model_name, revision: SimpleNamespace(sha="resolved"),
    )
    tokenizer = MuQRVQTokenizer("dummy", device="cpu")
    report = tokenizer.probe(torch.zeros(1, 100))
    assert report["raw_shape"] == [1, 2, 3]
    assert report["serialized_tokens_per_example"] == 6
    assert report["deterministic"]


def test_invalid_tokenizer_layout_is_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules, "muq", SimpleNamespace(MuQ=DummyMuQ))
    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info",
        lambda self, model_name, revision: SimpleNamespace(sha="resolved"),
    )
    tokenizer = MuQRVQTokenizer("dummy", device="cpu")
    tokenizer.model.model.get_targets = lambda waveform: (
        {"melspec_2048": torch.zeros(1, 24, dtype=torch.long)},
        waveform,
    )
    with pytest.raises(RuntimeError, match="block-major"):
        tokenizer.probe(torch.zeros(1, 100))


def test_shifted_crop_padding_is_measured_and_manifest_is_exact(
    monkeypatch, tmp_path
):
    record = CatalogueRecord(
        path="short.mp3", track_id="track", code="00000", duration=26.0
    )
    document = CatalogueDocument(
        document_index=0,
        record=record,
        start=24.0,
        duration=5.0,
        view_type="shifted",
        corpus_role="shifted_training",
    )
    monkeypatch.setattr(
        "para_audio_id.audio_lm.tokenization._load_document",
        lambda *args, **kwargs: torch.zeros(120_000).numpy(),
    )
    audio, padded = _load_view_document(
        document, audio_root=tmp_path, sample_rate=24_000
    )
    assert len(audio) == 120_000
    assert padded == 72_000

    manifest = tmp_path / "tracks.json"
    manifest.write_text('["track-a", "track-b"]')
    assert load_training_track_ids(manifest, expected_count=2) == [
        "track-a",
        "track-b",
    ]
    manifest.write_text('["track-a", "track-a"]')
    with pytest.raises(ValueError, match="unique"):
        load_training_track_ids(manifest, expected_count=2)
