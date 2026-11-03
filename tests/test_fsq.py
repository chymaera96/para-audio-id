import itertools
import json
from copy import deepcopy

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from para_audio_id.audio_lm.fsq import CNNFSQ, FSQ, FrozenFSQTokenizer, reconstruction_loss


@pytest.mark.parametrize("run,groups,frames", [(3, 2, 50), (4, 4, 50),
                                              (5, 1, 100), (6, 2, 100)])
def test_fsq_capacity_variants(tmp_path, run, groups, frames):
    cfg = yaml.safe_load(open(f"configs/fsq_tc{run}.yaml"))
    assert cfg["train"]["consistency_weight"] == 0.1
    assert cfg["train"]["max_steps"] == 50000
    cfg["model"]["channels"] = [16, 24, 32]
    model = CNNFSQ(cfg["model"])
    wave = torch.randn(2, 48000) * 0.1
    mel = model.features(wave)
    reconstructed, coordinates, ids = model(mel)
    assert mel.shape == reconstructed.shape == (2, 80, 200)
    assert ids.shape == (2, groups, frames)
    assert model.frame_rate == frames / 2
    reconstructed.abs().mean().backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    path = tmp_path / "tokenizer.ckpt"
    torch.save({"format": "cnn_fsq_tokenizer_v1", "config": cfg,
                "model": model.state_dict()}, path)
    frozen = FrozenFSQTokenizer(path, device="cpu")
    tokens = frozen.tokenize(wave)
    assert tokens.shape == (2, groups * frames)
    assert torch.equal(tokens, frozen.tokenize(wave))
    for group in range(groups):
        assert torch.equal(tokens[:, group::groups], ids[:, group] + group * 1024)


def test_fsq_grid_and_straight_through_gradient():
    q = FSQ([4, 4, 4, 4, 4])
    # Visit every symbol and prove the mixed-radix map is bijective on the grid.
    digits = torch.tensor(list(itertools.product(range(4), repeat=5)))
    ids = (digits * q.basis).sum(-1)
    assert torch.equal(ids.sort().values, torch.arange(1024))
    x = torch.randn(3, 50, 5, requires_grad=True)
    coordinates, tokens = q(x)
    assert tokens.shape == (3, 50, 1)
    assert 0 <= tokens.min() <= tokens.max() < 1024
    assert torch.equal(coordinates * 2, (coordinates * 2).round())
    coordinates.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_paired_clean_target_loss():
    target = torch.zeros(2, 80, 200)
    assert reconstruction_loss(target, torch.ones_like(target),
                               torch.full_like(target, 3)).item() == 2


def test_round_trip_frozen_tokenizer_and_encoder_gradients(tmp_path):
    cfg = yaml.safe_load(open("configs/fsq_tokenizer.yaml"))
    cfg["model"]["channels"] = [16, 24, 32]
    model = CNNFSQ(cfg["model"])
    waveform = torch.randn(2, 48000) * .1
    mel = model.features(waveform)
    assert mel.shape == (2, 80, 200)
    reconstruction, _, ids = model(mel)
    assert reconstruction.shape == mel.shape
    assert ids.shape == (2, 1, 50)
    reconstruction.abs().mean().backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    path = tmp_path / "tokenizer.ckpt"
    torch.save({"format": "cnn_fsq_tokenizer_v1", "config": cfg,
                "model": model.state_dict()}, path)
    frozen = FrozenFSQTokenizer(path, device="cpu")
    tokens = frozen.tokenize(waveform)
    assert torch.equal(tokens, frozen.tokenize(waveform))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert torch.equal(tokens, frozen.tokenize(waveform))
    assert torch.equal(tokens, ids[:, 0])
    assert tokens.shape == (2, 50)
    assert frozen.vocabulary.size == 1037
    assert not any(p.requires_grad for p in frozen.model.parameters())


def test_fsq_psi_configuration_and_evaluation_reload(tmp_path):
    from para_audio_id.audio_lm.profiles import resolve_training_config, evaluation_checkpoint_profile
    from para_audio_id.audio_lm.training import validate_tc18_query_configuration
    from para_audio_id.audio_lm.evaluation import _checkpoint_tokenizer

    cfg = yaml.safe_load(open("configs/fsq_tokenizer.yaml"))
    cfg["model"]["channels"] = [16, 24, 32]
    model = CNNFSQ(cfg["model"])
    path = tmp_path / "frozen.ckpt"
    torch.save({"format": "cnn_fsq_tokenizer_v1", "config": cfg,
                "model": model.state_dict()}, path)
    base = yaml.safe_load(open("configs/fma_large.yaml"))
    base["tokenizer"] = dict(kind="fsq", checkpoint=str(path), device="cpu")
    resolved = resolve_training_config(base, distillation_weight=0.0)
    assert resolved["train"]["max_steps"] == 225000
    assert resolved["train"]["id_digit_weight"] == 4
    frozen = FrozenFSQTokenizer(path, device="cpu")
    query = validate_tc18_query_configuration(resolved, frozen.spec, frozen.vocabulary)
    assert query["audio_targets"] == 50 and query["document_tokens"] == 58
    checkpoint = {
        "resolved_training_profile": resolved["resolved_training_profile"],
        "tokenizer_spec": frozen.spec.to_dict(), "tokenizer_fingerprint": frozen.spec.fingerprint,
        "query_spec": query, "hyper_parameters": resolved,
        "training_track_ids": list(range(25000)),
    }
    assert evaluation_checkpoint_profile(checkpoint) == resolved["resolved_training_profile"]
    resume_path = tmp_path / "psi.ckpt"
    torch.save(checkpoint, resume_path)
    resumed = resolve_training_config(base, checkpoint=resume_path)
    assert resumed["train"]["distillation"]["maximum_weight"] == 0.0
    assert resumed["resolved_training_profile"] == resolved["resolved_training_profile"]
    loaded = _checkpoint_tokenizer(checkpoint, "cpu")
    waveform = torch.randn(1, 48000)
    assert torch.equal(frozen.tokenize(waveform), loaded.tokenize(waveform))

    medium_cfg = deepcopy(base)
    medium_cfg["model"]["decoder"] = "medium"
    medium = resolve_training_config(medium_cfg, distillation_weight=0.0)
    assert medium["resolved_training_profile"]["decoder"]["name"] == "medium"
    assert medium["model"]["num_layers"] == 24
    assert medium["model"]["hidden_size"] == 1024
    assert medium["model"]["num_attention_heads"] == 16
    assert medium["train"]["max_steps"] == 225000
    checkpoint["resolved_training_profile"] = medium["resolved_training_profile"]
    checkpoint["hyper_parameters"] = medium
    torch.save(checkpoint, resume_path)
    resumed_medium = resolve_training_config(medium_cfg, checkpoint=resume_path)
    assert resumed_medium["model"]["hidden_size"] == 1024
    assert resumed_medium["train"]["distillation"]["maximum_weight"] == 0.0


def test_split_and_fixed_monitor_replacement_and_alignment(tmp_path):
    from para_audio_id.audio_lm.fsq_data import source_split, PairedCrops
    cfg = yaml.safe_load(open("configs/fsq_tokenizer.yaml"))
    data = cfg["data"]
    data.update(audio_root=str(tmp_path), catalogue=str(tmp_path / "catalogue.jsonl"),
                split_path=str(tmp_path / "split.json"), training_tracks=2, monitor_tracks=2)
    rows = []
    wave = np.sin(2 * np.pi * 220 * np.arange(6 * 24000) / 24000).astype("float32") * .3
    for i in range(8):
        sf.write(tmp_path / f"{i}.wav", wave, 24000)
        rows.append(dict(track_id=str(i), path=f"{i}.wav", duration=6., code=f"{i:05d}"))
    selected = np.random.default_rng(cfg["seed"]).permutation(8)
    bad = rows[selected[2]]["path"]
    sf.write(tmp_path / bad, np.zeros_like(wave), 24000)
    (tmp_path / "catalogue.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    split = source_split(cfg)
    assert len(split["monitor"]) == 2
    assert bad not in [r["path"] for r in split["monitor"]]
    assert split["monitor_preparation_audit"][0]["action"] == "replace_monitor_track"
    assert {r["track_id"] for r in split["training"]}.isdisjoint(
        r["track_id"] for r in split["monitor"])
    assert source_split(cfg) == split

    class Noise:
        def load_validation(self, key, samples):
            return np.random.default_rng(key).normal(size=samples).astype("float32")

    class RIR:
        def load_validation(self, key):
            return np.array([1., .5], dtype="float32"), "ir"

    dataset = PairedCrops(cfg, split["monitor"], Noise(), RIR(), monitor=True,
                           monitor_keys=split["monitor_crop_keys"])
    views = [dataset[i] for i in range(4)]
    assert len({v["start_sample"] for v in views}) == 1
    assert all(torch.equal(v["clean"], views[0]["clean"]) for v in views)
    assert torch.equal(views[0]["clean"], views[0]["degraded"])
    assert not torch.equal(views[3]["clean"], views[3]["degraded"])
