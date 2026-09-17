from copy import deepcopy

import pytest
import torch

from para_audio_id.audio_lm.continuation import continuation_multiplier, fork_payload
from para_audio_id.audio_lm.profiles import canonical_training_profile, resolve_training_config
from para_audio_id.config import load_config


@pytest.mark.parametrize("id_only", [False, True])
def test_full_state_fork_and_resumed_lr(tmp_path, id_only):
    cfg = resolve_training_config(load_config("configs/fma_large.yaml"))
    profile = canonical_training_profile(
        database_size=100000, decoder="medium", schedule="noise-rir",
        selected_codebooks=8, distillation_weight=0.0, devices=4,
    )
    cfg["resolved_training_profile"] = deepcopy(profile)
    cfg["train"]["run_id"] = "source"
    step = profile["schedule"]["max_steps"]
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    parameter.square().backward()
    optimizer.step()
    optimizer.param_groups[0]["lr"] = 1.5e-5
    optimizer.param_groups[0]["initial_lr"] = 3e-4
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 0.05)
    scheduler.last_epoch = step
    payload = {
        "hyper_parameters": cfg, "resolved_training_profile": profile,
        "global_step": step, "optimizer_states": [optimizer.state_dict()],
        "lr_schedulers": [scheduler.state_dict()],
        "callbacks": {"old": {"dirpath": "source"}},
        "state_dict": {"model.weight": parameter.detach().clone()},
        "loops": {"position": 123}, "torch_rng_state": torch.get_rng_state(),
    }
    fork = fork_payload(payload, run_id="continuation", source="source/last.ckpt",
                        source_sha256="test", id_only=id_only, keep_saved_lr=id_only)
    train = fork["hyper_parameters"]["train"]
    assert train["max_steps"] == step + 25000
    assert payload["resolved_training_profile"]["schedule"]["max_steps"] == step
    assert payload["callbacks"] and not fork["callbacks"]
    assert fork["loops"] == payload["loops"]
    assert torch.equal(fork["torch_rng_state"], payload["torch_rng_state"])
    assert torch.equal(fork["state_dict"]["model.weight"], payload["state_dict"]["model.weight"])
    path = tmp_path / "fork.ckpt"
    torch.save(fork, path)
    resolved = resolve_training_config(fork["hyper_parameters"], checkpoint=path)
    assert resolved["train"]["max_steps"] == step + 25000
    assert resolved["train"]["clean_audio_loss_weight"] == (0.0 if id_only else 1.0)
    assert payload["hyper_parameters"]["train"].get("clean_audio_loss_weight", 1.0) == 1.0
    assert resolved["train"]["schedule"] == cfg["train"]["schedule"]
    restored = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=3e-4)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored, lambda t: continuation_multiplier(t, train)
    )
    restored.load_state_dict(fork["optimizer_states"][0])
    resumed_scheduler.load_state_dict(fork["lr_schedulers"][0])
    assert restored.param_groups[0]["lr"] == pytest.approx(1.5e-5)
    for offset, expected in [(0, 1.5e-5), (1000, 3.25e-5), (2000, 5e-5), (25000, 5e-5)]:
        if id_only:
            expected = 1.5e-5
        assert continuation_multiplier(step + offset, train) * 3e-4 == pytest.approx(expected)
    restored.step()
    resumed_scheduler.step()
    assert restored.param_groups[0]["lr"] == pytest.approx(
        continuation_multiplier(step + 1, train) * 3e-4
    )
    assert torch.equal(restored.state_dict()["state"][0]["exp_avg"],
                       optimizer.state_dict()["state"][0]["exp_avg"])
    with pytest.raises(ValueError, match="different run"):
        fork_payload(payload, run_id="source", source="x", source_sha256="x")
