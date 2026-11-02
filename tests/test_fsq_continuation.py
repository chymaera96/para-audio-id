from copy import deepcopy

import numpy  # noqa: F401 -- load before torch for local runtime compatibility
import pytest
import torch

from train_fsq import restore_checkpoint


def checkpoint():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.000235)
    model(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    cfg = {"train": {"run_id": "baseline", "consistency_weight": 0.0,
                     "max_steps": 50000, "learning_rate": 0.0003}, "model": {"size": 3}}
    return {"config": cfg, "split": {"training": [1, 2], "monitor": [3]},
            "step": 16000, "wandb_id": "baseline-id",
            "model": deepcopy(model.state_dict()), "optimizer": deepcopy(optimizer.state_dict())}


def test_continuation_preserves_weights_moments_lr_and_step():
    payload = checkpoint()
    cfg = deepcopy(payload["config"])
    cfg["train"].update(run_id="consistency", consistency_weight=0.1)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    step, wandb_id = restore_checkpoint(payload, cfg, payload["split"], model,
                                        optimizer, continuation=True)
    assert step == 16000 and wandb_id is None
    assert optimizer.param_groups[0]["lr"] == 0.000235
    for name, value in model.state_dict().items():
        assert torch.equal(value, payload["model"][name])
    for key, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            assert torch.equal(value, payload["optimizer"]["state"][key][name])
    assert payload["config"]["train"]["consistency_weight"] == 0.0


def test_continuation_requires_new_name_and_rejects_other_changes():
    payload = checkpoint()
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="new --run-id"):
        restore_checkpoint(payload, payload["config"], payload["split"], model,
                           optimizer, continuation=True)
    cfg = deepcopy(payload["config"])
    cfg["train"].update(run_id="consistency", consistency_weight=0.1, learning_rate=0.001)
    with pytest.raises(ValueError, match="beyond permitted"):
        restore_checkpoint(payload, cfg, payload["split"], model, optimizer, continuation=True)


def test_normal_resume_keeps_wandb_run_and_exact_configuration():
    payload = checkpoint()
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    assert restore_checkpoint(payload, payload["config"], payload["split"], model,
                              optimizer) == (16000, "baseline-id")
    cfg = deepcopy(payload["config"])
    cfg["train"]["consistency_weight"] = 0.1
    with pytest.raises(ValueError, match="beyond permitted"):
        restore_checkpoint(payload, cfg, payload["split"], model, optimizer)
