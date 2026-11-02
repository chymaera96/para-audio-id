"""Check worker isolation after parent-only services have been initialized."""
import os
import sys

import numpy  # noqa: F401 -- load before torch for local runtime compatibility
import torch

from train_fsq import build_loader


_PARENT_RUNTIME_ACTIVE = False


class ForkSensitiveDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        if _PARENT_RUNTIME_ACTIVE:
            raise RuntimeError("Worker inherited a live parent-only service")
        # Exercise a lazy import in the worker, as in the reported failure.
        import joblib  # noqa: F401
        return index, os.getpid()


def test_workers_isolate_parent_runtime(monkeypatch):
    loader = build_loader(ForkSensitiveDataset(), {"batch_size": 2, "workers": 1},
                          torch.device("cpu"))
    # DataLoader workers have not started yet. This models wandb.init() running
    # between loader construction and its first iteration.
    monkeypatch.setattr(sys.modules[__name__], "_PARENT_RUNTIME_ACTIVE", True)
    batches = list(loader)
    assert torch.cat([batch[0] for batch in batches]).tolist() == [0, 1, 2, 3]
    assert all(int(pid) != os.getpid() for batch in batches for pid in batch[1])


def test_loader_without_workers():
    loader = build_loader(torch.utils.data.TensorDataset(torch.arange(4)),
                          {"batch_size": 2, "workers": 0}, torch.device("cpu"))
    assert torch.cat([batch[0] for batch in loader]).tolist() == [0, 1, 2, 3]
