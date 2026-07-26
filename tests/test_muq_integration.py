import os

import pytest
import torch

from para_audio_id.model import MuQEncoder


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_MUQ_INTEGRATION") != "1",
    reason="set RUN_MUQ_INTEGRATION=1 to load the real MuQ checkpoint",
)
def test_real_muq_forward_and_upper_block_backward():
    encoder = MuQEncoder("OpenMuQ/MuQ-large-msd-iter", encoder_dim=1024)
    audio = torch.zeros(1, 24_000)

    encoder.freeze_all()
    frozen = encoder(audio)
    assert frozen.ndim == 3
    assert not frozen.requires_grad

    upper = encoder.unfreeze_upper_fraction(0.25)
    encoder.train()
    output = encoder(audio)
    output.square().mean().backward()
    assert any(
        parameter.grad is not None
        for block in upper
        for parameter in block.parameters()
        if parameter.requires_grad
    )
