import json

import numpy as np

from para_audio_id.audio import BadFileRegistry, quantile_normalize
from para_audio_id.augment import convolve_ir


def test_silent_quantile_normalization_is_finite():
    output = quantile_normalize(np.zeros(100, dtype=np.float32))
    assert np.isfinite(output).all()
    assert output.dtype == np.float32


def test_bad_file_registry_is_persistent_and_deduplicated(tmp_path):
    path = tmp_path / "bad.jsonl"
    registry = BadFileRegistry(path)
    registry.add("001/a.mp3", ValueError("broken"))
    registry.add("001/a.mp3", RuntimeError("still broken"))
    assert BadFileRegistry(path).contains("001/a.mp3")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1


def test_ir_convolution_is_full_wet_peak_normalized():
    audio = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ir = np.array([0.5, 1.0], dtype=np.float32)
    output = convolve_ir(audio, ir)
    assert output.dtype == np.float32
    assert len(output) == len(audio)
    assert np.isclose(np.max(np.abs(output)), 1.0)
    assert np.allclose(output[:2], [0.5, 1.0])
