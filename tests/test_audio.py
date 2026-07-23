import json

import numpy as np

from para_audio_id.audio import BadFileRegistry, quantile_normalize


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
