import pytest

from para_audio_id.training import phase_two_due


def test_phase_two_begins_after_two_complete_exposures():
    assert not phase_two_due(0, 2)
    assert not phase_two_due(1, 2)
    assert phase_two_due(2, 2)
    assert phase_two_due(3, 2)
    with pytest.raises(ValueError, match="cannot be negative"):
        phase_two_due(0, -1)
