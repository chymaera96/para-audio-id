import numpy as np
import pytest

from scripts.summarize_fsq_diagnostics import paired_interval


def test_paired_intervals_identical_and_improved():
    a = np.array([True, False, True, False])
    assert paired_interval(a, a) == {'difference_pp': 0., 'ci95_low_pp': 0., 'ci95_high_pp': 0.}
    m = paired_interval(np.ones(20, dtype=bool), np.zeros(20, dtype=bool))
    assert m == {'difference_pp': 100., 'ci95_low_pp': 100., 'ci95_high_pp': 100.}
    with pytest.raises(ValueError):
        paired_interval([True], [])
