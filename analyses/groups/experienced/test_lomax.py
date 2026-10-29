import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from lomax import bands, fit  # noqa: E402


def test_fit_recovers_the_parameters():
    x = stats.lomax.rvs(c=1.7, scale=40, size=200_000, random_state=np.random.default_rng(0))
    alpha, scale = fit(x)
    assert alpha == pytest.approx(1.7, rel=0.03)
    assert scale == pytest.approx(40, rel=0.03)


def test_no_finite_maximum_for_a_tail_lighter_than_the_exponential():
    x = np.random.default_rng(2).gamma(4.0, 50.0, size=50_000)  # standard deviation / mean = 0.5
    assert np.isnan(fit(x)).all()


def test_a_weight_is_a_repetition():
    x, w = np.array([1.0, 2.0, 5.0, 20.0, 100.0]), np.array([3, 1, 2, 1, 4])
    assert fit(x, w) == pytest.approx(fit(np.repeat(x, w)))


def test_bands_contain_the_truth():
    rng = np.random.default_rng(1)
    x = stats.lomax.rvs(c=1.2, scale=15, size=20_000, random_state=rng)
    (a_lo, a_hi), (s_lo, s_hi) = bands(x, rng.integers(0, 2000, len(x)), rng, 200, (1, 99))
    assert a_lo < 1.2 < a_hi
    assert s_lo < 15 < s_hi
