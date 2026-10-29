import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from families import FAMILIES, aic, bands  # noqa: E402

N = 100_000
RNG = np.random.default_rng(3)
SAMPLES = {  # family: the parameters drawn, and a sample of N
    "exponential": ((50.0,), stats.expon(scale=50.0)),
    "gamma": ((2.0, 30.0), stats.gamma(2.0, scale=30.0)),
    "Weibull": ((1.5, 100.0), stats.weibull_min(1.5, scale=100.0)),
    "lognormal": ((4.0, 0.8), stats.lognorm(0.8, scale=np.exp(4.0))),
    "Burr XII": ((2.0, 1.5, 60.0), stats.burr12(2.0, 1.5, scale=60.0)),
    "Dagum": ((2.5, 0.8, 100.0), stats.burr(2.5, 0.8, scale=100.0)),
    "generalised gamma": ((0.8, 1.3, 50.0), stats.gengamma(0.8, 1.3, scale=50.0)),
}


@pytest.mark.parametrize("family", SAMPLES)
def test_fit_recovers_the_parameters(family):
    truth, law = SAMPLES[family]
    x = law.rvs(size=N, random_state=RNG)
    assert FAMILIES[family][0](x, np.ones(N)) == pytest.approx(truth, rel=0.06)


@pytest.mark.parametrize("family", SAMPLES)
def test_a_weight_is_a_repetition(family):
    sample = np.round(stats.gamma(1.5, scale=60.0).rvs(size=3000, random_state=np.random.default_rng(4))) + 1
    x, w = np.unique(sample, return_counts=True)
    estimate = FAMILIES[family][0]
    assert estimate(x, w) == pytest.approx(estimate(sample, np.ones(len(sample))), rel=1e-3)


def test_the_true_family_has_the_lowest_aic():
    x = stats.gamma(0.7, scale=80.0).rvs(size=N, random_state=RNG)
    w = np.ones(N)
    scores = {f: aic(f, x, w, FAMILIES[f][0](x, w)) for f in FAMILIES}
    assert min(scores, key=scores.get) == "gamma"


def test_bands_contain_the_truth():
    rng = np.random.default_rng(5)
    x = stats.gamma(2.0, scale=30.0).rvs(size=20_000, random_state=rng)
    (k_lo, k_hi), (s_lo, s_hi) = bands(FAMILIES["gamma"][0], x, rng.integers(0, 2000, len(x)), rng, 200, (0.5, 99.5))
    assert k_lo < 2.0 < k_hi
    assert s_lo < 30.0 < s_hi
