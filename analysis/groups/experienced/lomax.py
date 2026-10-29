"""Maximum likelihood fit of the Lomax distribution, S(x) = (1 + x / scale)^(-alpha) for x >= 0."""

import numpy as np
from scipy.optimize import minimize_scalar


def mean_log(x, w, scale):
    """Mean of log(1 + x / scale), observation i counted w[i] times."""
    return np.average(np.log1p(x / scale), weights=w)


def fit(x, w=None):
    """The (alpha, scale) that maximise the log-likelihood of the sample x, observation i counted w[i] times:

        l(alpha, scale) / n = log(alpha) - log(scale) - (alpha + 1) * s,   s = mean(log(1 + x / scale)).

    For a given scale, dl/dalpha = 0 gives alpha = 1 / s. Putting it back, the log-likelihood per
    observation is  -log(s) - log(scale) - 1 - s,  a function of the scale alone, which is
    maximised over log(scale) in a window of e^12 around the mean of x. When the maximum lies at the
    edge of the window the likelihood keeps rising towards the exponential limit (alpha and scale
    to infinity, alpha / scale = 1 / mean): there is no finite maximum and the result is (nan, nan)."""
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    centre = np.log(np.average(x, weights=w))

    def negative(log_scale):
        s = mean_log(x, w, np.exp(log_scale))
        return np.log(s) + log_scale + 1 + s

    best = minimize_scalar(negative, bounds=(centre - 12, centre + 12), method="bounded")
    if abs(best.x - centre) > 11.9:
        return np.nan, np.nan
    scale = np.exp(best.x)
    return 1 / mean_log(x, w, scale), scale


def bands(x, blocks, rng, n_boot, percentiles=(5, 95)):
    """5-95% bands of alpha and scale over n_boot draws of the dependency blocks: a block is drawn
    with all its runs (as in tamsd.ensemble), so the weight of a run is the number of times its block
    is drawn. `x` are the durations, `blocks` the block of each; the fit runs on the distinct values."""
    values, value = np.unique(x, return_inverse=True)
    ids, block = np.unique(blocks, return_inverse=True)
    draws = []
    for _ in range(n_boot):
        times = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)))
        draws.append(fit(values, np.bincount(value, weights=times[block], minlength=len(values))))
    return np.nanpercentile(draws, percentiles, axis=0).T.tolist()  # [[alpha lo, hi], [scale lo, hi]]
