"""Maximum likelihood fits of positive durations by an exponential, gamma, Weibull or lognormal law.

Every estimator takes the distinct values x and how many times each occurs, w, and returns the
parameters; a duration is taken as continuous. The exponential, gamma, Weibull and lognormal laws
have closed or one-dimensional equations; the three-parameter laws (Burr XII, Dagum, generalised
gamma) are maximised numerically.
"""

import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize
from scipy.special import digamma


def exponential(x, w):
    """scale = mean."""
    return (np.average(x, weights=w),)


def gamma(x, w):
    """(shape, scale). The scale equation gives scale = mean / shape; the shape solves
    log(shape) - digamma(shape) = log(mean) - mean(log x)."""
    mean = np.average(x, weights=w)
    s = np.log(mean) - np.average(np.log(x), weights=w)
    shape = brentq(lambda k: np.log(k) - digamma(k) - s, 1e-6, 1e6)
    return shape, mean / shape


def weibull(x, w):
    """(shape, scale). The scale equation gives scale^shape = mean(x^shape); the shape solves
    sum(x^k log x) / sum(x^k) - 1/k - mean(log x) = 0."""
    log_x = np.log(x)
    mean_log = np.average(log_x, weights=w)

    def score(k):
        xk = x**k
        return np.sum(w * xk * log_x) / np.sum(w * xk) - 1 / k - mean_log

    shape = brentq(score, 0.05, 20)
    return shape, np.average(x**shape, weights=w) ** (1 / shape)


def lognormal(x, w):
    """(mu, sigma) of log x: its mean and standard deviation."""
    log_x = np.log(x)
    mu = np.average(log_x, weights=w)
    return mu, np.sqrt(np.average((log_x - mu) ** 2, weights=w))


def numerical(law, starts):
    """Estimator of the weighted maximum likelihood of the scipy law `law` (shapes, then scale): the
    log-likelihood is maximised over the logs of the parameters by Nelder-Mead from each of `starts`
    (shapes; the scale starts at the median), keeping the best."""

    def estimate(x, w):
        median = x[np.searchsorted(np.cumsum(w), w.sum() / 2)]

        def negative(q):
            return -np.sum(w * law.logpdf(x, *np.exp(q[:-1]), scale=np.exp(q[-1])))

        best = min((minimize(negative, np.log([*shapes, median]), method="Nelder-Mead",
                             options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000}) for shapes in starts),
                   key=lambda r: r.fun)
        return tuple(np.exp(best.x))

    return estimate


# name: estimator, parameter names, the scipy law of a set of parameters
FAMILIES = {
    "exponential": (exponential, ("scale",), lambda p: stats.expon(scale=p[0])),
    "gamma": (gamma, ("shape", "scale"), lambda p: stats.gamma(p[0], scale=p[1])),
    "Weibull": (weibull, ("shape", "scale"), lambda p: stats.weibull_min(p[0], scale=p[1])),
    "lognormal": (lognormal, ("mu", "sigma"), lambda p: stats.lognorm(p[1], scale=np.exp(p[0]))),
    # S(x) = (1 + (x / scale)^c)^-k: flat at small x, a power law x^(-c k) at large x
    "Burr XII": (numerical(stats.burr12, [(1.5, 1.0), (3.0, 0.5)]), ("c", "k", "scale"),
                 lambda p: stats.burr12(p[0], p[1], scale=p[2])),
    # F(x) = (1 + (x / scale)^-c)^-k: a power law x^(-c) at large x
    "Dagum": (numerical(stats.burr, [(2.0, 1.0), (4.0, 0.5)]), ("c", "k", "scale"),
              lambda p: stats.burr(p[0], p[1], scale=p[2])),
    # f(x) ~ x^(c a - 1) exp(-(x / scale)^c): the gamma with a stretched exponential
    "generalised gamma": (numerical(stats.gengamma, [(1.0, 1.0), (0.5, 1.5)]), ("a", "c", "scale"),
                          lambda p: stats.gengamma(p[0], p[1], scale=p[2])),
}


def aic(family, x, w, parameters):
    """2 * parameters - 2 * log-likelihood of the sample (x, w)."""
    law = FAMILIES[family][2](parameters)
    return 2 * len(parameters) - 2 * float(np.sum(w * law.logpdf(x)))


def bands(estimate, x, blocks, rng, n_boot, percentiles=(5, 95)):
    """5-95% band of each parameter of `estimate` over n_boot draws of the dependency blocks: a
    block is drawn with all its runs (as in tamsd.ensemble), so the weight of a run is the number
    of times its block is drawn. `x` are the durations, `blocks` the block of each."""
    values, value = np.unique(x, return_inverse=True)
    ids, block = np.unique(blocks, return_inverse=True)
    draws = []
    for _ in range(n_boot):
        times = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)))
        draws.append(estimate(values, np.bincount(value, weights=times[block], minlength=len(values))))
    return np.percentile(draws, percentiles, axis=0).T.tolist()  # one [lo, hi] per parameter
