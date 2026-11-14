"""Glides and pauses of the selected flights, and their durations fitted by four laws.

Runs come from Vilpellet's segmenter (analyses/phases/durations.py): a glide is a transition run; a pause is a
climb run (consecutive runs of the pause phases joined). Search runs are in neither. Only whole phases count: a
phase touching the start or the end of a segment is left out. A duration is the number of 1 Hz fixes of the phase.

The laws are fitted by maximum likelihood on the durations above X0 = the shortest duration minus half a second,
conditional on being above it (durations are integers; X0 is the continuity correction):
  lomax        S(t | t > X0) = ((tau0 + t) / (tau0 + X0))^-mu               mu, tau0
  exponential  S(t | t > X0) = exp(-(t - X0) / theta)                       theta
  weibull      S(t) = exp(-(t / lam)^k), conditioned on t > X0                k, lam
  gamma        S(t) = Q(a, t / theta), conditioned on t > X0                  a, theta
The law of a group is the one with the lowest AIC. Confidence intervals: bootstrap over whole flights.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammainc, gammaincc, gammaincinv, gammainccinv, gammaln

RUNS = "derived/segmentation/vilpellet/phase_segments.parquet"
DISK = "/Volumes/SSD_DISANTE"
FOLDERS = {"para": "paragliders/ffvl_cfd_igc", "hang": "hang_gliders/delta_cfd_igc"}
GLIDE, PAUSE = "glide", "pause"
PAUSE_PHASES = ("climb",)  # the phases that make a pause
N_BOOT = 1000


def blocks(runs, pause_phases=PAUSE_PHASES):
    """Whole phases of a table of runs: one row per glide or pause with flight_id, segment_id, kind, t_start and
    duration_s (fixes); consecutive runs of the pause phases are one pause."""
    kinds = {"transition": GLIDE, **{p: PAUSE for p in pause_phases}}
    r = runs[runs["phase"].isin(kinds)].sort_values(["flight_id", "segment_id", "t_start"], kind="stable")
    kind = r["phase"].map(kinds).to_numpy()
    key = r["flight_id"].astype(str).to_numpy() + "/" + r["segment_id"].astype(str).to_numpy()
    head = np.r_[True, (kind[1:] != kind[:-1]) | (key[1:] != key[:-1])]
    # a run after an unlabelled one is not contiguous with it: break on any start not at the previous end + 1
    t0, t1 = r["t_start"].to_numpy(float), r["t_end"].to_numpy(float)
    head |= np.r_[True, t0[1:] != t1[:-1] + 1]
    b = np.cumsum(head) - 1
    df = pd.DataFrame({"b": b, "flight_id": r["flight_id"].astype(str).to_numpy(), "segment_id": r["segment_id"].to_numpy(int),
                       "kind": kind, "t_start": t0, "n": r["n_fixes"].to_numpy(float),
                       "cens": (r["left_censored"] | r["right_censored"]).to_numpy(bool)})
    out = df.groupby("b").agg(flight_id=("flight_id", "first"), segment_id=("segment_id", "first"), kind=("kind", "first"),
                              t_start=("t_start", "first"), duration_s=("n", "sum"), cens=("cens", "any"))
    return out[~out["cens"]].drop(columns="cens").reset_index(drop=True)


LAWS = ("lomax", "exponential", "weibull", "gamma")
N_PARAMS = {"lomax": 2, "exponential": 1, "weibull": 2, "gamma": 2}
PARAM_NAMES = {"lomax": ("mu", "tau0"), "exponential": ("theta",), "weibull": ("k", "lam"), "gamma": ("a", "theta")}


class Law:
    """A duration law conditional on exceeding x0: survival, quantiles, sampling, mean, equilibrium residual."""

    def __init__(self, name, params, x0):
        self.name, self.p, self.x0 = name, tuple(float(v) for v in params), float(x0)

    def isf(self, u):
        """Duration whose survival is u (0 < u <= 1; u = 1 gives x0)."""
        u, x0, p = np.asarray(u, float), self.x0, self.p
        if self.name == "lomax":
            return (p[1] + x0) * u ** (-1 / p[0]) - p[1]
        if self.name == "exponential":
            return x0 - p[0] * np.log(u)
        if self.name == "weibull":
            return p[1] * ((x0 / p[1]) ** p[0] - np.log(u)) ** (1 / p[0])
        return p[1] * gammainccinv(p[0], u * gammaincc(p[0], x0 / p[1]))

    def surv(self, t):
        """P(duration > t) given that it exceeds x0 (1 below x0)."""
        t, x0, p = np.maximum(np.asarray(t, float), self.x0), self.x0, self.p
        if self.name == "lomax":
            return ((p[1] + t) / (p[1] + x0)) ** -p[0]
        if self.name == "exponential":
            return np.exp(-(t - x0) / p[0])
        if self.name == "weibull":
            return np.exp(-((t / p[1]) ** p[0] - (x0 / p[1]) ** p[0]))
        return gammaincc(p[0], t / p[1]) / gammaincc(p[0], x0 / p[1])

    def sample(self, rng, n):
        """n durations in whole seconds, at least ceil(x0)."""
        return np.maximum(np.round(self.isf(rng.random(n))), np.ceil(self.x0)).astype(int)

    def _table(self):
        """The survival on a grid of levels, as (t, u) from u = 1 down to 1e-12."""
        if not hasattr(self, "_t"):
            u = np.geomspace(1, 1e-12, 3000)
            self._u, self._t = u, self.isf(u)
        return self._t, self._u

    def mean(self):
        t, u = self._table()
        return self.x0 + float(np.sum(0.5 * (u[1:] + u[:-1]) * np.diff(t)))

    def residual(self, rng):
        """Time left in the phase in progress at a random instant: density S(t) / E[T]."""
        t, u = self._table()
        area = np.r_[self.x0, self.x0 + np.cumsum(0.5 * (u[1:] + u[:-1]) * np.diff(t))]  # integral of S from 0 to t
        target = rng.random() * area[-1]
        if target <= self.x0:
            return target
        return float(np.interp(target, area, t))

    def to_dict(self):
        return {"name": self.name, "params": list(self.p), "x0": self.x0}

    @staticmethod
    def from_dict(d):
        return Law(d["name"], d["params"], d["x0"])


def loglik(name, p, x, c, x0):
    """Log likelihood of the values x with multiplicities c under the law conditional on exceeding x0."""
    n = np.sum(c)
    if name == "lomax":
        mu, tau0 = p
        return n * (np.log(mu) - np.log(tau0 + x0)) - (mu + 1) * np.sum(c * np.log((tau0 + x) / (tau0 + x0)))
    if name == "exponential":
        return -n * np.log(p[0]) - np.sum(c * (x - x0)) / p[0]
    if name == "weibull":
        k, lam = p
        return (n * (np.log(k) - np.log(lam)) + (k - 1) * np.sum(c * np.log(x / lam)) - np.sum(c * (x / lam) ** k) + n * (x0 / lam) ** k)
    a, th = p
    return np.sum(c * ((a - 1) * np.log(x) - x / th)) - n * (gammaln(a) + a * np.log(th)) - n * np.log(gammaincc(a, x0 / th))


def fit_law(name, x, c, x0, start=None):
    """Maximum likelihood parameters of one law for the values x with multiplicities c."""
    m = np.sum(c * x) / np.sum(c)
    if name == "exponential":
        return (np.sum(c * (x - x0)) / np.sum(c),)
    first = {"lomax": [3.0, max(m - x0, 1.0)], "weibull": [1.0, m], "gamma": [1.0, m]}[name]
    s0 = np.log(start if start is not None else first)
    lo, hi = np.log([1e-2, 1e-1]), np.log([3e3, 1e6])

    def nll(q):
        with np.errstate(all="ignore"):
            v = -loglik(name, np.exp(np.clip(q, lo, hi)), x, c, x0)
        return v if np.isfinite(v) else 1e300

    r = minimize(nll, s0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000})
    return tuple(np.exp(np.clip(r.x, lo, hi)))


def histogram(x, w=None):
    """Distinct values and their (weighted) multiplicities."""
    u, inv = np.unique(x, return_inverse=True)
    return u, np.bincount(inv, weights=w, minlength=len(u))


def fit_all(u, c, x0, start=None):
    """All four laws on the histogram (u, c): {name: (params, aic)}."""
    out = {}
    for name in LAWS:
        p = fit_law(name, u, c, x0, None if start is None else start.get(name))
        out[name] = (p, 2 * N_PARAMS[name] - 2 * loglik(name, p, u, c, x0))
    return out


def fit_phase(x, flight, n_boot=N_BOOT, seed=0):
    """Fits of the durations x (flight: the flight of each value) by the four laws; the law of the lowest AIC is `best`.

    Returns a dict: n, n_flights, x0, mean, fits {name: {params, aic, daic}}, best, mean_fit, and, with n_boot > 0, the
    2.5 and 97.5 percentiles of the parameters of the best law (`params_ci`), of its mean (`mean_fit_ci`), the fraction
    of resamples where each law is the best (`wins`) and the percentiles of the AIC of each law minus the best.
    """
    x = np.asarray(x, float)
    x0 = x.min() - 0.5
    _, f = np.unique(flight, return_inverse=True)
    nf = f.max() + 1
    u, c = histogram(x)
    fits = fit_all(u, c, x0)
    best = min(fits, key=lambda k: fits[k][1])
    law = Law(best, fits[best][0], x0)
    out = {"n": len(x), "n_flights": nf, "x0": x0, "mean": x.mean(), "best": best, "mean_fit": law.mean(),
           "fits": {k: {"params": list(p), "aic": a, "daic": a - fits[best][1]} for k, (p, a) in fits.items()}}
    if n_boot == 0:
        return out
    rng = np.random.default_rng(seed)
    start = {k: p for k, (p, _) in fits.items() if k != "exponential"}
    pars, means, daic = [], [], []
    wins = dict.fromkeys(LAWS, 0)
    for _ in range(n_boot):
        w = rng.multinomial(nf, np.full(nf, 1 / nf))[f]
        ub, cb = histogram(x[w > 0], w[w > 0].astype(float))
        fb = fit_all(ub, cb, x0, start)
        win = min(fb, key=lambda k: fb[k][1])
        wins[win] += 1
        pars.append(fb[best][0])
        means.append(Law(best, fb[best][0], x0).mean())
        daic.append([fb[k][1] - fb[win][1] for k in LAWS])
    out["params_ci"] = [list(np.percentile(np.array(pars)[:, i], [2.5, 97.5])) for i in range(N_PARAMS[best])]
    out["mean_fit_ci"] = list(np.percentile(means, [2.5, 97.5]))
    out["wins"] = {k: v / n_boot for k, v in wins.items()}
    out["daic_ci"] = {k: list(np.percentile(np.array(daic)[:, i], [2.5, 97.5])) for i, k in enumerate(LAWS)}
    return out


def survival(x):
    """Empirical survival P(X >= t) at the distinct values."""
    u, c = histogram(x)
    return u, 1 - np.r_[0, np.cumsum(c)[:-1]] / c.sum()
