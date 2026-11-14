"""Glides and pauses of the selected flights, and their durations fitted by a Lomax law and by an exponential.

Runs come from Vilpellet's segmenter (analyses/phases/durations.py): a glide is a transition run; a pause is
consecutive search and climb runs joined into one phase. Only whole phases count: a phase touching the start or
the end of a segment is left out. A duration is the number of 1 Hz fixes of the phase.

Both laws are fitted on the durations above X0 = the shortest duration minus half a second, conditional on being
above it (durations are integers; X0 is the continuity correction):
  Lomax        S(t | t > X0) = ((tau0 + t) / (tau0 + X0))^-mu    two parameters
  exponential  S(t | t > X0) = exp(-(t - X0) / theta)            one parameter
Confidence intervals: bootstrap over whole flights.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

RUNS = "derived/segmentation/vilpellet/phase_segments.parquet"
DISK = "/Volumes/SSD_DISANTE"
FOLDERS = {"para": "paragliders/ffvl_cfd_igc", "hang": "hang_gliders/delta_cfd_igc"}
GLIDE, PAUSE = "glide", "pause"
KIND = {"transition": GLIDE, "search": PAUSE, "climb": PAUSE}
N_BOOT = 1000


def blocks(runs):
    """Whole phases of a table of runs: one row per glide or pause with flight_id, segment_id, kind, t_start and
    duration_s (fixes); consecutive search and climb runs are one pause."""
    r = runs[runs["phase"].isin(KIND)].sort_values(["flight_id", "segment_id", "t_start"], kind="stable")
    kind = r["phase"].map(KIND).to_numpy()
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


def lomax_cond_mean(mu, tau0, x0):
    """Mean of a Lomax duration conditional on exceeding x0 (infinite for mu <= 1)."""
    return x0 + (tau0 + x0) / (mu - 1) if mu > 1 else np.inf


def lomax_loglik(x, c, mu, tau0, x0):
    """Log likelihood of the values x with multiplicities c."""
    b = tau0 + x0
    return np.sum(c) * (np.log(mu) - np.log(b)) - (mu + 1) * np.sum(c * np.log((tau0 + x) / b))


def exp_loglik(x, c, theta, x0):
    return -np.sum(c) * np.log(theta) - np.sum(c * (x - x0)) / theta


def fit_lomax(x, c, x0, start=None):
    """Maximum likelihood (mu, tau0) for the values x with multiplicities c."""
    m = np.sum(c * x) / np.sum(c)
    s0 = np.log(start) if start is not None else [np.log(3.0), np.log(max(m - x0, 1.0))]
    def nll(p):
        with np.errstate(all="ignore"):
            v = -lomax_loglik(x, c, np.exp(np.clip(p[0], -5, 8)), np.exp(np.clip(p[1], -5, 18)), x0)
        return v if np.isfinite(v) else 1e300

    r = minimize(nll, s0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000})
    return np.exp(np.clip(r.x, [-5, -5], [8, 18]))


def fit_exp(x, c, x0):
    return np.sum(c * (x - x0)) / np.sum(c)


def histogram(x, w=None):
    """Distinct values and their (weighted) multiplicities."""
    u, inv = np.unique(x, return_inverse=True)
    return u, np.bincount(inv, weights=w, minlength=len(u))


def fit_phase(x, flight, n_boot=N_BOOT, seed=0):
    """Fits of the durations x (flight: the flight of each value) by both laws, with bootstrap intervals.

    Returns a dict: n, n_flights, x0, mean, mu, tau0, theta, aic_lomax, aic_exp, their bootstrap percentiles
    (2.5 and 97.5) under `<name>_ci`, and the fraction of resamples where the Lomax has the lower AIC.
    """
    x = np.asarray(x, float)
    x0 = x.min() - 0.5
    _, f = np.unique(flight, return_inverse=True)
    nf = f.max() + 1
    u, c = histogram(x)
    mu, tau0 = fit_lomax(u, c, x0)
    theta = fit_exp(u, c, x0)
    aic_l = 4 - 2 * lomax_loglik(u, c, mu, tau0, x0)
    aic_e = 2 - 2 * exp_loglik(u, c, theta, x0)
    out = {"n": len(x), "n_flights": nf, "x0": x0, "mean": x.mean(), "mu": mu, "tau0": tau0, "theta": theta,
           "mean_lomax": lomax_cond_mean(mu, tau0, x0), "aic_lomax": aic_l, "aic_exp": aic_e}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        w = rng.multinomial(nf, np.full(nf, 1 / nf))[f]
        ub, cb = histogram(x[w > 0], w[w > 0].astype(float))
        mb, tb = fit_lomax(ub, cb, x0, start=[mu, tau0])
        th = fit_exp(ub, cb, x0)
        boot.append([mb, tb, th, lomax_cond_mean(mb, tb, x0), 4 - 2 * lomax_loglik(ub, cb, mb, tb, x0) - (2 - 2 * exp_loglik(ub, cb, th, x0))])
    boot = np.array(boot)
    for i, name in enumerate(["mu", "tau0", "theta", "mean_lomax", "daic"]):
        out[name + "_ci"] = tuple(np.percentile(boot[:, i], [2.5, 97.5]))
    out["lomax_better"] = float(np.mean(boot[:, 4] < 0))
    return out


def survival(x):
    """Empirical survival P(X >= t) at the distinct values."""
    u, c = histogram(x)
    return u, 1 - np.r_[0, np.cumsum(c)[:-1]] / c.sum()


def lomax_survival(t, mu, tau0, x0):
    return ((tau0 + t) / (tau0 + x0)) ** -mu


def exp_survival(t, theta, x0):
    return np.exp(-(t - x0) / theta)
