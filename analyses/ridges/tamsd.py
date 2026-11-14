"""Geometric TAMSD of curves sampled every STEP m of arc length, local slope alpha(s) and H on windows, with bootstrap.

A ball runs along each curve at constant speed V_KMH; a lag of k samples is an arc length s = k STEP and a
time tau = s / V. Per curve, delta2(k) = mean_i |r(i + k) - r(i)|^2 on the lags up to a quarter of the curve.
Across curves: equal weight per curve, lags with at least MIN_CURVES curves.
"""

import numpy as np
from numba import njit, prange

STEP = 5.0  # m
V_KMH = 35.0
N_LAGS = 80  # log-spaced lags up to a quarter of the longest curve
MAX_FRAC = 0.25  # lags up to this share of each curve
MIN_CURVES = 10
N_BOOT = 1000
FIT_FROM = 100.0  # m, no fit below four cells
WINDOWS = {"from 0.1 km": (100.0, np.inf), "0.1-1 km": (100.0, 1000.0), "1-10 km": (1000.0, 10000.0),
           "beyond 10 km": (10000.0, np.inf)}
SEED = 20260101


def lags(n_max, n=N_LAGS):
    """Log-spaced integer lags from 1 to the quarter of the longest curve (n_max samples)."""
    return np.unique(np.round(np.geomspace(1, max(n_max * MAX_FRAC, 2), n)).astype(np.int64))


@njit(cache=True, parallel=True)
def _tamsd(xy, ks):
    n = xy.shape[0]
    out = np.full(ks.size, np.nan)
    for j in prange(ks.size):
        k = ks[j]
        if k > n * 0.25 or k >= n:
            continue
        s = 0.0
        for i in range(n - k):
            dx, dy = xy[i + k, 0] - xy[i, 0], xy[i + k, 1] - xy[i, 1]
            s += dx * dx + dy * dy
        out[j] = s / (n - k)
    return out


def tamsd(xy, ks):
    """TAMSD (m^2) of one curve (n, 2) on the lags ks (samples); NaN beyond a quarter of the curve."""
    return _tamsd(np.ascontiguousarray(xy, np.float64), np.asarray(ks, np.int64))


def local_h(s, m):
    """alpha(s) = half the log-log slope of m, central differences (one-sided at the ends); NaN where m is not finite."""
    ls, lm = np.log(s), np.log(m)
    h = np.full(len(s), np.nan)
    ok = np.isfinite(lm)
    i = np.nonzero(ok)[0]
    if len(i) < 2:
        return h
    g = np.gradient(lm[i], ls[i]) / 2
    h[i] = g
    return h


def window_h(s, m, lo, hi):
    """Least-squares slope of ln m against ln s over lo <= s < hi, halved; NaN with fewer than 3 lags."""
    ok = (s >= lo) & (s < hi) & np.isfinite(m) & (m > 0)
    if ok.sum() < 3:
        return np.nan
    return np.polyfit(np.log(s[ok]), np.log(m[ok]), 1)[0] / 2


def weighted_means(rows, w):
    """Mean over curves with weights w (draws, curves) of rows (curves, lags; NaN = missing):
    returns means (draws, lags) with NaN where fewer than MIN_CURVES curves (with multiplicity), and the counts."""
    ok = np.isfinite(rows)
    r0 = np.where(ok, rows, 0.0)
    num = w @ r0
    den = w @ ok.astype(w.dtype)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(den >= MIN_CURVES, num / den, np.nan)
    return mean, den


def bootstrap_weights(group, n_boot=N_BOOT, seed=SEED):
    """(n_boot, curves) multiplicities: groups (trees) drawn with replacement, every member follows its group."""
    rng = np.random.default_rng(seed)
    _, inv = np.unique(group, return_inverse=True)
    g = inv.max() + 1
    out = np.empty((n_boot, len(group)), np.float32)
    for b in range(n_boot):
        out[b] = np.bincount(rng.integers(0, g, g), minlength=g)[inv]
    return out


def analyse(rows, ks, group, n_boot=N_BOOT):
    """Mean TAMSD, local slope and window fits with 95% bootstrap intervals.
    Returns (table of lags as dict of arrays, list of window rows (name, H, lo, hi))."""
    s = ks * STEP
    mean, den = weighted_means(rows, np.ones((1, len(rows)), np.float32))
    mean, den = mean[0], den[0]
    w = bootstrap_weights(group, n_boot)
    bm, _ = weighted_means(rows, w)
    h = local_h(s, mean)
    hb = np.array([local_h(s, m) for m in bm])
    ok = np.isfinite(rows)
    q = np.full((2, len(s)), np.nan)
    for j in range(len(s)):
        col = rows[ok[:, j], j]
        if len(col) >= MIN_CURVES:
            q[:, j] = np.percentile(col, [5, 95])
    tab = dict(s_m=s, tau_s=s / (V_KMH / 3.6), n=den.astype(int), mean=mean, q05=q[0], q95=q[1], H=h,
               H_lo=np.nanpercentile(hb, 2.5, axis=0), H_hi=np.nanpercentile(hb, 97.5, axis=0))
    wins = []
    for name, (lo, hi) in WINDOWS.items():
        est = window_h(s, mean, lo, hi)
        bs = np.array([window_h(s, m, lo, hi) for m in bm])
        bs = bs[np.isfinite(bs)]
        wins.append((name, lo / 1000, hi / 1000, est, *(np.percentile(bs, [2.5, 97.5]) if len(bs) > 20 else (np.nan, np.nan))))
    return tab, wins
