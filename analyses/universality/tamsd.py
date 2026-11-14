"""TA-MSD of the pieces of a flight on the 10-s grid, and the exponents read on a curve.

A piece is a stretch of a flight inside one segment of the cleaned track (the segments end at the gaps of the
logger) and inside the span of the flight, from its departure to its landing. Positions are interpolated linearly
on the grid GRID_S s. The MSD of a piece at the lag tau is the mean over its origins of |r(t + tau) - r(t)|^2;
it is kept as a sum and a number of origins, so that the pieces of a flight pool their origins. A lag tau needs a
piece of at least tau / SUPPORT s (the same rule as analyses/groups/longlags.py).

H is half the slope of log MSD against log tau, equal weight per lag; alpha(tau) = 2 H_loc(tau), with H_loc the
same slope within +/- HALF_DECADE of tau (the 3 nearest lags when fewer lie there).
"""

import numpy as np

from common.msd import tamsd_sums

GRID_S, SUPPORT, HALF_DECADE = 10, 0.8, 0.25
H_MAX = 10  # s: loggers up to this interval are on the grid at every instant
TAU_FIXED = 1000  # s: the lag of the fixed population
# lags: 10 s to 1000 s as analyses/groups/tamsd.py, then the same geometric step up to about 26 000 s
_STEP = TAU_FIXED ** (1 / 34)
LAGS = np.unique(np.r_[np.round(np.geomspace(1, 1000, 35)).astype(int) * 10, 100, 1000])
LAGS = LAGS[LAGS <= TAU_FIXED]
LONG_LAGS = np.unique(np.r_[LAGS, np.round(TAU_FIXED * _STEP ** np.arange(1, 17) / 10).astype(int) * 10])
SHORTEST = LONG_LAGS[0] / SUPPORT  # s: the shortest piece kept, long enough for the first lag


def grids(fixes, a, b, shortest=SHORTEST):
    """For one flight: every piece inside one segment and inside [a, b] (flight clock, s) that lasts at least
    `shortest` s, as its first instant and its positions E, N on the grid."""
    fixes = fixes.sort_values("t", kind="stable")
    t = fixes["t"].to_numpy(float)
    seg = fixes["segment_id"].to_numpy()
    for s in np.unique(seg):
        q = seg == s
        ts = t[q]
        g = np.arange(np.ceil(max(ts[0], a) / GRID_S) * GRID_S, min(ts[-1], b - 1e-6) + 1e-6, GRID_S)
        if len(g) < 2 or (len(g) - 1) * GRID_S < shortest:
            continue
        yield g[0], np.column_stack([np.interp(g, ts, fixes[c].to_numpy(float)[q]) for c in ("E", "N")])


def pieces(fixes, a, b, lags=LONG_LAGS):
    """The pieces of grids(): first instant, duration, and the sums and origins at `lags`; a lag the piece is too
    short for gets NaN and no origins."""
    out = []
    for start, pos in grids(fixes, a, b):
        d = lags // GRID_S
        ok = d < len(pos)
        sums, origins = np.full(len(d), np.nan), np.zeros(len(d), np.int64)
        sums[ok], origins[ok] = tamsd_sums(pos, d[ok])
        out.append((start, (len(pos) - 1) * GRID_S, sums, origins))
    return out


def msd_matrix(owner, duration, sums, origins, n, need):
    """MSD of n flights (rows) at each lag (columns): the pieces `owner` (flight row of each piece) that last at
    least `need[j]` s at the lag j pool their origins; NaN where a flight has no such piece."""
    s, o = np.zeros((n, sums.shape[1])), np.zeros((n, sums.shape[1]))
    for j in range(sums.shape[1]):
        keep = duration >= need[j]
        np.add.at(s[:, j], owner[keep], sums[keep, j])
        np.add.at(o[:, j], owner[keep], origins[keep, j])
    return np.where(o > 0, s / np.where(o > 0, o, 1), np.nan)


def fit_h(lags, msd):
    """Half the slope of log MSD against log lag, equal weight per lag; msd: (..., lags)."""
    x = np.log10(lags)
    return (np.log10(msd) @ ((x - x.mean()) / ((x - x.mean()) ** 2).sum())) / 2


def local_h(lags, msd):
    """H within +/- HALF_DECADE of each lag; the 3 nearest lags when fewer lie there; msd: (..., lags)."""
    x = np.log10(lags)
    op = np.zeros((len(x), len(x)))
    for i, c in enumerate(x):
        near = np.abs(x - c) <= HALF_DECADE + 1e-9
        if near.sum() < 3:
            near = np.isin(np.arange(len(x)), np.argsort(np.abs(x - c), kind="stable")[:3])
        xc = x[near] - x[near].mean()
        op[i, near] = xc / (xc**2).sum()
    return (np.log10(msd) @ op.T) / 2


def alpha(lags, msd):
    """alpha(tau) = 2 H_loc(tau); msd: (..., lags)."""
    return 2 * local_h(lags, msd)
