"""Box dimension and divider dimension of curves sampled every few metres, pooled over curves, with bootstrap.

Box counting: N(eps) = number of squares of side eps, on a grid shifted by N_OFFSETS fixed offsets, that hold a sample;
the mean over the offsets is used. Divider: length L(eps) of the curve walked with a compass of opening eps, from each
end (mean of the two). Local dimension D(eps) = -d ln N / d ln eps (box) or 1 - d ln L / d ln eps (divider).

N(eps) of an open curve has the form a / eps^D + END_BOX; END_BOX is removed before the fits (tests: line, fBm).

Pooling: a length class holds the curves of at least `lmin` m and the scales eps <= lmin / EPS_OVER_LEN, so that the same
curves are used at every scale of the class. N and L are divided by the length of the curve (N is proportional to the
length of a self-similar curve at fixed eps) and averaged with equal weight per curve. The bootstrap resamples whole
groups (trees, flights), as tamsd.py. The chord sqrt(MSD(s)) of the same curves ties eps to the arc length s.
"""

import numpy as np
from numba import njit, prange

import tamsd

EPS_MIN_M = 200.0  # twice the 100 m smoothing cut-off of the crests
EPS_MAX_M = 20000.0
PER_DECADE = 12  # box sizes per decade
N_OFFSETS = 8  # random shifts of the grid, in units of eps
EPS_OVER_LEN = 8  # a class of curves of at least lmin uses eps <= lmin / EPS_OVER_LEN
CLASS_MIN_KM = (2, 4, 8, 16, 32, 64)  # lmin of the length classes
MIN_GROUPS = 25  # a class is used on a window only if it holds at least this many trees (flights)
END_BOX = 1  # boxes of the two ends of an open curve: N - END_BOX is proportional to L / eps (exact for a line)
N_BOOT = tamsd.N_BOOT
SEED = 20260116


def eps_grid(lo=EPS_MIN_M, hi=EPS_MAX_M, per_decade=PER_DECADE):
    """Log-spaced box sizes from lo up to hi."""
    return lo * 10 ** (np.arange(int(np.floor(np.log10(hi / lo) * per_decade + 1e-9)) + 1) / per_decade)


def offsets(n=N_OFFSETS, seed=SEED):
    """(n, 2) grid shifts in units of eps, in [0, 1)."""
    return np.random.default_rng(seed).random((n, 2))


@njit(cache=True)
def _boxes(xy, a, b, eps, ox, oy):
    x0, y0 = xy[a, 0], xy[a, 1]
    x1, y1 = x0, y0
    for i in range(a, b):
        x0, x1 = min(x0, xy[i, 0]), max(x1, xy[i, 0])
        y0, y1 = min(y0, xy[i, 1]), max(y1, xy[i, 1])
    nx, ny = int((x1 - x0) / eps) + 2, int((y1 - y0) / eps) + 2
    grid = np.zeros(nx * ny, np.uint8)
    n, pi = 0, -1
    for i in range(a, b):
        ix = int((xy[i, 0] - x0) / eps + ox)
        iy = int((xy[i, 1] - y0) / eps + oy)
        k = ix * ny + iy
        if k != pi:
            pi = k
            if grid[k] == 0:
                grid[k] = 1
                n += 1
    return n


@njit(cache=True, parallel=True)
def count_many(xy, off, eps, offs):
    """Mean over the offsets of the number of occupied boxes: (curves, len(eps)). Curve i is xy[off[i]:off[i + 1]]."""
    nc, ne, no = len(off) - 1, len(eps), len(offs)
    tot = np.zeros((nc, ne, no))
    for t in prange(nc * ne * no):
        c, r = t // (ne * no), t % (ne * no)
        e, o = r // no, r % no
        tot[c, e, o] = _boxes(xy, off[c], off[c + 1], eps[e], offs[o, 0], offs[o, 1])
    return tot.sum(2) / no


@njit(cache=True)
def _walk(xy, a, b, eps, rev):
    n = b - a
    j0 = a + n - 1 if rev else a
    s = -1 if rev else 1
    cx, cy = xy[j0, 0], xy[j0, 1]
    qx, qy = cx, cy
    j, steps = 1, 0
    while j < n:
        px, py = xy[j0 + s * j, 0], xy[j0 + s * j, 1]
        if np.hypot(px - cx, py - cy) >= eps:
            dx, dy, fx, fy = px - qx, py - qy, qx - cx, qy - cy
            A, B, C = dx * dx + dy * dy, fx * dx + fy * dy, fx * fx + fy * fy - eps * eps
            t = min(max((-B + np.sqrt(max(B * B - A * C, 0.0))) / A, 0.0), 1.0)
            cx, cy = qx + t * dx, qy + t * dy
            qx, qy = cx, cy
            steps += 1
        else:
            qx, qy = px, py
            j += 1
    return steps * eps + np.hypot(qx - cx, qy - cy)


@njit(cache=True, parallel=True)
def divider_many(xy, off, eps):
    """Length (m) of every curve walked with a compass of opening eps (mean of the walks from each end): (curves, len(eps))."""
    nc, ne = len(off) - 1, len(eps)
    out = np.zeros((nc, ne))
    for t in prange(nc * ne):
        c, e = t // ne, t % ne
        out[c, e] = 0.5 * (_walk(xy, off[c], off[c + 1], eps[e], False) + _walk(xy, off[c], off[c + 1], eps[e], True))
    return out


def count_curves(curves, eps, offs):
    """count_many and divider_many of a list of (n, 2) arrays: returns N, L (curves, len(eps))."""
    off = np.r_[0, np.cumsum([len(c) for c in curves])].astype(np.int64)
    xy = np.ascontiguousarray(np.concatenate(curves))
    return count_many(xy, off, eps, offs), divider_many(xy, off, eps)


def local_slope(x, y):
    """d ln y / d ln x by central differences on the points where y is finite and positive."""
    out = np.full(len(x), np.nan)
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() >= 2:
        out[ok] = np.gradient(np.log(y[ok]), np.log(x[ok]))
    return out


def window_slope(x, y, lo, hi):
    """Least-squares slope of ln y against ln x over lo <= x <= hi; NaN with fewer than 3 points."""
    ok = (x >= lo) & (x <= hi) & np.isfinite(y) & (y > 0)
    return np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0] if ok.sum() >= 3 else np.nan


def dim_box(slope):
    return -slope


def dim_divider(slope):
    return 1 - slope


def pooled(v, w):
    """Weighted mean over curves of v (curves, scales) for weights w (draws, curves): (draws, scales)."""
    return (w @ v) / w.sum(1)[:, None]


def analyse_class(eps, n_box, n_div, length, rows, s, group, lmin, n_boot=N_BOOT, seed=SEED, ratio=EPS_OVER_LEN, end_box=END_BOX):
    """Pooled curves of the length class `lmin` (m): mean N/length, L/length and TAMSD over the curves of at least lmin,
    for the data (row 0) and n_boot bootstrap resamples of the groups. Scales above lmin / ratio are NaN.
    Returns a dict: eps, s, n, box, div (ln-free means, draws x eps), msd (draws x s)."""
    sel = length >= lmin
    g = group[sel]
    w = np.vstack([np.ones((1, sel.sum()), np.float32), tamsd.bootstrap_weights(g, n_boot, seed)])
    box, div = pooled((n_box[sel] - end_box) / length[sel, None], w), pooled(n_div[sel] / length[sel, None], w)
    top = eps > lmin / ratio
    box[:, top] = np.nan
    div[:, top] = np.nan
    msd, _ = tamsd.weighted_means(rows[sel], w)
    return dict(eps=eps, s=s, n=int(sel.sum()), n_groups=len(np.unique(g)), lmin=lmin, box=box, div=div, msd=msd)


def chord_to_s(s, msd):
    """Function eps -> arc length s with sqrt(msd(s)) = eps (log-log interpolation; NaN outside the range)."""
    ok = np.isfinite(msd) & (msd > 0)
    lc, ls = 0.5 * np.log(msd[ok]), np.log(s[ok])
    return lambda e: np.exp(np.interp(np.log(e), lc, ls, left=np.nan, right=np.nan))


def s_to_chord(s, msd):
    ok = np.isfinite(msd) & (msd > 0)
    ls, lc = np.log(s[ok]), 0.5 * np.log(msd[ok])
    return lambda x: np.exp(np.interp(np.log(x), ls, lc, left=np.nan, right=np.nan))


def local_table(res):
    """Local D(eps) box and divider, and 1/alpha(s(eps)) with alpha the local H of the same curves, each as (draws, eps)."""
    eps, s, msd = res["eps"], res["s"], res["msd"]
    box = np.array([dim_box(local_slope(eps, y)) for y in res["box"]])
    div = np.array([dim_divider(local_slope(eps, y)) for y in res["div"]])
    inv = np.full_like(box, np.nan)
    for i, m in enumerate(msd):
        alpha = tamsd.local_h(s, m)
        ok = np.isfinite(alpha) & np.isfinite(m) & (m > 0)
        if ok.sum() >= 2:
            inv[i] = 1 / np.interp(np.log(eps), 0.5 * np.log(m[ok]), alpha[ok], left=np.nan, right=np.nan)
    return box, div, inv


def eps_range(chord, lo_s, hi_s, eps_min=EPS_MIN_M):
    """Box sizes of the arc-length window [lo_s, hi_s): chord(lo_s) (at least eps_min) to chord(hi_s) (inf if beyond the data)."""
    lo, hi = chord(lo_s), chord(hi_s) if np.isfinite(hi_s) else np.inf
    return max(eps_min, lo if np.isfinite(lo) else eps_min), hi if np.isfinite(hi) else np.inf


def window_table(res, e_lo, e_hi):
    """Dimensions on the box sizes e_lo <= eps <= e_hi of a class (None if fewer than 3 grid sizes fall in it):
    arrays (draws,) of D_box, D_div and 1/H, the H being the TAMSD slope of the same curves on the arc lengths
    s(e_lo) to s(e_hi) given by their chord, and the eps and s ranges actually used."""
    eps, s, msd = res["eps"], res["s"], res["msd"]
    inside = (eps >= e_lo * (1 - 1e-9)) & (eps <= e_hi * (1 + 1e-9)) & np.isfinite(res["box"][0])
    if inside.sum() < 3:
        return None
    e0, e1 = eps[inside][0], eps[inside][-1]
    to_s = chord_to_s(s, msd[0])
    s_lo, s_hi = to_s(e0), to_s(e1)
    box = np.array([dim_box(window_slope(eps, y, e0, e1)) for y in res["box"]])
    div = np.array([dim_divider(window_slope(eps, y, e0, e1)) for y in res["div"]])
    h = np.array([tamsd.window_h(s, m, s_lo, s_hi * (1 + 1e-9)) for m in msd])
    return dict(eps_lo=e0, eps_hi=e1, s_lo=s_lo, s_hi=s_hi, n_eps=int(inside.sum()), box=box, div=div, h=h, inv_h=1 / h)


def pick_class(classes, e_hi, ratio=EPS_OVER_LEN):
    """Key (lmin) of the smallest class that reaches e_hi (lmin / ratio >= e_hi) with MIN_GROUPS groups; else the largest
    class with MIN_GROUPS groups; None if no class has them. `classes` maps lmin to the result of analyse_class."""
    ok = [l for l, r in sorted(classes.items()) if r["n_groups"] >= MIN_GROUPS]
    full = [l for l in ok if l / ratio >= e_hi]
    return full[0] if full else (ok[-1] if ok else None)


def interval(x):
    x = x[1:]
    x = x[np.isfinite(x)]
    return tuple(np.percentile(x, [2.5, 97.5])) if len(x) > 20 else (np.nan, np.nan)
