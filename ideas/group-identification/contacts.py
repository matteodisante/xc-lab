"""Phases 1-3 of the note on one date, and the surrogate pairs of the null model.

Input: the flight table and the grid of grid.day(). Phase 1 finds the candidate pairs with a
KD-tree on one point per minute (Lemma "no false negative"), Phase 2 scans their delays
(Algorithm "scan"), Phase 3 turns the contact into pair intervals. The minimum duration tau*
is not applied here: it comes from the null model, over all dates.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SEARCH, GLIDE = 1, 2  # state codes of grid.py
K = 1e8  # metres between two minutes in the KD-tree, far more than D_c
CHUNK = 2000  # window instants scanned at once
APART_BIN = 250  # m, bins of the distance at the same instant in a profile
VALUES = ["X", "Y", "Z", "z", "ve", "vn", "state"]


class Tracks:
    """The grid of one date as dense arrays: row k of flight f is the instant first[f] + k step,
    NaN where the flight has no position."""

    def __init__(self, grid, step):
        self.step, self.first, self.values = step, {}, {}
        for f, g in grid.groupby("f", sort=False):
            t = g["t"].to_numpy()
            v = np.full(((t.max() - t.min()) // step + 1, len(VALUES)), np.nan)
            v[(t - t.min()) // step] = g[VALUES].to_numpy(float)
            self.first[f], self.values[f] = t.min(), v

    def shifted(self, f, r, key):
        """Add under `key` the copy of flight f shifted in time, x(t) = x_f(t + r)."""
        self.first[key], self.values[key] = self.first[f] - r, self.values[f]

    def at(self, f, t):
        v, k = self.values[f], (t - self.first[f]) // self.step
        inside = (k >= 0) & (k < len(v))
        out = np.full(t.shape + (v.shape[1],), np.nan)
        out[inside] = v[k[inside]]
        return out


def minute_points(grid, delta, v_max):
    """Phase 1, step 1: one point per whole minute of each extended piece. A piece is a run of
    grid rows of one segment of one flight with no step faster than v_max, extended by
    holding its first and last positions; its minutes go from delta + 60 s before its first
    instant to delta + 60 s after its last. Returns the flight, minute (UTC s) and position
    of each point."""
    grid = grid.sort_values(["f", "seg", "t"], kind="stable")
    f, seg, t = (grid[c].to_numpy() for c in ("f", "seg", "t"))
    p = grid[["X", "Y", "Z"]].to_numpy(float)
    fast = np.linalg.norm(np.diff(p, axis=0), axis=1) > v_max * np.diff(t)
    starts = np.flatnonzero(np.r_[True, (f[1:] != f[:-1]) | (seg[1:] != seg[:-1]) | fast])
    owner, minute, xyz = [], [], []
    for a, b in zip(starts, np.r_[starts[1:], len(t)]):
        tc = np.arange(np.ceil((t[a] - delta - 60) / 60) * 60, t[b - 1] + delta + 60 + 1e-6, 60).astype(np.int64)
        owner.append(np.full(len(tc), f[a]))
        minute.append(tc)
        xyz.append(np.column_stack([np.interp(tc, t[a:b], p[a:b, c]) for c in range(3)]))
    return np.concatenate(owner), np.concatenate(minute), np.concatenate(xyz)


def candidates(grid, delta, eps_h, v_max):
    """Phase 1: the candidate pairs i < j and the minutes tc at which their pieces are within
    D_c = eps_h + v_max (delta + 60 s), from a single KD-tree on (X, Y, Z, K tc / 60)."""
    owner, minute, xyz = minute_points(grid, delta, v_max)
    tree = cKDTree(np.column_stack([xyz, K * (minute - minute.min()) / 60]))
    pairs = tree.query_pairs(eps_h + v_max * (delta + 60), output_type="ndarray")
    a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
    other = a != b
    found = pd.DataFrame({"i": np.minimum(a, b)[other], "j": np.maximum(a, b)[other], "tc": minute[pairs[other, 0]]})
    return found.drop_duplicates().sort_values(["i", "j", "tc"]).reset_index(drop=True)


def window(minutes, step):
    """The window W_ij: the grid instants within 30 s of a minute that passed the filter."""
    return np.unique((minutes[:, None] + np.arange(-30, 31, step)[None, :]).ravel())


def scan(tracks, i, j, instants, delta, eps_h, eps_z, cos_max, profile=None):
    """Phase 2 on one pair: the instants t of i in `instants` with m_ij(t) = 1, the delay l* of
    the accepted correspondence closest in 3D at each, whether either pilot is in search there, and
    the instants of j with m_ji = 1. The heading condition holds only when both glide.
    With `profile`, also adds to profile["lags"] the correspondences found at each delay, from
    -delta to delta, and to profile["apart"] the distance of i and j at the same instant, in
    bins of APART_BIN m (the last bin holds the rest), at the instants with m_ij = 1."""
    lags = tracks.step * np.arange(-(delta // tracks.step), delta // tracks.step + 1)
    hits, best, search, other = [], [], [], []
    for start in range(0, len(instants), CHUNK):
        ti = instants[start:start + CHUNK]
        xi = tracks.at(i, ti)
        ti, xi = ti[np.isfinite(xi[:, 0])], xi[np.isfinite(xi[:, 0])]
        tj = ti[:, None] + lags[None, :]
        xj = tracks.at(j, tj)
        d = np.linalg.norm(xi[:, None, :3] - xj[..., :3], axis=2)
        dz = np.abs(xi[:, None, 3] - xj[..., 3])
        glide = (xi[:, None, 6] == GLIDE) & (xj[..., 6] == GLIDE)
        aligned = xi[:, None, 4] * xj[..., 4] + xi[:, None, 5] * xj[..., 5] > cos_max
        ok = (d < eps_h) & (dz < eps_z) & (~glide | aligned)
        hit = ok.any(axis=1)
        k = np.where(ok, np.hypot(d, dz), np.inf)[hit].argmin(axis=1)
        hits.append(ti[hit])
        best.append(lags[k])
        search.append((xi[hit, 6] == SEARCH) | (xj[hit, k, 6] == SEARCH))
        other.append(tj[ok])
        if profile is not None:
            profile["lags"] += ok.sum(axis=0)
            now = d[hit, len(lags) // 2]
            now = now[np.isfinite(now)]
            np.add.at(profile["apart"], np.minimum(now // APART_BIN, len(profile["apart"]) - 1).astype(int), 1)
    return (np.concatenate(hits), np.concatenate(best), np.concatenate(search),
            np.unique(np.concatenate(other)) if other else np.array([], np.int64))


def runs_of(times, step, gap):
    """Maximal runs of sorted grid instants, joined across gaps shorter than `gap`, as the
    half-open intervals [on, off)."""
    cut = np.flatnonzero(np.diff(times) - step >= gap)
    return times[np.r_[0, cut + 1]], times[np.r_[cut, len(times) - 1]] + step


def pair_intervals(tracks, i, j, instants, span, p):
    """Phase 3 on one pair: the intervals of contact a_ij = m_ij or m_ji, restricted to `span`
    (lo, hi), the time both flights are between departure and landing, with gaps shorter than
    p["gap"] joined; and the signed median delay of each (positive when j follows i), over
    the instants of i with m_ij = 1 and no pilot in search (NaN when there are none)."""
    ti, lag, search, tj = scan(tracks, i, j, instants, p["delta"], p["eps_h"], p["eps_z"], np.cos(np.radians(p["theta_max"])))
    a = np.union1d(ti, tj)
    a = a[(a >= span[0]) & (a <= span[1])]
    if not a.size:
        return []
    out = []
    for on, off in zip(*runs_of(a, tracks.step, p["gap"])):
        inside = (ti >= on) & (ti < off) & ~search
        out.append((int(on), int(off), float(np.median(lag[inside])) if inside.any() else np.nan))
    return out


def real_intervals(table, grid, tracks, p):
    """Phases 1-3 on every candidate pair of the date: a, b (rows of the flight table, a < b),
    t_on, t_off, ell_med (positive when b follows a) and d_dec, the distance of the takeoffs."""
    rows = []
    found = candidates(grid, p["delta"], p["eps_h"], p["v_max"])
    for (i, j), c in found.groupby(["i", "j"], sort=False):
        span = (max(table.departure[i], table.departure[j]), min(table.landing[i], table.landing[j]))
        for on, off, ell in pair_intervals(tracks, i, j, window(c["tc"].to_numpy(), tracks.step), span, p):
            rows.append((i, j, on, off, ell))
    out = pd.DataFrame(rows, columns=["a", "b", "t_on", "t_off", "ell_med"])
    out = out.astype({"a": np.int64, "b": np.int64, "t_on": np.int64, "t_off": np.int64, "ell_med": float})
    out["d_dec"] = takeoff_distance(table, out["a"].to_numpy(), out["b"].to_numpy())
    return out


def takeoff_distance(table, a, b):
    x0 = table[["X0", "Y0", "Z0"]].to_numpy(float)
    return np.linalg.norm(x0[a] - x0[b], axis=1)


def exposure(table, a, b, r=0):
    """Time both flights spend between departure and landing, with flight b shifted by r."""
    lo = np.maximum(table.departure.to_numpy()[a], table.departure.to_numpy()[b] - r)
    hi = np.minimum(table.landing.to_numpy()[a], table.landing.to_numpy()[b] - r)
    return np.maximum(hi - lo, 0.0)


def near_pairs(table, r_s):
    """Pairs a < b of flights with grid instants whose takeoffs are less than r_s apart."""
    ok = np.flatnonzero(table.n_grid.to_numpy() > 0)
    a, b = np.triu_indices(len(ok), 1)
    a, b = ok[a], ok[b]
    near = takeoff_distance(table, a, b) < r_s
    return a[near], b[near]


def surrogates(table, rng, p, step):
    """The surrogate pairs of the null model on one date: up to p["n_null"] pairs with takeoffs
    within p["r_s"], drawn at random, each as (i, j, r): the copy of j shifted by
    r = (k0_j - k0_i) - theta, theta drawn from the takeoff delays of these pairs,
    |r| >= p["r_min"], r a multiple of the step."""
    a, b = near_pairs(table, p["r_s"])
    if not a.size:
        return []
    k0 = table.takeoff.to_numpy()
    delays = np.r_[k0[b] - k0[a], k0[a] - k0[b]]
    out = []
    for n in rng.permutation(len(a))[:p["n_null"]]:
        i, j = (a[n], b[n]) if rng.random() < 0.5 else (b[n], a[n])
        r = (k0[j] - k0[i]) - rng.choice(delays, 50)
        r = r[np.abs(r) >= p["r_min"]]
        if r.size:
            out.append((i, j, int(np.round(r[0] / step) * step)))
    return out


def surrogate(table, grid, tracks, i, j, r, p):
    """Add to `tracks` the copy of j shifted by r; return its key, the instants of the copy
    where the filter passes against i (None if nowhere), and the span when both fly."""
    copy = ("shift", j)
    tracks.shifted(j, r, copy)
    g = grid[grid.f.isin([i, j])].copy()
    g.loc[g.f == j, "t"] -= r
    g.loc[g.f == j, "f"] = -1
    found = candidates(g, p["delta"], p["eps_h"], p["v_max"])  # the copy is flight -1, so found.i == -1
    lo = max(table.departure[i], table.departure[j] - r)
    hi = min(table.landing[i], table.landing[j] - r)
    instants = None if found.empty or hi <= lo else window(found["tc"].to_numpy(), tracks.step)
    return copy, instants, (lo, hi)


def null_intervals(table, grid, tracks, rng, p):
    """The null model on one date: the surrogate pairs (i, j, r, exposure) and the intervals of
    their contacts (i, j, duration)."""
    pairs, durations = [], []
    for i, j, r in surrogates(table, rng, p, tracks.step):
        copy, instants, (lo, hi) = surrogate(table, grid, tracks, i, j, r, p)
        pairs.append((i, j, r, max(hi - lo, 0.0)))
        if instants is None:
            continue
        for on, off, _ in pair_intervals(tracks, copy, i, instants, (lo, hi), p):
            durations.append((i, j, off - on))
    return pd.DataFrame(pairs, columns=["i", "j", "r", "exposure"]), pd.DataFrame(durations, columns=["i", "j", "duration"])
