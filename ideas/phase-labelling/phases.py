"""Vilpellet's flight-phase labelling, rewritten from his reference inference script.

Every window counts fixes and every parameter was fitted at 1 Hz, so the input must be a
regular 1-s sequence; regular_grid builds one from fixes at any cadence.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

STATES = ("climb", "search", "transition")
ALPHA = {"para": 0.20960958, "hang": 0.17398039}  # rad per fix
SMOOTH, TURN_MEAN, WINDOW, BETA, VOTE = 6, 6, 30, 0.9, 90
# Rows: climb, search, transition. Columns: the 8 symbols (straight, rising, one-sign)
# in binary order, (0,0,0) to (1,1,1).
EMISSION = np.array([
    [1.54243294e-11, 5.07106916e-04, 8.72065952e-05, 9.98979845e-01,
     8.52234426e-21, 1.25568196e-12, 7.85199401e-13, 4.25841826e-04],
    [3.25680580e-01, 1.77503496e-01, 4.95899060e-01, 7.21547291e-04,
     6.96932383e-05, 1.93093305e-05, 1.06025552e-04, 2.89805651e-07],
    [6.93534587e-06, 1.04473011e-08, 2.91554932e-06, 1.93493842e-10,
     7.38362033e-01, 1.40643862e-02, 2.41340621e-01, 6.22309818e-03],
])
TRANSITION = np.array([
    [0.98961359, 0.00346214, 0.00692427],
    [0.00666667, 0.99, 0.00333333],
    [0.00099934, 0.00712054, 0.99188012],
])
INITIAL = np.array([5.66075578e-02, 9.43392442e-01, 5.88699965e-29])


def enu(lat, lon, h):
    """East, north, up in metres from the first fix (WGS84)."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    n = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    ecef = np.stack([(n + h) * np.cos(phi) * np.cos(lam), (n + h) * np.cos(phi) * np.sin(lam),
                     (n * (1 - e2) + h) * np.sin(phi)], axis=-1)
    p0, l0 = phi[0], lam[0]
    rot = np.array([[-np.sin(l0), np.cos(l0), 0],
                    [-np.sin(p0) * np.cos(l0), -np.sin(p0) * np.sin(l0), np.cos(p0)],
                    [np.cos(p0) * np.cos(l0), np.cos(p0) * np.sin(l0), np.sin(p0)]])
    return ((ecef - ecef[0]) @ rot.T).T


def turn_sign_persistence(turn, window=WINDOW, beta=BETA):
    """1 where, among the window//2 largest |turn| of the window, one sign holds at least
    ceil(beta * window//2) of them. The window at fix k spans k-15 to k+14, reflected at the
    ends, as in the reference."""
    half = window // 2
    padded = np.pad(turn, (half, half), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)[: len(turn)]
    top = np.argpartition(-np.abs(windows), kth=half - 1, axis=1)[:, :half]
    signs = np.sign(np.take_along_axis(windows, top, axis=1))
    agree = np.maximum((signs > 0).sum(axis=1), (signs < 0).sum(axis=1))
    return (agree >= np.ceil(beta * half)).astype(int)


def features(t, e, n, u, alpha):
    """The three binary features of every fix, with the turn and vertical speed behind them."""
    pos = pd.DataFrame({"e": e, "n": n, "u": u})
    pos = pos.rolling(SMOOTH, win_type="gaussian", center=True).mean(std=SMOOTH / 6)
    heading = np.arctan2(np.diff(pos["n"]), np.diff(pos["e"]))
    raw = np.full(len(t), np.nan)
    raw[1:-1] = (heading[1:] - heading[:-1] + np.pi) % (2 * np.pi) - np.pi
    turn = pd.Series(raw).rolling(TURN_MEAN, center=True, min_periods=1).mean().to_numpy()
    vz = np.full(len(t), np.nan)
    vz[1:] = np.diff(pos["u"]) / np.diff(t)
    vz_mean = pd.Series(vz).rolling(WINDOW, center=True, min_periods=1).mean().to_numpy()
    straight = ((turn < alpha) & (turn > -alpha)).astype(float)
    straight_share = pd.Series(straight).rolling(WINDOW, center=True, min_periods=1).mean().to_numpy()
    obs = np.column_stack([straight_share > BETA, vz_mean > 0, turn_sign_persistence(turn)]).astype(int)
    return obs, turn, vz_mean


def viterbi(obs, emission=EMISSION, transition=TRANSITION, initial=INITIAL):
    """Most likely state sequence of the three binary features of every fix."""
    return viterbi_symbols(obs @ np.array([4, 2, 1]), emission, transition, initial)


def viterbi_symbols(symbols, emission, transition, initial):
    """Most likely state sequence of a symbol sequence; scores renormalised at every step,
    ties to the lower state."""
    b = emission[:, symbols].T
    k_len, s_len = b.shape
    psi = np.zeros((k_len, s_len), dtype=int)
    delta = initial * b[0]
    for k in range(1, k_len):
        scores = delta[:, None] * transition
        psi[k] = scores.argmax(axis=0)
        delta = scores.max(axis=0) * b[k]
        delta = delta / delta.sum()
    path = np.empty(k_len, dtype=int)
    path[-1] = delta.argmax()
    for k in range(k_len - 1, 0, -1):
        path[k - 1] = psi[k, path[k]]
    return path


def majority(states, width=VOTE):
    """Centred rolling majority; a tie goes to the lowest state present in the sequence."""
    present = np.unique(states)
    counts = np.column_stack([
        pd.Series((states == s).astype(int)).rolling(width, center=True, min_periods=1).sum()
        for s in present])
    return present[counts.argmax(axis=1)]


def label(t, e, n, u, alpha):
    """Final states of a regular 1-s sequence, and the Viterbi path before the vote."""
    obs, _, _ = features(t, e, n, u, alpha)
    path = viterbi(obs)
    return majority(path), path


def regular_grid(t, xyz, max_gap, cubic=False):
    """Pieces of a regular 1-s grid over the fixes: a gap longer than max_gap seconds ends a
    piece. Returns (grid times, interpolated xyz) per piece; fixes at whole seconds."""
    t = np.asarray(t, dtype=float)
    cut = np.flatnonzero(np.diff(t) > max_gap) + 1
    pieces = []
    for idx in np.split(np.arange(len(t)), cut):
        if len(idx) < 2:
            continue
        ts = t[idx]
        grid = np.arange(ts[0], ts[-1] + 0.5, 1.0)
        if cubic and len(idx) >= 4:
            vals = CubicSpline(ts, xyz[:, idx], axis=1)(grid)
        else:
            vals = np.vstack([np.interp(grid, ts, row[idx]) for row in xyz])
        pieces.append((grid, vals))
    return pieces
