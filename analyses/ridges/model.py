"""Intermittent ball on a crest path.

The ball moves along the path at the speed v for a glide time, stops for a pause time, moves again, and so on.
Glide and pause times are independent Lomax durations above X0 (pauses.py): T = (tau0 + X0) U^(-1/mu) - tau0,
rounded to whole seconds. The first phase is drawn in equilibrium: glide with probability E[G] / (E[G] + E[P]),
its remaining time from the density S(t) / E[T]. Positions are given every 1 s, from the start of the path until
the ball reaches its end.
"""

import numpy as np

import flights as fl
from pauses import lomax_cond_mean


def draw(rng, n, mu, tau0, x0):
    """n durations (whole seconds, at least ceil(x0)) of the Lomax law conditional on exceeding x0."""
    t = (tau0 + x0) * rng.random(n) ** (-1 / mu) - tau0
    return np.maximum(np.round(t), np.ceil(x0)).astype(int)


def equilibrium_residual(rng, mu, tau0, x0):
    """Remaining time of the phase in progress at a random instant: density S(t) / E[T] (needs mu > 1)."""
    e, b = lomax_cond_mean(mu, tau0, x0), tau0 + x0
    ue = rng.random() * e
    if ue <= x0:
        return ue
    y = (ue - x0) * (mu - 1) / b
    return b * (1 - y) ** (-1 / (mu - 1)) - tau0


def moving_series(rng, length_m, v, glide, pause=None):
    """Boolean array, one value per second: True while the ball moves, until it has covered `length_m`.
    glide, pause: (mu, tau0, x0); pause None means a ball that never stops."""
    need = length_m / v  # seconds of motion
    if pause is None:
        return np.ones(int(np.ceil(need)), bool)
    eg, ep = lomax_cond_mean(*glide), lomax_cond_mean(*pause)
    moves = rng.random() < eg / (eg + ep)
    first = max(int(np.round(equilibrium_residual(rng, *(glide if moves else pause)))), 1)
    seq, flags, total = [np.array([first])], [np.array([moves])], first * moves
    while total < need:
        moves = not flags[-1][-1]
        n = int((need - total) / eg) + 10  # pairs of phases, from the one after the last
        g, p = draw(rng, n, *glide), draw(rng, n, *pause)
        d = np.empty(2 * n, int)
        d[0::2], d[1::2] = (g, p) if moves else (p, g)
        f = np.tile([moves, not moves], n)
        seq.append(d)
        flags.append(f)
        total += d[f].sum()
    m = np.repeat(np.concatenate(flags), np.concatenate(seq))
    return m[: np.searchsorted(np.cumsum(m), np.ceil(need)) + 1]


def positions(path, rng, v, glide, pause=None):
    """1-Hz positions of the intermittent ball on `path` (points every fl.STEP_M of arc length)."""
    arc = fl.STEP_M * np.arange(len(path))
    m = moving_series(rng, arc[-1], v, glide, pause)
    s = np.minimum(v * np.cumsum(m), arc[-1])
    return np.c_[np.interp(s, arc, path[:, 0]), np.interp(s, arc, path[:, 1])]


def null_positions(length_m, rng, v, glide, pause, kind):
    """1-Hz positions of a null model that keeps the durations and the speed of the crest model and changes the
    geometry of the motion. straight: every glide goes on along one fixed line. random: each glide is a straight
    segment in a direction drawn uniformly on the circle, independent of the previous one."""
    m = moving_series(rng, length_m, v, glide, pause)
    if kind == "straight":
        x = v * np.cumsum(m)
        return np.c_[x, np.zeros_like(x)]
    glide_index = np.cumsum(m & ~np.r_[False, m[:-1]])  # the glide each second of motion belongs to
    angle = rng.uniform(0, 2 * np.pi, glide_index[-1] + 1)[glide_index]
    return np.cumsum(v * m[:, None] * np.c_[np.cos(angle), np.sin(angle)], axis=0)


def model_rows(paths, rng, v, glide, pause, n_real=3, kind="crest"):
    """TAMSD on the 1-s lags fl.LAGS_T of the ball on each path, the pairs of n_real realisations pooled. kind
    crest: the ball runs on the path; straight and random: null models with the length of the path (null_positions)."""
    if kind == "crest":
        walk = lambda p: positions(p, rng, v, glide, pause)
    else:
        walk = lambda p: null_positions(fl.STEP_M * (len(p) - 1), rng, v, glide, pause, kind)
    return np.array([fl.pooled([walk(p) for _ in range(n_real)], fl.LAGS_T) for p in paths])
