"""Synthetic scenarios behind the figures of the note, and the group sweep on pair intervals."""

from dataclasses import dataclass

import numpy as np


def runs(mask, t, step):
    """Maximal runs of True in ``mask``, as half-open intervals ``[start, end)`` of ``t``."""
    edges = np.diff(np.concatenate([[0], np.asarray(mask, dtype=int), [0]]))
    starts = t[np.flatnonzero(edges == 1)]
    ends = t[np.flatnonzero(edges == -1) - 1] + step
    return list(zip(starts.tolist(), ends.tolist()))


# Pair contact: j flies the route of i with a delay and a lateral offset.

PAIR = dict(speed=8.0, amplitude=600.0, wavelength=8000.0, delay=120.0, delta=300.0, eps_h=500.0,
            t_join=300.0, t_leave=800.0, offset=250.0, drift=3.0, step=5.0, t_end=1200.0)


def _route(s, amplitude, wavelength):
    y = amplitude * np.sin(2 * np.pi * s / wavelength)
    dy = amplitude * 2 * np.pi / wavelength * np.cos(2 * np.pi * s / wavelength)
    normal = np.stack([-dy, np.ones_like(s)]) / np.hypot(1, dy)
    return np.stack([s, y]), normal


def pair_tracks(p=PAIR):
    """Time grid and horizontal tracks (2, n) of i and j, in metres."""
    t = np.arange(0.0, p["t_end"] + p["step"], p["step"])
    xi, _ = _route(p["speed"] * t, p["amplitude"], p["wavelength"])
    base, normal = _route(p["speed"] * (t - p["delay"]), p["amplitude"], p["wavelength"])
    offset = p["offset"] + p["drift"] * (np.maximum(0, p["t_join"] - t) + np.maximum(0, t - p["t_leave"]))
    return t, xi, base - offset * normal


def correspondences(t, xi, xj, delta, eps_h):
    """Boolean matrix M[a, b]: (t[a], t[b]) is a correspondence, and the distance matrix."""
    dist = np.hypot(xi[0][:, None] - xj[0][None, :], xi[1][:, None] - xj[1][None, :])
    return (np.abs(t[None, :] - t[:, None]) <= delta) & (dist < eps_h), dist


# m_ij and m_ji differ: i waits in one place and j flies through it; j follows the route of i.

WAIT = dict(t_leave=600.0, t_pass=500.0, speed=10.0, delta=120.0, eps_h=500.0, step=1.0, t_end=700.0)
FOLLOW = dict(t_leave=600.0, delay=90.0, speed=10.0, delta=120.0, eps_h=500.0, step=1.0, t_end=700.0)


def wait_tracks(p=WAIT):
    """Time grid and horizontal tracks (2, n) of i, who circles at the origin until t_leave (NaN after), and
    of j, who flies east through the origin at t_pass."""
    t = np.arange(0.0, p["t_end"] + p["step"], p["step"])
    xi = np.where(t <= p["t_leave"], 0.0, np.nan) * np.ones((2, 1))
    xj = np.stack([p["speed"] * (t - p["t_pass"]), np.zeros_like(t)])
    return t, xi, xj


def follow_tracks(p=FOLLOW):
    """Time grid and tracks (2, n) of i, who flies east until t_leave, and of j, on the same route `delay` later
    (NaN while a pilot is not flying)."""
    t = np.arange(0.0, p["t_end"] + p["step"], p["step"])
    x = lambda s, on: np.where(on, p["speed"] * s, np.nan) * np.array([[1.0], [0.0]])
    return t, x(t, t <= p["t_leave"]), x(t - p["delay"], (t >= p["delay"]) & (t <= p["t_leave"] + p["delay"]))


# Altitude as a clock: i and j share a thermal, j lower by dz; scan the delay l.

THERMAL = dict(radius=25.0, period=20.0, climb=2.0, eps_h=500.0, eps_z=300.0, delta=300.0, dz=200.0,
               z_j0=1200.0, t_in=-120.0, glide_speed=9.0, sink=1.0, phase_i=1.0)


@dataclass
class ThermalScan:
    lags: np.ndarray
    xj: np.ndarray
    zj: np.ndarray
    xi: np.ndarray
    zi: float
    horizontal: np.ndarray
    ok_h: np.ndarray
    ok_z: np.ndarray

    @property
    def both(self):
        return self.ok_h & self.ok_z


def thermal_scan(p=THERMAL, step=1.0):
    """Position of j at t + l for every delay l, and which tolerance holds.

    j glides into the thermal at delay ``t_in`` and circles for the rest of the scan; i is on
    the same circle at time t, ``dz`` above j.
    """
    lags = np.arange(-p["delta"], p["delta"] + step / 2, step)
    omega = 2 * np.pi / p["period"]
    r = p["radius"]
    x, y, z = r * np.cos(omega * lags), r * np.sin(omega * lags), p["z_j0"] + p["climb"] * lags
    before = lags < p["t_in"]
    x_in, y_in = r * np.cos(omega * p["t_in"]), r * np.sin(omega * p["t_in"])
    x = np.where(before, x_in - p["glide_speed"] * (p["t_in"] - lags), x)
    y = np.where(before, y_in, y)
    z = np.where(before, p["z_j0"] + p["climb"] * p["t_in"] + p["sink"] * (p["t_in"] - lags), z)
    xi = np.array([r * np.cos(p["phase_i"]), r * np.sin(p["phase_i"])])
    zi = p["z_j0"] + p["dz"]
    horizontal = np.hypot(x - xi[0], y - xi[1])
    return ThermalScan(lags, np.stack([x, y]), z, xi, zi, horizontal,
                       horizontal < p["eps_h"], np.abs(zi - z) < p["eps_z"])


# Groups: configurations, episodes and blocks from pair intervals.

def minutes(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


DAY_FLIGHTS = {  # takeoff, departure (exit from the R_dep disc), landing
    1: ("11:00", "11:20", "15:00"),
    5: ("11:15", "11:45", "14:30"),
    8: ("11:05", "11:30", "16:00"),
    14: ("12:30", "12:50", "15:30"),
    22: ("11:10", "11:35", "16:00"),
}
DAY_PAIRS = [((1, 8), "11:40", "12:55"), ((8, 22), "11:40", "14:10"), ((1, 22), "12:05", "12:30"),
             ((8, 14), "13:30", "14:10"), ((8, 22), "15:00", "15:40")]


@dataclass(frozen=True)
class Configuration:
    members: frozenset
    start: int
    end: int


def example_day():
    """The example day of the note, in minutes since midnight."""
    flights = {f: tuple(minutes(x) for x in v) for f, v in DAY_FLIGHTS.items()}
    pairs = [(e, minutes(a), minutes(b)) for e, a, b in DAY_PAIRS]
    return flights, pairs


def active_components(flights, pairs, t):
    """Connected components with at least two flights of the active graph G(t)."""
    nodes = [f for f, (_, dep, end) in flights.items() if dep <= t < end]
    parent = {n: n for n in nodes}

    def find(n):
        while parent[n] != n:
            n = parent[n]
        return n

    for (a, b), on, off in pairs:
        if on <= t < off:
            parent[find(a)] = find(b)
    groups = {}
    for n in nodes:
        groups.setdefault(find(n), set()).add(n)
    return {frozenset(g) for g in groups.values() if len(g) > 1}


def sweep(flights, pairs):
    """Configurations, the episode of each configuration, and the dependency blocks."""
    times = sorted({x for _, a, b in pairs for x in (a, b)} | {x for _, d, e in flights.values() for x in (d, e)})
    open_, done, links = {}, [], []
    for te in times:
        comps = active_components(flights, pairs, te)
        closed = [Configuration(m, open_.pop(m), te) for m in list(open_) if m not in comps]
        done += closed
        for m in comps - set(open_):
            open_[m] = te
            links += [(c, Configuration(m, te, -1)) for c in closed if c.members & m]
    configs = sorted(done, key=lambda c: (c.start, sorted(c.members)))
    parent = {c: c for c in configs}

    def find(c):
        while parent[c] != c:
            c = parent[c]
        return c

    by_start = {(c.members, c.start): c for c in configs}
    for prev, nxt in links:
        parent[find(prev)] = find(by_start[(nxt.members, nxt.start)])
    roots = []
    for c in configs:
        if find(c) not in roots:
            roots.append(find(c))
    episode = {c: roots.index(find(c)) + 1 for c in configs}

    owner = {f: f for f in flights}

    def root(f):
        while owner[f] != f:
            f = owner[f]
        return f

    for e in set(episode.values()):
        members = sorted(set().union(*(c.members for c in configs if episode[c] == e)))
        for f in members[1:]:
            owner[root(f)] = root(members[0])
    blocks = {}
    for f in flights:
        blocks.setdefault(root(f), set()).add(f)
    return configs, episode, sorted(map(sorted, blocks.values()))


def label(flights, configs, f, t):
    """The membership gamma_f(t): 'L', 'S', a Configuration, or None outside the flight."""
    takeoff, dep, end = flights[f]
    if t < takeoff or t >= end:
        return None
    if t < dep:
        return "L"
    for c in configs:
        if f in c.members and c.start <= t < c.end:
            return c
    return "S"
