"""Raw IGC tracks as input of flights.py: the check of the cleaned fixes against the raw ones, and the tests.

A flight is read, cleaned (increasing times, spikes), cut to the time in the air and given as the arrays of
flights.process: times, Lambert-93 positions and the index of the piece (a new one after a jump).
"""

import re
from pathlib import Path

import numpy as np

import flights as fl
from flights import _runs

MAX_JUMP_KMH = 150.0  # a faster step starts a new piece
AIR_KMH, AIR_S, AIR_WIN_S = 15.0, 30, 10  # in the air: speed over AIR_WIN_S above AIR_KMH for AIR_S


_B = re.compile(rb"^B(\d\d)(\d\d)(\d\d)(\d\d)(\d{5})([NS])(\d{3})(\d{5})([EW])([AV])", re.M)


def read_igc(path):
    """Time (s, midnight unwrapped), latitude, longitude, 3D flag of every B record."""
    b = _B.findall(Path(path).read_bytes().replace(b"\r", b"\n"))
    if not b:
        return (np.empty(0),) * 4
    f = lambda i: np.array([int(r[i]) for r in b], float)
    t = f(0) * 3600 + f(1) * 60 + f(2)
    lat = (f(3) + f(4) / 60000) * np.where([r[5] == b"N" for r in b], 1, -1)
    lon = (f(6) + f(7) / 60000) * np.where([r[8] == b"E" for r in b], 1, -1)
    t = t + 86400 * np.cumsum(np.r_[0, np.diff(t) < -43200])
    return t, lat, lon, np.array([r[9] == b"A" for r in b])


def clean(t, lat, lon, valid):
    """Fixes with increasing time and a position, as time and Lambert-93 coordinates."""
    ok = (lat != 0) & (lon != 0)  # fixes flagged V keep their horizontal position
    t, lat, lon = t[ok], lat[ok], lon[ok]
    if len(t) == 0:
        return t, np.empty((0, 2))
    inc = np.r_[True, t[1:] > np.maximum.accumulate(t)[:-1]]
    t, lat, lon = t[inc], lat[inc], lon[inc]
    return t, np.c_[fl.lambert93(lat, lon)]


def _speed_kmh(t, xy):
    return 3.6 * np.hypot(*np.diff(xy, axis=0).T) / np.diff(t)


def despike(t, xy):
    """Drop a fix whose steps in and out are both faster than MAX_JUMP_KMH."""
    for _ in range(3):
        v = _speed_kmh(t, xy) > MAX_JUMP_KMH
        bad = np.r_[False, v[:-1] & v[1:], False]
        if not bad.any():
            break
        t, xy = t[~bad], xy[~bad]
    return t, xy



def cut_air(t, xy):
    """Fixes between the first and the last stretch of AIR_S seconds faster than AIR_KMH; None if no stretch."""
    g = np.arange(t[0], t[-1] + 1)
    x, y = np.interp(g, t, xy[:, 0]), np.interp(g, t, xy[:, 1])
    h = AIR_WIN_S // 2
    i, j = np.clip(np.arange(len(g)) - h, 0, len(g) - 1), np.clip(np.arange(len(g)) + h, 0, len(g) - 1)
    speed = 3.6 * np.hypot(x[j] - x[i], y[j] - y[i]) / np.maximum(j - i, 1)
    a, b = _runs(speed > AIR_KMH)
    keep = b - a >= AIR_S
    if not keep.any():
        return None
    inside = (t >= g[a[keep][0]]) & (t <= g[b[keep][-1] - 1])
    return t[inside], xy[inside]




def raw_flight(path):
    """Arrays t, xy, seg of the raw track in the air, or None if there is none."""
    t, lat, lon, valid = read_igc(path)
    if len(t) < 2:
        return None
    t, xy = clean(t, lat, lon, valid)
    if len(t) < 2:
        return None
    air = cut_air(*despike(t, xy))
    if air is None:
        return None
    t, xy = air
    jump = np.r_[0, _speed_kmh(t, xy) > MAX_JUMP_KMH]
    return {"t": t, "xy": xy, "seg": np.cumsum(jump)}
