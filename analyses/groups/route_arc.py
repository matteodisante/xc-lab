"""The route of each piece of S, A and G, and its MSD in arc length.

A piece is a piece of the cache of longlags.py (declared open paragliders, logger interval up to longlags.MAX_DT s).
Its path is the track on a 1-s grid (cubic spline through the fixes, as phase_terms.py reads them); the route is the
path with each climb run of our refit labels replaced by its chord (first to last fix of the run; consecutive runs
joined; a run cut by the piece boundary has its chord from the boundary). Search runs stay in the route. Both the
path ("arc", circles kept) and the route are resampled every DS_M metres of arc length and measured as time pieces
are: TA-MSD at the lags of longlags.LONG_LAGS read as lag / 10 steps of DS_M, so that the machinery of cohorts.py
applies unchanged (the lag k s of the grid is k / 10 DS_M metres).

Per piece: flight, start, duration (s), the lengths of the path and of the route (m), the seconds dropped by the
chords, and per kind (arc, route) the steps of the grid, the sums and the origins of the TA-MSD.

    uv run python route_arc.py --declared-open --dt5 [--sample N]    # reads the fixes, about an hour; resumable
        # one part per PART flights in CACHE_DIR; a part that exists is skipped (--sample: scratch folder, no resume)
"""

import sys
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import longlags
import phase_terms
from common.legacy import stream_flights
from common.msd import tamsd_sums
from tamsd import FIXES, LABELS

DS_M = 100.0  # m: the step of the arc-length grid, about the median ground speed times the 10-s grid of the time pieces
STEP_S = 1.0  # s: the grid of the path
MARGIN_STEPS = 3  # fixes read beyond each end of a piece, for the spline
PART = 2000  # flights per saved part
WORKERS = 4
KINDS = ("arc", "route")
POPS = ("solo", "group", "group_solo")
SAMPLE = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else 0
ROOT = Path("/Volumes/SSD_DATA/xc-lab/groups")  # the caches, in folders named by their parameters
CACHE_DIR = ROOT / LABELS.name / f"route_ds{DS_M:g}_{phase_terms.SET}_dt{longlags.MAX_DT}" / (f"sample{SAMPLE}" if SAMPLE else "full")
LAGS = longlags.LONG_LAGS // 10  # steps of the grid


def climbs_by_flight(flights):
    """The climb runs of the refit of each flight (cache number), as (t_start, t_end) rows, runs less than a logger
    interval apart joined."""
    runs = phase_terms.read_runs("refit", flights)
    runs = runs[runs["phase"] == "climb"]
    dt = flights["dt_native"].reindex(runs["fid"]).to_numpy()
    out = {}
    for flight, r in runs.assign(dt=dt).groupby("flight"):
        a, b = r["t_start"].to_numpy(float), r["t_end"].to_numpy(float)
        new = np.r_[True, a[1:] - b[:-1] > r["dt"].to_numpy()[1:] + 1e-6]
        out[flight] = np.c_[a[new], np.maximum.reduceat(b, np.flatnonzero(new))]
    return out


def resample(xy):
    """The points of the polyline `xy` every DS_M of arc length, and its length."""
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    g = np.arange(0.0, s[-1] + 1e-9, DS_M)
    return np.column_stack([np.interp(g, s, xy[:, 0]), np.interp(g, s, xy[:, 1])]), s[-1]


def measure(pos):
    """Steps of the grid, sums and origins of the TA-MSD at LAGS (NaN and 0 where the lag is out of reach)."""
    ok = LAGS < len(pos)
    sums, origins = np.full(len(LAGS), np.nan), np.zeros(len(LAGS), np.int64)
    if len(pos) > 1:
        sums[ok], origins[ok] = tamsd_sums(pos, LAGS[ok])
    return len(pos) - 1, sums, origins


def route_piece(t, xy, a, b, climbs):
    """One piece [a, b] (s) of a segment with fixes at `t` (positions `xy`): the path and the route, each as
    (steps, sums, origins) and its length, and the seconds the chords drop."""
    g = np.arange(a, b + 1e-6, STEP_S)
    path = CubicSpline(t, xy)(g)
    keep = np.ones(len(g), bool)
    for t0, t1 in climbs:
        keep &= ~((g > max(t0, a)) & (g < min(t1, b)))
    out = {}
    for kind, line in (("arc", path), ("route", path[keep])):
        pos, length = resample(line)
        out[kind] = (length, *measure(pos))
    return out, int((~keep).sum())


def job(fixes, rows, climbs):
    """The rows (index in the cache, start, duration) of one flight, measured on its fixes."""
    fixes = fixes.sort_values("t", kind="stable")
    seg = fixes["segment_id"].to_numpy()
    t, xy = fixes["t"].to_numpy(float), fixes[["E", "N"]].to_numpy(float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    out = []
    for i, a, d in rows:
        b = a + d
        for s in np.unique(seg):
            q = np.flatnonzero(seg == s)
            if t[q[0]] - 1e-6 <= a and b <= t[q[-1]] + 1e-6:
                lo, hi = np.searchsorted(t[q], a - MARGIN_STEPS * dt), np.searchsorted(t[q], b + MARGIN_STEPS * dt, "right")
                q = q[lo:hi]
                q = q[np.r_[True, np.diff(t[q]) > 0]]
                found, dropped = route_piece(t[q], xy[q], a, b, climbs[(climbs[:, 1] > a) & (climbs[:, 0] < b)])
                out.append((i, dropped, found))
                break
    return out


def pieces_by_flight(p):
    """{flight: [(row, start, duration), ...]} for the pieces of POPS of the cache."""
    rows = np.flatnonzero(np.isin(p["population"], POPS))
    table = pd.DataFrame({"row": rows, "flight": p["flight"][rows], "start": p["start"][rows], "duration": p["duration"][rows]})
    return {f: list(zip(g["row"], g["start"], g["duration"])) for f, g in table.groupby("flight")}


def pack(done):
    """The arrays of a part from its (row, dropped, found) triples."""
    row, dropped, found = zip(*done)
    out = {"row": np.array(row), "dropped_s": np.array(dropped)}
    for kind in KINDS:
        length, steps, sums, origins = zip(*(f[kind] for f in found))
        out |= {f"length_{kind}": np.array(length), f"steps_{kind}": np.array(steps), f"sums_{kind}": np.array(sums),
                f"origins_{kind}": np.array(origins)}
    return out


def run():
    flights = phase_terms.kept_flights()
    p = longlags.load_pieces()
    todo, climbs = pieces_by_flight(p), climbs_by_flight(flights)
    wanted = set(list(todo)[:SAMPLE]) if SAMPLE else set(todo)
    number = {int(fid): n for fid, n in flights["flight"].items()}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    state = {"chunk": 0, "pending": set(), "done": [], "seen": 0}

    def collect(done):
        state["done"] += [x for f in done for x in f.result()]

    def save():
        collect(wait(state["pending"])[0])
        state["pending"] = set()
        if state["done"]:
            np.savez(CACHE_DIR / f"part{state['chunk']:04d}.npz", **pack(state["done"]))
            print("part", state["chunk"], "saved,", state["seen"], "flights", flush=True)
        state["done"] = []

    with ProcessPoolExecutor(WORKERS) as pool:
        for fixes in stream_flights(FIXES["para"], ["segment_id", "t", "E", "N"]):
            flight = number.get(int(fixes["flight_id"].iat[0]))
            if flight not in wanted:
                continue
            chunk = state["seen"] // PART
            state["seen"] += 1
            if chunk != state["chunk"]:
                save()
                state["chunk"] = chunk
            if (CACHE_DIR / f"part{chunk:04d}.npz").exists():
                continue
            state["pending"].add(pool.submit(job, fixes, todo[flight], climbs.get(flight, np.zeros((0, 2)))))
            if len(state["pending"]) >= 4 * WORKERS:
                done, state["pending"] = wait(state["pending"], return_when=FIRST_COMPLETED)
                collect(done)
        save()


def load():
    """The rows of the parts of CACHE_DIR, sorted by row of the cache: a dict of arrays."""
    parts = [np.load(f) for f in sorted(CACHE_DIR.glob("part*.npz"))]
    out = {k: np.concatenate([q[k] for q in parts]) for k in parts[0].files}
    order = np.argsort(out["row"])
    return {k: v[order] for k, v in out.items()}


if __name__ == "__main__":
    assert longlags.DT5, "run with --dt5"
    run()
