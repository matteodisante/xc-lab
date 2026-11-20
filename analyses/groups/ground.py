"""Ground class of each flight of the pieces cache: plains, hills, low mountains, high mountains.

The ground under a flight is the median of the elevation of the 200 m blocks under its points on the 10-s grid
(analyses/universality/terrain.py: DEM, grid, edges, classes, MIN_ON_DEM, all taken from there), keeping only the
points with t in [departure + TRIM D, landing - TRIM D], D = landing - departure: the first and last TRIM of the
duration are reaching and leaving the flying zone. Points off the DEM are dropped; a flight with fewer than
MIN_ON_DEM of its kept points on the DEM is unknown. The median of all the points (untrimmed) is saved beside it.

Flights: those of the pieces cache of cohorts.py (longlags.PIECES), paragliders of the run.

    uv run --with rasterio --with pyproj python ground.py --declared-open [--sample N] [--workers W]
        # writes out/<run>/open_declared/ground.parquet (ground_sample<N>.parquet with --sample)
"""

import importlib.util
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

import longlags
from common.legacy import LocalFrame, enu_to_geodetic, stream_flights
from tamsd import FIXES, LABELS, OUT

TRIM = 0.1  # share of the duration dropped at each end of the flight

_spec = importlib.util.spec_from_file_location("universality_terrain", Path(__file__).parents[1] / "universality/terrain.py")
terrain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(terrain)

_STATE = {}


def _init():
    from pyproj import Transformer

    _STATE["dem"] = terrain.load()
    _STATE["to_l93"] = Transformer.from_crs(4326, 2154, always_xy=True)


def _job(flight, fixes, frame, a, b):
    """(flight, then (median, points, points on the DEM) of the trimmed and of all the points)."""
    t, g = [], []
    for k, east, north, z in terrain.grid_points(fixes):
        lat, lon = enu_to_geodetic(east, north, z, LocalFrame(*frame))
        x, y = _STATE["to_l93"].transform(lon, lat)
        t.append(k * terrain.GRID_S)
        g.append(terrain.ground(_STATE["dem"], x, y))
    t, g = (np.concatenate(v) if v else np.empty(0) for v in (t, g))
    cut = TRIM * (b - a)
    return (flight, terrain.median_ground(g[(t >= a + cut) & (t <= b - cut)]), terrain.median_ground(g))


def measure(flights, limit=None, workers=os.cpu_count()):
    """The ground table of `flights` (run table rows of paragliders): flight, median_ground, n_points, n_on_dem,
    class, and the same of all the points with the suffix _all."""
    meta = pd.read_parquet(FIXES["para"].parent / "flights_meta.parquet", columns=["flight_id", "lat0", "lon0", "alt0"])
    f = flights.astype({"flight_id": str}).merge(meta.astype({"flight_id": str}), on="flight_id")
    todo = {str(i): r for i, r in zip(f["flight_id"], f.itertuples())}
    found = {}

    def store(future):
        flight, kept, every = future.result()
        found[flight] = (*kept, *every)

    with ProcessPoolExecutor(workers, initializer=_init) as pool:
        pending, sent = set(), 0
        for fixes in stream_flights(FIXES["para"], ["segment_id", "t", "E", "N", "z"]):
            r = todo.get(str(fixes["flight_id"].iat[0]))
            if r is None:
                continue
            frame = (r.lat0, r.lon0, 0.0 if pd.isna(r.alt0) else r.alt0)
            pending.add(pool.submit(_job, r.flight, fixes, frame, r.departure - r.utc0, r.landing - r.utc0))
            sent += 1
            if sent % 2000 == 0:
                print(sent, "flights sent", flush=True)
            if len(pending) >= 3 * workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    store(future)
            if limit and sent >= limit:
                break
        for future in pending:
            store(future)
    out = pd.DataFrame({"flight": f["flight"]})
    got = pd.DataFrame.from_dict(found, orient="index", columns=["median_ground", "n_points", "n_on_dem",
                                                                 "median_ground_all", "n_points_all", "n_on_dem_all"])
    out = out.join(got, on="flight").fillna({"n_points": 0, "n_on_dem": 0, "n_points_all": 0, "n_on_dem_all": 0})
    out["class"] = terrain.classify(out["median_ground"], out["n_points"], out["n_on_dem"])
    out["class_all"] = terrain.classify(out["median_ground_all"], out["n_points_all"], out["n_on_dem_all"])
    return out


if __name__ == "__main__":
    sample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else os.cpu_count()
    ids = np.unique(np.load(longlags.PIECES)["flight"])
    table = pd.read_parquet(LABELS / "flights.parquet")
    table = table[table["flight"].isin(ids) & (table["discipline"] == "para")]
    OUT.mkdir(parents=True, exist_ok=True)
    out = measure(table, sample, workers)
    if sample:  # the first flights of the file only
        out = out[out["n_points_all"] > 0]
    out.to_parquet(OUT / (f"ground_sample{sample}.parquet" if sample else "ground.parquet"), index=False)
    known = out[out["class"] != terrain.UNKNOWN]
    print(len(out), "flights;", len(out) - len(known), "unknown;", "changed class with the trim:",
          f"{(out['class'] != out['class_all']).mean():.3f}")
    print(out["class"].value_counts().to_string())
