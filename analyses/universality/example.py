"""One real flight for the figure of the terrain rule: the ground elevation under each 10-s point, the recorded
altitude, the median ground elevation and its class. Written to out/<run>/example_flight.npz (no identifier).

The flight: the first paraglider of the cache that is open, entirely on the DEM, in the class EXAMPLE_CLASS, with a
median in the middle of its class and a duration between EXAMPLE_HOURS.

    uv run python example.py    # after measure.py; needs SSD_DISANTE
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import measure
import openflights
import terrain
from common.legacy import LocalFrame, enu_to_geodetic, stream_flights

EXAMPLE_CLASS, EXAMPLE_HOURS, MID_M = "low mountains", (2.5, 3.5), (1000.0, 1300.0)


def pick(table):
    m = table[table["measured"] & (table["discipline"] == "para")]
    m = m[openflights.is_open(m["flight_type"], m["ends_km"], m["path_km"])]
    hours = (m["landing"] - m["departure"]) / 3600
    cls = terrain.classify(m["median_elev"], m["n_points"], m["n_on_dem"])
    m = m[(cls == EXAMPLE_CLASS) & (m["n_on_dem"] == m["n_points"]) & hours.between(*EXAMPLE_HOURS) & m["median_elev"].between(*MID_M)]
    return m.iloc[0]


def main(out):
    from pyproj import Transformer

    row = pick(pd.read_parquet(measure.CACHE / "flights.parquet"))
    to_l93, dem = Transformer.from_crs(4326, 2154, always_xy=True), terrain.load()
    for fixes in stream_flights(measure.FIXES["para"], ["segment_id", "t", "E", "N", "z"]):
        if str(fixes["flight_id"].iat[0]) == row["flight_id"]:
            break
    k, ground, z = [], [], []
    for kk, e, n, zz in terrain.grid_points(fixes):
        lat, lon = enu_to_geodetic(e, n, zz, LocalFrame(row["lat0"], row["lon0"], 0.0 if pd.isna(row["alt0"]) else row["alt0"]))
        ground.append(terrain.ground(dem, *to_l93.transform(lon, lat)))
        k.append(kk), z.append(zz)
    k, ground, z = (np.concatenate(a) for a in (k, ground, z))
    median = float(np.median(ground[np.isfinite(ground)]))
    assert median == row["median_elev"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "example_flight.npz", t=k * terrain.GRID_S, ground=ground, z=z, median=median)


if __name__ == "__main__":
    main(Path(__file__).with_name("out") / "full")
