"""One pass over the cleaned fixes of both disciplines: for every eligible declared open flight, its ends, its
pieces on the 10-s grid (tamsd.py) and the median ground elevation under its points (terrain.py).

Flights: the table of the reference run of methods/group-identification (flights.parquet, both disciplines: the
interval of the logger, the departure, the landing, the date), read only for those columns. Eligible: a logger of
at most tamsd.H_MAX s, grid positions, a departure. The span of a flight is from its departure to its landing.
Not measured: flights that are not declared open (openflights.py). The opening cut is made later (analysis.py),
on the ends saved here.

    uv run python measure.py [--sample N] [--workers W]    # after terrain.py --build; needs SSD_DISANTE
                                                           # writes CACHE (full) or CACHE/sample<N>
"""

import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

import openflights
import tamsd
import terrain
from common.legacy import LocalFrame, enu_to_geodetic, stream_flights

DISK = Path("/Volumes/SSD_DISANTE")
RUN = DISK / "xc-lab/group-identification/runs/d120_eh500_ez300_th45_g120_q0.1"
FOLDERS = {"para": DISK / "paragliders/ffvl_cfd_igc", "hang": DISK / "hang_gliders/delta_cfd_igc"}
FIXES = {d: p / "derived/fixes.parquet" for d, p in FOLDERS.items()}
CACHE = DISK / "xc-lab/universality/flights_grid10_support0.8_hmax10_declared_open_dem200m"


def flight_table(run=RUN):
    """Every flight of the run table, both disciplines: the run columns, catalogue type and pilot, path, takeoff
    frame, and the flags `declared` and `eligible`."""
    f = pd.read_parquet(run / "flights.parquet", columns=["discipline", "flight", "flight_id", "date", "utc0",
                                                          "dt_native", "n_grid", "departure", "landing"])
    f["flight_id"] = f["flight_id"].astype(str)
    parts = []
    for d, folder in FOLDERS.items():
        c = pd.read_csv(folder / "catalog/catalog.csv", usecols=["flight_id", "flight_type", "pilot"], dtype=str)
        m = pd.read_parquet(folder / "derived/flights_meta.parquet",
                            columns=["flight_id", "path_km", "lat0", "lon0", "alt0"]).astype({"flight_id": str})
        x = f[f["discipline"] == d]
        x = x.merge(c.drop_duplicates("flight_id"), on="flight_id", how="left")
        parts.append(x.merge(m.drop_duplicates("flight_id"), on="flight_id", how="left"))
    f = pd.concat(parts, ignore_index=True)
    name = f["pilot"].where(f["pilot"].notna() & (f["pilot"].str.strip() != ""), "flight " + f["flight_id"])
    f["pilot"] = name
    f["declared"] = openflights.declared(f["flight_type"])
    f["eligible"] = ((f["dt_native"] <= tamsd.H_MAX + 1e-6) & (f["n_grid"] > 0) & np.isfinite(f["departure"])
                     & f[["lat0", "lon0"]].notna().all(axis=1))
    return f.sort_values(["discipline", "flight"], kind="stable").reset_index(drop=True)


_STATE = {}


def _init():
    from pyproj import Transformer

    _STATE["dem"] = terrain.load()
    _STATE["to_l93"] = Transformer.from_crs(4326, 2154, always_xy=True)


def _job(row, fixes, frame, a, b):
    """Ends, pieces and ground elevation of one flight; `row` is its row in the flight table."""
    t, e, n = (fixes[c].to_numpy(float) for c in ("t", "E", "N"))
    i, j = int(np.argmin(t)), int(np.argmax(t))
    ends_km = float(np.hypot(e[j] - e[i], n[j] - n[i])) / 1000
    ground = []
    for _, east, north, z in terrain.grid_points(fixes):
        lat, lon = enu_to_geodetic(east, north, z, LocalFrame(*frame))
        x, y = _STATE["to_l93"].transform(lon, lat)
        ground.append(terrain.ground(_STATE["dem"], x, y))
    median, n_points, n_on = terrain.median_ground(np.concatenate(ground) if ground else np.empty(0))
    return row, ends_km, tamsd.pieces(fixes, a, b), median, n_points, n_on


def run(table, path, limit=None, workers=os.cpu_count()):
    """The pass; saves flights.parquet (the table with ends_km, median_elev, n_points, n_on_dem, measured) and
    pieces.npz (row of the flight, start, duration, sums, origins) in `path`."""
    path.mkdir(parents=True, exist_ok=True)
    todo = table[table["declared"] & table["eligible"]]
    key = {(d, str(i)): r for r, d, i in zip(todo.index, todo["discipline"], todo["flight_id"])}
    ends, ground, found = {}, {}, []

    def store(future):
        row, ends_km, ps, median, n_points, n_on = future.result()
        ends[row], ground[row] = ends_km, (median, n_points, n_on)
        found.extend((row, *p) for p in ps)

    with ProcessPoolExecutor(workers, initializer=_init) as pool:
        for d in FIXES:
            pending, sent = set(), 0
            for fixes in stream_flights(FIXES[d], ["segment_id", "t", "E", "N", "z"]):
                row = key.get((d, str(fixes["flight_id"].iat[0])))
                if row is None:
                    continue
                r = table.loc[row]
                frame = (r["lat0"], r["lon0"], 0.0 if pd.isna(r["alt0"]) else r["alt0"])
                pending.add(pool.submit(_job, row, fixes, frame, r["departure"] - r["utc0"], r["landing"] - r["utc0"]))
                sent += 1
                if sent % 2000 == 0:
                    print(d, sent, "flights sent", flush=True)
                if len(pending) >= 3 * workers:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        store(future)
                if limit and sent >= limit:
                    break
            for future in pending:
                store(future)
    table = table.assign(ends_km=pd.Series(ends), measured=table.index.isin(list(ends)))
    g = pd.DataFrame.from_dict(ground, orient="index", columns=["median_elev", "n_points", "n_on_dem"])
    table = table.join(g)
    table.to_parquet(path / "flights.parquet", index=False)
    row, start, duration, sums, origins = zip(*found)
    np.savez(path / "pieces.npz", row=np.array(row, np.int64), start=np.array(start), duration=np.array(duration),
             sums=np.array(sums), origins=np.array(origins))
    (path / "PARAMETERS.md").write_text(parameters_text(limit))


def parameters_text(limit):
    return f"""# Flights, pieces and ground elevation of the declared open flights

Written by `analyses/universality/measure.py` of xc-lab (do not edit by hand).{" Sample: the first " + str(limit) + " flights of each discipline in the file." if limit else ""}

- Flights: table of the run `{RUN.name}` of methods/group-identification (both disciplines), declared open
  (`openflights.DECLARED_OPEN`), eligible = logger interval <= {tamsd.H_MAX} s, grid positions, a departure.
- Pieces: stretches of a segment inside [departure, landing], on the {tamsd.GRID_S}-s grid, at least {tamsd.SHORTEST:g} s,
  with the sums and the origins of the TAMSD at the lags `tamsd.LONG_LAGS` ({tamsd.LONG_LAGS[0]} to {tamsd.LONG_LAGS[-1]} s).
- Ground: `terrain.py`, elevation grid in `{terrain.CACHE.name}`; median over all the points of the flight on the
  grid, points outside the DEM dropped (`n_points`, `n_on_dem`).
- Files: `flights.parquet` (row = position in the table; `ends_km`, `median_elev`, `n_points`, `n_on_dem` only for
  `measured` flights), `pieces.npz` (`row`, `start`, `duration`, `sums`, `origins`; sums and origins have one column per lag).
"""


if __name__ == "__main__":
    arg = lambda name, default: int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
    sample = arg("--sample", 0)
    run(flight_table(), CACHE / (f"sample{sample}" if sample else ""), sample or None, arg("--workers", os.cpu_count()))
