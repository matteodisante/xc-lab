"""Phase 0 of the note on the cleaned tracks of the old repository: one date on the grid.

The tracks are the cleaned fixes that soaring (tag legacy-final) wrote on SSD_DISANTE: per
segment a uniform grid at the logger interval, time t from the trimmed start, smoothed east
and north E, N in the frame of the first fix, GNSS altitude z and velocity v_E, v_N. The UTC
time of a fix is the first IGC fix (read by the old viewer into its index) plus the trimmed
start plus t. Phases are Vilpellet's runs, which soaring decoded on 1 Hz flights only.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lab.legacy import LocalFrame, enu_to_geodetic, geodetic_to_ecef

DISK = Path("/Volumes/SSD_DISANTE")
DERIVED = {"para": DISK / "paragliders/ffvl_cfd_igc/derived", "hang": DISK / "hang_gliders/delta_cfd_igc/derived"}
INDEX = DERIVED["para"] / "viewer/thermal-planes/thermal-cells.sqlite3"
INDEX_NAMES = {"paragliders": "para", "hang gliders": "hang"}
RUNS = "segmentation/vilpellet/phase_segments.parquet"
STATES = {"climb": 0, "search": 1, "transition": 2}
UNCLASSIFIED = 3
COLUMNS = ["flight_id", "segment_id", "t", "E", "N", "z", "v_E", "v_N"]


def archive():
    """Every retained flight with a UTC clock: discipline, flight_id, utc0 (UTC seconds at
    t = 0), date (the UTC day of utc0), lat0, lon0, alt0, dt_native (the logger interval
    inferred by the cleaning, s), and the row groups of fixes.parquet that hold its fixes."""
    db = sqlite3.connect(f"file:{INDEX}?mode=ro", uri=True)
    clocks = pd.read_sql_query("SELECT discipline, flight_id, start_utc FROM flights WHERE start_utc IS NOT NULL", db)
    groups = pd.read_sql_query("SELECT discipline, flight_id, row_group FROM flight_groups", db)
    db.close()
    for t in (clocks, groups):
        t["discipline"] = t["discipline"].map(INDEX_NAMES)
    groups = groups.groupby(["discipline", "flight_id"])["row_group"].agg(sorted).rename("row_groups")
    tables = []
    for d, path in DERIVED.items():
        meta = pd.read_parquet(path / "flights_meta.parquet",
                               columns=["flight_id", "drop_stage", "ground_phase_start_s", "lat0", "lon0", "alt0",
                                        "dt_native_s"])
        meta = meta[meta["drop_stage"].isna()].drop(columns="drop_stage").assign(discipline=d)
        meta = meta.rename(columns={"dt_native_s": "dt_native"})
        tables.append(meta)
    flights = pd.concat(tables).merge(clocks, on=["discipline", "flight_id"], validate="one_to_one")
    flights = flights.join(groups, on=["discipline", "flight_id"], how="inner")
    flights["utc0"] = flights["start_utc"] + flights["ground_phase_start_s"]
    flights["date"] = pd.to_datetime(flights["utc0"], unit="s").dt.strftime("%Y-%m-%d")
    columns = ["discipline", "flight_id", "utc0", "date", "lat0", "lon0", "alt0", "dt_native", "row_groups"]
    return flights[columns].sort_values(["date", "utc0"], kind="stable").reset_index(drop=True)


def phase_runs(discipline):
    """Vilpellet's runs of one discipline: flight_id, segment_id, state code, and t_start,
    t_end of the first and last fix of the run on the flight clock."""
    runs = pd.read_parquet(DERIVED[discipline] / RUNS, columns=["flight_id", "segment_id", "phase", "t_start", "t_end"])
    runs["state"] = runs.pop("phase").map(STATES).astype(np.int8)
    return runs


def read_fixes(flights):
    """The cleaned fixes of some flights of one discipline (rows of archive())."""
    file = pq.ParquetFile(DERIVED[flights["discipline"].iat[0]] / "fixes.parquet")
    ids = pa.array(flights["flight_id"].tolist())
    parts = []
    for g in sorted({g for gs in flights["row_groups"] for g in gs}):
        table = file.read_row_group(g, columns=COLUMNS)
        parts.append(table.filter(pc.is_in(table["flight_id"], value_set=ids)))
    return pa.concat_tables(parts).to_pandas()


def states(seg, t, runs):
    """State code of each fix (sorted by segment and time) from the runs of its flight."""
    if runs.empty:
        return np.full(len(t), UNCLASSIFIED, np.int8)
    runs = runs.sort_values(["segment_id", "t_start"])
    starts = runs["segment_id"].to_numpy() * 1e6 + runs["t_start"].to_numpy()
    ends = runs["segment_id"].to_numpy() * 1e6 + runs["t_end"].to_numpy()
    key = seg * 1e6 + t
    k = np.maximum(np.searchsorted(starts, key, side="right") - 1, 0)
    inside = (key >= starts[k]) & (key <= ends[k])
    return np.where(inside, runs["state"].to_numpy()[k], UNCLASSIFIED).astype(np.int8)


def on_grid(u, seg, values, near_values, start, step, h_max):
    """The grid instants (multiples of step) from `start` that a flight covers, segment by
    segment: linear interpolation of `values` between two fixes at most h_max apart (or the
    fix itself at its own instant), and `near_values` from the nearer fix."""
    out = []
    for s in np.unique(seg):
        q = np.flatnonzero(seg == s)
        us = u[q]
        g = np.arange(np.ceil(max(us[0], start) / step) * step, us[-1] + 1e-6, step)
        if not g.size:
            continue
        a = np.searchsorted(us, g, side="right") - 1
        b = np.minimum(a + 1, len(us) - 1)
        h = us[b] - us[a]
        w = np.divide(g - us[a], h, out=np.zeros_like(g), where=h > 0)
        ok = (g == us[a]) | ((h > 0) & (h <= h_max))
        lerp = values[q][a] * (1 - w[:, None]) + values[q][b] * w[:, None]
        near = np.where(w < 0.5, a, b)
        out.append((g[ok].astype(np.int64), np.full(ok.sum(), s), lerp[ok], near_values[q][near][ok]))
    return out


def flight_grid(fixes, row, runs, step, h_max, r_dep):
    """Phase 0 for one flight: its times (takeoff, departure, landing, in UTC seconds) and its
    grid rows; None for a local flight (never farther than r_dep from its first fix) and for
    a flight with no grid instant after departure (fixes more than h_max apart)."""
    fixes = fixes.sort_values("t", kind="stable")
    t = fixes["t"].to_numpy(float)
    seg = fixes["segment_id"].to_numpy()
    e, n, z = (fixes[c].to_numpy(float) for c in ("E", "N", "z"))
    u = row.utc0 + t
    times = {"takeoff": u[0], "departure": np.inf, "landing": u[-1], "labelled": not runs.empty}
    far = np.flatnonzero(np.hypot(e, n) > r_dep)
    if not far.size:
        return times, None
    times["departure"] = u[far[0]]
    lat, lon = enu_to_geodetic(e, n, z, LocalFrame(row.lat0, row.lon0, row.alt0))
    xyz = np.column_stack(geodetic_to_ecef(lat, lon, 0.0))
    v_e, v_n = fixes["v_E"].to_numpy(float), fixes["v_N"].to_numpy(float)
    speed = np.maximum(np.hypot(v_e, v_n), 1e-9)
    near = np.column_stack([v_e / speed, v_n / speed, states(seg, t, runs)])
    pieces = on_grid(u, seg, np.column_stack([xyz, z]), near, u[far[0]], step, h_max)
    if not sum(len(p[0]) for p in pieces):
        return times, None
    g, s, lerp, nv = (np.concatenate(p) for p in zip(*pieces))
    rows = pd.DataFrame({"seg": s.astype(np.int16), "t": g, "X": lerp[:, 0], "Y": lerp[:, 1], "Z": lerp[:, 2],
                         "z": lerp[:, 3], "ve": nv[:, 0], "vn": nv[:, 1], "state": nv[:, 2].astype(np.int8)})
    return times, rows


def day(flights, runs, step, h_max, r_dep):
    """Phase 0 for the flights of one date (rows of archive()) and the runs of each discipline.

    Returns the flight table (its rows, with takeoff, departure and landing in UTC seconds;
    departure inf for a local flight; X0, Y0, Z0 the Earth-centred takeoff) and the grid: one
    row per instant of the common grid, a multiple of `step` UTC seconds, from the departure
    to the landing, at which the flight has a position. `f` is the row of the flight table.
    Positions X, Y, Z are Earth-centred, of the point of the WGS84 ellipsoid below the fix
    (h = 0); z the altitude; ve, vn the unit horizontal heading; state the phase code.
    """
    table, grids = [], []
    for d, group in flights.groupby("discipline", sort=False):
        fixes = read_fixes(group)
        by_flight = dict(tuple(fixes.groupby("flight_id", sort=False)))
        mine = runs[d][runs[d]["flight_id"].isin(group["flight_id"])]
        runs_of = dict(tuple(mine.groupby("flight_id", sort=False)))
        for row in group.itertuples():
            times, rows = flight_grid(by_flight[row.flight_id], row, runs_of.get(row.flight_id, mine.iloc[:0]),
                                      step, h_max, r_dep)
            if rows is not None:
                grids.append(rows.assign(f=np.int32(len(table))))
            table.append({"discipline": d, "flight_id": row.flight_id, "utc0": row.utc0, "dt_native": row.dt_native,
                          "n_grid": 0 if rows is None else len(rows), **times})
    table = pd.DataFrame(table)
    origin = flights.set_index(["discipline", "flight_id"]).loc[pd.MultiIndex.from_frame(table[["discipline", "flight_id"]])]
    table[["X0", "Y0", "Z0"]] = np.column_stack(geodetic_to_ecef(origin["lat0"].to_numpy(), origin["lon0"].to_numpy(), 0.0))
    grid = pd.concat(grids, ignore_index=True) if grids else None
    return table, grid
