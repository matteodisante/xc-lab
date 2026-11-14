"""Ground speed during the glides of the selected flights.

For each flight, the length and the duration of its steps (1 s) that lie inside a transition run of Vilpellet's
segmenter, read from the cleaned fixes of the old repository (E, N, 1 Hz). The mean ground speed of a group of
flights is total length over total time.

    uv run python glide_speed.py CACHE_DIR      # reads CACHE_DIR/flights.csv, writes CACHE_DIR/glide_speed.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

import pauses as pa

SEG, T = 1000, 1_000_000  # key = (flight code * SEG + segment) * T + t, exact in int64


def keys(flight, segment, t):
    return (flight.astype(np.int64) * SEG + segment) * T + np.rint(t).astype(np.int64)


def glide_steps(fixes_file, runs, flight_ids):
    """Per flight id: summed length (m) and number of steps (s) inside the transition runs of `runs`.

    fixes_file: fixes.parquet with flight_id, segment_id, t, E, N (rows of a flight contiguous, in time order).
    """
    code = pd.Series(np.arange(len(flight_ids)), index=sorted(str(i) for i in flight_ids))
    g = runs[runs["phase"] == "transition"].copy()
    g = g[g["flight_id"].isin(flight_ids)]
    fid = g["flight_id"].map(code).to_numpy()
    start = keys(fid, g["segment_id"].to_numpy(np.int64), g["t_start"].to_numpy(float))
    end = keys(fid, g["segment_id"].to_numpy(np.int64), g["t_end"].to_numpy(float))
    order = np.argsort(start)
    start, end = start[order], end[order]
    wanted = pa_set(flight_ids)
    
    dist, steps = {}, {}
    f = pq.ParquetFile(fixes_file)
    for i in range(f.metadata.num_row_groups):
        t = f.read_row_group(i, columns=["flight_id", "segment_id", "t", "E", "N"])
        t = t.filter(pc.is_in(t["flight_id"], value_set=wanted))
        if t.num_rows == 0:
            continue
        d = t.to_pandas()
        flight = d["flight_id"].map(code).to_numpy()
        k = keys(flight, d["segment_id"].to_numpy(np.int64), d["t"].to_numpy(float))
        step = np.hypot(np.diff(d["E"].to_numpy(float)), np.diff(d["N"].to_numpy(float)))
        j = np.searchsorted(start, k[:-1], side="right") - 1
        inside = (j >= 0) & (k[1:] - k[:-1] == 1) & (k[1:] <= end[np.maximum(j, 0)])
        u, inv = np.unique(flight[:-1][inside], return_inverse=True)
        for fl_id, s, n in zip(u, np.bincount(inv, weights=step[inside]), np.bincount(inv)):
            dist[fl_id] = dist.get(fl_id, 0.0) + s
            steps[fl_id] = steps.get(fl_id, 0) + n
    names = code.index.to_numpy()
    return pd.DataFrame({"flight_id": names[list(dist)], "dist_m": list(dist.values()), "time_s": [steps[k] for k in dist]})


def pa_set(ids):
    import pyarrow as pyarrow

    return pyarrow.array([str(i) for i in ids])


if __name__ == "__main__":
    cache = Path(sys.argv[1])
    sel = pd.read_csv(cache / "flights.csv", dtype={"flight": str})
    sel = sel[(sel["status"] == "ok") & (sel["dt_median_s"] == 1)]
    out = []
    for disc, short in (("paragliders", "para"), ("hang_gliders", "hang")):
        ids = set(sel.loc[sel["discipline"] == disc, "flight"])
        derived = Path(pa.DISK) / pa.FOLDERS[short] / "derived"
        runs = pd.read_parquet(derived / pa.RUNS.replace("derived/", ""))
        runs["flight_id"] = runs["flight_id"].astype(str)
        r = glide_steps(derived / "fixes.parquet", runs, ids)
        r.insert(0, "discipline", disc)
        out.append(r)
        print(disc, len(r), "flights", flush=True)
    pd.concat(out).to_csv(cache / "glide_speed.csv", index=False)
