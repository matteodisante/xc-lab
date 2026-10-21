"""Distance covered by each episode: the longest straight line that one of its members flew
while in it, from where it entered a group stretch of the episode to where it left it.

    uv run python distances.py [RUN]    # from this folder, after run.py; writes RUN/episode_distance.parquet

RUN is the folder of a run; by default that of the starting values on SSD_DISANTE.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import grid
import run as runs

BATCH = 300  # flights read together


def group_stretches(membership, flights):
    """Consecutive G rows of a flight, which lie in one episode: flight, episode_id, and the
    first and last instant on the flight clock."""
    m = membership.sort_values(["flight", "t_start"], kind="stable")
    run = (m["state"] != m.groupby("flight")["state"].shift()).cumsum()
    g = m[m["state"] == "G"].groupby(run[m["state"] == "G"]).agg(
        flight=("flight", "first"), episode_id=("episode_id", "first"), a=("t_start", "min"), b=("t_end", "max"))
    utc0 = flights.set_index("flight")["utc0"].reindex(g["flight"]).to_numpy()
    return g.assign(a=g["a"] - utc0, b=g["b"] - utc0).reset_index(drop=True)


def stretch_distances(fixes, stretches):
    """Straight-line distance between the positions at the two ends of each stretch."""
    fixes = fixes.sort_values("t", kind="stable")
    t = fixes["t"].to_numpy(float)
    e, n = fixes["E"].to_numpy(float), fixes["N"].to_numpy(float)
    ends = np.r_[stretches["a"].to_numpy(), np.minimum(stretches["b"].to_numpy(), t[-1])]
    x, y = np.interp(ends, t, e), np.interp(ends, t, n)
    k = len(stretches)
    return np.hypot(x[k:] - x[:k], y[k:] - y[:k])


def distances(folder):
    flights = pd.read_parquet(folder / "flights.parquet")
    stretches = group_stretches(pd.read_parquet(folder / "membership.parquet"), flights)
    archive = grid.archive().merge(flights[["discipline", "flight_id", "flight"]], on=["discipline", "flight_id"])
    archive = archive[archive["flight"].isin(stretches["flight"])]
    archive = archive.assign(first=archive["row_groups"].map(min)).sort_values(["discipline", "first"])
    by_flight = dict(tuple(stretches.groupby("flight")))
    out = []
    for _, discipline in archive.groupby("discipline"):
        for start in range(0, len(discipline), BATCH):  # a few row groups at a time
            group = discipline.iloc[start:start + BATCH]
            fixes = grid.read_fixes(group).merge(group[["flight_id", "flight"]], on="flight_id")
            for flight, f in fixes.groupby("flight"):
                mine = by_flight[flight]
                out.append(mine.assign(distance_m=stretch_distances(f, mine)))
    s = pd.concat(out, ignore_index=True)
    best = s.loc[s.groupby("episode_id")["distance_m"].idxmax(), ["episode_id", "distance_m", "flight"]]
    return best.sort_values("episode_id").reset_index(drop=True)


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else runs.DATA / "runs" / runs.run_name(runs.PARAMS)
    d = distances(folder)
    d.to_parquet(folder / "episode_distance.parquet", index=False)
    print(d["distance_m"].describe().round(0).to_string())
