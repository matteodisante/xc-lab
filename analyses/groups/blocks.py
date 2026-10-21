"""Bands of H and H_loc with dependency blocks coarser than the episodes.

The bands of tamsd.py resample the blocks of the note: flights tied by episodes, a solo flight
alone. Flights of the same day and site share the weather even when they never meet, so here
a block also takes every flight whose takeoff lies within SITE_M of another one's on the same
day (chained), or every flight of the day. Same pieces, seed and bootstrap as tamsd.py, so the
row "episodes" repeats its bands.

    uv run python blocks.py [--run NAME]    # from this folder, after tamsd.py; writes out/<run>/blocks.json
"""

import json

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from tamsd import CACHE, FIXES, LABELS, LAGS, OUT, SEED, summarize

SITE_M = 3000.0  # the departure radius of the note: takeoffs this close on one day are one site
EARTH_M = 6_371_000.0
DAY_APART_M = 1e8  # days are laid this far apart on the plane, so no site spans two days
AT = (100, 1000)  # lags of the H_loc quoted


def components(n, pairs):
    """Connected components of n nodes joined by the rows of `pairs`."""
    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def merged(*labels):
    """Blocks of flights tied by sharing any of the `labels` (arrays over the flights)."""
    n = len(labels[0])
    pairs = []
    for label in labels:
        _, k = np.unique(label, return_inverse=True)
        first = np.full(k.max() + 1, -1)
        first[k[::-1]] = np.arange(n)[::-1]  # the first flight of each label
        pairs.append(np.column_stack([np.arange(n), first[k]]))
    return components(n, np.concatenate(pairs))


def sites(lat, lon, date):
    """Site of each flight: takeoffs within SITE_M of each other on the same day, chained."""
    lat, lon = np.radians(lat), np.radians(lon)
    day = pd.factorize(np.asarray(date))[0]
    xy = np.column_stack([EARTH_M * lon * np.cos(lat) + DAY_APART_M * day, EARTH_M * lat])
    return components(len(xy), cKDTree(xy).query_pairs(SITE_M, output_type="ndarray"))


def quoted(r):
    """H with its band, H_loc at AT with their bands."""
    k = [int(np.searchsorted(LAGS, x)) for x in AT]
    return {"flights": r["flights"], "H": r["H"], "H_band": r["H_band"],
            "H_loc": [r["H_loc"][i] for i in k],
            "H_loc_band": [[r["H_loc_band"][0][i], r["H_loc_band"][1][i]] for i in k]}


if __name__ == "__main__":
    flights = pd.read_parquet(LABELS / "flights.parquet")
    block = pd.read_parquet(LABELS / "flight_summary.parquet").set_index("flight")["block"]
    takeoff = pd.concat([pd.read_parquet(FIXES[d].parent / "flights_meta.parquet", columns=["flight_id", "lat0", "lon0"])
                         .assign(discipline=d) for d in FIXES])
    flights = flights.astype({"flight_id": str}).merge(takeoff.astype({"flight_id": str}), how="left",
                                                       on=["discipline", "flight_id"], validate="one_to_one")
    assert flights[["lat0", "lon0"]].notna().all().all()
    episode = block.reindex(flights["flight"]).to_numpy()
    variants = {"episodes": episode,
                "day and site": merged(episode, sites(flights["lat0"], flights["lon0"], flights["date"])),
                "day": merged(episode, flights["date"].to_numpy())}
    data = np.load(CACHE / "pieces.npz")
    measured = {d: {k.split("/")[1]: data[k] for k in data.files if k.startswith(d + "/")} for d in FIXES}
    out = {}
    for name, b in variants.items():
        table = pd.DataFrame({"flight": flights["flight"], "block": b})
        rng = np.random.default_rng(SEED)
        out[name] = {"blocks": int(len(np.unique(b)))}
        for d in FIXES:
            out[name][d] = {pop: quoted(r) for pop, r in summarize(measured[d], table, rng).items()}
            for pop, r in out[name][d].items():
                (lo, hi), (l1, h1) = r["H_band"], r["H_loc_band"][1]
                print(f"{name:13s} {d} {pop:10s} H = {r['H']:.3f} [{lo:.3f}, {hi:.3f}]   "
                      f"H_loc(1000 s) = {r['H_loc'][1]:.3f} [{l1:.3f}, {h1:.3f}]")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "blocks.json").write_text(json.dumps(out, indent=1))
