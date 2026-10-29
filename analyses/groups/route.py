"""Group time against the same route flown alone on another day: does the straighter progress of
groups come from the route they fly, or from flying together?

Pieces of group time are those of tamsd.py (paragliders, at least TAU_MAX / SUPPORT long). The
candidates are the flights that never had a contact of any length (intervals_raw of the run) and
took off from the same site on another date. A site joins the takeoffs within SITE_LINK m,
chained, across all dates; its flights are placed in one east-north frame, centred on the mean of
its takeoffs, by the offset of their own origin (the rotation between the frames, below 1e-3 rad,
is neglected). A candidate flew the same route when at least COVER of the points of the piece, one
every MATCH_S s, have its track within RHO m, and it passed nearest to the first of them before the
last; its segment between those two points is measured as the pieces are (10-s grid, TA-MSD at
LAGS) if it lasts at least TAU_MAX / SUPPORT and lies in one piece of its flight.

Comparison: stratum = the piece of group time, with its matched solo segments, at most MAX_MATCHES
per piece, the most covered first. The ensemble MSD of the pieces is compared with that of the
segments reweighted to one piece per stratum (zones.standardized): each side is a mean of many
MSDs before its logarithm. A stratified mean of log MSD differences would set the log of one noisy
MSD against the log of a mean of several, which raises the second side at the long lags. Two sets
of matches: any date, and the same season and hour (month within MONTHS, local hour of the middle
within HOURS). Bands: N_BOOT draws of the blocks of decompose.flight_table. Baseline: the same
matched pieces against the solo flights of their site-day and hour (the comparison of the report,
on these pieces), so that the two differences refer to the same group time.

    uv run python route.py [--run NAME] [--sample N] [--rho M] [--cover SHARE]    # after tamsd.py
    # the first run reads the fixes; matches and results are named by the sample and the parameters
"""

import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import decompose
import zones
from blocks import EARTH_M, components, with_takeoffs
from common.legacy import stream_flights
from common.msd import tamsd_sums
from tamsd import (CACHE, FIXES, GRID_S, H_MAX, LABELS, LAGS, OUT, SEED, SUPPORT, TAU_MAX, grids, read_pieces, save,
                   stretches)

def option(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


SITE_LINK = 1000.0  # m: takeoffs this close are one site, across dates
RHO_DEFAULT, COVER_DEFAULT = 500.0, 0.8
RHO = option("--rho", RHO_DEFAULT)  # m: a point of the piece is on the route when the candidate passed this close (eps_h)
COVER = option("--cover", COVER_DEFAULT)  # share of the points of the piece that must be on the candidate's track
MATCH_S = 30  # s between the points of the piece that are checked
MAX_MATCHES = 20  # solo segments kept per piece, the most covered first
MONTHS, HOURS = 1, 2  # same season and hour: month and local hour of the middle within these
SHORTEST = TAU_MAX / SUPPORT
TRACKS = CACHE / "route_tracks.npz"


def eligible(flights, summary, raw):
    """Paraglider flights on the grid of the labels: those with group time, and those that never
    had a contact of any length (no row in the raw intervals of the run)."""
    f = flights.merge(summary, on="flight")
    ok = (f["discipline"] == "para") & (f["dt_native"] <= H_MAX + 1e-6) & (f["n_grid"] > 0)
    touched = np.union1d(raw["a"].to_numpy(), raw["b"].to_numpy())
    alone = ok & (f["n_entries"] == 0) & ~f["flight"].isin(touched) & ~f["local"]
    return f.loc[ok & (f["n_entries"] > 0), "flight"].to_numpy(), f.loc[alone, "flight"].to_numpy()


def site_frames(lat0, lon0):
    """Site of every takeoff (degrees) and the east and north offsets, in m, of its origin from the
    centre of its site."""
    lat, lon = np.radians(lat0), np.radians(lon0)
    xy = np.column_stack([EARTH_M * lon * np.cos(lat), EARTH_M * lat])
    site = components(len(xy), cKDTree(xy).query_pairs(SITE_LINK, output_type="ndarray"))
    lat_c = pd.Series(lat).groupby(site).transform("mean").to_numpy()
    lon_c = pd.Series(lon).groupby(site).transform("mean").to_numpy()
    return site, EARTH_M * np.cos(lat_c) * (lon - lon_c), EARTH_M * (lat - lat_c)


def coverage(points, tree, owner, exclude=(), workers=-1):
    """For every flight owning points of `tree` (owner: its flight per point), the share of
    `points` with one of its points within RHO; flights in `exclude` are left out."""
    hits = tree.query_ball_point(points, RHO, workers=workers)  # threads share one copy of the tree
    n = np.fromiter((len(h) for h in hits), int, len(hits))
    if not n.sum():
        return pd.Series(dtype=float)
    found = np.concatenate([np.asarray(h, np.int64) for h in hits])
    pairs = np.unique(np.column_stack([np.repeat(np.arange(len(points)), n), owner[found]]), axis=0)
    share = pd.Series(pairs[:, 1]).value_counts() / len(points)
    return share.drop(list(exclude), errors="ignore")


def segment(pos, first, last):
    """Indices of the points of `pos` nearest to `first` and to `last`, when both lie within RHO
    and the first comes before the last (the route flown the same way); otherwise None."""
    d1, d2 = np.linalg.norm(pos - first, axis=1), np.linalg.norm(pos - last, axis=1)
    i, j = int(d1.argmin()), int(d2.argmin())
    return (i, j) if d1[i] <= RHO and d2[j] <= RHO and j > i else None


def _job(flight, fixes, spans):
    return flight, [(s.population, start, pos.astype(np.float32)) for s, start, pos in grids(fixes, spans, SHORTEST)]


def read_tracks(flights, spans):
    """Positions on the 10-s grid of every piece of `spans`, from one pass over the fixes, cached in
    TRACKS: flight, population, start (flight clock), offset and length in `pos`, and pos."""
    if TRACKS.exists():
        data = np.load(TRACKS)
        return {k: data[k] for k in data.files}
    ids = flights.loc[flights["discipline"] == "para"].set_index("flight_id")["flight"]
    by_flight = dict(tuple(spans.groupby("flight")))
    rows = []
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        pending = set()
        for fixes in stream_flights(FIXES["para"], ["segment_id", "t", "E", "N"]):
            flight = ids.get(str(fixes["flight_id"].iat[0]))
            if flight in by_flight:
                pending.add(pool.submit(_job, flight, fixes, by_flight[flight]))
            if len(pending) >= 4 * os.cpu_count():
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                rows.extend(r for f in done for r in [f.result()])
        rows.extend(f.result() for f in pending)
    pieces = [(flight, pop, start, pos) for flight, found in rows for pop, start, pos in found]
    n = np.array([len(p[3]) for p in pieces])
    out = {"flight": np.array([p[0] for p in pieces], np.int64), "population": np.array([p[1] for p in pieces]),
           "start": np.array([p[2] for p in pieces]), "offset": np.r_[0, np.cumsum(n)[:-1]], "n": n,
           "pos": np.concatenate([p[3] for p in pieces])}
    np.savez(TRACKS, **out)
    return out


def matches(tracks, flights, sample=None):
    """Every match of a piece of group time with a solo segment: piece, solo piece, i, j (the
    segment in the solo piece) and the share of the piece covered."""
    info = flights.set_index("flight")
    site, de, dn = site_frames(info["lat0"].to_numpy(float), info["lon0"].to_numpy(float))
    frame = pd.DataFrame({"site": site, "de": de, "dn": dn}, index=info.index).reindex(tracks["flight"])
    date = info["date"].reindex(tracks["flight"]).to_numpy()
    piece_site = frame["site"].to_numpy()
    shift = frame[["de", "dn"]].to_numpy()
    pos = lambda k: tracks["pos"][tracks["offset"][k]:tracks["offset"][k] + tracks["n"][k]] + shift[k]
    group = np.flatnonzero(tracks["population"] == "group")
    if sample:
        group = np.sort(np.random.default_rng(SEED).choice(group, min(sample, len(group)), replace=False))
    solo = np.flatnonzero(tracks["population"] == "solo")
    out = []
    for s in np.intersect1d(piece_site[group], piece_site[solo]):
        mine = solo[piece_site[solo] == s]
        points = np.concatenate([pos(k) for k in mine])
        owner = np.repeat(tracks["flight"][mine], tracks["n"][mine])
        tree = cKDTree(points)
        for g in group[piece_site[group] == s]:
            route = pos(g)[::MATCH_S // GRID_S]
            same_day = tracks["flight"][mine][date[mine] == date[g]]
            share = coverage(route, tree, owner, same_day)
            kept = 0
            for flight, c in share[share >= COVER].sort_values(ascending=False, kind="stable").items():
                for k in mine[tracks["flight"][mine] == flight]:
                    ij = segment(pos(k), route[0], route[-1])
                    if ij and (ij[1] - ij[0]) * GRID_S >= SHORTEST:
                        out.append((g, k, *ij, c))
                        kept += 1
                        break
                if kept == MAX_MATCHES:
                    break
        del points, owner, tree  # one site in memory at a time
    return pd.DataFrame(out, columns=["piece", "solo", "i", "j", "cover"])


def units(tracks, found, flights, block, season=False):
    """decompose.Units of the matched pieces (population "group") and their solo segments
    ("route"), the piece as stratum; with season, only the matches of the same season and hour."""
    info = flights.set_index("flight")
    utc0 = info["utc0"].reindex(tracks["flight"]).to_numpy()
    mid_g = utc0[found["piece"]] + tracks["start"][found["piece"]] + tracks["n"][found["piece"]] * GRID_S / 2
    mid_s = utc0[found["solo"]] + tracks["start"][found["solo"]] + (found["i"] + found["j"]).to_numpy() * GRID_S / 2
    if season:
        month = lambda t: pd.to_datetime(t, unit="s", utc=True).month.to_numpy()
        dm = np.abs(month(mid_g) - month(mid_s))
        keep = (np.minimum(dm, 12 - dm) <= MONTHS) & (np.abs(decompose.local_hour(mid_g) - decompose.local_hour(mid_s)) <= HOURS)
        found = found[keep]

    def msd(k, i=0, j=None):
        p = tracks["pos"][tracks["offset"][k]:tracks["offset"][k] + tracks["n"][k]][i:j]
        sums, origins = tamsd_sums(p.astype(float), LAGS // GRID_S)
        return sums / origins

    pieces = np.unique(found["piece"])
    rows = [("group", g, tracks["flight"][g], msd(g)) for g in pieces]
    rows += [("route", r.piece, tracks["flight"][r.solo], msd(r.solo, r.i, r.j + 1)) for r in found.itertuples()]
    pop, stratum, flight, values = zip(*rows)
    return decompose.Units(np.array(pop), np.array(stratum), block.reindex(np.array(flight)).to_numpy(),
                           np.array(values)), found


def compare(u, counts, populations=("group", "route")):
    """The pieces and the other population reweighted to them: H_loc over every lag and fitted H of
    each, and Delta H, Delta H_loc at 100 and 1000 s and over every lag (P minus Q), with bands."""
    mp, mq, _, _ = zones.standardized(u, *populations, counts)
    (hp, fp), (hq, fq) = zones.curve(mp), zones.curve(mq)
    k = {x: int(np.searchsorted(LAGS, x)) for x in (100, 1000)}
    return {"Delta H": decompose.described(fp - fq),
            **{f"H_loc({x} s)": decompose.described((hp - hq)[:, i]) for x, i in k.items()},
            "Delta H_loc": (hp - hq)[0].tolist(), "Delta H_loc band": np.nanpercentile((hp - hq)[1:], [5, 95], axis=0).tolist(),
            "H_loc": {"group": hp[0].tolist(), "route": hq[0].tolist()}, "H": {"group": float(fp[0]), "route": float(fq[0])}}


def baseline(found, tracks, info, block, counts, solo=None):
    """The same pieces of group time against the solo flights (never in a group) of their site-day
    and hour, from the pieces of tamsd.py: the comparison of the report, on these pieces only.
    With `solo` (flight numbers), only those solo flights."""
    p = read_pieces(CACHE / "pieces.npz")["para"]
    mine = pd.MultiIndex.from_arrays([tracks["flight"][found["piece"]], tracks["start"][found["piece"]]])
    allowed = np.ones(len(p["flight"]), bool) if solo is None else np.isin(p["flight"], solo)
    keep = ((p["population"] == "solo") & allowed) | ((p["population"] == "group")
                                          & pd.MultiIndex.from_arrays([p["flight"], p["start"]]).isin(mine))
    u = decompose.msd_units({k: v[keep] for k, v in p.items()}, "site_hour", info, block)
    return {"pieces": int(keep[p["population"] == "group"].sum()), **compare(u, counts, ("group", "solo"))}


def figure(result, path):
    """Left: H_loc of the matched pieces of group time and of their routes flown alone. Right: the
    difference over every lag, any date and same season and hour, with 5-95% bands."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), constrained_layout=True)
    r = result["any date"]
    for pop, color, text in (("group", "#eb6834", "group time"), ("route", "#2a78d6", "same route, alone, another day")):
        axes[0].plot(LAGS, r["H_loc"][pop], color=color, marker="o", ms=3, lw=1.3, label=text)
    axes[0].set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$H_{\rm loc}(\tau)$")
    axes[0].legend(frameon=False, fontsize=8)
    for key, color in (("any date", "#8c2f0f"), ("same season and hour", "#f2a065")):
        e = result[key]
        axes[1].fill_between(LAGS, *np.array(e["Delta H_loc band"]), color=color, alpha=0.25, lw=0)
        axes[1].plot(LAGS, e["Delta H_loc"], color=color, marker="o", ms=3, lw=1.3, label=f"{key}: {e['pieces']} pieces")
    axes[1].axhline(0, color="#9a9893", lw=0.8)
    axes[1].set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$\Delta H_{\rm loc}(\tau)$, group $-$ route")
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, path)


if __name__ == "__main__":
    sample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None
    flights = with_takeoffs(pd.read_parquet(LABELS / "flights.parquet"))
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    group, alone = eligible(flights, summary, pd.read_parquet(LABELS / "intervals_raw.parquet"))
    spans = None
    if not TRACKS.exists():  # the stretches are needed only to build the cache
        spans = stretches(flights, pd.read_parquet(LABELS / "membership.parquet"), summary)
        spans = spans[((spans["population"] == "group") & spans["flight"].isin(group))
                      | ((spans["population"] == "solo") & spans["flight"].isin(alone))]
    tracks = read_tracks(flights, spans)
    tag = ("" if sample is None else f"_sample{sample}") \
        + ("" if (RHO, COVER) == (RHO_DEFAULT, COVER_DEFAULT) else f"_rho{RHO:g}_cover{COVER:g}")
    cached = CACHE / f"route_matches{tag}.parquet"
    if not cached.exists():
        matches(tracks, flights, sample).to_parquet(cached)
    found = pd.read_parquet(cached)
    info, block, n_blocks = decompose.flight_table(flights, summary)
    counts = decompose.block_counts(n_blocks, np.random.default_rng(SEED))
    pieces = int((tracks["population"] == "group").sum()) if sample is None else sample
    result = {"group pieces": pieces, "alone flights": int(len(alone)), "matches": int(len(found)),
              "median cover": float(found["cover"].median())}
    for key, season in (("any date", False), ("same season and hour", True)):
        u, kept = units(tracks, found, flights, block, season)
        result[key] = {"pieces": int(kept["piece"].nunique()), "matches": int(len(kept)),
                       "median matches per piece": float(kept.groupby("piece").size().median()), **compare(u, counts),
                       "baseline": baseline(kept, tracks, info, block, counts)}
    name = f"route{tag}"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    figure(result, OUT / f"{name}.pdf")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}
                     | {k: {s: result[k][s] for s in ("pieces", "matches", "Delta H", "H_loc(100 s)", "H_loc(1000 s)")}
                        for k in ("any date", "same season and hour")}, indent=1))
