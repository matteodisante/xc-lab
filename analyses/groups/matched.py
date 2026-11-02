"""The controls of the pieces of group time (G) and of the solo time of the same flights (A): one
match table on the route only, and the counts of pieces that have a control under each caliper.

A segment of a pool piece (route.TRACKS, population "solo": flights with no contact of any length)
matches a query piece of G or A when route.COVER-like rules hold: at least MIN_COVER of the points
of the query, one every route.MATCH_S s, have the pool track within route.RHO m, the pool flight
flew them the same way (route.segment) and the segment lasts at least route.SHORTEST. The date is
other than that of the query; there is no condition on season, hour, wing or site. A query keeps one
match per pool flight, the most covered. Calipers, cover and isolation are filters on the table:

    dmonth   circular difference of the months of the middles (0-6)
    dhour    circular difference of the local time of day (Europe/Paris) of the middles, hours
    class_*  catalog wing class; level_* beginners or experienced (NaN when the class is in neither)
    isolated every point of the segment, one every route.MATCH_S s, had nobody within
             isolated.ISOLATION_M at the same minute (isolated.isolated)

Positions are on the sphere (isolated.sphere); one tree holds all the pool points.

    uv run python matched.py --open [--sample N]    # after tamsd.py and route.py (or on its own for the tracks)
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import isolated
import route
from blocks import with_takeoffs
from tamsd import CACHE, GRID_S, LABELS, OUT, SEED, SUPPORT, read_flights
from isolated import MIN_PIECES, WORKERS, levels, sphere, wing_class

MIN_COVER = 0.7  # the lowest cover of the robustness; the table keeps every match from here
MAIN_COVER, COVERS = 0.8, (0.7, 0.8, 0.9)
DATASETS = {"1000": 1000.0, "2250": 2250.0}  # name: lag T; pieces and segments last at least T / SUPPORT
LEVEL_NAMES = ("beginners", "experienced", "all")
# name: (month within, hour within, same wing "class" or "level"); the order is that of the choice
CALIPERS = {"month1_hour1_class": (1, 1, "class"), "month1_hour1_level": (1, 1, "level"),
            "month2_hour2_class": (2, 2, "class"), "month2_hour2_level": (2, 2, "level")}
COLUMNS = ["side", "piece", "solo", "i", "j", "cover"]


def month_gap(a, b):
    """Circular difference (0-6) of the months of two instants (UTC s, arrays)."""
    month = lambda t: pd.to_datetime(np.asarray(t, float), unit="s", utc=True).month.to_numpy()
    d = np.abs(month(a) - month(b))
    return np.minimum(d, 12 - d)


def time_of_day(utc):
    """Local time of day (Europe/Paris) of UTC seconds, in hours (float)."""
    t = pd.to_datetime(np.asarray(utc, float), unit="s", utc=True).tz_convert("Europe/Paris")
    return (t.hour + t.minute / 60 + t.second / 3600).to_numpy(float)


def hour_gap(a, b):
    """Absolute difference of the local time of day of two instants (UTC s), in hours, across midnight."""
    d = np.abs(time_of_day(a) - time_of_day(b))
    return np.minimum(d, 24 - d)


def segment_isolated(flags, i, j, step):
    """Whether every checked point of the segment [i, j] (grid indices) is isolated; `flags` holds
    one flag per `step` grid points from the start of the solo piece."""
    return bool(flags[-(-i // step):j // step + 1].all())


def select(table, dataset, caliper, cover=MAIN_COVER, isolation=False):
    """The rows of the match table that are controls under `caliper` (a key of CALIPERS) at `cover`,
    for pieces and segments of the dataset; with `isolation`, only isolated segments."""
    month, hour, by = CALIPERS[caliper]
    shortest = DATASETS[dataset] / SUPPORT
    same = table[f"{by}_piece"].notna() & (table[f"{by}_piece"] == table[f"{by}_solo"])
    keep = ((table["cover"] >= cover - 1e-9) & (table["dur_piece"] >= shortest) & (table["dur_solo"] >= shortest)
            & (table["dmonth"] <= month) & (table["dhour"] <= hour + 1e-9) & same)
    return table[keep & table["isolated"]] if isolation else table[keep]


def tally(controls):
    """Per level of the query piece (beginners, experienced, all): pieces with at least one control
    and the median number of controls per piece."""
    out = {}
    for name in LEVEL_NAMES:
        c = controls if name == "all" else controls[controls["level_piece"] == name]
        per = c.groupby("piece").size()
        out[name] = {"pieces": int(len(per)), "median controls": float(per.median()) if len(per) else None}
    return out


def choose_caliper(pieces, minimum=MIN_PIECES):
    """The first caliper of `pieces` (name: {level: count}, in the order of CALIPERS) with at least
    `minimum` pieces in beginners and in experienced; None when there is none."""
    for name, by_level in pieces.items():
        if all(by_level[level] >= minimum for level in ("beginners", "experienced")):
            return name
    return None


def queries_of(tracks, info, level):
    """Number of query pieces of each side and level that last long enough for each dataset."""
    pop = tracks["population"]
    dur = (tracks["n"] - 1) * GRID_S
    lv = level.reindex(tracks["flight"]).to_numpy()
    out = {}
    for side, label in (("G", "group"), ("A", "group_solo")):
        out[side] = {d: {name: int(((pop == label) & (dur >= T / SUPPORT) & ((lv == name) | (name == "all"))).sum())
                         for name in LEVEL_NAMES} for d, T in DATASETS.items()}
    return out


def match_table(tracks, flights, queries, sample=None):
    """Rows (COLUMNS) of the matches of the query pieces (indices of `tracks`) with the pool pieces.
    With `sample`, that many pieces of each side chosen at random."""
    info = flights.set_index("flight").reindex(tracks["flight"])
    lat0, lon0, date = (info[c].to_numpy() for c in ("lat0", "lon0", "date"))
    lat0, lon0 = lat0.astype(float), lon0.astype(float)
    n, flight, step = tracks["n"], tracks["flight"], route.MATCH_S // GRID_S

    def pos(k):
        p = tracks["pos"][tracks["offset"][k]:tracks["offset"][k] + n[k]].astype(float)
        return sphere(lat0[k], lon0[k], p[:, 0], p[:, 1])

    pool = np.flatnonzero(tracks["population"] == "solo")
    first = np.r_[0, np.cumsum(n[pool])]
    points = np.empty((first[-1], 3))
    for a, b, k in zip(first[:-1], first[1:], pool):
        points[a:b] = pos(k)
    owner = np.repeat(np.arange(len(pool), dtype=np.int32), n[pool])
    print(f"pool: {len(pool)} pieces, {len(np.unique(flight[pool]))} flights, {first[-1]} points", file=sys.stderr, flush=True)
    tree = cKDTree(points, copy_data=False, balanced_tree=False, compact_nodes=False)
    rng = np.random.default_rng(SEED)
    out = []
    for side, label in (("G", "group"), ("A", "group_solo")):
        mine = queries[tracks["population"][queries] == label]
        if sample:
            mine = np.sort(rng.choice(mine, min(sample, len(mine)), replace=False))
        for d, g in enumerate(mine):
            path = pos(g)[::step]
            h0, h1 = tree.query_ball_point(path[[0, -1]], route.RHO, workers=WORKERS)
            near = np.intersect1d(owner[h0], owner[h1])  # both ends on the track: needed by route.segment
            near = near[date[pool[near]] != date[g]]
            if len(near):
                k = pool[near]
                sub = np.concatenate([points[first[c]:first[c + 1]] for c in near])
                share = route.coverage(path, cKDTree(sub), np.repeat(k, n[k]), workers=1)
                seen = set()
                for kk in share[share >= MIN_COVER - 1e-9].sort_values(ascending=False, kind="stable").index:
                    if flight[kk] in seen:
                        continue
                    c = np.searchsorted(pool, kk)
                    ij = route.segment(points[first[c]:first[c + 1]], path[0], path[-1])
                    if ij and (ij[1] - ij[0]) * GRID_S >= route.SHORTEST:
                        seen.add(flight[kk])
                        out.append((side, g, kk, *ij, share[kk]))
            if d % 1000 == 0:
                print(f"{side} {d}/{len(mine)}: {len(out)} matches", file=sys.stderr, flush=True)
    return pd.DataFrame(out, columns=COLUMNS)


def describe(found, tracks, flights, summary_level, cls):
    """`found` with the gaps, the wing of both sides, the durations and the isolation flag."""
    info = flights.set_index("flight").reindex(tracks["flight"])
    utc0 = info["utc0"].to_numpy(float)
    g, s = found["piece"].to_numpy(), found["solo"].to_numpy()
    mid_g = utc0[g] + tracks["start"][g] + tracks["n"][g] * GRID_S / 2
    mid_s = utc0[s] + tracks["start"][s] + (found["i"] + found["j"]).to_numpy() * GRID_S / 2
    f = tracks["flight"]
    found = found.assign(dmonth=month_gap(mid_g, mid_s).astype(np.int8), dhour=hour_gap(mid_g, mid_s),
                         class_piece=cls.reindex(f[g]).to_numpy(), class_solo=cls.reindex(f[s]).to_numpy(),
                         level_piece=summary_level.reindex(f[g]).to_numpy(), level_solo=summary_level.reindex(f[s]).to_numpy(),
                         dur_piece=(tracks["n"][g] - 1) * GRID_S, dur_solo=((found["j"] - found["i"]) * GRID_S).to_numpy())
    flags = isolated.solo_flags(tracks, flights, np.unique(s), isolated.read_index(flights))
    step = route.MATCH_S // GRID_S
    return found.assign(isolated=[segment_isolated(flags[a], i, j, step) for a, i, j in zip(s, found["i"], found["j"])])


def counts(table, queries):
    """The table of pieces and controls per side, dataset, caliper and level (isolation off, main cover),
    the chosen caliper, and its counts with isolation on and at covers MIN_COVER and 0.9."""
    grid = {side: {d: {c: tally(select(table[table["side"] == side], d, c)) for c in CALIPERS} for d in DATASETS}
            for side in ("G", "A")}
    pieces = {c: {lv: grid["G"]["2250"][c][lv]["pieces"] for lv in LEVEL_NAMES} for c in CALIPERS}
    name = choose_caliper(pieces)
    meets = name is not None
    name = name or list(CALIPERS)[-1]  # none reaches MIN_PIECES: the loosest, flagged
    month, hour, by = CALIPERS[name]
    variants = {"isolation on": dict(isolation=True), "cover 0.7": dict(cover=0.7), "cover 0.9": dict(cover=0.9)}
    robust = {key: {side: {d: tally(select(table[table["side"] == side], d, name, **kw)) for d in DATASETS}
                    for side in ("G", "A")} for key, kw in variants.items()}
    return {"min pieces": MIN_PIECES, "main cover": MAIN_COVER, "rho": route.RHO, "min cover of the table": MIN_COVER,
            "calipers": {c: dict(zip(("month within", "hour within", "same"), v)) for c, v in CALIPERS.items()},
            "query pieces": queries, "pieces with a control": grid,
            "chosen caliper": {"name": name, "month within": month, "hour within": hour, "same": by,
                               "meets min pieces": meets, "G pieces on dataset 2250": pieces[name]},
            "chosen caliper, robustness": robust}


def show(result):
    grid = result["pieces with a control"]
    print(f"{'side':4} {'data':5} {'caliper':20} " + " ".join(f"{lv[:5]:>12}" for lv in LEVEL_NAMES) + "   (pieces / median controls)")
    for side in grid:
        for d in grid[side]:
            for c, by in grid[side][d].items():
                cells = " ".join(f"{by[lv]['pieces']:>6}/{by[lv]['median controls'] or 0:>5.1f}" for lv in LEVEL_NAMES)
                print(f"{side:4} {d:5} {c:20} {cells}")
    print("chosen:", json.dumps(result["chosen caliper"]))
    for key, by_side in result["chosen caliper, robustness"].items():
        for side, by_d in by_side.items():
            for d, by in by_d.items():
                print(f"{key:13} {side} {d:5} " + " ".join(f"{lv[:5]}={by[lv]['pieces']}/{by[lv]['median controls'] or 0:.1f}" for lv in LEVEL_NAMES))


if __name__ == "__main__":
    sample = route.option("--sample", 0) or None
    flights = with_takeoffs(read_flights())
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    if not route.TRACKS.exists():
        group, alone = route.eligible(flights, summary, pd.read_parquet(LABELS / "intervals_raw.parquet"))
        from tamsd import stretches
        spans = route.track_spans(stretches(flights, pd.read_parquet(LABELS / "membership.parquet"), summary), group, alone)
    else:
        spans = None
    tracks = route.read_tracks(flights, spans)
    pop = pd.Series(tracks["population"]).value_counts().to_dict()
    print("tracks:", pop, file=sys.stderr, flush=True)
    tag = f"_sample{sample}" if sample else ""
    path = CACHE / f"comparisons_matches{tag}.parquet"
    level, cls = levels(flights), wing_class(flights)
    if not path.exists():
        queries = np.flatnonzero(np.isin(tracks["population"], ["group", "group_solo"]))
        describe(match_table(tracks, flights, queries, sample), tracks, flights, level, cls).to_parquet(path)
    table = pd.read_parquet(path)
    result = {"matches": int(len(table)), "pieces with a match": table.groupby("side")["piece"].nunique().to_dict(),
              "pool flights": int(np.unique(tracks["flight"][table["solo"]]).size),
              **counts(table, queries_of(tracks, flights, level))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"matched_counts{tag}.json").write_text(json.dumps(result, indent=1))
    print({k: result[k] for k in ("matches", "pieces with a match", "pool flights")})
    show(result)
