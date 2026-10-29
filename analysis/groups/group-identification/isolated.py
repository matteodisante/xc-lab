"""Group time against truly solo flights on the same route, on other days: how much of the
group - solo difference remains once route, season, hour and wing level are the same?

Pieces of group time and candidates are those of route.py. A segment k of a solo piece matches the
piece g when all hold:
1. route: at least route.COVER of the points of g, one every route.MATCH_S s, have the track of
   the candidate within route.RHO m, k flew them the same way (route.segment) and lasts at least
   route.SHORTEST;
2. date: other than that of g, whatever the site;
3. season and hour: the middles of g and k are in months within route.MONTHS (circular) and in
   local hours within route.HOURS;
4. wing level: the same (beginners A, B; experienced C, D, CCC of the FFVL catalog);
5. solo: the flight never had a contact of any length (population "solo" of route.py);
6. isolation: at every point of k, one every route.MATCH_S s, no other flight of the archive, alone
   or in a group, para or hang, was within ISOLATION_M at the same instant, within TIME_S (the
   index holds one position per minute).

All matches are kept. The statistics compare, per level, the matched pieces with their segments, each
piece weighing the same, over the lags of tamsd.LAGS (main series) and over longer lags on the
pieces and segments that last long enough (long series), then the time in each phase.

Positions are on the sphere (x, y, z in m), so that distances are the same everywhere. The index of
all the flights of the archive, one position per minute, is read once from the fixes.

    uv run python isolated.py [--run NAME] [--sample N]    # after tamsd.py and route.py
"""

import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

import budget
import decompose
import performance
import route
import zones
from blocks import EARTH_M, with_takeoffs
from lab.legacy import stream_flights
from lab.msd import tamsd_sums
from phase_msd import RUNS
from tamsd import CACHE, COLORS, FIXES, GRID_S, LABELS, LAGS, N_BOOT, OUT, SEED, SUPPORT, TAU_MAX, save

sys.path.insert(0, str(Path(__file__).parents[1] / "experienced"))
from population import CATALOG, LEVELS, wing_class  # noqa: E402

STEP_S = 60  # s between the positions of the index
INDEX = CACHE / "archive_index.npz"
ISOLATION_M = 10_000.0  # m: nobody within this of any point of the segment
TIME_S = 60  # s: "at the same instant" for the isolation, within this
WORKERS = 6  # threads of the queries, processes of the isolation
LONG_CANDIDATES = (2000, 3000, 5000, 8000)  # s: the longest lag of the long series is the largest with enough pieces
MIN_PIECES = 200  # matched pieces per level the long series must keep
LAGS_EXTENDED = np.unique(np.r_[np.round(np.geomspace(1, 1000, 35)).astype(int) * 10, 100, 1000])  # s: the spacing of tamsd.LAGS, further
LAGS_EXTENDED = LAGS_EXTENDED[LAGS_EXTENDED <= max(LONG_CANDIDATES)]
assert (LAGS_EXTENDED[LAGS_EXTENDED <= TAU_MAX] == LAGS).all()
CHUNK = 100  # draws handled together
AT = (100, 1000)  # s: lags of H_loc quoted
RANK = 1e6  # s: flights are told apart in the sorted keys of window_sums by this


def sphere(lat0, lon0, east, north):
    """Positions on the sphere of radius EARTH_M, in m, of the points at `east` and `north` m of
    (lat0, lon0) on its tangent plane: array (n, 3)."""
    lat, lon = np.radians(lat0), np.radians(lon0)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    e = np.array([-np.sin(lon), np.cos(lon), 0.0])
    n = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    p = EARTH_M * up + np.outer(east, e) + np.outer(north, n)
    return EARTH_M * p / np.linalg.norm(p, axis=1, keepdims=True)


def _index_job(flight, fixes, utc0, lat0, lon0):
    """Positions every STEP_S s of the clock (UTC multiples of STEP_S) inside each segment of one
    flight: minute, position (n, 3)."""
    fixes = fixes.sort_values("t", kind="stable")
    t, seg = fixes["t"].to_numpy(float), fixes["segment_id"].to_numpy()
    minute, xyz = [], []
    for s in np.unique(seg):
        q = seg == s
        ts = t[q]
        g = np.arange(np.ceil((utc0 + ts[0]) / STEP_S) * STEP_S, utc0 + ts[-1] + 1e-6, STEP_S) - utc0
        if len(g) == 0:
            continue
        east, north = (np.interp(g, ts, fixes[c].to_numpy(float)[q]) for c in ("E", "N"))
        minute.append(np.round((utc0 + g) / STEP_S).astype(np.int32))
        xyz.append(sphere(lat0, lon0, east, north).astype(np.float32))
    if not minute:
        return flight, np.empty(0, np.int32), np.empty((0, 3), np.float32)
    return flight, np.concatenate(minute), np.concatenate(xyz)


def read_index(flights, limit=None):
    """Every flight of the archive on the minute grid: flight, minute (UTC / STEP_S) and position
    (n, 3), sorted by minute; cached in INDEX unless `limit` keeps only the first flights."""
    if limit is None and INDEX.exists():
        data = np.load(INDEX)
        return {k: data[k] for k in data.files}
    info = flights.set_index(["discipline", "flight_id"])
    rows = []
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        pending = set()
        for discipline, path in FIXES.items():
            for k, fixes in enumerate(stream_flights(path, ["segment_id", "t", "E", "N"])):
                if limit and k >= limit:
                    break
                key = (discipline, str(fixes["flight_id"].iat[0]))
                if key not in info.index:
                    continue
                f = info.loc[key]
                pending.add(pool.submit(_index_job, int(f["flight"]), fixes, f["utc0"], f["lat0"], f["lon0"]))
                if len(pending) >= 4 * os.cpu_count():
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    rows.extend(r.result() for r in done)
        rows.extend(r.result() for r in pending)
    flight = np.concatenate([np.full(len(m), f, np.int32) for f, m, _ in rows])
    minute, xyz = np.concatenate([m for _, m, _ in rows]), np.concatenate([p for _, _, p in rows])
    order = np.argsort(minute, kind="stable")
    out = {"flight": flight[order], "minute": minute[order], "xyz": xyz[order]}
    if limit is None:
        np.savez(INDEX, **out)
    return out


def levels(flights):
    """Wing level of every paraglider flight (beginners, experienced), by flight number; the others
    are left out."""
    of = {cls: level for level, classes in LEVELS.items() for cls in classes}
    return wing_class(flights).map(of).dropna()


def season_ok(mid_g, mid_s):
    """Whether instants (UTC s, arrays) are in the same season and hour: month within route.MONTHS
    (circular) and local hour within route.HOURS."""
    month = lambda t: pd.to_datetime(t, unit="s", utc=True).month.to_numpy()
    dm = np.abs(month(mid_g) - month(mid_s))
    return (np.minimum(dm, 12 - dm) <= route.MONTHS) & (np.abs(decompose.local_hour(mid_g) - decompose.local_hour(mid_s)) <= route.HOURS)


def matches(tracks, flights, level, sample=None):
    """Every match of a piece of group time with a segment of a solo piece under rules 1-5: level,
    piece, solo piece, i, j (the segment in the solo piece) and the share of the piece covered. A
    piece has at most one match per solo flight, the most covered. With `sample`, that many pieces
    of group time chosen at random."""
    info = flights.set_index("flight").reindex(tracks["flight"])
    lat0, lon0, utc0 = (info[c].to_numpy(float) for c in ("lat0", "lon0", "utc0"))
    date, lv = info["date"].to_numpy(), level.reindex(tracks["flight"]).to_numpy()
    month = pd.to_datetime(utc0, unit="s", utc=True).month.to_numpy()
    start, n, flight = tracks["start"], tracks["n"], tracks["flight"]

    def pos(k):
        p = tracks["pos"][tracks["offset"][k]:tracks["offset"][k] + n[k]].astype(float)
        return sphere(lat0[k], lon0[k], p[:, 0], p[:, 1])

    group = np.flatnonzero((tracks["population"] == "group") & pd.notna(lv))
    if sample:
        group = np.sort(np.random.default_rng(SEED).choice(group, min(sample, len(group)), replace=False))
    out, step = [], route.MATCH_S // GRID_S
    for name in LEVELS:
        mine = np.flatnonzero((tracks["population"] == "solo") & (lv == name))
        pieces = group[lv[group] == name]
        first = np.r_[0, np.cumsum(n[mine])]
        print(f"{name}: {len(pieces)} pieces, {len(mine)} solo pieces, {first[-1]} points", file=sys.stderr, flush=True)
        points = np.empty((first[-1], 3))
        for a, b, k in zip(first[:-1], first[1:], mine):
            points[a:b] = pos(k)
        owner = np.repeat(np.arange(len(mine), dtype=np.int32), n[mine])
        tree = cKDTree(points, copy_data=False, balanced_tree=False, compact_nodes=False)
        for d, g in enumerate(pieces):
            path = pos(g)[::step]
            h0, h1 = tree.query_ball_point(path[[0, -1]], route.RHO, workers=WORKERS)
            near = np.intersect1d(owner[h0], owner[h1])  # both ends on the track: needed by route.segment
            k = mine[near]
            dm = np.abs(month[k] - month[g])
            keep = (date[k] != date[g]) & (np.minimum(dm, 12 - dm) <= route.MONTHS + 1)  # the exact test follows
            near, k = near[keep], k[keep]
            if len(k):
                sub = np.concatenate([points[first[c]:first[c + 1]] for c in near])
                share = route.coverage(path, cKDTree(sub), np.repeat(k, n[k]), workers=1)
                rows = []
                for kk in share[share >= route.COVER].sort_values(ascending=False, kind="stable").index:
                    c = np.searchsorted(mine, kk)
                    ij = route.segment(points[first[c]:first[c + 1]], path[0], path[-1])
                    if ij and (ij[1] - ij[0]) * GRID_S >= route.SHORTEST:
                        rows.append((kk, *ij, share[kk]))
                if rows:
                    r = pd.DataFrame(rows, columns=["solo", "i", "j", "cover"])
                    mid_s = utc0[r["solo"]] + start[r["solo"]] + (r["i"] + r["j"]).to_numpy() * GRID_S / 2
                    r = r[season_ok(np.full(len(r), utc0[g] + start[g] + n[g] * GRID_S / 2), mid_s)]
                    r = r[~pd.Series(flight[r["solo"]]).duplicated().to_numpy()]  # one match per solo flight
                    out.extend((name, g, *row) for row in r.itertuples(index=False))
            if d % 500 == 0:
                print(f"{name} {d}/{len(pieces)}: {len(out)} matches", file=sys.stderr, flush=True)
        del tree, points, owner  # one level in memory at a time
    return pd.DataFrame(out, columns=["level", "piece", "solo", "i", "j", "cover"])


def isolated(tree, other, xyz, t, flight):
    """Per point of `xyz` (instants `t`, UTC s): whether no flight of the index other than `flight`
    was within ISOLATION_M at that instant, within TIME_S. `tree` holds the positions and the time
    (scaled so that TIME_S counts as ISOLATION_M) of the index slice `other` (its flights and
    positions)."""
    q = np.column_stack([xyz, t * ISOLATION_M / TIME_S])
    hits = tree.query_ball_point(q, ISOLATION_M, p=np.inf)  # the box, refined on the distance
    n = np.fromiter((len(h) for h in hits), int, len(hits))
    alone = np.ones(len(t), bool)
    if n.sum():
        found = np.concatenate([np.asarray(h, np.int64) for h in hits])
        point = np.repeat(np.arange(len(t)), n)
        near = np.linalg.norm(other["xyz"][found] - xyz[point], axis=1) <= ISOLATION_M
        alone[point[near & (other["flight"][found] != flight)]] = False
    return alone


def _day_job(flight, minute, xyz, pieces):
    """The flags of isolated() for each of `pieces` (flight, instants, positions), against the
    index slice of one day."""
    other = {"flight": flight, "xyz": xyz.astype(float)}
    tree = cKDTree(np.column_stack([other["xyz"], minute * STEP_S * ISOLATION_M / TIME_S]))
    return [isolated(tree, other, p, t, f) for f, t, p in pieces]


def solo_flags(tracks, flights, solo, index):
    """For each solo piece of `solo`, in a dict: the flags of isolated() at its points every
    route.MATCH_S s from its start. One task per date, each given only the slice of the index of
    its day."""
    info = flights.set_index("flight").reindex(tracks["flight"])
    lat0, lon0, utc0 = (info[c].to_numpy(float) for c in ("lat0", "lon0", "utc0"))
    step = route.MATCH_S // GRID_S
    days = pd.Series(solo).groupby(info["date"].to_numpy()[solo]).agg(list)

    def jobs():
        for ks in days:
            pieces = []
            for k in ks:
                idx = np.arange(0, tracks["n"][k], step)
                p = tracks["pos"][tracks["offset"][k]:tracks["offset"][k] + tracks["n"][k]][idx].astype(float)
                pieces.append((int(tracks["flight"][k]), utc0[k] + tracks["start"][k] + idx * GRID_S,
                               sphere(lat0[k], lon0[k], p[:, 0], p[:, 1])))
            t = np.concatenate([p[1] for p in pieces]) / STEP_S
            lo, hi = np.searchsorted(index["minute"], [t.min() - 2, t.max() + 2])
            yield ks, (index["flight"][lo:hi], index["minute"][lo:hi], index["xyz"][lo:hi], pieces)

    flags, pending = {}, {}
    with ProcessPoolExecutor(WORKERS) as pool:
        for d, (ks, args) in enumerate(jobs()):
            pending[pool.submit(_day_job, *args)] = ks
            if len(pending) >= 2 * WORKERS:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    flags.update(zip(pending.pop(f), f.result()))
            if d % 200 == 0:
                print(f"isolation: day {d}/{len(days)}", file=sys.stderr, flush=True)
        for f in wait(pending)[0]:
            flags.update(zip(pending[f], f.result()))
    return flags


def keep_isolated(found, tracks, flights):
    """The matches of `found` whose segment is isolated at every point checked."""
    solo = np.unique(found["solo"])
    flags = solo_flags(tracks, flights, solo, read_index(flights))
    step = route.MATCH_S // GRID_S
    ok = [flags[r.solo][-(-r.i // step):r.j // step + 1].all() for r in found.itertuples()]
    return found[ok].reset_index(drop=True)


def cached(path, make):
    """The table saved in `path`, made and saved first when missing."""
    if not path.exists():
        make().to_parquet(path)
    table = pd.read_parquet(path)
    if "level" not in table.columns:
        raise SystemExit(f"{path} comes from an older version of this module: delete it")
    return table


def count(found, tracks):
    """Per level and for all: pieces with a match, matches, distinct solo flights, median matches
    per piece."""
    out = {}
    for name in (*LEVELS, "all"):
        f = found if name == "all" else found[found["level"] == name]
        out[name] = {"pieces with a match": int(f["piece"].nunique()), "matches": int(len(f)),
                     "solo flights": int(np.unique(tracks["flight"][f["solo"]]).size),
                     "median matches per piece": float(f.groupby("piece").size().median())}
    return out


def phase_stats():
    """name: (source, numerator, denominator, scale) of the phase statistics; the source is "b" for
    the terms of budget.py and "p" for those of performance.py."""
    b, p = budget.STATS, performance.STATS
    sink = decompose.PERFORMANCE[decompose.SINK]
    return {"time in glide (transition) (%)": ("b", *b["time in transition (%)"][:3]),
            "time in search (%)": ("b", "search_s", "s", 100),
            "time in climb (%)": ("b", *b["time in climb (%)"][:3]),
            "searches that end in a climb (%)": ("b", *b["searches that end in a climb (%)"][:3]),
            "climbs per hour": ("b", *b["climbs per hour"][:3]),
            "climb rate (m/s)": ("p", *p["climb rate (m/s)"][:3]),
            "glide speed (km/h)": ("p", *p["glide speed (km/h)"][:3]),
            "glide ratio": ("p", *p["glide ratio"][:3]),
            decompose.SINK: ("p", *sink[:3])}


PHASES = phase_stats()
TERMS = sorted({(src, c) for src, num, den, _ in PHASES.values() for c in (num, den)})


def run_terms(flights):
    """The terms of the runs of the paragliders, by flight and instant of the middle of the run on
    the flight clock: the frames of budget.py ("b") and of performance.py ("p")."""
    numbers = flights.loc[flights["discipline"] == "para", ["flight_id", "flight"]].astype({"flight_id": str})
    runs = pd.read_parquet(RUNS["para"]).astype({"flight_id": str}).merge(numbers, on="flight_id")
    b = budget.columns(runs)
    b = b.assign(t_mid=(b["t_start"] + b["t_end"]) / 2, search_s=b["found_s"] + b["vain_s"])
    p = pd.read_parquet(CACHE / "performance_terms.parquet")
    return {"b": b, "p": p[p["xc_origins"] == 0]}


def window_sums(table, terms, flight, a, b):
    """For each window (flight, a, b; flight clock, s): the sum of `terms` over the rows of `table`
    (flight, t_mid) with t_mid in [a, b): the runs that survival.assign gives to a stretch [a, b)."""
    table = table.sort_values(["flight", "t_mid"], kind="stable")
    key = table["flight"].to_numpy(float) * RANK + table["t_mid"].to_numpy(float)
    total = np.vstack([np.zeros(len(terms)), np.cumsum(table[terms].to_numpy(float), axis=0)])
    lo = np.searchsorted(key, flight * RANK + a, "left")
    hi = np.searchsorted(key, flight * RANK + b, "left")
    return total[hi] - total[lo]


def phase_terms(tables, flight, a, b):
    """The sums of TERMS (columns, in order) over each window (flight, a, b)."""
    out = np.zeros((len(flight), len(TERMS)))
    for src in ("b", "p"):
        cols = [i for i, (s, _) in enumerate(TERMS) if s == src]
        out[:, cols] = window_sums(tables[src], [TERMS[i][1] for i in cols], flight, a, b)
    return out


def draw_counts(n, rng):
    """N_BOOT draws of n items with replacement: how many times each is drawn (n, N_BOOT)."""
    out = np.empty((n, N_BOOT), np.uint8)
    for a in range(0, N_BOOT, CHUNK):
        out[:, a:a + CHUNK] = rng.multinomial(n, np.full(n, 1 / n), size=min(CHUNK, N_BOOT - a)).T
    return out


def bootstrap_weights(pilot, block, rng):
    """Weights of the flights (rows of `pilot` and `block`: their pilot and block codes) for the
    point estimate (column 0, all 1) and for N_BOOT draws of pilots and blocks, independent
    (pigeonhole): c_pilot * c_block. Returns the function (rows, columns) -> weights, and the number
    of columns."""
    cp, cb = draw_counts(pilot.max() + 1, rng), draw_counts(block.max() + 1, rng)

    def of(rows, cols):
        w = np.ones((len(rows), len(cols)))
        d = cols[cols > 0] - 1
        w[:, cols > 0] = cp[pilot[rows]][:, d].astype(float) * cb[block[rows]][:, d]
        return w

    return of, N_BOOT + 1


def year_weights(year, years):
    """Weights of the flights (rows of `year`: their year) that leave out one of `years` in each
    column; same form as bootstrap_weights."""
    return (lambda rows, cols: (year[rows][:, None] != years[cols][None, :]).astype(float)), len(years)


def engine(piece, flight, a, b, rows_piece, rows_flight, weights):
    """The estimate of each column k of `a` (terms of the pieces) and `b` (terms of the matches),
    reweighted to the piece. Match m joins piece `piece[m]` and solo flight `flight[m]` (indices
    into `a` and into `rows_flight`); `rows_*` are the rows of the weights of the pieces' flights
    and the solo flights. With w_g the weight of piece g and w_s that of a control s, for each
    column of `weights` = (of, n):
      group[k]   = sum_g w_g a_gk
      control[k] = sum_g w_g (sum_s w_s b_sk / sum_s w_s)
      total      = sum_g w_g
    over the pieces whose weight and that of their controls are positive. Arrays (k, n), (k, n), (n,)."""
    of, n = weights
    size, order = (len(a), len(rows_flight)), np.lexsort((flight, piece))
    piece, flight, b = piece[order], flight[order], b[order]
    indptr = np.r_[0, np.cumsum(np.bincount(piece, minlength=len(a)))]
    controls = csr_matrix((np.ones(len(piece)), flight, indptr), shape=size)
    terms = [csr_matrix((b[:, k], flight, indptr), shape=size) for k in range(b.shape[1])]
    group, control, total = (np.zeros((len(terms), n)), np.zeros((len(terms), n)), np.zeros(n))
    for c0 in range(0, n, CHUNK):
        cols = np.arange(c0, min(c0 + CHUNK, n))
        wp, wf = of(rows_piece, cols), of(rows_flight, cols)
        mass = controls @ wf
        ok = (wp > 0) & (mass > 0)
        w = np.where(ok, wp, 0.0)
        share = np.divide(w, mass, out=np.zeros_like(w), where=ok)
        group[:, cols], total[cols] = a.T @ w, w.sum(axis=0)
        for k, term in enumerate(terms):
            control[k, cols] = ((term @ wf) * share).sum(axis=0)
    return group, control, total


def msd_of(pos, lags):
    """TA-MSD of the 10-s positions `pos` at `lags` (s), NaN where the lag reaches their length."""
    d = lags // GRID_S
    ok = d < len(pos)
    out = np.full(len(lags), np.nan)
    if ok.any():
        sums, origins = tamsd_sums(pos.astype(float), d[ok])
        out[ok] = sums / origins
    return out


def measure(found, tracks, runs):
    """What the statistics read, for each match of `found` and for each piece of group time that
    has one (`pieces`, sorted): the TA-MSD over LAGS_EXTENDED and the sums of TERMS over the window
    (the piece, or the segment of the solo piece). Keys ending in _g are per piece, in _s per match."""
    n, start, flight = tracks["n"], tracks["start"], tracks["flight"]
    pieces, owner = np.unique(found["piece"], return_inverse=True)
    pos = lambda k, i=0, j=None: tracks["pos"][tracks["offset"][k]:tracks["offset"][k] + n[k]][i:j]
    solo, i, j = (found[c].to_numpy() for c in ("solo", "i", "j"))
    out = {"level": found["level"].to_numpy(), "owner": owner, "flight_g": flight[pieces], "flight_s": flight[solo],
           "piece_s": (n[found["piece"]] - 1) * GRID_S, "segment_s": (j - i) * GRID_S,
           "msd_g": np.array([msd_of(pos(g), LAGS_EXTENDED) for g in pieces]),
           "msd_s": np.array([msd_of(pos(k, a, b + 1), LAGS_EXTENDED) for k, a, b in zip(solo, i, j)]),
           "terms_g": phase_terms(runs, flight[pieces], start[pieces], start[pieces] + (n[pieces] - 1) * GRID_S),
           "terms_s": phase_terms(runs, flight[solo], start[solo] + i * GRID_S, start[solo] + j * GRID_S)}
    seen = TERMS.index(("b", "s"))  # seconds of runs: nothing where the flight has no phases
    out["phases_g"], out["phases_s"] = out["terms_g"][:, seen] > 0, out["terms_s"][:, seen] > 0
    return out


def codes(flights, summary, universe):
    """For each flight of `universe` (flight numbers): the dense codes of its pilot and of its block
    (decompose.flight_table), and its year. A flight with no pilot is a pilot of its own."""
    catalog = pd.read_csv(CATALOG, usecols=["flight_id", "pilot"], dtype=str).set_index("flight_id")["pilot"]
    ids = flights.set_index("flight")["flight_id"].reindex(universe)
    pilot = ids.map(catalog)
    pilot = pilot.where(pilot.notna() & (pilot.str.strip() != ""), "flight " + ids)
    block = decompose.flight_table(flights, summary)[1].reindex(universe).to_numpy()
    year = pd.to_datetime(flights.set_index("flight")["date"].reindex(universe)).dt.year.to_numpy()
    return np.unique(pilot.to_numpy(str), return_inverse=True)[1], np.unique(block, return_inverse=True)[1], year


def long_enough(data, tau):
    """The matches whose piece and segment both last at least tau / SUPPORT s."""
    return (data["piece_s"] >= tau / SUPPORT) & (data["segment_s"] >= tau / SUPPORT)


def long_tau(data):
    """The largest of LONG_CANDIDATES that leaves at least MIN_PIECES matched pieces in every
    level when only the pieces and segments that last tau / SUPPORT s are kept; None when none."""
    best = None
    for tau in LONG_CANDIDATES:
        left = {name: int(np.unique(data["owner"][long_enough(data, tau) & (data["level"] == name)]).size)
                for name in LEVELS}
        print(f"tau_max {tau} s: matched pieces left {left}", file=sys.stderr, flush=True)
        if min(left.values()) >= MIN_PIECES:
            best = tau
    return best


def sums_by_level(data, mask, a, b, cols, weights):
    """engine() per level on the matches of `mask`, for the columns `cols` of `a` (terms of the
    pieces) and `b` (terms of the matches); "all" adds the levels, whose pieces are separate."""
    out = {}
    for name in LEVELS:
        m = mask & (data["level"] == name)
        if not m.any():
            out[name] = (np.zeros((len(cols), weights[1])), np.zeros((len(cols), weights[1])), np.zeros(weights[1]))
            continue
        piece, p = np.unique(data["owner"][m], return_inverse=True)
        solo, s = np.unique(data["row_s"][m], return_inverse=True)
        out[name] = engine(p, s, a[piece][:, cols], b[m][:, cols], data["row_g"][piece], solo, weights)
    out["all"] = tuple(sum(x) for x in zip(*out.values()))
    return out


def curve(msd, lags):
    """zones.curve of the rows of `msd` (draws x lags), NaN for the rows that are not all finite."""
    ok = np.isfinite(msd).all(axis=1)
    h, f = np.full(msd.shape, np.nan), np.full(len(msd), np.nan)
    if ok.any():
        h[ok], f[ok] = zones.curve(msd[ok], lags)
    return h, f


def sides(sums, lags):
    """MSD, H_loc and H of the group and of the control, and their difference, for each column
    (draw or year) of `sums` = (group, control, total) of engine()."""
    group, control, total = sums
    with np.errstate(invalid="ignore", divide="ignore"):
        mg, mc = (group / total).T, (control / total).T
    (hg, fg), (hc, fc) = curve(mg, lags), curve(mc, lags)
    return {"group": {"msd": mg, "H_loc": hg, "H": fg}, "control": {"msd": mc, "H_loc": hc, "H": fc},
            "delta": {"H_loc": hg - hc, "H": fg - fc}}


def quote(boot, years=None):
    """The point estimate (first row of `boot`), the 5-95% band of the other rows and, when given,
    the minimum and maximum over the rows of `years`."""
    out = {"value": np.asarray(boot[0]).tolist(), "band": np.nanpercentile(boot[1:], [5, 95], axis=0).tolist()}
    if years is not None:
        out["years"] = [np.nanmin(years, axis=0).tolist(), np.nanmax(years, axis=0).tolist()]
    return out


def msd_report(boot, year, lags):
    """Group, control and difference over `lags`: MSD, H_loc and H with the band of the draws and
    the range over the years, and H_loc of the difference at AT and at the last lag."""
    b, y = sides(boot, lags), sides(year, lags)
    out = {"lags": lags.tolist()}
    for side in ("group", "control", "delta"):
        out[side] = {k: quote(b[side][k], y[side][k]) for k in ("H_loc", "H")}
    for side in ("group", "control"):
        out[side]["msd"] = quote(b[side]["msd"])
    at = {f"H_loc({x} s)": int(np.searchsorted(lags, x)) for x in sorted({*AT, int(lags[-1])}) if x in lags}
    out["delta"]["at"] = {k: quote(b["delta"]["H_loc"][:, i], y["delta"]["H_loc"][:, i]) for k, i in at.items()}
    return out


def phase_report(boot, year):
    """For each statistic of PHASES: group, control and difference, with the band of the draws and
    the range over the years. Ratios of the sums, as in budget.py and performance.py."""
    def ratio(sums, spec):
        (group, control, _), (src, num, den, scale) = sums, spec
        i, j = TERMS.index((src, num)), TERMS.index((src, den))
        with np.errstate(invalid="ignore", divide="ignore"):
            return scale * group[i] / group[j], scale * control[i] / control[j]

    out = {}
    for name, spec in PHASES.items():
        (gb, cb), (gy, cy) = ratio(boot, spec), ratio(year, spec)
        out[name] = {"group": quote(gb, gy), "control": quote(cb, cy), "delta": quote(gb - cb, gy - cy)}
    return out


def level_counts(data, m, pilot, tau):
    """Counts for the matches of the mask `m`: pieces, matches, distinct solo flights and pilots of
    each side, median matches per piece, what has phases, the pieces of the long series."""
    owner = data["owner"][m]
    pieces, per_piece = np.unique(owner, return_counts=True)
    both = data["phases_s"][m] & data["phases_g"][owner]
    out = {"pieces with a match": int(pieces.size), "matches": int(m.sum()),
           "solo flights": int(np.unique(data["flight_s"][m]).size),
           "pilots, group side": int(np.unique(pilot[data["row_g"][pieces]]).size),
           "pilots, control side": int(np.unique(pilot[data["row_s"][m]]).size),
           "median matches per piece": float(np.median(per_piece)),
           "pieces with phases": int(data["phases_g"][pieces].sum()),
           "segments with phases": int(data["phases_s"][m].sum()),
           "pieces in the phase statistics": int(np.unique(owner[both]).size)}
    out["pieces of the long series"] = int(np.unique(owner[long_enough(data, tau)[m]]).size) if tau else 0
    return out


def analyse(found, tracks, flights, summary, runs):
    """Per level and for all, the statistics of the matches of `found`: counts, the main series (lags
    of tamsd.LAGS, every match), the long series (the matches that last long enough, over longer
    lags) and the phases. Bands: the same N_BOOT draws of pilots and blocks for everything; the
    range over the years leaves out one year at a time, pieces and controls."""
    data = measure(found, tracks, runs)
    universe = np.unique(np.r_[data["flight_g"], data["flight_s"]])
    pilot, block, year = codes(flights, summary, universe)
    data["row_g"], data["row_s"] = (np.searchsorted(universe, data[k]) for k in ("flight_g", "flight_s"))
    boot = bootstrap_weights(pilot, block, np.random.default_rng(SEED))
    years = np.unique(year)
    leaving = year_weights(year, years)
    every = np.ones(len(found), bool)
    both = lambda mask, a, b, cols: (sums_by_level(data, mask, a, b, cols, boot), sums_by_level(data, mask, a, b, cols, leaving))
    series, lags = {}, {"main": LAGS}
    series["main"] = both(every, data["msd_g"], data["msd_s"], np.flatnonzero(LAGS_EXTENDED <= TAU_MAX))
    tau = long_tau(data)
    if tau:
        lags["long"] = LAGS_EXTENDED[LAGS_EXTENDED <= tau]
        series["long"] = both(long_enough(data, tau), data["msd_g"], data["msd_s"], np.arange(len(lags["long"])))
    phased = data["phases_s"] & data["phases_g"][data["owner"]]
    phases = both(phased, data["terms_g"], data["terms_s"], np.arange(len(TERMS)))
    result = {"long_tau_max": tau, "years": years.tolist(), "levels": {}}
    for name in (*LEVELS, "all"):
        m = every if name == "all" else data["level"] == name
        result["levels"][name] = {
            "counts": level_counts(data, m, pilot, tau),
            "msd": {k: msd_report(s[0][name], s[1][name], lags[k]) for k, s in series.items()},
            "phases": phase_report(phases[0][name], phases[1][name])}
    return result


def explained(result, found, tracks, flights, summary, level):
    """Per level, the share of the group - solo difference at 1000 s that the route explains: the
    formula of \\RouteExplained (report/tables.py), 1 - Delta H_loc(1000 s) over its value for the
    same pieces against the solo flights of their site-day and hour (route.baseline), within the level."""
    info, block, n_blocks = decompose.flight_table(flights, summary)
    counts = np.ones((2, n_blocks), np.int32)  # the point estimate only
    out = {}
    for name in (*LEVELS, "all"):
        f = found if name == "all" else found[found["level"] == name]
        solo = level.index if name == "all" else level.index[level == name]
        base = route.baseline(f, tracks, info, block, counts, solo)["H_loc(1000 s)"]["value"]
        delta = result["levels"][name]["msd"]["main"]["delta"]["at"]["H_loc(1000 s)"]["value"]
        out[name] = {"pieces": int(f["piece"].nunique()), "delta baseline": base, "delta isolated": delta,
                     "share explained (%)": 100 * (1 - delta / base)}
    return out


LEVEL_COLORS = {"beginners": "#1baf7a", "experienced": "#c24a1d", "all": "#555555"}
DRAWN = ("H_loc(100 s)", "H_loc(1000 s)", *decompose.DRAWN[2:], "time in search (%)")  # decompose.DRAWN plus the search


def _delta(level, stat):
    """The quote of Delta `stat` (an H_loc at a lag, or a phase statistic) of a level of the result."""
    return level["msd"]["main"]["delta"]["at"][stat] if stat.startswith("H_loc") else level["phases"][stat]["delta"]


def figure_dhloc(result, path):
    """Delta H_loc(tau), group minus control, one curve per level with its 5-95% band: the main
    series solid, the long series dashed, the range over the years as thin lines."""
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for name, color in LEVEL_COLORS.items():
        for kind, ls in (("main", "-"), ("long", "--")):
            r = result["levels"][name]["msd"].get(kind)
            if r is None:
                continue
            lags, d = np.array(r["lags"]), r["delta"]["H_loc"]
            ax.fill_between(lags, *d["band"], color=color, alpha=0.15, lw=0)
            for edge in d["years"]:
                ax.plot(lags, edge, color=color, lw=0.6, alpha=0.6, ls=ls)
            ax.plot(lags, d["value"], color=color, lw=1.6, ls=ls, label=name if kind == "main" else None)
    ax.axhline(0, color="#9a9893", lw=0.8)
    ax.set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$\Delta H_{\rm loc}(\tau)$, group $-$ control")
    ax.legend(frameon=False, fontsize=8)
    save(fig, path)


def figure(result, path):
    """Left: H_loc(tau) of the group and of the control, one panel per level, the long series dashed.
    Right: the Delta of the statistics of DRAWN, one panel each, one row per level, with the 5-95%
    band (thick) and the range over the years (thin)."""
    fig = plt.figure(figsize=(14, 6.2), constrained_layout=True)
    sub_left, sub_right = fig.subfigures(1, 2, width_ratios=[1, 2.6])
    axes = sub_left.subplots(len(LEVEL_COLORS), 1, sharex=True, sharey=True)
    for ax, name in zip(axes, LEVEL_COLORS):
        for pop, color, text in (("group", COLORS["group"], "group time"), ("control", COLORS["solo"], "same route, alone")):
            for kind, ls in (("main", "-"), ("long", "--")):
                r = result["levels"][name]["msd"].get(kind)
                if r is None:
                    continue
                h, lags = r[pop]["H_loc"], np.array(r["lags"])
                ax.fill_between(lags, *h["band"], color=color, alpha=0.15, lw=0)
                ax.plot(lags, h["value"], color=color, lw=1.4, ls=ls, label=text if kind == "main" else None)
        ax.set(xscale="log", title=f"{name}: {result['levels'][name]['counts']['pieces with a match']} pieces")
        ax.title.set_fontsize(9)
        ax.set_ylabel(r"$H_{\rm loc}(\tau)$")
    axes[0].legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel(r"$\tau$ (s)")
    grid = sub_right.subplots(2, 4, squeeze=False)
    for ax, stat in zip(grid.flat, DRAWN):
        for y, name in enumerate(LEVEL_COLORS):
            d = _delta(result["levels"][name], stat)
            (lo, hi), (y0, y1) = d["band"], d["years"]
            ax.plot([y0, y1], [y, y], color=LEVEL_COLORS[name], lw=0.8, alpha=0.7)
            ax.errorbar(d["value"], y, xerr=[[d["value"] - lo], [hi - d["value"]]], fmt="o", ms=5, color=LEVEL_COLORS[name],
                        elinewidth=1.8, capsize=2)
        ax.axvline(0, color="#999999", lw=0.8, ls=":")
        ax.set_ylim(len(LEVEL_COLORS) - 0.5, -0.5)
        ax.set_yticks(range(len(LEVEL_COLORS)), list(LEVEL_COLORS) if ax in grid[:, 0] else [])
        ax.set_title("Δ " + stat.replace("(%)", "(points)"), fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="x", color="#e6e6e6", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    sample = route.option("--sample", 0) or None
    flights = with_takeoffs(pd.read_parquet(LABELS / "flights.parquet"))
    assert route.TRACKS.exists(), "run route.py first"
    tracks = route.read_tracks(flights, None)
    tag = "" if sample is None else f"_sample{sample}"
    candidates = cached(CACHE / f"isolated_candidates{tag}.parquet", lambda: matches(tracks, flights, levels(flights), sample))
    found = cached(CACHE / f"isolated_matches{tag}.parquet", lambda: keep_isolated(candidates, tracks, flights))
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    result = analyse(found, tracks, flights, summary, run_terms(flights))
    result["share explained"] = explained(result, found, tracks, flights, summary, levels(flights))
    result["matches before isolation"] = count(candidates, tracks)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"isolated{tag}.json").write_text(json.dumps(result, indent=1))
    figure(result, OUT / f"isolated{tag}.pdf")
    figure_dhloc(result, OUT / f"isolated_dhloc{tag}.pdf")
    print(json.dumps({"long_tau_max": result["long_tau_max"], "share explained": result["share explained"],
                      **{k: v["counts"] for k, v in result["levels"].items()}}, indent=1))
