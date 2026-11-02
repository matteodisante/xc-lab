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

    uv run python matched.py --open [--sample N] [--rho M]    # after tamsd.py and route.py (or on its own for the tracks)

The statistics (--stats, on the chosen caliper, cover MAIN_COVER, isolation off, per level of the query piece):
G against its controls (S|G) and A, the solo time of the same flights, against theirs (S|A), and the
difference of the two Delta. Unit = a piece or a control segment. Each side is a mean over units reweighted to
the piece (isolated.engine: weight w_g of the piece, per piece the mean over its controls), with the
logarithm, the slopes and the ratios taken after the means:

    MSD      dataset `1000` (lags to 1000 s, pieces and segments of at least 1250 s) and the long series
             (lags to the largest of LONG_LAGS up to 2760 s that keeps MIN_PIECES pieces in the level and
             side, pieces and segments of at least T / SUPPORT); H_loc(1000 s), H_loc(1840 s), fitted H;
    phases   the seven statistics of raw.STATISTICS on the pieces and segments of dataset `1000` that have
             phase time: r = n / T and q = d / T per unit, the control side the mean of r and of q over the
             controls of a piece, stat = sum_g w_g rbar_g / sum_g w_g qbar_g;
    survival of climbs, searches that end in a climb and glides (level "all"): the runs of a piece weigh 1,
             those of a control 1 / (controls of its piece), and the bootstrap weights on top.

Bands: twoway.py (every flight, group or solo, weighs the count of its pilot, block or cell, so a piece
enters a draw with the controls drawn with it), the same draws for every number; for every Delta also the
minimum and maximum over the years leaving one year out.

    uv run python matched.py --open --stats [--sample N]    # after the match tables (--rho 300 and 1000 for the robustness)
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import budget
import curves
import isolated
import longlags
import raw
import route
import survival
import twoway
from blocks import with_takeoffs
from phase_msd import PHASES
from tamsd import CACHE, COLORS, GRID_S, LABELS, N_BOOT, OUT, SEED, SUPPORT, read_flights, save
from isolated import MIN_PIECES, WORKERS, levels, sphere, wing_class

MIN_COVER = 0.7  # the lowest cover of the robustness; the table keeps every match from here
MAIN_COVER, COVERS = 0.8, (0.7, 0.8, 0.9)
DATASETS = {"1000": 1000.0, "2760": 2760.0}  # name: lag T; pieces and segments last at least T / SUPPORT
LEVEL_NAMES = ("beginners", "experienced", "all")
# name: (month within, hour within, same wing "class" or "level"); the order is that of the choice
CALIPERS = {"month1_hour1_class": (1, 1, "class"), "month1_hour1_level": (1, 1, "level"),
            "month2_hour2_class": (2, 2, "class"), "month2_hour2_level": (2, 2, "level")}
COLUMNS = ["side", "piece", "solo", "i", "j", "cover"]
RHO_TAG = "" if route.RHO == route.RHO_DEFAULT else f"_rho{route.RHO:g}"  # the tables of other rho keep their own names


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
    for pieces and segments of the dataset (a key of DATASETS, or a lag T in s); with `isolation`, only isolated segments."""
    month, hour, by = CALIPERS[caliper]
    shortest = (DATASETS[dataset] if isinstance(dataset, str) else dataset) / SUPPORT
    same = table["level_piece"].notna() & (table[f"{by}_piece"] == table[f"{by}_solo"])  # classes of isolated.LEVELS only
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
    mid_g = utc0[g] + tracks["start"][g] + (tracks["n"][g] - 1) * GRID_S / 2
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
    pieces = {c: {lv: grid["G"]["2760"][c][lv]["pieces"] for lv in LEVEL_NAMES} for c in CALIPERS}
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
                               "meets min pieces": meets, "G pieces on dataset 2760": pieces[name]},
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


# ---- the statistics (--stats)

LONG_T = 2760  # s: the longest lag of the long series
LAGS = longlags.LONG_LAGS[longlags.LONG_LAGS <= LONG_T]
QUOTED = (1000, 1840)  # s: the lags of the quoted H_loc, on the main and on the long series
NAMES = list(raw.STATISTICS)
SIDES = ("G", "S|G", "A", "S|A")
CONTRASTS = ("G - S|G", "A - S|A", "(G - S|G) - (A - S|A)")
SIDE_STYLE = {"G": (COLORS["group"], "-"), "S|G": (COLORS["solo"], "-"), "A": (COLORS["group_solo"], "-"), "S|A": ("#16427a", "--")}
CONTRAST_COLORS = dict(zip(CONTRASTS, (COLORS["group"], COLORS["group_solo"], "#555555")))
METRICS = ("H_loc(1000 s)", "H_loc(1840 s)", "H (10 s to 1000 s)", "H (10 s to T)", *NAMES)
FOREST = METRICS[:2] + METRICS[4:]  # drawn: the two H_loc and the seven statistics
SURVIVAL_GRID = survival.GRID
LINES = {"main": {}, "caliper month1_hour1_level": dict(caliper="month1_hour1_level"),
         "caliper month2_hour2_class": dict(caliper="month2_hour2_class"),
         "caliper month2_hour2_level": dict(caliper="month2_hour2_level"), "isolation on": dict(isolation=True),
         "cover 0.7": dict(cover=0.7), "cover 0.9": dict(cover=0.9), "rho 300 m": dict(rho=300), "rho 1000 m": dict(rho=1000)}


class Context:
    """What every number shares: the tracks, the codes and weights of the flights (one universe, so that the
    draws are the same for every table), the terms of the runs and the counted runs sorted by middle."""

    def __init__(self, tracks, flights, summary):
        self.tracks = tracks
        self.universe = np.unique(tracks["flight"])
        pilot, block, year = isolated.codes(flights, summary, self.universe)
        self.pilot, self.years = pilot, np.unique(year)
        cell, of_pilot, of_block = twoway.cell_codes(pilot, block)
        self.boot = twoway.weights(cell, twoway.counts(of_pilot, of_block, np.random.default_rng(SEED)))
        self.leaving = isolated.year_weights(year, self.years)
        self.tables, self.defs = isolated.run_terms(flights), raw.stat_defs()
        self.scale = np.array([next(s[n][2] for s in self.defs.values() if n in s) for n in NAMES], float)
        b = self.tables["b"]
        counted = b[b["phase"].ne("search") | b["found"].eq(1)]
        counted = counted.assign(key=counted["flight"].to_numpy(float) * isolated.RANK + counted["t_mid"].to_numpy(float))
        self.runs = counted.sort_values("key", kind="stable").reset_index(drop=True)

    def row(self, flight):
        """Row of the universe (codes, weights) of flight numbers."""
        return np.searchsorted(self.universe, flight)

    def rates(self, flight, a, b):
        """For windows (flight, a, b on the flight clock): the phase time T_u and the columns (r, q) of the NAMES
        (budget.unit_rates), 0 where T_u = 0."""
        T = isolated.window_sums(self.tables["b"], ["s"], flight, a, b)[:, 0]
        r, q = np.zeros((len(T), len(NAMES))), np.zeros((len(T), len(NAMES)))
        for src, stats in self.defs.items():
            cols = sorted({c for num, den, *_ in stats.values() for c in (num, den)})
            sums = pd.DataFrame(isolated.window_sums(self.tables[src], cols, flight, a, b), columns=cols)
            rr, qq = budget.unit_rates(sums, np.arange(len(T)), pd.Series(T), stats)
            for name in stats:
                r[:, NAMES.index(name)], q[:, NAMES.index(name)] = rr[name], qq[name]
        ok = (T > 0)[:, None]
        return T, np.hstack([np.where(ok, r, 0.0), np.where(ok, q, 0.0)])


class Measured:
    """The MSD at LAGS and the rates of the pieces and of the control segments of the rows `wanted` of a match
    table (the row labels of the table index them)."""

    def __init__(self, ctx, wanted):
        tr = ctx.tracks
        self.rows, self.piece = wanted.index.to_numpy(), np.unique(wanted["piece"].to_numpy())
        pos = lambda k, i=0, j=None: tr["pos"][tr["offset"][k]:tr["offset"][k] + tr["n"][k]][i:j]
        s, i, j = (wanted[c].to_numpy() for c in ("solo", "i", "j"))
        g = self.piece
        self.msd_g = np.array([isolated.msd_of(pos(k), LAGS) for k in g])
        self.msd_s = np.array([isolated.msd_of(pos(k, a, b + 1), LAGS) for k, a, b in zip(s, i, j)])
        self.win_g = (tr["flight"][g], tr["start"][g], tr["start"][g] + (tr["n"][g] - 1) * GRID_S)
        self.win_s = (tr["flight"][s], tr["start"][s] + i * GRID_S, tr["start"][s] + j * GRID_S)
        (self.T_g, self.rq_g), (self.T_s, self.rq_s) = ctx.rates(*self.win_g), ctx.rates(*self.win_s)

    def at(self, sel):
        """Positions of the pieces of `sel` (sorted unique) and of its rows in the arrays."""
        pieces = np.unique(sel["piece"].to_numpy())
        return pieces, np.searchsorted(self.piece, pieces), np.searchsorted(self.rows, sel.index.to_numpy())


def of_level(sel, level):
    """The rows of `sel` whose query piece is of `level` ("all": every row)."""
    return sel if level == "all" else sel[sel["level_piece"] == level]


def long_lag(table, line, level):
    """The largest lag T of LONG_LAGS from 1000 s up to LONG_T for which the pieces of `level` with a control
    (both of at least T / SUPPORT s) number at least MIN_PIECES; None when there is none."""
    for T in LAGS[::-1]:
        if T >= 1000 and of_level(select(table, int(T), **line), level)["piece"].nunique() >= MIN_PIECES:
            return int(T)
    return None


def common_long_lag(table, line, level):
    """The smaller long_lag of the sides G and A of `table`, so that G - S|G and A - S|A share their lags; None
    when either has none."""
    lags = [long_lag(table[table["side"] == side], line, level) for side in ("G", "A")]
    return None if None in lags else min(lags)


def run_engine(ctx, sel, a, b, years):
    """isolated.engine on the rows `sel` (one per piece and pool flight), with `a` (one row per unique piece of
    `sel`) and `b` (one per row of `sel`): the sums for the bootstrap draws and, with `years`, for the years."""
    pieces, p = np.unique(sel["piece"].to_numpy(), return_inverse=True)
    flights, s = np.unique(ctx.tracks["flight"][sel["solo"].to_numpy()], return_inverse=True)
    args = (p, s, a, b, ctx.row(ctx.tracks["flight"][pieces]), ctx.row(flights))
    return isolated.engine(*args, ctx.boot), (isolated.engine(*args, ctx.leaving) if years else None)


def msd_series(ctx, M, sel, T, years):
    """Group and control MSD, H_loc and H over the lags up to T (isolated.sides) for the draws and the years."""
    lags = LAGS[LAGS <= T]
    _, pos, rows = M.at(sel)
    boot, year = run_engine(ctx, sel, M.msd_g[pos][:, :len(lags)], M.msd_s[rows][:, :len(lags)], years)
    return {"lags": lags, "T": T, "pieces": int(len(pos)), "controls": int(len(sel)), "boot": isolated.sides(boot, lags),
            "year": isolated.sides(year, lags) if years else None}


def phased(M, sel):
    """The rows of `sel` with phase time on both sides."""
    rows = np.searchsorted(M.rows, sel.index.to_numpy())
    return sel[(M.T_g[np.searchsorted(M.piece, sel["piece"].to_numpy())] > 0) & (M.T_s[rows] > 0)]


def phase_series(ctx, M, sel, years):
    """The NAMES of the pieces and of the controls of `sel` (rows with phase time on both sides): estimator
    stat = scale * sum_g w_g rbar_g / sum_g w_g qbar_g. Arrays (draws, statistics)."""
    sel = phased(M, sel)
    _, pos, rows = M.at(sel)
    boot, year = run_engine(ctx, sel, M.rq_g[pos], M.rq_s[rows], years)
    k = len(NAMES)

    def ratios(sums):
        if sums is None:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            return {"group": (ctx.scale[:, None] * sums[0][:k] / sums[0][k:]).T,
                    "control": (ctx.scale[:, None] * sums[1][:k] / sums[1][k:]).T}

    return {"pieces": int(len(pos)), "controls": int(len(sel)), "boot": ratios(boot), "year": ratios(year)}


def window_runs(ctx, flight, a, b):
    """The counted runs whose middle lies in each window (flight, a, b): the window of each run and its row in
    ctx.runs."""
    key = ctx.runs["key"].to_numpy()
    lo = np.searchsorted(key, flight * isolated.RANK + a, "left")
    n = np.searchsorted(key, flight * isolated.RANK + b, "left") - lo
    return np.repeat(np.arange(len(n)), n), np.repeat(lo - np.r_[0, np.cumsum(n)[:-1]], n) + np.arange(n.sum())


def duration_counts(unit, x, n_units, grid=SURVIVAL_GRID):
    """Per unit, the number of the durations `x` above each point of `grid`, then their total: (n_units, grid + 1)."""
    k = np.searchsorted(grid, x, "left")
    m = np.zeros((n_units, len(grid) + 1))
    np.add.at(m, (unit, k), 1.0)
    above = np.cumsum(m[:, ::-1], axis=1)[:, ::-1]
    return np.column_stack([above[:, 1:], above[:, 0]])


def phase_counts(ctx, win, phases=PHASES):
    """The duration_counts of each phase (trimmed as durations.trim) for the windows `win`, side by side."""
    unit, idx = window_runs(ctx, *win)
    x, phase = ctx.runs["n_fixes"].to_numpy(float)[idx], ctx.runs["phase"].to_numpy()[idx]
    out = []
    for p in phases:
        m = phase == p
        keep = np.isin(x[m], raw.trim(x[m]))
        out.append(duration_counts(unit[m][keep], x[m][keep], len(win[0])))
    return np.hstack(out)


def survival_series(ctx, M, sel):
    """P(duration > t) of the runs of the pieces of `sel` (rows with phase time on both sides) and of their
    controls, each run of a piece weighing 1 and each run of a control 1 / (controls of its piece), by phase,
    with the pointwise band of the draws; and the runs counted."""
    sel = phased(M, sel)
    _, pos, rows = M.at(sel)
    a = phase_counts(ctx, tuple(w[pos] for w in M.win_g))
    b = phase_counts(ctx, tuple(w[rows] for w in M.win_s))
    (group, control, _), _ = run_engine(ctx, sel, a, b, False)
    width, out = len(SURVIVAL_GRID) + 1, {}
    for k, phase in enumerate(PHASES):
        out[phase] = {}
        for side, sums, units in (("G", group, a), ("S|G", control, b)):
            s = sums[k * width:(k + 1) * width]
            with np.errstate(invalid="ignore", divide="ignore"):
                curve = s[:-1] / s[-1]
            out[phase][side] = {"grid": SURVIVAL_GRID.tolist(), "value": curve[:, 0].tolist(),
                                "band": twoway.share_band(curve[:, 0], curve[:, 1:].T),
                                "runs": int(units[:, k * width + width - 1].sum())}
    return out


def estimate(ctx, M, table, side, line, levels=LEVEL_NAMES, years=True, survival_too=False):
    """Per level, for the pieces of `side` and their controls under `line` (caliper, cover, isolation): the main
    series on dataset `1000`, the long series, the phase statistics, the counts (and the survival)."""
    t, out = table[table["side"] == side], {}
    for level in levels:
        main = of_level(select(t, "1000", **line), level)
        T = common_long_lag(table, line, level)
        e = {"main": msd_series(ctx, M, main, 1000, years) if len(main) else None, "long_T": T,
             "long": msd_series(ctx, M, of_level(select(t, T, **line), level), T, years) if T else None,
             "phases": phase_series(ctx, M, main, years) if len(main) else None}
        used = phased(M, main) if len(main) else main
        e["counts"] = counts_of(ctx, main, used, e["long"], side)
        if survival_too and len(main):
            e["survival"] = survival_series(ctx, M, main)
        out[level] = e
    return out


def counts_of(ctx, main, used, long, side):
    """Pieces, controls, distinct pool flights and pilots of each side, on the main series, those with phase time
    (`used`) and on the long series."""
    tr = ctx.tracks
    flight = tr["flight"][np.unique(main["piece"].to_numpy())]
    return {"pieces": int(main["piece"].nunique()), "controls": int(len(main)),
            "pool flights": int(np.unique(tr["flight"][main["solo"].to_numpy()]).size),
            f"flights, {side} side": int(np.unique(flight).size),
            f"pilots, {side} side": int(np.unique(ctx.pilot[ctx.row(flight)]).size),
            "pilots, control side": int(np.unique(ctx.pilot[ctx.row(tr["flight"][main["solo"].to_numpy()])]).size),
            "pieces with phases": int(used["piece"].nunique()), "controls with phases": int(len(used)),
            "pieces of the long series": long["pieces"] if long else 0}


def draws(e):
    """The scalar METRICS of one side at one level of `estimate`: {metric: {"group"/"control": (draws, years)}},
    the draws with the point estimate first, years None when not computed; metrics that do not exist are left out."""
    out = {}
    for ser, h_name, f_name, lag in ((e["main"], METRICS[0], METRICS[2], QUOTED[0]), (e["long"], METRICS[1], METRICS[3], QUOTED[1])):
        if ser is None:
            continue
        lags = list(ser["lags"])
        pick = lambda field, idx=None: {w: tuple(None if r is None else (r[w][field] if idx is None else r[w][field][:, idx])
                                                 for r in (ser["boot"], ser["year"])) for w in ("group", "control")}
        if lag in lags:
            out[h_name] = pick("H_loc", lags.index(lag))
        out[f_name] = pick("H")
    if e["phases"] is not None:
        for j, name in enumerate(NAMES):
            out[name] = {w: tuple(None if r is None else r[w][:, j] for r in (e["phases"]["boot"], e["phases"]["year"]))
                         for w in ("group", "control")}
    return out


def quoted(pair):
    """twoway.quote of a (draws, years) pair (no years when there are none)."""
    return twoway.quote(*pair)


def contrasts(dg, da=None):
    """The Delta of `draws` of a side (group - control) and, when `da` (the other side) is given, the other Delta
    and their difference, per metric: {contrast: {metric: (draws, years)}}."""
    delta = lambda d, m: tuple(None if x is None else x - y for x, y in zip(d[m]["group"], d[m]["control"]))
    out = {CONTRASTS[0]: {m: delta(dg, m) for m in dg}}
    if da is not None:
        out[CONTRASTS[1]] = {m: delta(da, m) for m in da}
        out[CONTRASTS[2]] = {m: tuple(None if x is None else x - y for x, y in zip(out[CONTRASTS[0]][m], out[CONTRASTS[1]][m]))
                             for m in dg if m in da}
    return out


def curve_of(ser, which):
    """MSD and H_loc with their bands, and the fitted H with its band, of the group or the control of an msd_series."""
    b = ser["boot"][which]
    band = lambda a: twoway.band(a[0], a[1:])
    return {"lags": ser["lags"].tolist(), "T": ser["T"], "pieces": ser["pieces"], "controls": ser["controls"], "msd": b["msd"][0].tolist(),
            "msd_band": band(b["msd"]), "H_loc": b["H_loc"][0].tolist(), "H_loc_band": band(b["H_loc"]),
            "H": float(b["H"][0]), "H_band": band(b["H"])}


def delta_curve(ser):
    """Delta H_loc (group - control) at every lag of an msd_series, with its band."""
    return {"lags": ser["lags"].tolist(), **twoway.quote(ser["boot"]["delta"]["H_loc"])}


def draw_msd(est, folder, tag):
    """The MSD and the H_loc figure of G, S|G, A and S|A (level "all") of both series, fitted H in the legend."""
    for key, name in (("main", "1000"), ("long", "2760")):
        drawn = []
        for side, label, which in (("G", "G", "group"), ("G", "S|G", "control"), ("A", "A", "group"), ("A", "S|A", "control")):
            ser = est[side]["all"][key]
            if ser is None:
                continue
            c = curve_of(ser, which)
            lo, hi = c["H_band"]
            n = f"{c['pieces']:,} pieces" if which == "group" else f"{c['controls']:,} controls"
            drawn.append(curves.series(c, f"{label} ({n}): H = {c['H']:.3f} [{lo:.3f}, {hi:.3f}]", *SIDE_STYLE[label]))
        if drawn:
            T = max(c["T"] for c in drawn)
            title = f"matched, lags to {T:,} s, pieces of at least {T / SUPPORT:g} s"
            curves.figures(drawn, folder / f"matched{tag}_{name}", title)
            right = [{**delta_curve(est[side]["all"][key]), "label": c, "color": CONTRAST_COLORS[c]}
                     for side, c in (("G", CONTRASTS[0]), ("A", CONTRASTS[1])) if est[side]["all"][key]]
            curves.delta_figure([{**c, "label": c["label"].split(" (")[0]} for c in drawn], right,
                                folder / f"matched{tag}_{name}_delta", title)


def draw_survival(surv, path):
    """P(duration > t) of G, S|G, A and S|A, one panel per phase (log-log), the 5-95% band, no fit."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for ax, phase in zip(axes, PHASES):
        for side, label in (("G", "G"), ("G", "S|G"), ("A", "A"), ("A", "S|A")):
            s = surv[side][phase]["G" if label in ("G", "A") else "S|G"]
            t, v, band = np.array(s["grid"]), np.array(s["value"]), np.array(s["band"])
            ok = v > 0
            color, ls = SIDE_STYLE[label]
            ax.fill_between(t[ok], np.maximum(band[0][ok], 1e-9), band[1][ok], color=color, alpha=0.15, lw=0)
            ax.plot(t[ok], v[ok], color=color, ls=ls, lw=1.3, label=f"{label} ({s['runs']:,} runs)")
        ax.set(xscale="log", yscale="log", title=raw.SURVIVAL_PHASES[phase], xlabel=r"$\tau$ (s)",
               ylabel=rf"$P(\tau_{{\mathrm{{{phase}}}}} > \tau)$", ylim=(1e-5, 1.2))
        ax.legend(frameon=False, fontsize=7, loc="lower left")
        ax.grid(color="#eeeeee", lw=0.5)
    save(fig, path)


def draw_forest(rows, path, title, stats=FOREST, per_row=3, width=None):
    """One panel per statistic of `stats`, one row per entry of `rows` = [(label, colour, {stat: quote}) or
    None for a gap]: a dot with the 5-95% band (thick) and, when the quote has them, the range over the years (thin)."""
    y, ys = 0.0, []
    for r in rows:
        ys.append(y)
        y += 0.5 if r is None else 1.0
    n = -(-len(stats) // per_row)
    fig, axes = plt.subplots(n, per_row, figsize=(width or 3.0 * per_row + 2.6, n * (0.2 * y + 1.2)), squeeze=False)
    for ax in axes.flat[len(stats):]:
        ax.axis("off")
    for ax, stat in zip(axes.flat, stats):
        for r, y0 in zip(rows, ys):
            if r is None or r[2].get(stat) is None:
                continue
            v = r[2][stat]
            lo, hi = v["band"]
            if "years" in v:
                ax.plot(v["years"], [y0, y0], color=r[1], lw=0.8, alpha=0.8, marker="|", ms=7)
            ax.errorbar(v["value"], y0, xerr=[[v["value"] - lo], [hi - v["value"]]], fmt="o", ms=5, color=r[1], elinewidth=1.8, capsize=2)
        ax.axvline(0, color="#999999", lw=0.8, ls=":")
        ax.set_ylim(y - 0.4, -0.6)
        ax.set_yticks([y0 for r, y0 in zip(rows, ys) if r], [r[0] for r in rows if r] if ax in axes[:, 0] else [])
        ax.set_title("Δ " + stat.replace("(%)", "(points)"), fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", color="#e6e6e6", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"{title} (5-95% bands)", fontsize=10)
    save(fig, path)


def robustness(ctx, table, tables_rho, chosen):
    """Delta(G - S|G) of H_loc(1000 s), H_loc(1840 s) and the seven statistics at level "all" for each line of
    LINES: the point estimate and the band of the draws. The tables of other rho are those of `tables_rho`."""
    out = {}
    for name, change in LINES.items():
        change = dict(change)
        rho = change.pop("rho", None)
        tab = table if rho is None else tables_rho.get(rho)
        if tab is None:
            print(f"robustness: no table for {name}", file=sys.stderr)
            continue
        line = {"caliper": chosen, "cover": MAIN_COVER, "isolation": False} | change
        M = Measured(ctx, select(tab, "1000", **line))
        e = estimate(ctx, M, tab, "G", line, ("all",), years=False)["all"]
        d = contrasts(draws(e))[CONTRASTS[0]]
        out[name] = {"line": line | {"rho": rho or route.RHO},
                     "pieces": e["counts"]["pieces"], "long T": e["long_T"],
                     "quotes": {m: twoway.quote(d[m][0]) for m in FOREST if m in d}}
    return out


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    return int(x) if isinstance(x, np.integer) else x


def run_stats(ctx, table, tables_rho, chosen, tag):
    """The whole of the statistics at the chosen caliper: JSON, figures and the robustness lines."""
    line = {"caliper": chosen, "cover": MAIN_COVER, "isolation": False}
    wanted = select(table, "1000", **line)
    M = Measured(ctx, wanted)
    est = {side: estimate(ctx, M, table, side, line, survival_too=True) for side in ("G", "A")}
    result = {"line": line, "rho": route.RHO, "n_boot": N_BOOT, "years": ctx.years.tolist(), "levels": {}}
    forest = []
    for level in LEVEL_NAMES:
        g, a = (draws(est[s][level]) for s in ("G", "A"))
        d = contrasts(g, a)
        result["levels"][level] = {
            "counts": {s: est[s][level]["counts"] for s in ("G", "A")},
            "long T": {s: est[s][level]["long_T"] for s in ("G", "A")},
            "sides": {label: {m: quoted(draws_[m][which]) for m in draws_} | {"curves": {
                key: curve_of(est[s][level][key], which) for key in ("main", "long") if est[s][level][key]}}
                for label, s, which, draws_ in (("G", "G", "group", g), ("S|G", "G", "control", g),
                                                ("A", "A", "group", a), ("S|A", "A", "control", a))},
            "deltas": {c: {m: quoted(v) for m, v in dm.items()} for c, dm in d.items()},
            "delta curves": {c: {key: delta_curve(est[s][level][key]) for key in ("main", "long") if est[s][level][key]}
                             for s, c in (("G", CONTRASTS[0]), ("A", CONTRASTS[1]))}}
        for c in CONTRASTS:
            forest.append((f"{level}: {c}", CONTRAST_COLORS[c], result["levels"][level]["deltas"][c]))
        forest.append(None)
    result["survival"] = {s: est[s]["all"]["survival"] for s in ("G", "A")}
    result["robustness"] = robustness(ctx, table, tables_rho, chosen)
    OUT.mkdir(parents=True, exist_ok=True)
    draw_msd(est, OUT, tag)
    draw_survival(result["survival"], OUT / f"matched{tag}_survival.pdf")
    draw_forest(forest[:-1], OUT / f"matched{tag}_forest.pdf", "Open paragliders: group against the same route alone")
    rows = [(n, CONTRAST_COLORS[CONTRASTS[0]], r["quotes"]) for n, r in result["robustness"].items()]
    draw_forest(rows, OUT / f"matched{tag}_robustness.pdf", "Delta(G - S|G), level all, by line")
    (OUT / f"matched{tag}.json").write_text(json.dumps(jsonable(result), indent=1))
    return result


def show_stats(result):
    for level, r in result["levels"].items():
        print(level, json.dumps(r["counts"]), "long T", r["long T"])
        for label, s in r["sides"].items():
            print(f"  {label:4}", {m: round(s[m]["value"], 3) for m in METRICS if m in s})
        for c, dm in r["deltas"].items():
            print(f"  {c}")
            for m, v in dm.items():
                print(f"    {m:36} {v['value']:8.3f} band {np.round(v['band'], 3).tolist()} years {np.round(v['years'], 3).tolist()}")
    for name, r in result["robustness"].items():
        print(name, r["pieces"], {m: round(q["value"], 3) for m, q in r["quotes"].items()})


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
    tag = RHO_TAG + (f"_sample{sample}" if sample else "")
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
    if "--stats" in sys.argv:
        tag = f"_sample{sample}" if sample else ""
        others = {r: CACHE / f"comparisons_matches_rho{r}{tag}.parquet" for r in (300, 1000)}
        rho = {r: pd.read_parquet(p) for r, p in others.items() if p.exists()}
        show_stats(run_stats(Context(tracks, flights, summary), table, rho, result["chosen caliper"]["name"], tag))
