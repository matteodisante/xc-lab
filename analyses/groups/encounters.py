"""Encounters: when two gliders not in contact come close, how often do they join? Between groups.

Candidates. The method links any two flights of a date, whatever their discipline and the distance
of their takeoffs (r_s only enters the null model that fixes tau*), at the grid instants where both
have a position: after the departure (first fix farther than r_dep from the takeoff), until the
landing, across fixes at most h_max apart; its pair intervals are restricted to the time both fly and
kept if they last at least tau*. Encounters use the same flights and instants (grid.day on all the
flights of the date), so a pair that can never be linked is never counted as "did not join". The
method asks the headings within theta_max only when both glide; encounters ask it for every pass.

Pair pass. A maximal run of grid instants at which the horizontal distance of two flights (on the
ellipsoid, h = 0) is below R_ENC and the pair has no pair interval of the run; runs of the same pair
separated by less than the gap g of the run are one, since g is the time below which the method
itself treats a break of contact as no break. It starts where the pair comes from afar: at the grid
instant before it both flights have a position and are at least R_ENC apart, otherwise it is not an
approach (two gliders crossing the departure circle together start close) and is dropped.

Unit pass. The unit of a glider at the first instant of a pair pass is the row of groups.parquet it
belongs to then, or the glider itself if solo, so that a group meeting a glider is one encounter and
not one per member. The pair passes between the same two units that overlap in time are one unit pass
(a pass between two members of one group is not an encounter and is dropped). Its d_min, the
vertical separation |dz| and the heading difference are those of its closest pair of members, at the
instant of that distance (heading: displacement from the previous to the next grid instant); the type
is solo-solo, solo-group or group-group. Only passes with a heading difference of at most theta_max
of the run are kept, as the method links only gliders on similar headings.

First approach. No pair interval between any members of the two units starts before the pass: a
pair of units that were in contact earlier that day is a reunion, not a first approach.

Joined. A pair interval between any member of one unit and any member of the other starts during the
pass or within delta of the run after its end, the largest delay of a correspondence: a contact
that the method would date to the pass, as the glider behind reaches the place of the other within
delta.

R_ENC, measured in each run and written in encounters.json: the 95th percentile of the horizontal
distance of the two flights at the first instant of the pair intervals of the run (beyond it few
contacts begin). Measured on the full run: R_ENC = 1946 m.

Independent chances. A unit of n gliders meeting one of m offers n m member pairs, so it joins more
often than two solo gliders without any attraction. The expected joins of a unit pass are
1 - (1 - p)^(n m), p the solo-solo P(joined) of its d_min bin (and |dz| class when split). Members
fly together, so their chances are not independent and this is an upper bound on what the number of
gliders alone gives: observed over expected above 1 is a lower bound on attraction.

Summary. By type: the passes, the share joined and P(joined) in N_BINS equal bins of d_min from 0
to R_ENC, also for |dz| below and above eps_z; the ratios of solo-group and group-group to solo-solo,
pooled over d_min with the solo-solo weights of the bins (direct standardisation, on the bins where
both types have passes), and the observed joins over the independent chances (also solo-group by
size of the group, 2 or 3 and more). Bands: 5-95% of N_BOOT draws of the dates, the dependency block (a thermal
day shared by the pilots of one date).

    uv run python encounters.py [--run NAME] [--sample N]    # from this folder, after cohesion.py
    uv run python encounters.py --redraw                     # summary and figure again, from the saved pieces
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from breathing import N_BOOT, NO_RUNS, stat
from cohesion import contacts, grid
from marking import group_at, spans_of
from tamsd import LABELS, OUT, SEED, save

PCT_R = 95  # percentile that gives R_ENC
N_BINS = 8  # equal bins of d_min from 0 to R_ENC
BLOCK_S = 3600  # s, close pairs are searched one block of time at a time
PAIR = 1 << 20  # key of a pair of flights: first * PAIR + second
UNIT = 1 << 18  # offset of a unit code (a group id, or minus the flight number minus 1), so that it is positive
SPAN = 1 << 32  # s, larger than any time of a date: separates the pairs of units when merging
COUNTS = ("pair_passes", "same_unit", "start_close", "unit_passes", "not_first_approach", "no_heading", "above_theta_max")
TYPES = ("solo-solo", "solo-group", "group-group")
COLORS = {"solo-solo": "#2a78d6", "solo-group": "#8e5ad7", "group-group": "#eb6834"}
SELECTIONS = {"all": (0, 1), "below": (0,), "above": (1,)}  # |dz| below / above eps_z


def group_ids(flight, t, spans):
    """The group (-1 solo) of each flight at each time, from `spans` (marking.spans_of)."""
    out = np.full(len(t), -1)
    for f in np.intersect1d(flight, list(spans)):
        m = flight == f
        out[m] = group_at(t[m], spans[f])
    return out


def close_points(g, r, step):
    """The pairs of grid rows of `g` (sorted by f, t) of different flights at the same instant less
    than r apart, found hour by hour: row a, row b and their distance."""
    t, xyz = g["t"].to_numpy(), g[["X", "Y", "Z"]].to_numpy()
    f = g["f"].to_numpy()
    block = (t - t.min()) // BLOCK_S
    out = []
    for b in np.unique(block):
        rows = np.flatnonzero(block == b)
        tree = cKDTree(np.column_stack([xyz[rows], contacts.K * (t[rows] - t[rows].min()) / step]))
        pairs = rows[tree.query_pairs(r, output_type="ndarray")]
        pairs = pairs[f[pairs[:, 0]] != f[pairs[:, 1]]]
        out.append((pairs[:, 0], pairs[:, 1], np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)))
    return tuple(np.concatenate(x) for x in zip(*out))


def inside(key, t, intervals):
    """Whether each close point (sorted by key, t) lies in a pair interval [on, off) of its pair;
    `intervals` maps a key to the array (m, 2) of the (on, off) of its pair, sorted."""
    out = np.zeros(len(t), bool)
    for k, iv in intervals.items():
        lo, hi = np.searchsorted(key, k, "left"), np.searchsorted(key, k, "right")
        for on, off in iv:
            out[lo:hi] |= (t[lo:hi] >= on) & (t[lo:hi] < off)
    return out


def passes(key, t, d, intervals, step, gap):
    """The pair passes of the close points (sorted by key, t) of a date: key, first instant `start`,
    last instant `end`, the index `first` of the first close point and `i` of the closest one, and its
    distance `d`. Close points inside a pair interval are left out; runs of a pair less than `gap`
    apart (the time missing between two instants) are one."""
    idx = np.flatnonzero(~inside(key, t, intervals))
    k, s, e = key[idx], t[idx], d[idx]
    new = np.zeros(len(idx), bool)
    new[:1] = True
    new[1:] = (k[1:] != k[:-1]) | (s[1:] - s[:-1] - step >= gap)
    first = np.flatnonzero(new)
    last = np.r_[first[1:], len(idx)][:len(first)] - 1
    best = np.lexsort((e, np.cumsum(new) - 1))[first]  # the closest point of each run
    return {"key": k[first], "start": s[first], "end": s[last], "first": idx[first], "i": idx[best], "d": e[best]}


def genuine(xyz, f, t, rows_a, rows_b, r, step):
    """Whether a pass is an approach: at the grid instant before the grid rows `rows_a`, `rows_b` (the
    two flights at the first instant) both flights have a position and are at least r apart."""
    pa, pb = np.maximum(rows_a - 1, 0), np.maximum(rows_b - 1, 0)
    ok = (f[pa] == f[rows_a]) & (t[pa] == t[rows_a] - step) & (f[pb] == f[rows_b]) & (t[pb] == t[rows_b] - step)
    return ok & (np.linalg.norm(xyz[pa] - xyz[pb], axis=1) >= r)


def unit_codes(flight, group):
    """The unit of each glider: its group id if it is in a group (>= 0), else -(flight + 1)."""
    return np.where(group >= 0, group, -(flight + 1))


def merge_units(ua, ub, start, end, d):
    """The unit passes of pair passes (units ua, ub at the start, first and last instant, closest
    distance): those between the same two units that overlap in time are one. Returns the index of
    the closest pair pass of each unit pass, its first and its last instant."""
    key = (np.minimum(ua, ub) + UNIT) * PAIR + np.maximum(ua, ub) + UNIT
    o = np.lexsort((start, key))
    k, s, e = key[o], start[o], end[o]
    rank = np.cumsum(np.r_[True, k[1:] != k[:-1]]) - 1 if len(o) else np.zeros(0, np.int64)
    top = np.maximum.accumulate(rank * SPAN + e)
    new = np.zeros(len(o), bool)
    new[:1] = True
    new[1:] = rank[1:] * SPAN + s[1:] > top[:-1]
    first = np.flatnonzero(new)
    best = np.lexsort((d[o], np.cumsum(new) - 1))[first]
    return o[best], np.minimum.reduceat(s, first), np.maximum.reduceat(e, first)


def first_contact(a, b, earliest):
    """The earliest start of a pair interval between a flight of `a` and one of `b` (inf if none);
    `earliest` maps the key of a pair of flights to the start of its first interval."""
    return min((earliest.get(min(x, y) * PAIR + max(x, y), np.inf) for x in a for y in b), default=np.inf)


def heading(xyz, f, t, rows, step):
    """The displacement from the previous to the next grid instant of the flight at each grid row (NaN
    where either is missing)."""
    lo, hi = np.maximum(rows - 1, 0), np.minimum(rows + 1, len(t) - 1)
    ok = (f[lo] == f[rows]) & (f[hi] == f[rows]) & (t[hi] - t[lo] == 2 * step)
    return np.where(ok[:, None], xyz[hi] - xyz[lo], np.nan)


def angle(u, v):
    """The angle (deg) between rows of u and v; NaN for a zero vector."""
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (u * v).sum(axis=1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def open_day(archive_day, mine, p, only=None):
    """grid.day of the flights `mine` of a date (all, or those with a number in `only`): the flight
    table, the grid sorted by (f, t) and the flight number of each table row; None without a grid."""
    if only is not None:
        mine = mine[mine["flight"].isin(only)]
    a = archive_day.merge(mine[["discipline", "flight_id", "flight"]], on=["discipline", "flight_id"])
    table, g = grid.day(a, {d: NO_RUNS for d in grid.DERIVED}, p["step"], p["h_max"], p["r_dep"])
    if g is None:
        return None
    number = a.set_index(["discipline", "flight_id"]).loc[pd.MultiIndex.from_frame(table[["discipline", "flight_id"]]), "flight"]
    return table, g.sort_values(["f", "t"]).reset_index(drop=True), number.to_numpy()


def interval_table(iv):
    """The pair intervals of a date as {key: array (m, 2) of (on, off)}, sorted."""
    a, b = np.minimum(iv["a"], iv["b"]), np.maximum(iv["a"], iv["b"])
    key = a.to_numpy().astype(np.int64) * PAIR + b.to_numpy()
    return {k: np.array(sorted(zip(on, off))) for k, on, off in
            ((k, iv["t_on"].to_numpy()[key == k], iv["t_off"].to_numpy()[key == k]) for k in np.unique(key))}


def start_distances(iv, tracks, row_of):
    """The horizontal distance (m) of the two flights at the first instant of each pair interval, for the
    intervals whose first instant has a position of both."""
    out = []
    for a, b, t in iv[["a", "b", "t_on"]].itertuples(index=False):
        if row_of[a] in tracks.values and row_of[b] in tracks.values:
            x, y = (tracks.at(row_of[f], np.array([t]))[0, :3] for f in (a, b))
            out.append(np.linalg.norm(x - y))
    out = np.array(out)
    return out[np.isfinite(out)]  # an instant without a position of either flight gives no distance


def one_date(g, number, intervals, day, r, p):
    """One date: its kept unit passes (see the header) as a dict of arrays type, d, dz, joined, the sizes
    n <= m of the two units, and
    the counts (COUNTS) of the passes found and dropped. `day` are the rows of groups.parquet."""
    step = p["step"]
    f, t, z = (g[c].to_numpy() for c in ("f", "t", "z"))
    xyz = g[["X", "Y", "Z"]].to_numpy()
    ra, rb, d = close_points(g, r, step)
    fa, fb = number[f[ra]], number[f[rb]]
    swap = fa > fb
    ra, rb, fa, fb = (np.where(swap, y, x) for x, y in ((ra, rb), (rb, ra), (fa, fb), (fb, fa)))
    key = fa.astype(np.int64) * PAIR + fb
    o = np.lexsort((t[ra], key))
    ra, rb, fa, fb, key, d = ra[o], rb[o], fa[o], fb[o], key[o], d[o]
    pp = passes(key, t[ra], d, intervals, step, p["gap"])
    i, s = pp["i"], pp["start"]
    a, b = fa[i], fb[i]
    spans = spans_of(day)
    ua, ub = (unit_codes(x, group_ids(x, s, spans)) for x in (a, b))
    same = ua == ub
    far = genuine(xyz, f, t, ra[pp["first"]], rb[pp["first"]], r, step)
    keep = np.flatnonzero(~same & far)
    closest, start, end = merge_units(ua[keep], ub[keep], s[keep], pp["end"][keep], pp["d"][keep])
    c = keep[closest]
    members = dict(zip(day["group_id"], day["members"]))
    of = lambda u: members[u] if u >= 0 else (-(u + 1),)
    size = np.array([[len(of(x)), len(of(y))] for x, y in zip(ua[c], ub[c])]).reshape(-1, 2)
    earliest = {k: iv[0, 0] for k, iv in intervals.items()}
    first = np.array([first_contact(of(x), of(y), earliest) for x, y in zip(ua[c], ub[c])])
    theta = angle(heading(xyz, f, t, ra[i[c]], step), heading(xyz, f, t, rb[i[c]], step))
    new = first >= start  # no interval between the units before the pass
    ok = new & (theta <= p["theta_max"])
    out = {"type": ((ua[c] >= 0).astype(int) + (ub[c] >= 0))[ok], "d": pp["d"][c][ok],
           "dz": np.abs(z[ra[i[c]]] - z[rb[i[c]]])[ok], "joined": (first <= end + p["delta"])[ok],
           "n": size.min(axis=1)[ok], "m": size.max(axis=1)[ok]}
    return out, np.array([len(i), same.sum(), (~same & ~far).sum(), len(c), (~new).sum(),
                          (new & ~np.isfinite(theta)).sum(), (new & (theta > p["theta_max"])).sum()])


def measure_starts(flights, intervals, dates, p):
    """The distance at the first instant of the pair intervals of the dates (pass 1 of the run)."""
    archive = grid.archive()
    by_date = dict(tuple(archive[archive["date"].isin(dates)].groupby("date")))
    flights_of = dict(tuple(flights[flights["date"].isin(dates)].groupby("date")))
    by_day = dict(tuple(intervals.groupby("date")))
    out = []
    for number, date in enumerate(dates):
        iv = by_day.get(date)
        if iv is None:
            continue
        opened = open_day(by_date[date], flights_of[date], p, set(iv["a"]) | set(iv["b"]))
        if opened is not None:
            table, g, num = opened
            out.append(start_distances(iv, contacts.Tracks(g, p["step"]), {n: k for k, n in enumerate(num)}))
        if number % 200 == 0:
            print(f"{number}/{len(dates)} dates, {sum(len(x) for x in out)} interval starts", flush=True)
    return np.concatenate(out)


def measure(flights, groups, intervals, dates, p, r):
    """one_date for each date: the kept unit passes of all the dates with their date index and the
    summed counts."""
    archive = grid.archive()
    by_date = dict(tuple(archive[archive["date"].isin(dates)].groupby("date")))
    flights_of = dict(tuple(flights[flights["date"].isin(dates)].groupby("date")))
    by_day, iv_day = dict(tuple(groups.groupby("date"))), dict(tuple(intervals.groupby("date")))
    pieces, counts = [], np.zeros(len(COUNTS), int)
    for number, date in enumerate(dates):
        opened = open_day(by_date[date], flights_of[date], p)
        if opened is not None:
            table, g, num = opened
            iv = iv_day.get(date)
            out, c = one_date(g, num, interval_table(iv) if iv is not None else {}, by_day.get(date, groups.iloc[:0]), r, p)
            pieces.append({**out, "date": np.full(len(out["d"]), number)})
            counts += c
        if number % 100 == 0:
            print(f"{number}/{len(dates)} dates, {counts[3]} unit passes, {sum(len(x['d']) for x in pieces)} kept", flush=True)
    return {k: np.concatenate([x[k] for x in pieces]) for k in pieces[0]}, counts


def tally(passes_, n_dates, r, eps_z, ks):
    """The unit passes and those that joined, by date, type, |dz| below / above eps_z, bin of d_min
    and number of member pairs n m (the values `ks`): two arrays (dates, 3, 2, N_BINS, len(ks))."""
    p = passes_
    cell = (p["date"], p["type"], (p["dz"] >= eps_z).astype(int), np.minimum((p["d"] / r * N_BINS).astype(int), N_BINS - 1),
            np.searchsorted(ks, p["n"].astype(int) * p["m"]))
    n, j = (np.zeros((n_dates, 3, 2, N_BINS, len(ks)), np.float32) for _ in range(2))
    np.add.at(n, cell, 1)
    np.add.at(j, tuple(c[p["joined"]] for c in cell), 1)
    return n, j


def over_expected(nk, jk, ks, k, v, pooled, pairs=None):
    """Observed joins of type k over those expected from independent chances, 1 - (1 - p)^(n m) per
    unit pass with p the solo-solo P(joined) of its d_min bin, of its |dz| class too unless `pooled`;
    for the |dz| classes `v` and the numbers of pairs `pairs` (a mask on ks, default all). `nk`, `jk`
    (..., 3, 2, N_BINS, len(ks)), summed over dates. Returns the ratio, observed, expected."""
    n, j = nk.sum(axis=-1), jk.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = j[..., 0, :, :] / n[..., 0, :, :]
        if pooled:
            p = np.broadcast_to(j[..., 0, :, :].sum(axis=-2, keepdims=True) / n[..., 0, :, :].sum(axis=-2, keepdims=True), p.shape)
    ok = np.isfinite(p)  # a bin without solo-solo passes has no expectation: left out of both sums
    chance = 1 - (1 - np.where(ok, p, 0.0))[..., None] ** ks
    keep = (ok[..., None] * (np.ones(len(ks), bool) if pairs is None else pairs))[..., list(v), :, :]
    obs = (jk[..., k, :, :, :][..., list(v), :, :] * keep).sum(axis=(-3, -2, -1))
    exp = (nk[..., k, :, :, :][..., list(v), :, :] * chance[..., list(v), :, :] * keep).sum(axis=(-3, -2, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return obs / exp, obs, exp


def standardised(n, j, k):
    """P(joined) of type k over that of solo-solo (type 0) with the solo-solo weights of the bins,
    on the bins where both have passes: n, j (..., 3, N_BINS) give (...)."""
    both = (n[..., 0, :] > 0) & (n[..., k, :] > 0)
    w = np.where(both, n[..., 0, :], 0.0)
    p0, pk = (np.divide(j[..., m, :], n[..., m, :], out=np.zeros(n[..., m, :].shape), where=n[..., m, :] > 0) for m in (0, k))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (w * pk).sum(axis=-1) / (w * p0).sum(axis=-1)


def summarize(nk, jk, ks, rng):
    """By selection of |dz| (all, below, above) and type: the passes, the share joined and P(joined)
    by bin of d_min; the ratios of solo-group and group-group to solo-solo; their observed joins over
    independent chances, also for solo-group of 2 and of 3 or more members; with their bands over
    the dates. `nk`, `jk` (dates, 3, 2, N_BINS, len(ks)) from tally."""
    w = rng.multinomial(len(nk), np.full(len(nk), 1 / len(nk)), size=N_BOOT).astype(float)
    n, j = nk.sum(axis=-1), jk.sum(axis=-1)
    real = [x.sum(axis=0, dtype=float) for x in (nk, jk)]
    draw = [np.tensordot(w, x.reshape(len(x), -1), axes=(1, 0)).reshape(len(w), *x.shape[1:]) for x in (nk, jk)]
    out = {}
    for name, v in SELECTIONS.items():
        cells = [(x[:, :, list(v)].sum(axis=2)) for x in (n, j)]
        (nv, jv), (nd, jd) = ([x.sum(axis=0) for x in cells], [np.tensordot(w, x, axes=(1, 0)) for x in cells])
        with np.errstate(divide="ignore", invalid="ignore"):
            share, share_d = jv.sum(-1) / nv.sum(-1), jd.sum(-1) / nd.sum(-1)
            curve, curve_d = jv / nv, jd / nd

        def chance(k, pairs=None):
            (value, obs, exp), (draws, _, _) = (over_expected(*x, ks, k, v, name == "all", pairs) for x in (real, draw))
            return {**stat(float(value), draws), "observed": float(obs), "expected": float(exp)}

        out[name] = {"passes": {TYPES[k]: int(nv[k].sum()) for k in range(3)},
                     "share": {TYPES[k]: stat(float(share[k]), share_d[:, k]) for k in range(3)},
                     "curve": {TYPES[k]: [{"n": int(nv[k, b]), **stat(float(curve[k, b]), curve_d[:, k, b])} for b in range(N_BINS)]
                               for k in range(3)},
                     "ratio": {TYPES[k]: stat(float(standardised(nv, jv, k)), standardised(nd, jd, k)) for k in (1, 2)},
                     "obs_exp": {TYPES[k]: chance(k) for k in (1, 2)},
                     "size": {"solo-pair": chance(1, ks == 2), "solo-3+": chance(1, ks >= 3)}}
    return out


def plot(r, path):
    """Left: P(joined) against the closest distance by type; right: for solo-group and group-group,
    the ratio to solo-solo (standardised over d_min) and the observed joins over independent
    chances, with the line at 1."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.9))
    edges = np.linspace(0, r["params"]["R_ENC_m"], N_BINS + 1)
    x = (edges[1:] + edges[:-1]) / 2
    for k in TYPES:
        c = r["summary"]["all"]["curve"][k]
        a.fill_between(x, *np.array([y["band"] for y in c]).T, color=COLORS[k], alpha=0.15, lw=0)
        a.plot(x, [y["value"] for y in c], color=COLORS[k], marker="o", ms=3, lw=1.2, label=k)
    a.set(xlabel="closest distance $d_{min}$ (m)", ylabel="P(joined)", ylim=(0, None))
    a.legend(frameon=False, fontsize=8)
    for i, k in enumerate(TYPES[1:]):
        for dx, key, marker in ((-0.1, "ratio", "o"), (0.1, "obs_exp", "s")):
            v = r["summary"]["all"][key][k]
            b.errorbar(i + dx, v["value"], yerr=[[v["value"] - v["band"][0]], [v["band"][1] - v["value"]]], fmt=marker, ms=4, lw=1.2,
                       color=COLORS[k], mfc=COLORS[k] if key == "ratio" else "white")
    b.plot([], [], "o", color="#555555", label="over solo-solo")
    b.plot([], [], "s", color="#555555", mfc="white", label="over independent chances")
    b.axhline(1, color="#999999", lw=0.6)
    b.set_xticks(range(2), TYPES[1:])
    b.set(xlim=(-0.5, 1.5), ylabel="observed / reference", ylim=(0, None))
    b.legend(frameon=False, fontsize=8)
    save(fig, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="read by tamsd")
    parser.add_argument("--sample", type=int, help="a random sample of dates (seed 0)")
    parser.add_argument("--redraw", action="store_true", help="summary and figure again, from the saved pieces")
    args = parser.parse_args()
    name = "encounters" + (f"_sample{args.sample}" if args.sample else "")
    OUT.mkdir(parents=True, exist_ok=True)
    params = json.loads((LABELS / "days" / "params.json").read_text())
    if args.redraw:
        saved = np.load(OUT / f"{name}_pieces.npz")
        pieces = {k: saved[k] for k in ("type", "d", "dz", "joined", "date", "n", "m")}
        starts, counts, n_dates = (saved[k] for k in ("starts", "counts", "n_dates"))
    else:
        flights = pd.read_parquet(LABELS / "flights.parquet")
        groups = pd.read_parquet(LABELS / "groups.parquet")
        intervals = pd.read_parquet(LABELS / "pair_intervals.parquet")
        count = flights.groupby("date").size()
        dates = sorted(count[count >= 2].index)
        if args.sample:
            dates = sorted(np.random.default_rng(0).choice(dates, args.sample, replace=False))
        starts = measure_starts(flights, intervals[intervals["date"].isin(dates)], dates, params)
        r = float(np.percentile(starts, PCT_R))
        print(f"{len(starts)} interval starts: R_ENC = {r:.0f} m", flush=True)
        pieces, counts = measure(flights, groups[groups["date"].isin(dates)], intervals[intervals["date"].isin(dates)], dates, params, r)
        n_dates = np.array(len(dates))
        np.savez(OUT / f"{name}_pieces.npz", starts=starts, counts=counts, n_dates=n_dates, **pieces)
    r = float(np.percentile(starts, PCT_R))
    ks = np.unique(pieces["n"].astype(int) * pieces["m"])
    nk, jk = tally(pieces, int(n_dates), r, params["eps_z"], ks)
    result = {"summary": summarize(nk, jk, ks, np.random.default_rng(SEED))}
    pairs = pieces["n"].astype(int) * pieces["m"]
    pct = (10, 25, 50, 75, 90, 95)
    result["measured"] = {"interval_starts": len(starts), **dict(zip(COUNTS, map(int, counts))), "unit_passes_kept": len(pieces["d"]),
                          "member_pairs": {t: {"median": float(np.median(pairs[pieces["type"] == k])), "p90": float(np.percentile(pairs[pieces["type"] == k], 90))}
                                           for k, t in enumerate(TYPES)},
                          "percentiles": list(pct), "start_distance_m": np.percentile(starts, pct).tolist()}
    result["params"] = {"R_ENC_m": r, "percentile_R": PCT_R, "delta_s": params["delta"], "gap_s": params["gap"], "n_bins": N_BINS,
                        "theta_max": params["theta_max"], "eps_z": params["eps_z"], "step": params["step"],
                        "n_boot": N_BOOT, "dates": int(n_dates)}
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / f"{name}.pdf")
    print(json.dumps(result["summary"]["all"]["ratio"]), flush=True)
