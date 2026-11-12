"""The group, or the gliders around it? Intra-group against inter-group effect, on the open flights.

Positions: the archive index of isolated.py (every flight, para and hang, one position per minute, on
the sphere). An outsider of a pilot at a minute is any other flight of the index at that minute that
is not in the pilot's group then. The direction of motion u of a pilot is the displacement of its
position between t - T_DIR_S / 2 and t + T_DIR_S / 2 (of the centroid of its group while it is in
one), kept when it is at least MIN_DISP_M long. An outsider is ahead when its position relative to the
pilot has a positive component along u, behind otherwise; d_a and d_b are the distances to the nearest
one of each kind, infinite when none lies within R_MAX.

Step A: length and width of every group (extent of its members along and across u) and the gap from
its leader to the nearest outsider ahead. Step B: the pieces of dataset 1000 of the raw design
(populations S and A pooled as "solo", and G as "group"), each with the share of its minutes with an
outsider ahead (f_a) and behind (f_b) within R; classes ALONE, AHEAD, BEHIND, MIXED; H_loc(1000 s) and
the phase statistics by state and class, five contrasts, raw and within the flight. Step C: searches
and glides by the gap at their start, and H_loc(1000 s) by the median gap of the piece.
Bands: twoway.py, the same draws for every number.

    uv run python intergroup.py --open [--sample DAYS]    # after tamsd.py, longlags.py, performance.py with --open
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import budget
import isolated
import longlags
import raw
import survival
import tamsd
import twoway
from blocks import with_takeoffs
from isolated import STEP_S
from phase_msd import RUNS
from tamsd import CACHE, LABELS, LAGS, OUT, SEED, read_flights

T_DIR_S = 600  # s: the window of the direction of motion
MIN_DISP_M = 500.0  # m: a displacement shorter than this over the window gives no direction
R_MAX = 20_000.0  # m: gaps farther than this are "none"
RADII = (1000.0, 2000.0, 5000.0, 10000.0)  # m: the radii of the shares and of the classes
R_MAIN = 5000.0  # m: the radius of the classes of the main line
F_LOW, F_HIGH = 0.1, 0.5  # share of minutes: "nobody" and "most of the time"
MIN_DEFINED = 0.8  # share of the minutes of a piece that must have a direction
BINS = (0.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0)  # m: the bins of the gaps; "none" follows
H_BINS = (0.0, 2000.0, 5000.0, 10000.0, 20000.0)  # m: the bins of the median gap of a piece for H_loc; "none" follows
AT = 1000  # s: the lag of H_loc
CLASSES = ("ALONE", "AHEAD", "BEHIND", "MIXED")
STATES = ("solo", "group")
CONTRASTS = {  # name: ((state, class), (state, class))
    "intra-group": (("group", "ALONE"), ("solo", "ALONE")),
    "inter-group": (("solo", "AHEAD"), ("solo", "ALONE")),
    "crowd control": (("solo", "BEHIND"), ("solo", "ALONE")),
    "information, solo": (("solo", "AHEAD"), ("solo", "BEHIND")),
    "information, group": (("group", "AHEAD"), ("group", "BEHIND")),
    "both": (("group", "AHEAD"), ("solo", "ALONE")),
}
PHASE_STATS = ("time in climb (%)", "searches that end in a climb (%)", "glide speed (km/h)", raw.decompose.SINK)
HALF = T_DIR_S // STEP_S // 2  # minutes on each side of the instant
KEY = 1 << 26  # flight * KEY + minute is a key
SPAN = 4e9  # s: flight * SPAN + t is a key of the membership
FAR = 1e9  # m: scale of the minute in the trees, so that only the same minute is near
DAY = 1440  # minutes


def tangent_unit(vec, pos):
    """The unit vector of `vec` (n, 3) projected on the tangent plane at `pos`; NaN where the
    projection is shorter than MIN_DISP_M or `vec` is NaN."""
    n = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    t = vec - np.sum(vec * n, axis=1, keepdims=True) * n
    length = np.linalg.norm(t, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = t / length
    u[~(length[:, 0] >= MIN_DISP_M)] = np.nan
    return u


def displacement(flight, minute, xyz):
    """For rows sorted by (flight, minute): the position at minute + HALF minus that at minute - HALF
    of the same flight, NaN where either is missing."""
    key = flight.astype(np.int64) * KEY + minute
    last = len(key) - 1
    lo, hi = (np.minimum(np.searchsorted(key, key + s), last) for s in (-HALF, HALF))
    ok = (key[lo] == key - HALF) & (key[hi] == key + HALF)
    out = np.full((len(key), 3), np.nan)
    out[ok] = xyz[hi[ok]] - xyz[lo[ok]]
    return out


def group_of(flight, minute, mem):
    """The group id of each (flight, minute) from the stretches of group time `mem` (sorted keys of
    starts, flights, ends, ids); -1 when not in a group."""
    start, mem_flight, end, gid = mem
    if not len(start):
        return np.full(len(flight), -1, np.int64)
    t = minute.astype(float) * STEP_S
    i = np.searchsorted(start, flight * SPAN + t, "right") - 1
    j = np.maximum(i, 0)
    ok = (i >= 0) & (mem_flight[j] == flight) & (t < end[j])
    return np.where(ok, gid[j], -1)


def membership_table(membership):
    """The stretches of group time of the membership table, sorted, as group_of() takes them."""
    g = membership[membership["state"] == "G"].sort_values(["flight", "t_start"], kind="stable")
    f = g["flight"].to_numpy(np.int64)
    return f * SPAN + g["t_start"].to_numpy(float), f, g["t_end"].to_numpy(float), g["group_id"].to_numpy(np.int64)


def day_frame(flight, minute, xyz, mem):
    """The rows of the index of one day (and its margins) sorted by (flight, minute), with the group
    id, the direction of the pilot (u_own) and that of its group (u_grp, from the mean displacement of
    its members at the minute)."""
    o = np.lexsort((minute, flight))
    f, m, x = flight[o].astype(np.int64), minute[o].astype(np.int64), xyz[o].astype(float)
    gid = group_of(f, m, mem)
    d = displacement(f, m, x)
    u_grp = np.full((len(f), 3), np.nan)
    g = np.flatnonzero(gid >= 0)
    if len(g):
        code = np.unique(gid[g] * KEY + m[g], return_inverse=True)[1]
        ok = np.isfinite(d[g, 0])
        total, n = np.zeros((code.max() + 1, 3)), np.zeros(code.max() + 1)
        np.add.at(total, code[ok], d[g][ok])
        np.add.at(n, code[ok], 1)
        u_grp[g] = tangent_unit(total[code] / np.maximum(n[code], 1)[:, None], x[g])
    return {"flight": f, "minute": m, "xyz": x, "gid": gid, "u_own": tangent_unit(d, x), "u_grp": u_grp}


def direction(fr, rows):
    """The direction of each row of the frame: that of its group when it is in one, else its own."""
    return np.where(fr["gid"][rows, None] >= 0, fr["u_grp"][rows], fr["u_own"][rows])


def gaps(fr, rows, chunk=3000):
    """For the rows `rows` of a day frame, the distance (m) to the nearest outsider within R_MAX: ahead
    (d_a), behind (d_b), ahead and in another group (d_ag), ahead and solo (d_as), and the cross-track
    offset of the nearest one ahead (xt); inf (NaN for xt) where there is none; and whether the
    direction is defined. Outsiders: other flights of the frame at the same minute, not of the group of
    the row."""
    origin = fr["minute"].min()
    tree = cKDTree(np.column_stack([fr["xyz"], (fr["minute"] - origin) * FAR]), balanced_tree=False, compact_nodes=False)
    u = direction(fr, rows)
    out = {k: np.full(len(rows), np.inf) for k in ("d_a", "d_b", "d_ag", "d_as")}
    out["xt"], out["defined"] = np.full(len(rows), np.nan), np.isfinite(u[:, 0])
    for a in range(0, len(rows), chunk):
        r = rows[a:a + chunk]
        q = np.column_stack([fr["xyz"][r], (fr["minute"][r] - origin) * FAR])
        pairs = cKDTree(q).sparse_distance_matrix(tree, R_MAX, output_type="ndarray")
        i, j, dist = pairs["i"], pairs["j"], pairs["v"]
        me = r[i]
        keep = out["defined"][a + i] & (fr["flight"][j] != fr["flight"][me])
        keep &= ~((fr["gid"][me] >= 0) & (fr["gid"][j] == fr["gid"][me]))
        i, j, dist, me = i[keep], j[keep], dist[keep], me[keep]
        along = np.sum((fr["xyz"][j] - fr["xyz"][me]) * u[a + i], axis=1)
        ahead, k = along > 0, a + i
        np.minimum.at(out["d_a"], k[ahead], dist[ahead])
        np.minimum.at(out["d_b"], k[~ahead], dist[~ahead])
        other = ahead & (fr["gid"][j] >= 0)
        np.minimum.at(out["d_ag"], k[other], dist[other])
        solo = ahead & (fr["gid"][j] < 0)
        np.minimum.at(out["d_as"], k[solo], dist[solo])
        nearest = ahead & (dist == out["d_a"][k])
        out["xt"][k[nearest]] = np.sqrt(np.maximum(dist[nearest] ** 2 - along[nearest] ** 2, 0))
    return out


def group_geometry(fr, wanted, lo, hi):
    """Per (group, minute) of the groups `wanted` (mask by group id) with at least two members and a
    direction, for the minutes in [lo, hi): n members, length (extent along u), width (across), and the
    gaps of gaps() from the leader (the member farthest along u)."""
    g = np.flatnonzero((fr["gid"] >= 0) & wanted[np.maximum(fr["gid"], 0)] & (fr["minute"] >= lo) & (fr["minute"] < hi))
    if not len(g):
        return None
    key, first, k = np.unique(fr["gid"][g] * KEY + fr["minute"][g], return_index=True, return_inverse=True)
    n = np.bincount(k)
    u = fr["u_grp"][g][first]
    xyz = fr["xyz"][g]
    centroid = np.zeros((len(key), 3))
    np.add.at(centroid, k, xyz)
    centroid /= n[:, None]
    up = centroid / np.linalg.norm(centroid, axis=1, keepdims=True)
    side = np.cross(up, u)
    rel = xyz - centroid[k]
    s, c = np.sum(rel * u[k], axis=1), np.sum(rel * side[k], axis=1)
    frame = pd.DataFrame({"k": k, "s": np.where(np.isfinite(s), s, -np.inf), "c": c, "row": g})
    by = frame.groupby("k")
    length, width = by["s"].agg(np.ptp), by["c"].agg(np.ptp)
    leader = frame.loc[by["s"].idxmax(), "row"].to_numpy()
    ok = (n >= 2) & np.isfinite(u[:, 0])
    out = gaps(fr, leader[ok])
    return {"gid": key[ok] // KEY, "minute": key[ok] % KEY, "n": n[ok], "length": length.to_numpy()[ok],
            "width": width.to_numpy()[ok], **{k_: v for k_, v in out.items() if k_ != "defined"}}


def run_days(index, mem, wanted, qflight, qminute, days):
    """The geometry of every day of `days` (UTC day numbers): for the queries (flight, minute; sorted by
    minute) the gaps of gaps(), and for the groups `wanted` the rows of group_geometry(). Returns the
    queries' result (dict of arrays) and the group rows (dict of arrays)."""
    res = {k: np.full(len(qminute), v, np.float32) for k, v in (("d_a", np.inf), ("d_b", np.inf), ("xt", np.nan))}
    res["defined"], res["found"] = np.zeros(len(qminute), bool), np.zeros(len(qminute), bool)
    rows = []
    for n, day in enumerate(days):
        lo, hi = day * DAY, (day + 1) * DAY
        a, b = np.searchsorted(index["minute"], [lo - HALF - 1, hi + HALF + 1])
        if b <= a:
            continue
        fr = day_frame(index["flight"][a:b], index["minute"][a:b], index["xyz"][a:b], mem)
        qa, qb = np.searchsorted(qminute, [lo, hi])
        if qb > qa:
            key = fr["flight"] * KEY + fr["minute"]
            want = qflight[qa:qb].astype(np.int64) * KEY + qminute[qa:qb]
            at = np.minimum(np.searchsorted(key, want), len(key) - 1)
            found = key[at] == want
            r = gaps(fr, at[found])
            sel = qa + np.flatnonzero(found)
            res["found"][sel] = True
            for k in ("d_a", "d_b", "xt", "defined"):
                res[k][sel] = r[k]
        g = group_geometry(fr, wanted, lo, hi)
        if g:
            rows.append(g)
        if n % 200 == 0:
            print(f"day {n}/{len(days)}", file=sys.stderr, flush=True)
    keys = rows[0].keys() if rows else ()
    return res, {k: np.concatenate([r[k] for r in rows]) for k in keys}


def piece_minutes(utc0, flight, start, duration):
    """The minutes of the index inside each piece: the piece number and the minute, flat."""
    t0 = utc0[flight] + start
    m0, m1 = np.ceil(t0 / STEP_S).astype(np.int64), np.floor((t0 + duration) / STEP_S).astype(np.int64)
    n = np.maximum(m1 - m0 + 1, 0)
    piece = np.repeat(np.arange(len(flight)), n)
    return piece, m0[piece] + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)


def piece_summary(piece, n_pieces, d_a, d_b, defined):
    """Per piece: the share of its minutes with a direction, f_a(R) and f_b(R) for the RADII (shares of
    the minutes with a direction), and the median gap ahead and behind over them."""
    n = np.bincount(piece, minlength=n_pieces)
    nd = np.bincount(piece, weights=defined, minlength=n_pieces)
    out = {"defined": np.divide(nd, n, out=np.zeros(n_pieces), where=n > 0)}
    for side, d in (("a", d_a), ("b", d_b)):
        for r in RADII:
            out[f"f_{side}_{r:g}"] = np.divide(np.bincount(piece, weights=defined & (d <= r), minlength=n_pieces), nd,
                                               out=np.full(n_pieces, np.nan), where=nd > 0)
        med = pd.Series(np.where(defined, d, np.nan)).groupby(piece).median()
        out[f"med_{side}"] = med.reindex(range(n_pieces)).to_numpy()
    return out


def neighbourhood(f_a, f_b):
    """The class of each piece from its shares of minutes with an outsider ahead and behind."""
    out = np.full(len(f_a), "MIXED", object)
    out[(f_a < F_LOW) & (f_b < F_LOW)] = "ALONE"
    out[(f_b >= F_HIGH) & (f_a < F_LOW)] = "BEHIND"
    out[f_a >= F_HIGH] = "AHEAD"
    return out


def hloc_weights(lags=LAGS, at=AT):
    """The vector c with H_loc(at) = c . log10 MSD: the slope of tamsd.local_h on the lags within
    +- tamsd.HALF_DECADE of `at`, halved."""
    x = np.log10(lags)
    near = np.abs(x - np.log10(at)) <= tamsd.HALF_DECADE + 1e-9
    c = np.zeros(len(lags))
    c[near] = (x[near] - x[near].mean()) / np.sum((x[near] - x[near].mean()) ** 2) / 2
    return c


def flight_msd(flight, sums, origins):
    """Per flight with pieces: its MSD at LAGS pooling the origins of its pieces."""
    f, k = np.unique(flight, return_inverse=True)
    s, o = np.zeros((len(f), sums.shape[1])), np.zeros((len(f), sums.shape[1]))
    np.add.at(s, k, sums)
    np.add.at(o, k, origins)
    return f, s / o


def weighted_mean(values, cell, counts, chunk=250):
    """The mean of the rows of `values` weighted by 1 (row 0 of the result) and by the count of their
    cell in each draw (`counts`: draws x cells); the rows after: (1 + draws, columns)."""
    out = [np.asarray(values).mean(axis=0, keepdims=True)]
    for a in range(0, len(counts), chunk):
        w = counts[a:a + chunk][:, cell].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append((w @ values) / w.sum(axis=1, keepdims=True))
    return np.vstack(out)


def cell_hloc(f, msd, code, counts):
    """H_loc(AT) of a cell (flights `f`, their MSD): the point estimate and the draws (1 + draws,)."""
    return np.log10(weighted_mean(msd, code[f], counts)) @ hloc_weights()


def within_flight(fp, mp, fq, mq, code, counts):
    """The difference of H_loc(AT) within the flights that have both cells: the slope of the mean over
    those flights of log10 MSD_P - log10 MSD_Q; (1 + draws,) and the number of flights."""
    both, ip, iq = np.intersect1d(fp, fq, return_indices=True)
    if not len(both):
        return np.full(1 + len(counts), np.nan), 0
    return weighted_mean(np.log10(mp[ip]) - np.log10(mq[iq]), code[both], counts) @ hloc_weights(), len(both)


def quote(rows):
    """twoway.quote of a (1 + draws, ...) array, NaN-safe: value and 90% band, or None."""
    rows = np.asarray(rows, float)
    if not np.isfinite(rows[0]).all():
        return None
    return twoway.quote(rows)


def cell_counts(flight, code, pilot):
    """Pieces, flights and pilots of a cell: `flight` is the flight of each of its pieces."""
    f = np.unique(flight)
    return {"pieces": int(len(flight)), "flights": int(len(f)), "pilots": int(len(np.unique(pilot[code[f]])))}


def hloc_cells(pc, cls, code, pilot, counts):
    """The cells (state, class) of the pieces `pc` (a dict of arrays: flight, state, sums, origins, use)
    with classes `cls`: counts and H_loc(AT) of each, and the contrasts of CONTRASTS, raw and within the
    flight, with the bands of the draws `counts` of the cells of `code` (flight -> cell)."""
    msd, out = {}, {"cells": {}, "contrasts": {}}
    for state in STATES:
        for c in CLASSES:
            m = (pc["state"] == state) & (cls == c) & pc["use"]
            f, v = flight_msd(pc["flight"][m], pc["sums"][m][:, :len(LAGS)], pc["origins"][m][:, :len(LAGS)])
            msd[state, c] = (f, v)
            out["cells"].setdefault(state, {})[c] = {**cell_counts(pc["flight"][m], code, pilot),
                                                     "H_loc": quote(cell_hloc(f, v, code, counts)) if len(f) else None}
    for name, (P, Q) in CONTRASTS.items():
        res = {}
        if len(msd[P][0]) and len(msd[Q][0]):
            res["raw"] = quote(cell_hloc(*msd[P], code, counts) - cell_hloc(*msd[Q], code, counts))
            d, n = within_flight(*msd[P], *msd[Q], code, counts)
            res["flight"], res["flights both"] = quote(d), n
        out["contrasts"][name] = res
    out["classes"] = {c: {s: int(((pc["state"] == s) & (cls == c) & pc["use"]).sum()) for s in STATES} for c in CLASSES}
    return out


def step_b(pc, tables, defs, code, pilot, counts, radius):
    """hloc_cells() for the classes at `radius`, and the phase statistics of PHASE_STATS of each cell
    and contrast (raw.phase_statistics; within the flight, phase_within)."""
    cls = neighbourhood(pc[f"f_a_{radius:g}"], pc[f"f_b_{radius:g}"])
    out, phase = hloc_cells(pc, cls, code, pilot, counts), {}
    for state in STATES:
        for c in CLASSES:
            m = (pc["state"] == state) & (cls == c) & pc["use"]
            if m.any():
                phase[state, c] = raw.phase_statistics(tables, defs, (pc["flight"][m], pc["start"][m], pc["start"][m] + pc["duration"][m]),
                                                       pd.Series(code), counts, PHASE_STATS)
                out["cells"][state][c]["phase"] = {n: phase[state, c]["average first"][n] for n in PHASE_STATS}
    for name, (P, Q) in CONTRASTS.items():
        if P in phase and Q in phase:
            out["contrasts"][name]["phase raw"] = {n: quote(phase[P]["draws"][:, j] - phase[Q]["draws"][:, j])
                                                   for j, n in enumerate(PHASE_STATS)}
            out["contrasts"][name]["phase flight"] = phase_within(tables, defs, pc, cls, P, Q, code, counts)
    return out


def phase_within(tables, defs, pc, cls, P, Q, code, counts):
    """The phase statistics of PHASE_STATS within the flight: for the flights with phase time in both
    cells P and Q, the mean over them of the difference of the flight's statistic (ratio of its sums)."""
    stats = {}
    for cell in (P, Q):
        m = (pc["state"] == cell[0]) & (cls == cell[1]) & pc["use"]
        time, sums = raw.unit_terms(tables, defs, (pc["flight"][m], pc["start"][m], pc["start"][m] + pc["duration"][m]))
        value = {}
        for src, d in defs.items():
            for n, (num, den, scale, _) in d.items():
                if n in PHASE_STATS:
                    value[n] = scale * (sums[src][num] / sums[src][den].where(sums[src][den] > 0)).reindex(time.index[time > 0])
        stats[cell] = pd.DataFrame(value)
    both = stats[P].index.intersection(stats[Q].index)
    diff = (stats[P].loc[both, list(PHASE_STATS)] - stats[Q].loc[both, list(PHASE_STATS)]).to_numpy()
    out = {}
    for j, n in enumerate(PHASE_STATS):
        ok = np.isfinite(diff[:, j])
        out[n] = quote(weighted_mean(diff[ok, j:j + 1], code[both.to_numpy()[ok]], counts)[:, 0]) if ok.sum() > 1 else None
        out[n] = out[n] and {**out[n], "flights": int(ok.sum())}
    return out


def bin_of(d, edges=BINS):
    """The bin of each gap: i for edges[i] <= d < edges[i+1], len(edges) - 1 for "none" (d >= the last
    edge, inf)."""
    return np.minimum(np.searchsorted(edges, d, "right") - 1, len(edges) - 1)


def event_curves(ev, code, counts):
    """For events with columns flight, state (0 solo, 1 group), d_a, d_b and the terms num and den of
    one statistic (arrays): per state, side ("a", "b") and bin of BINS (plus none) the ratio
    sum(num) / sum(den) of the events whose gap falls in the bin, with its band; "<state> a-b" is the
    difference of the two sides, bin by bin. The ratio of sums, each cell of the bootstrap weighing its
    events."""
    nb = len(BINS)
    ratio, n = {}, {}
    for side in ("a", "b"):
        k = ev["state"] * nb + bin_of(ev[f"d_{side}"])
        tot = np.zeros((len(counts[0]), 2 * nb, 2))
        np.add.at(tot, (code[ev["flight"]], k), np.column_stack([ev["num"], ev["den"]]))
        n[side] = np.bincount(k, minlength=2 * nb)
        num = np.vstack([tot[..., 0].sum(0), counts.astype(float) @ tot[..., 0]])
        den = np.vstack([tot[..., 1].sum(0), counts.astype(float) @ tot[..., 1]])
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio[side] = num / den
    out = {}
    for s, state in enumerate(STATES):
        cols = slice(s * nb, (s + 1) * nb)
        for side in ("a", "b"):
            out[f"{state} {side}"] = [{"bin": i, "events": int(n[side][s * nb + i]), "estimate": quote(ratio[side][:, s * nb + i])}
                                      for i in range(nb)]
        diff = ratio["a"][:, cols] - ratio["b"][:, cols]
        out[f"{state} a-b"] = [quote(diff[:, i]) for i in range(nb)]
    return out


def hloc_by_gap(pc, code, pilot, counts):
    """H_loc(AT) of the usable pieces by the bin of H_BINS of their median gap ahead (side "a") and
    behind ("b") and state, with the difference of the two sides bin by bin."""
    out, nb = {}, len(H_BINS)
    cells = {}
    for side in ("a", "b"):
        pc[f"bin_{side}"] = np.where(np.isnan(pc[f"med_{side}"]), -1, bin_of(pc[f"med_{side}"], H_BINS))
    for state in STATES:
        for side in ("a", "b"):
            rows = []
            for i in range(nb):
                m = (pc["state"] == state) & (pc[f"bin_{side}"] == i) & pc["use"]
                f, v = flight_msd(pc["flight"][m], pc["sums"][m][:, :len(LAGS)], pc["origins"][m][:, :len(LAGS)])
                cells[state, side, i] = (f, v)
                rows.append({"bin": i, **cell_counts(pc["flight"][m], code, pilot), "H_loc": quote(cell_hloc(f, v, code, counts)) if len(f) else None})
            out[f"{state} {side}"] = rows
        out[f"{state} a-b"] = [quote(cell_hloc(*cells[state, "a", i], code, counts) - cell_hloc(*cells[state, "b", i], code, counts))
                               if len(cells[state, "a", i][0]) and len(cells[state, "b", i][0]) else None for i in range(nb)]
    return out


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def share_within(d, defined, radii=RADII):
    """The share of the minutes with a direction that have a gap of at most each radius."""
    n = max(int(defined.sum()), 1)
    return {f"{r:g}": float((defined & (d <= r)).sum() / n) for r in radii}


def quantiles(x):
    """Median and quantiles 10 and 90 of the finite values of x, and the share of x that is infinite."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    q = np.percentile(x[ok], [10, 50, 90]).tolist() if ok.any() else [None] * 3
    return {"q10": q[0], "median": q[1], "q90": q[2], "none": float(1 - ok.mean()) if len(x) else None}


def step_a(g, pieces_min):
    """The geometry of the groups (rows `g` of group_geometry) and the shares of time with an outsider
    ahead within each radius, for group time (by group minute, from the leader), and for the minutes of
    the pieces of G, A and S (from the pilot), `pieces_min`: {name: (d_a, defined)}."""
    out = {"group minutes": int(len(g["gid"])), "groups": int(len(np.unique(g["gid"]))),
           "length": quantiles(g["length"]), "width": quantiles(g["width"]),
           "gap ahead": quantiles(g["d_a"]), "gap ahead, other group": quantiles(g["d_ag"]),
           "gap ahead, solo": quantiles(g["d_as"]), "cross-track": quantiles(g["xt"][np.isfinite(g["xt"])]),
           "share, group minutes": share_within(g["d_a"], np.ones(len(g["d_a"]), bool)),
           "share, group minutes, other group": share_within(g["d_ag"], np.ones(len(g["d_a"]), bool)),
           "share, group minutes, solo": share_within(g["d_as"], np.ones(len(g["d_a"]), bool)),
           "size": {int(k): int(v) for k, v in zip(*np.unique(np.minimum(g["n"], 7), return_counts=True))}}
    for name, (d, defined) in pieces_min.items():
        out[f"share, {name} minutes"] = {**share_within(d, defined), "minutes": int(defined.sum())}
    return out


def light_tables(flights):
    """The terms of isolated.run_terms ("b": budget.py, "p": performance.py) for the flights of `flights`
    only, read with a filter on the flight so that the other runs are never in memory."""
    ids = flights.loc[flights["discipline"] == "para", ["flight_id", "flight"]].astype({"flight_id": str})
    keep = ids["flight_id"].tolist()
    runs = pd.read_parquet(RUNS["para"], columns=["flight_id", "segment_id", "phase", "t_start", "t_end", "n_fixes"],
                           filters=[("flight_id", "in", keep)]).astype({"flight_id": str}).merge(ids, on="flight_id")
    b = budget.columns(runs)
    b = b.assign(t_mid=(b["t_start"] + b["t_end"]) / 2, search_s=b["found_s"] + b["vain_s"])
    p = pd.read_parquet(CACHE / "performance_terms.parquet", filters=[("flight", "in", [int(f) for f in ids["flight"]])])
    return {"b": b, "p": p[p["xc_origins"] == 0]}


def events(flights, membership, summary, utc0):
    """The searches and the glides of the open paragliders with the minute of their start, and the
    phase tables of those flights: ({"search": frame, "glide": frame}, tables). A search carries
    num = found, den = 1; a glide chord, time and height lost."""
    spans = tamsd.stretches(flights, membership, summary)
    tables = light_tables(flights)
    b = survival.assign(tables["b"][tables["b"]["phase"] == "search"], spans)
    g = tables["p"][tables["p"]["transition_s"] > 0]
    ev = {"search": pd.DataFrame({"flight": b["flight"], "start": b["t_start"], "population": b["population"],
                                  "num": b["found"], "den": b["searches"]}),
          "glide": pd.DataFrame({"flight": g["flight"], "start": g["t_mid"] - g["transition_s"] / 2, "population": g["population"],
                                 "num": g["chord_m"], "den": g["transition_s"], "loss": g["loss_m"]})}
    for e in ev.values():
        e["minute"] = np.round((utc0[e["flight"].to_numpy()] + e["start"].to_numpy()) / STEP_S).astype(np.int64)
    return ev, tables


def option(name):
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else 0


if __name__ == "__main__":
    assert "--open" in sys.argv, "this analysis is on the open flights: pass --open"
    sample = option("--sample")
    tag = f"_sample{sample}" if sample else ""
    tamsd.folders()
    flights = with_takeoffs(read_flights())
    everything = pd.read_parquet(LABELS / "flights.parquet")
    membership = pd.read_parquet(LABELS / "membership.parquet")
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    groups = pd.read_parquet(LABELS / "groups.parquet")
    size = everything["flight"].max() + 1
    utc0 = np.zeros(size)
    utc0[everything["flight"].to_numpy()] = everything["utc0"].to_numpy()
    p = dict(np.load(longlags.PIECES))
    keep = longlags.long_enough(p, np.isin(p["population"], ["solo", "group", "group_solo"]), tamsd.TAU_MAX)
    p = {k: v[keep] for k, v in p.items()}
    p["state"] = np.where(p["population"] == "group", "group", "solo")
    n_pieces = len(p["flight"])
    piece, minute = piece_minutes(utc0, p["flight"], p["start"], p["duration"])
    ev, tables = events(flights, membership, summary, utc0)
    keys = np.unique(np.concatenate([p["flight"][piece] * KEY + minute] + [e["flight"].to_numpy(np.int64) * KEY + e["minute"].to_numpy() for e in ev.values()]))
    keys = keys[np.argsort(keys % KEY, kind="stable")]  # by minute, then flight (keys were sorted by flight)
    qflight, qminute = keys // KEY, keys % KEY
    days = np.unique(qminute // DAY)
    if sample:
        days = np.sort(np.random.default_rng(SEED).choice(days, min(sample, len(days)), replace=False))
    wanted = np.zeros(groups["group_id"].max() + 1, bool)
    wanted[groups["group_id"].to_numpy()] = [bool(set(flights["flight"]).intersection(m)) for m in groups["members"]]
    cache = CACHE / f"intergroup_geometry{tag}.npz"
    if not cache.exists():
        index = isolated.read_index(everything)
        res, g = run_days(index, membership_table(membership), wanted, qflight, qminute, days)
        del index
        np.savez(cache, **{f"q_{k}": v for k, v in res.items()}, **{f"g_{k}": v for k, v in g.items()})
    saved = np.load(cache)
    res = {k[2:]: saved[k] for k in saved.files if k.startswith("q_")}
    g = {k[2:]: saved[k] for k in saved.files if k.startswith("g_")}
    order = qminute * KEY + qflight  # sorted: queries are by minute

    def look(f, m):
        """The result rows of (flight, minute) pairs: gaps, and whether the minute was measured."""
        w = np.minimum(np.searchsorted(order, m * KEY + f), len(order) - 1)
        ok = (order[w] == m * KEY + f) & res["found"][w] & res["defined"][w]
        return w, ok

    w, defined = look(p["flight"][piece], minute)
    d_a, d_b = (res[k][w].astype(float) for k in ("d_a", "d_b"))
    p.update(piece_summary(piece, n_pieces, d_a, d_b, defined))
    p["use"] = p["defined"] >= MIN_DEFINED
    if sample:  # only the pieces whose minutes all lie in the days of the sample
        p["use"] &= np.bincount(piece, weights=~np.isin(minute // DAY, days), minlength=n_pieces) == 0
    print("pieces", n_pieces, "usable", int(p["use"].sum()), file=sys.stderr, flush=True)
    code_s, of_pilot, of_block = twoway.cells(flights, summary)
    code = np.full(size, -1, np.int64)
    code[code_s.index.to_numpy()] = code_s.to_numpy()
    counts = twoway.counts(of_pilot, of_block, np.random.default_rng(SEED))
    defs = raw.stat_defs()
    result = {"params": {"T_DIR_S": T_DIR_S, "MIN_DISP_M": MIN_DISP_M, "R_MAX": R_MAX, "F_LOW": F_LOW, "F_HIGH": F_HIGH,
                         "MIN_DEFINED": MIN_DEFINED, "RADII": RADII, "R_MAIN": R_MAIN, "BINS": BINS, "H_BINS": H_BINS,
                         "sample days": sample or None, "pieces": n_pieces, "pieces used": int(p["use"].sum())}}
    mins = {}
    for name, pop in (("G pieces", "group"), ("A", "group_solo"), ("S", "solo")):
        mm = (p["population"] == pop)[piece] & p["use"][piece]
        mins[name] = (d_a[mm], defined[mm])
    result["step A"] = step_a(g, mins)
    result["step B"] = {f"{r:g}": step_b(p, tables, defs, code, of_pilot, counts, r) for r in RADII}
    result["step C"] = {"hloc": hloc_by_gap(p, code, of_pilot, counts), "events": {}}
    for name, e in ev.items():
        w, ok = look(e["flight"].to_numpy(np.int64), e["minute"].to_numpy())
        if sample:
            ok &= np.isin(e["minute"].to_numpy() // DAY, days)
        sel = {"flight": e["flight"].to_numpy()[ok], "state": (e["population"].to_numpy()[ok] == "group").astype(int),
               "d_a": res["d_a"][w][ok].astype(float), "d_b": res["d_b"][w][ok].astype(float)}
        terms = {"search found": ("num", "den")} if name == "search" else {"glide: chord over time": ("num", "den"), "glide: loss over chord": ("loss", "num")}
        for stat, (a, b) in terms.items():
            result["step C"]["events"][stat] = event_curves({**sel, "num": e[a].to_numpy(float)[ok], "den": e[b].to_numpy(float)[ok]},
                                                            code, counts)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"intergroup{tag}.json").write_text(json.dumps(jsonable(result), indent=1))
    print("done", file=sys.stderr)
