"""Marking: do pilots start climbing where a glider outside their group is already climbing? Between groups.

Population: the climb starts of the labelled flights (1 Hz with phases) of each date: the first
instant of each climb run (rotation.climb_runs), with the position there. The runs and the
positions are read on the 5 s grid of the run, which is enough for a place to within a few tens of
m and a time to 5 s. A pilot is a group member at an instant if the flight belongs to a group of
groups.parquet then (t - DT, see below), otherwise solo.

Markers. The other labelled flights of the date that are not in the pilot's group at t - DT, the
instant before the window (nor the pilot itself): a glider that comes into contact with the pilot
only after that is a case of marking, while one in contact at t would be excluded exactly when it
climbs in the same thermal. The pilot's kind is its state at t - DT for the same reason; a start
within DT of the first instant of its flight has no such instant and is dropped (counted in
"measured"). A marker is "of another group" at an instant if it is in a group then, else "solo".
The start at t is marked if a marker is in a climb run at some instant of [t - DT, t] at a
horizontal distance of at most R from the pilot's position at t (positions on the ellipsoid, h = 0).

Constants, the 90th percentiles over every pair of climbers of a group thermal (exits.py: groups of
n >= 2, every member labelled, at least MIN_GROUP s) of the horizontal distance between their
entries (first climb instant in the thermal, each at its own position) and of the delay between them:
the scale of "the same thermal". They are measured in each run and written in marking.json.

Null. The markers' tracks and climb runs shifted in time by +s and by -s (both, averaged), the pilot
unchanged, for the s of S_GRID_MIN (s = 0 is the real share). A marker shifted outside its flight
has no fix and marks nothing; whether it is of another group or solo is read at its own climbing
instant. The share against s is kept for every kind of pilot and marker. Two quantities summarise it:
- "now": the real share over the share at S_NEAR, the smallest s of S_GRID_MIN with s >= DT, whose
  window [t - DT, t] does not overlap the real one. It says how much a glider climbing right then
  adds over one that climbed at the same place a few minutes before or after; the markers leaving
  their flights hardly dilute the share at so small a shift.
- "persistence": s_half, the s at which the share falls to half its real value (linear interpolation
  on S_GRID_MIN; NaN if it does not within the grid). The share keeps falling at large s because
  the shifted markers also leave their flights and the thermals have a daily cycle, so s_half is
  an upper bound on how long a place keeps producing climbs.
Measured on the full run: R = 829 m, DT = 152 s, S_NEAR = 5 min.

Bands: 5-95% of N_BOOT draws of the dates, which are the dependency block here (a thermal day
shared by the pilots of one date). Excess over the null is marking or two pilots finding one active
thermal; the data do not separate them.

    uv run python marking.py [--run NAME] [--sample N]    # from this folder, after cohesion.py
"""

import argparse
import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from breathing import MIN_GROUP, N_BOOT, NO_RUNS, band, observed, stat
from cohesion import contacts, grid
from exits import NO_RUNS as NO_CLIMB, climbers, exits, thermals
from rotation import climb_runs, climbing
from tamsd import LABELS, OUT, SEED, save

PERCENTILE = 90  # of the distance and of the delay between entries: R and DT
S_GRID_MIN = (0, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120)  # shifts of the null, min
KINDS = ("group", "solo")  # kind of pilot
LABEL = {"group": "group members", "solo": "solo pilots"}
TYPES = ("group", "solo", "any")  # kind of marker
TYPE_LABEL = {"group": "another group", "solo": "solo glider", "any": "any"}
COLORS = {"group": "#eb6834", "solo": "#2a78d6"}
CHUNK = 200  # starts handled together
CODE = 1 << 20  # group * CODE + flight codes a member; flight numbers are below it


def entry_pairs(t, xyz):
    """The distance (m) and the delay (s) between the entries of every pair of climbers: entry times
    `t` (k,) and positions `xyz` (k, 3)."""
    i, j = np.triu_indices(len(t), 1)
    return np.linalg.norm(xyz[i] - xyz[j], axis=1), np.abs(t[i] - t[j])


def measure_entries(groups, flights, step, p):
    """The distances and the delays between the entries of the pairs of climbers of every group
    thermal of `groups` (exits.thermals), each climber at its own entry, for the pairs whose two
    entries have a position."""
    runs = climb_runs(flights, {f for m in groups["members"] for f in m})
    dist, delay = [], []
    for group, real, flight_of, tracks, row_of in observed(groups, flights, p):
        t = group.t_start + step * np.arange((group.t_end - group.t_start) // step)
        z = np.stack([tracks.at(row_of[f], t)[:, 3] if row_of.get(f) in tracks.values else np.full(len(t), np.nan)
                      for f in group.members])
        f = climbing(group, step, runs).mean(axis=1)
        for g0, a, e, g1 in thermals(f, step):
            got = np.array([exits(t, z[k], runs.get(m, NO_CLIMB), a, e) for k, m in enumerate(group.members)])
            c = climbers({"exit_z": got[:, 3]})
            if len(c) > 1:
                entry = (np.round(got[c, 0] / step) * step).astype(np.int64)
                xyz = np.stack([tracks.at(row_of[group.members[k]], np.array([s]))[0, :3] for k, s in zip(c, entry)])
                d, w = entry_pairs(got[c, 0], xyz)
                dist.append(d)
                delay.append(w)
    dist, delay = np.concatenate(dist), np.concatenate(delay)
    return dist[np.isfinite(dist)], delay[np.isfinite(dist)]  # an entry without a position gives no distance


def marked(start, point, codes, r, dt, shifts):
    """Whether each start is marked, for each shift: an array (starts, 3, shifts) of 0, 1/2 and 1
    (the mean over +s and -s) for the markers of another group, solo markers and any. `start` has
    t, xyz, flight and group (-1 solo) of the starts; `point` has t, xyz, flight and grouped of the
    climbing fixes of all the flights; `codes` are the group * CODE + flight of the members of the
    groups. A marker is a flight other than the pilot and not in its group; its fix at time tau is
    at tau + s once shifted by +s, and counts if that lies in [t - dt, t] and its xyz is within
    r of the start."""
    tree = cKDTree(point.xyz)
    out = np.zeros((len(start.t), 3, len(shifts)))
    reach = (-max(shifts), max(shifts) + dt)  # t - tau of the fixes that any shift can use
    for a in range(0, len(start.t), CHUNK):
        rows = np.arange(a, min(a + CHUNK, len(start.t)))
        near = tree.query_ball_point(start.xyz[rows], r)
        i = np.repeat(rows, [len(x) for x in near])
        j = np.concatenate(near).astype(int)
        late = start.t[i] - point.t[j]  # t - tau
        ok = (late >= reach[0]) & (late <= reach[1]) & (point.flight[j] != start.flight[i])
        ok[ok] = ~np.isin(start.group[i[ok]] * CODE + point.flight[j[ok]], codes)
        i, j, late = i[ok], j[ok], late[ok]
        kind = np.where(point.grouped[j], 0, 1)
        for k, s in enumerate(shifts):
            for sign in (s, -s):
                hit = (late - sign >= 0) & (late - sign <= dt)
                seen = np.zeros((len(start.t), 2), bool)
                seen[i[hit], kind[hit]] = True
                out[:, :2, k] += seen / 2
                out[:, 2, k] += seen.any(axis=1) / 2
    return out


def spans_of(day):
    """By flight, the array (k, 3) of the (t_start, t_end, group_id) of its groups of a date."""
    rows = [(f, g.t_start, g.t_end, g.group_id) for g in day.itertuples() for f in g.members]
    out = {}
    for f, a, b, i in rows:
        out.setdefault(f, []).append((a, b, i))
    return {f: np.array(v) for f, v in out.items()}


def group_at(t, spans):
    """The group id of a flight with `spans` at each time of `t` (-1 if none)."""
    out = np.full(len(t), -1)
    for a, b, i in spans:
        out[(t >= a) & (t <= b)] = int(i)
    return out


def climb_fixes(run, track, step):
    """The climb starts and the climbing fixes of a flight: the times (grid instants) and positions
    of the first instant of each run `run` (starts, ends) and of every grid instant inside the runs,
    where the flight has a position. `track(t)` gives the position (len(t), 3)."""
    start, end = run
    first = (np.round(start / step) * step).astype(np.int64)
    inside = np.concatenate([np.arange(np.ceil(a / step), np.ceil(b / step), dtype=np.int64) * step for a, b in zip(start, end)])
    return [(t, xyz[np.isfinite(xyz[:, 0])], np.isfinite(xyz[:, 0])) for t, xyz in ((x, track(x)) for x in (first, inside))]


def one_date(day, mine, runs, tracks, row_of, r, dt, shifts, step):
    """The marking of the climb starts of one date: the number of starts of group members and of
    solo pilots (2,) and the marks summed over them (2, 3, shifts); also the starts without a
    position and those dropped because the flight starts after t - dt (2,). The kind of the pilot
    and the group that excludes markers are read at t - dt. `day` are the groups of the date,
    `mine` its labelled flights."""
    spans = spans_of(day)
    codes = np.array([g.group_id * CODE + f for g in day.itertuples() for f in g.members], np.int64)
    starts, points = [], []
    lost = np.zeros(2, int)
    for f in mine["flight"]:
        if f not in runs or row_of.get(f) not in tracks.values:
            continue
        (t0, x0, ok0), (t1, x1, ok1) = climb_fixes(runs[f], lambda t: tracks.at(row_of[f], t)[:, :3], step)
        t0 = t0[ok0]
        keep = t0 - dt >= tracks.first[row_of[f]]  # the flight exists at t - dt
        lost += (~ok0).sum(), (~keep).sum()
        group = group_at(t0[keep] - dt, spans.get(f, ()))
        starts.append(SimpleNamespace(t=t0[keep], xyz=x0[keep], flight=np.full(keep.sum(), f), group=group))
        grouped = group_at(t1[ok1], spans.get(f, ())) >= 0
        points.append(SimpleNamespace(t=t1[ok1], xyz=x1, flight=np.full(ok1.sum(), f), grouped=grouped))
    n, marks = np.zeros(2), np.zeros((2, 3, len(shifts)))
    if not starts or not any(len(p.t) for p in points):
        return n, marks, lost
    cat = lambda xs, name: np.concatenate([getattr(x, name) for x in xs])
    start = SimpleNamespace(**{k: cat(starts, k) for k in ("t", "xyz", "flight", "group")})
    point = SimpleNamespace(**{k: cat(points, k) for k in ("t", "xyz", "flight", "grouped")})
    m = marked(start, point, codes, r, dt, shifts)
    for k, mask in enumerate((start.group >= 0, start.group < 0)):
        n[k], marks[k] = mask.sum(), m[mask].sum(axis=0)
    return n, marks, lost


def measure(flights, groups, dates, step, p, r, dt, shifts):
    """one_date for each date: the starts (dates, 2), the marks (dates, 2, 3, shifts) and the
    starts without a position and dropped (2,)."""
    runs = climb_runs(flights, set(flights.loc[flights["labelled"] & flights["date"].isin(dates), "flight"]))
    archive = grid.archive()
    by_date = dict(tuple(archive[archive["date"].isin(dates)].groupby("date")))
    flights_of = dict(tuple(flights[flights["date"].isin(dates)].groupby("date")))
    by_day = dict(tuple(groups.groupby("date")))
    n, marks, lost = [], [], np.zeros(2, int)
    for number, date in enumerate(dates):
        mine = flights_of[date]
        mine = mine[mine["labelled"]]
        flight_of = dict(zip(mine["flight_id"], mine["flight"]))
        a = by_date[date].merge(mine[["discipline", "flight_id"]], on=["discipline", "flight_id"])
        table, g = grid.day(a, {d: NO_RUNS for d in grid.DERIVED}, p["step"], p["h_max"], p["r_dep"])
        if g is None:
            n.append(np.zeros(2))
            marks.append(np.zeros((2, 3, len(shifts))))
            continue
        tracks = contacts.Tracks(g, p["step"])
        row_of = {flight_of[fid]: k for k, fid in enumerate(table["flight_id"])}
        day = by_day.get(date, groups.iloc[:0])
        d, m, l = one_date(day, mine, runs, tracks, row_of, r, dt, shifts, step)
        n.append(d)
        marks.append(m)
        lost += l
        if number % 100 == 0:
            print(f"{number}/{len(dates)} dates, {int(np.sum(n))} starts", flush=True)
    return np.array(n), np.array(marks), lost


def half_shift(share, shifts_min):
    """The first s (min) at which `share` (..., shifts) falls to half its value at s = 0, by linear
    interpolation between the s of the grid; NaN where it does not within the grid."""
    s = np.asarray(shifts_min, float)
    r = share / share[..., :1]
    below = r <= 0.5
    k = np.maximum(below.argmax(axis=-1), 1)[..., None]
    r0, r1 = (np.take_along_axis(r, k + d, axis=-1)[..., 0] for d in (-1, 0))
    s0, s1 = s[k[..., 0] - 1], s[k[..., 0]]
    return np.where(below.any(axis=-1), s0 + (0.5 - r0) / (r1 - r0) * (s1 - s0), np.nan)


def summarize(n, marks, lost, dt, rng, shifts_min=S_GRID_MIN):
    """The curves (share of marked starts against s) and, by kind of pilot and kind of marker, the
    real share, the share at S_NEAR (the smallest s with 60 s >= dt), their ratio ("now") and s_half
    (see the header), with their bands. `n` (dates, 2), `marks` (dates, 2, 3, shifts)."""
    np.seterr(divide="ignore", invalid="ignore")
    w = rng.multinomial(len(n), np.full(len(n), 1 / len(n)), size=N_BOOT).astype(float)
    n = np.concatenate([n, n.sum(axis=1, keepdims=True)], axis=1)
    marks = np.concatenate([marks, marks.sum(axis=1, keepdims=True)], axis=1)
    kinds = (*KINDS, "all")
    value = marks.sum(axis=0) / n.sum(axis=0)[:, None, None]
    draws = np.einsum("bd,dkts->bkts", w, marks) / (w @ n)[:, :, None, None]
    near = next(c for c, s in enumerate(shifts_min) if 60 * s >= dt)
    half, half_draws = half_shift(value, shifts_min), half_shift(draws, shifts_min)
    out = {"S_near_min": shifts_min[near], "no_position": int(lost[0]), "dates": len(n),
           "curve": {k: {t: [stat(float(value[i, j, c]), draws[:, i, j, c]) for c in range(len(shifts_min))]
                         for j, t in enumerate(TYPES)} for i, k in enumerate(kinds)},
           "summary": {}}
    for i, k in enumerate(kinds):
        out["summary"][k] = {"starts": int(n[:, i].sum())}
        for j, t in enumerate(TYPES):
            out["summary"][k][t] = {"real": stat(float(value[i, j, 0]), draws[:, i, j, 0]),
                                    "near": stat(float(value[i, j, near]), draws[:, i, j, near]),
                                    "now": stat(float(value[i, j, 0] / value[i, j, near]), draws[:, i, j, 0] / draws[:, i, j, near]),
                                    "s_half": stat(float(half[i, j]), half_draws[:, i, j])}
    return out


def plot(r, path):
    """Left: share of marked starts (any marker) against the shift s, for group members and solo
    pilots, with the line at S_NEAR and the half of the real share (dotted) where the share crosses
    it; right: the "now" ratio, real share over the share at S_NEAR, by kind of marker."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.9))
    s = np.array(r["params"]["shifts_min"])
    for k in KINDS:
        c = r["curve"][k]["any"]
        a.fill_between(s, *np.array([x["band"] for x in c]).T, color=COLORS[k], alpha=0.15, lw=0)
        a.plot(s, [x["value"] for x in c], color=COLORS[k], marker="o", ms=3, lw=1.2, label=LABEL[k])
        half = c[0]["value"] / 2
        a.axhline(half, color=COLORS[k], ls=":", lw=0.8)
        a.plot(r["summary"][k]["any"]["s_half"]["value"], half, "x", color=COLORS[k], ms=7)
    a.axvline(r["S_near_min"], color="#999999", ls="--", lw=0.8)
    a.set(xlabel="shift of the markers, $s$ (min)", ylabel="share of climb starts marked", ylim=(0, None))
    a.legend(frameon=False, fontsize=8)
    for i, k in enumerate(KINDS):
        for j, t in enumerate(TYPES):
            x = r["summary"][k][t]["now"]
            b.errorbar(j + (i - 0.5) * 0.2, x["value"], yerr=[[x["value"] - x["band"][0]], [x["band"][1] - x["value"]]],
                       fmt="o", ms=4, lw=1.2, color=COLORS[k], label=LABEL[k] if j == 0 else None)
    b.axhline(1, color="#999999", lw=0.6)
    b.set_xticks(range(len(TYPES)), [TYPE_LABEL[t] for t in TYPES])
    b.set(xlabel="markers", ylabel=f"real / share at $s$ = {r['S_near_min']:g} min")
    b.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="read by tamsd")
    parser.add_argument("--sample", type=int, help="a random sample of dates (seed 0)")
    parser.add_argument("--redraw", action="store_true", help="summary and figure again, from the saved pieces")
    args = parser.parse_args()
    name = "marking" + (f"_sample{args.sample}" if args.sample else "")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.redraw:
        saved = np.load(OUT / f"{name}_pieces.npz")
        n, marks, lost, dist, delay = (saved[k] for k in ("n", "marks", "lost", "dist", "delay"))
    else:
        params = json.loads((LABELS / "days" / "params.json").read_text())
        step = params["step"]
        flights = pd.read_parquet(LABELS / "flights.parquet")
        groups = pd.read_parquet(LABELS / "groups.parquet")
        labelled = flights.set_index("flight")["labelled"]
        counts = flights[flights["labelled"]].groupby("date").size()
        dates = sorted(counts[counts >= 2].index)
        if args.sample:
            dates = sorted(np.random.default_rng(0).choice(dates, args.sample, replace=False))
        of_dates = groups["date"].isin(dates)
        scale = groups[of_dates & (groups["t_end"] - groups["t_start"] >= MIN_GROUP)]
        scale = scale[scale["members"].map(lambda m: bool(labelled.loc[list(m)].all()))].reset_index(drop=True)
        assert scale["date"].is_monotonic_increasing  # observed() goes date by date
        dist, delay = measure_entries(scale, flights, step, params)
        r, dt = (float(np.percentile(x, PERCENTILE)) for x in (dist, delay))
        print(f"{len(dist)} pairs of entries: R = {r:.0f} m, DT = {dt:.0f} s", flush=True)
        n, marks, lost = measure(flights, groups[of_dates], dates, step, params, r, dt, [60 * s for s in S_GRID_MIN])
        np.savez(OUT / f"{name}_pieces.npz", n=n, marks=marks, lost=lost, dist=dist, delay=delay)
    pct = (10, 25, 50, 75, 90, 95)
    r, dt = (float(np.percentile(x, PERCENTILE)) for x in (dist, delay))
    result = summarize(n, marks, lost, dt, np.random.default_rng(SEED))
    result["measured"] = {"dropped_starts": int(lost[1]), "pairs": len(dist), "percentiles": list(pct),
                          "distance_m": np.percentile(dist, pct).tolist(), "delay_s": np.percentile(delay, pct).tolist()}
    result["params"] = {"R_m": r, "DT_s": dt, "percentile": PERCENTILE, "shifts_min": list(S_GRID_MIN), "min_group_s": MIN_GROUP, "n_boot": N_BOOT}
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / f"{name}.pdf")
    print(json.dumps({"S_near_min": result["S_near_min"], "summary": result["summary"]}), flush=True)
