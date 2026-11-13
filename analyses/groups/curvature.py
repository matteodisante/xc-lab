"""Curvature: do the leaders of a thermal keep adjusting the circle? An analogue of Flack et al. 2018 (storks).

Population, group thermals, leaders (before, after, at random among the climbers) and bands: those of
exits.py. Measured on the native 1 Hz fixes of each member that climbs in a group thermal (E, N, z of the
cleaned flights, the interpolated fixes dropped), on the seconds inside its own climb runs there.

Per contiguous 1 Hz stretch (no gap of more than 1 s; shorter than the window dropped), a Savitzky-Golay
filter of order 3 and window SG_WINDOW s gives x', y', x'', y'' (deriv = 1, 2) and z'; the signed
curvature is kappa = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2), its rate d kappa / dt is np.gradient over the
stretch. An instant is dropped if the horizontal speed is below V_MIN at it or at a neighbour. A member
in a thermal has the median |d kappa / dt|, the median |kappa| (1 / radius), the mean climb rate z', over
its valid seconds, at least MIN_SECONDS of them. Two windows: "own", all the climb seconds of the member
in the thermal; "common", only the seconds where the leader and the follower both have a valid one.

Rows, one per follower-thermal: r = log10(m_follower / m_leader) of each measure; share of the rows where
the leader's |d kappa / dt| exceeds the follower's (r < 0), against the leader at random (0.5).
Sensitivity: the median r of |d kappa / dt| (leader before, own window) for other windows and V_MIN.
Centring: median |d kappa / dt| over the first FIRST_S s of the climb of a member (from the start of its
first run in the thermal), against after, over every climber of every thermal.

    uv run python curvature.py [--run NAME] [--sample N] [--replot]    # from this folder, after exits.py's inputs
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import savgol_filter

from breathing import MIN_GROUP, N_BOOT
from rotation import climb_runs
from exits import COLORS, LEADERS, N_NULL, climbers, measure, median, share
from tamsd import FIXES, LABELS, OUT, SEED, save

SG_WINDOW = 7  # s; a turn lasts about 20 s, a window of a third of it keeps the circle and averages the noise
SG_ORDER = 3  # lowest order that gives a second derivative with a curvature of its own
V_MIN = 3  # m/s; kappa blows up where the speed goes to 0
MIN_SECONDS = 30  # valid seconds for a member (or a pair) to count: a median of at least about one turn
FIRST_S = 60  # s; the "centring" part of a climb
MIN_PART = 20  # valid seconds in each part of a climb for the first-minute check
# (window, V_MIN) of the sensitivity check, the base case first
SENSITIVITY = ((SG_WINDOW, V_MIN), (5, V_MIN), (11, V_MIN), (SG_WINDOW, 2), (SG_WINDOW, 5))
PAD = max(w for w, _ in SENSITIVITY) // 2 + 2  # s of fixes read around a thermal so that the filter sees its ends
MEASURES = ("dk", "ka", "vz")  # |d kappa / dt|, |kappa|, climb rate
NAMES = {"dk": "$|d\\kappa/dt|$", "ka": "$|\\kappa|$", "vz": "climb rate"}
HIST_EDGES = np.arange(-1, 1.001, 0.05)  # bins of r in the figure
AGG = {"dk": np.median, "ka": np.median, "vz": np.mean}  # per member-thermal, over its valid seconds
SIZES = {"2": lambda n: n == 2, "3+": lambda n: n >= 3, "all": lambda n: n >= 2}
READ = ["flight_id", "segment_id", "t", "E", "N", "z", "interpolated"]


def rates(E, N, Z, seg, w, vertical=False):
    """Speed, signed curvature, its rate and (with `vertical`) the climb rate at each fix of a series
    with the stretches of constant `seg` and a 1 s step; NaN where a stretch is shorter than `w`."""
    out = np.full((4, len(E)), np.nan)
    cut = np.flatnonzero(np.r_[True, (np.diff(seg) != 0)])
    for a, b in zip(cut, np.r_[cut[1:], len(E)]):
        if b - a < w:
            continue
        d = lambda v, k: savgol_filter(v[a:b], w, SG_ORDER, deriv=k)
        x1, y1, x2, y2 = d(E, 1), d(N, 1), d(E, 2), d(N, 2)
        sp = np.hypot(x1, y1)
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = (x1 * y2 - y1 * x2) / sp ** 3
        out[:3, a:b] = sp, kappa, np.gradient(kappa)
        if vertical:
            out[3, a:b] = d(Z, 1)
    return out


def valid(speed, vmin):
    """Instants whose speed and that of both neighbours reach vmin."""
    ok = speed >= vmin
    ok[1:-1] &= ok[:-2] & ok[2:]
    ok[[0, -1]] = False
    return ok


def one_member(t, E, N, Z, seg, t0, t1, run, count):
    """The measures of one member in a thermal [t0, t1) from its fixes (1 Hz, UTC s `t`) and its climb
    `run` (starts, ends): None if too few valid seconds, else a dict (see measure_all); `count` adds up
    the climb seconds, those with a stretch, and those below V_MIN."""
    lo, hi = np.searchsorted(t, [t0 - PAD, t1 + PAD])
    t, E, N, Z, seg = t[lo:hi], E[lo:hi], N[lo:hi], Z[lo:hi], seg[lo:hi]
    start, end = run
    i = np.searchsorted(start, t, side="right") - 1
    inside = (t >= t0) & (t < t1) & (i >= 0) & (t < end[np.maximum(i, 0)])
    count["run"] += float(np.clip(np.minimum(end, t1) - np.maximum(start, t0), 0, None).sum())
    if len(t) == 0 or not inside.any():
        return None
    # a break is a gap of more than 1 s or a new segment
    seg = np.cumsum(np.r_[0, (np.diff(t) != 1) | (np.diff(seg) != 0)])
    medians, filtered = [], {}
    for w, vmin in SENSITIVITY:
        if w not in filtered:
            filtered[w] = rates(E, N, Z, seg, w, vertical=(w == SG_WINDOW))
        sp, ka, dk, vz = filtered[w]
        if not medians:
            known = inside & np.isfinite(sp)
            count["stretch"] += float(known.sum())
            count["slow"] += float((known & (sp < vmin)).sum())
        ok = inside & valid(sp, vmin)
        medians.append(np.median(np.abs(dk[ok])) if ok.sum() >= MIN_SECONDS else np.nan)
        if len(medians) == 1:
            base = ok, sp, np.abs(ka), np.abs(dk), vz
    ok, sp, ka, dk, vz = base
    if ok.sum() < MIN_SECONDS:
        return {"sens": np.array(medians)}
    sec = t[ok].astype(np.int32)
    return {"sens": np.array(medians), "sec": sec, "dk": dk[ok].astype(np.float32), "ka": ka[ok].astype(np.float32),
            "vz": vz[ok].astype(np.float32), "period": float(np.median(2 * np.pi / (ka[ok] * sp[ok])))}


def flight_stream(path, ids):
    """(flight_id, fixes) of the flights `ids` (sorted array of str) in a fixes.parquet whose row groups
    are ordered by flight_id; the row groups that hold none of them are not read."""
    file = pq.ParquetFile(path)
    meta = file.metadata
    column = [meta.row_group(0).column(i).path_in_schema for i in range(meta.row_group(0).num_columns)].index("flight_id")
    carry = None
    for g in range(meta.num_row_groups):
        s = meta.row_group(g).column(column).statistics
        if s is not None and np.searchsorted(ids, s.min) >= np.searchsorted(ids, s.max, side="right"):
            if carry is not None:
                yield carry
                carry = None
            continue
        frame = file.read_row_group(g, columns=READ).to_pandas()
        frame = frame[frame["flight_id"].isin(ids) & ~frame["interpolated"]]
        if carry is not None:
            frame = pd.concat([carry[1], frame], ignore_index=True)
            carry = None
        if frame.empty:
            continue
        cut = np.flatnonzero(frame["flight_id"].to_numpy()[1:] != frame["flight_id"].to_numpy()[:-1]) + 1
        parts = np.split(np.arange(len(frame)), cut)
        for k, p in enumerate(parts):
            piece = (frame["flight_id"].iat[p[0]], frame.iloc[p])
            if k == len(parts) - 1:
                carry = piece
            else:
                yield piece
    if carry is not None:
        yield carry


def measure_all(ths, flights, runs):
    """One entry per (thermal, climbing member), in order: the dict of one_member (None where the
    member has no fix or no valid seconds), plus the counters of the climb seconds."""
    info = flights.set_index("flight")
    tasks, order = {}, {}
    for i, r in enumerate(ths):
        for k in climbers(r):
            f = r["members"][k]
            order[(i, k)] = len(order)
            tasks.setdefault(f, []).append((i, k, order[(i, k)]))
    found = [None] * len(order)
    count = {"run": 0.0, "stretch": 0.0, "slow": 0.0}
    by_id = {}
    for d in FIXES:
        mine = {str(info.at[f, "flight_id"]): f for f in tasks if info.at[f, "discipline"] == d}
        by_id[d] = mine
    done = 0
    for d, mine in by_id.items():
        ids = np.array(sorted(mine))
        for fid, fl in flight_stream(FIXES[d], ids):
            f = mine[fid]
            fl = fl.sort_values("t", kind="stable")
            t = fl["t"].to_numpy(float) + info.at[f, "utc0"]
            cols = [fl[c].to_numpy(float) for c in ("E", "N", "z")]
            for i, k, j in tasks[f]:
                found[j] = one_member(t, *cols, fl["segment_id"].to_numpy(), ths[i]["t0"], ths[i]["t1"], runs[f], count)
            done += 1
            if done % 2000 == 0:
                print(f"{done}/{len(tasks)} flights", flush=True)
    return found, order, count


def pair_rows(ths, lead, order, found):
    """One row per follower-thermal whose leader and follower have a measure: block, n, index of the
    leader, index of the follower."""
    rows = []
    for i, (r, l) in enumerate(zip(ths, lead)):
        c = climbers(r)
        if l in c:
            rows += [(r["block"], r["n"], order[(i, l)], order[(i, k)]) for k in c[c != l]]
    return np.array(rows, int).reshape(-1, 4)


def logratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((a > 0) & (b > 0), np.log10(b / a), np.nan)


def table(rows, found, own, cache):
    """The columns of the rows of pair_rows: r of the MEASURES in the own window, in the common window,
    and of |d kappa / dt| for each case of SENSITIVITY (own window): (rows, 3 + 3 + len(SENSITIVITY))."""
    jl, jf = rows[:, 2], rows[:, 3]
    out = np.full((len(rows), 6 + len(SENSITIVITY)), np.nan)
    out[:, :3] = logratio(own[jl], own[jf])
    sens = np.array([m["sens"] if m else np.full(len(SENSITIVITY), np.nan) for m in found])
    out[:, 6:] = logratio(sens[jl], sens[jf])
    common = np.full((len(rows), 6), np.nan)
    for n, (a, b) in enumerate(zip(jl, jf)):
        if (a, b) not in cache:
            x, y = found[a], found[b]
            cache[a, b] = np.full(6, np.nan)
            if x and y and "sec" in x and "sec" in y:
                _, p, q = np.intersect1d(x["sec"], y["sec"], assume_unique=True, return_indices=True)
                if len(p) >= MIN_SECONDS:
                    cache[a, b] = np.array([AGG[k](m[k][i]) for m, i in ((x, p), (y, q)) for k in MEASURES])
        common[n] = cache[a, b]
    common = logratio(common[:, :3], common[:, 3:])
    out[:, 3:6] = common
    return out


def r_stats(x, rows, w):
    """Median of r and its bands, the rows, and the share of rows with r < 0 (leader above)."""
    ok = np.isfinite(x)
    return {"median": median(x, rows[:, 0], w), "rows": int(ok.sum()),
            "leader_above": share((x[ok] < 0).astype(float), rows[ok, 0], w) if ok.any() else None}


def summarize(ths, found, order, count, nb, rng):
    np.seterr(divide="ignore", invalid="ignore")
    w = rng.multinomial(nb, np.full(nb, 1 / nb), size=N_BOOT).astype(float)
    own = np.array([[AGG[k](m[k]) for k in MEASURES] if m and "sec" in m else [np.nan] * 3 for m in found])
    cache = {}
    leaders = {d: np.array([r[d] for r in ths]) for d in LEADERS}
    found_c = [climbers(r) for r in ths]
    rows = {d: pair_rows(ths, leaders[d], order, found) for d in LEADERS}
    rows["random"] = np.concatenate([pair_rows(ths, np.array([rng.choice(c) if len(c) else -1 for c in found_c]), order, found)
                                     for _ in range(N_NULL)])
    out = {"leaders": {}, "hist": {}, "params": {
        "sg_window_s": SG_WINDOW, "sg_order": SG_ORDER, "v_min_ms": V_MIN, "min_seconds": MIN_SECONDS, "first_s": FIRST_S,
        "min_part_s": MIN_PART, "sensitivity_window_vmin": list(SENSITIVITY), "n_null": N_NULL, "min_group_s": MIN_GROUP,
        "hist_edges": HIST_EDGES.tolist()}}
    tables = {d: table(r, found, own, cache) for d, r in rows.items()}
    for d, r in rows.items():
        t = tables[d]
        out["leaders"][d] = {}
        for size, sel in SIZES.items():
            m = sel(r[:, 1])
            res = {win: {q: r_stats(t[m, 3 * a + b], r[m], w) for b, q in enumerate(MEASURES)} for a, win in enumerate(("own", "common"))}
            res["pairs"] = int(m.sum())
            if d == "before" and size == "all":
                res["sensitivity"] = {f"w{win}_v{v}": r_stats(t[m, 6 + c], r[m], w) for c, (win, v) in enumerate(SENSITIVITY)}
            out["leaders"][d][size] = res
        x = t[:, 0][np.isfinite(t[:, 0])]
        out["hist"][d] = (np.histogram(x, HIST_EDGES)[0] / max(len(x), 1)).tolist()
    # centring: the first FIRST_S s of a climb against the rest, per member-thermal
    first, later, blk = [], [], []
    for (i, k), j in order.items():
        m = found[j]
        if m and "sec" in m:
            early = m["sec"] - ths[i]["entry_t"][k] < FIRST_S
            if early.sum() >= MIN_PART and (~early).sum() >= MIN_PART:
                first.append(np.median(m["dk"][early]))
                later.append(np.median(m["dk"][~early]))
                blk.append(ths[i]["block"])
    first, later, blk = np.array(first), np.array(later), np.array(blk)
    ratio = logratio(later, first)
    out["centring"] = {"members": len(first), "first_median": float(np.median(first)), "later_median": float(np.median(later)),
                       "log10_first_over_later": r_stats(ratio, np.column_stack([blk]), w) | {"hist": np.histogram(ratio, HIST_EDGES)[0].tolist()}}
    good = [m for m in found if m and "sec" in m]
    out["plausibility"] = {
        "member_thermals": len(found), "with_valid_seconds": len(good),
        "median_radius_m": float(np.median(1 / own[np.isfinite(own[:, 1]), 1])),
        "median_turn_period_s": float(np.median([m["period"] for m in good])),
        "median_abs_dkdt_per_m_s": float(np.nanmedian(own[:, 0])), "median_climb_rate_ms": float(np.nanmedian(own[:, 2])),
        "median_valid_seconds": float(np.median([len(m["sec"]) for m in good])),
        "share_dropped_by_vmin": count["slow"] / count["stretch"], "share_climb_seconds_in_stretches": count["stretch"] / count["run"]}
    out["members_by_size"] = {s: {"thermals": int(sum(sel(r["n"]) for r in ths)),
                                  "member_thermals": int(sum(len(c) for r, c in zip(ths, found_c) if sel(r["n"])))}
                              for s, sel in SIZES.items()}
    return out


def plot(r, path):
    """Left: distribution of r for |d kappa / dt| (leader before and after, at random), own window; middle:
    median r by measure, own (full marks) and common (open) window, by size; right: log10 of |d kappa / dt|
    in the first minute of a climb over after it."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.9))
    mid = (HIST_EDGES[:-1] + HIST_EDGES[1:]) / 2
    for d, ls in zip((*LEADERS, "random"), ("-", "-", ":")):
        a.step(mid, r["hist"][d], where="mid", color=COLORS[d], ls=ls, lw=1.3, label=f"leader {d}" if d != "random" else "leader at random")
    a.axvline(0, color="#999999", lw=0.6)
    a.set(xlabel="$\\log_{10}$ ($|d\\kappa/dt|$ of the follower / of the leader)", ylabel="share of followers per bin")
    a.legend(frameon=False, fontsize=8)
    marks = {"2": "o", "3+": "^"}
    for i, d in enumerate(LEADERS):
        for j, size in enumerate(marks):
            for k, win in enumerate(("own", "common")):
                s = [r["leaders"][d][size][win][q]["median"] for q in MEASURES]
                v = np.array([x["value"] for x in s])
                bd = np.array([x["band"] for x in s]).T
                b.errorbar(np.arange(3) + (2 * i + j - 1.5) * 0.12 + k * 0.04, v, yerr=[v - bd[0], bd[1] - v], fmt=marks[size], ms=4, lw=1,
                           color=COLORS[d], mfc=COLORS[d] if k == 0 else "white", label=f"leader {d}, {size} pilots" if k == 0 else None)
    s = r["leaders"]["random"]["all"]["own"]
    for i, q in enumerate(MEASURES):
        b.hlines(s[q]["median"]["value"], i - 0.4, i + 0.4, color="#666666", ls="--", lw=1.6, zorder=0.5,
                 label="leader at random" if i == 0 else None)
    b.axhline(0, color="#cccccc", lw=0.6, zorder=0)
    b.set_xticks(range(3), [NAMES[q] for q in MEASURES])
    b.set(ylabel="median $\\log_{10}$ (follower / leader)")
    b.margins(y=0.3)
    for filled, text in ((True, "own window (filled)"), (False, "common window (open)")):
        b.plot([], [], "o", ms=4, color="#555555", mfc="#555555" if filled else "white", label=text)
    b.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    h = r["centring"]["log10_first_over_later"]["hist"]
    c.step(mid, np.array(h) / max(sum(h), 1), where="mid", color="#333333", lw=1.3)
    c.axvline(0, color="#999999", lw=0.6)
    c.axvline(r["centring"]["log10_first_over_later"]["median"]["value"], color="#eb6834", lw=1)
    c.set(xlabel=f"$\\log_{{10}}$ ($|d\\kappa/dt|$ in the first {FIRST_S} s of the climb / after)", ylabel="share of member-thermals per bin")
    save(fig, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="read by tamsd")
    parser.add_argument("--sample", type=int, help="a random sample of dates (seed 0)")
    parser.add_argument("--replot", action="store_true", help="redraw the figure from the saved JSON, no computation")
    args = parser.parse_args()
    if args.replot:
        name = f"curvature_sample{args.sample}" if args.sample else "curvature"
        plot(json.loads((OUT / f"{name}.json").read_text()), OUT / f"{name}.pdf")
        raise SystemExit
    params = json.loads((LABELS / "days" / "params.json").read_text())
    flights = pd.read_parquet(LABELS / "flights.parquet")
    groups = pd.read_parquet(LABELS / "groups.parquet")
    labelled = flights.set_index("flight")["labelled"]
    groups = groups[groups["t_end"] - groups["t_start"] >= MIN_GROUP]
    groups = groups[groups["members"].map(lambda m: bool(labelled.loc[list(m)].all()))].reset_index(drop=True)
    if args.sample:
        dates = np.random.default_rng(0).choice(sorted(groups["date"].unique()), args.sample, replace=False)
        groups = groups[groups["date"].isin(dates)].reset_index(drop=True)
    assert groups["date"].is_monotonic_increasing  # observed() goes date by date
    block = pd.read_parquet(LABELS / "episodes.parquet").set_index("episode_id")["block"]
    ids, groups["block"] = np.unique(block.reindex(groups["episode_id"]).to_numpy(), return_inverse=True)
    ths = measure(groups, flights, params["step"], params)
    runs = climb_runs(flights, {f for m in groups["members"] for f in m})
    found, order, count = measure_all(ths, flights, runs)
    result = summarize(ths, found, order, count, len(ids), np.random.default_rng(SEED))
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"curvature_sample{args.sample}" if args.sample else "curvature"
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / f"{name}.pdf")
    print(json.dumps({k: result[k] for k in ("members_by_size", "plausibility", "centring")}), flush=True)
