"""Exits: do followers leave a thermal at the leader's height, below it or above it? The scale of one group.

Population: the groups of breathing.py (n >= 2, every member a 1 Hz flight with phases, lasting at
least MIN_GROUP s), in sizes 2, 3+ and all, on the grid instants of the group, with the passage
times x of rotation.positions and the climbing state of each member (rotation.climbing).

Group thermal. The glides are the runs of f = 0 (no member in a climb run, f the climbing share)
lasting at least MIN_RUN s; a group thermal is what lies between two consecutive glides, so some
member climbs at both of its ends and f = 0 gaps shorter than MIN_RUN s stay inside it. A member
climbs in it if one of its climb runs overlaps it and it has a fix at its exit. Entry: the start of its first run there; exit:
the end of its last (UTC s, the altitude interpolated on the grid); gain: z_exit - z_entry.

Leader. The head (smallest x) that is most often so over the ordered instants of the glide before
the thermal ("before") or of the glide after it ("after"), ties to the head nearest the thermal;
none if no instant is ordered. A follower is any other member that climbs in the thermal; the
leader must climb in it. A thermal gives one row per follower.

Measures, by size, per leader definition (before, after) and for a leader drawn at random among
the climbers of each thermal, N_NULL draws stacked (a check: dz is symmetric about 0):
- dz = z_exit(follower) - z_exit(leader) and dt = t_exit(follower) - t_exit(leader): the medians;
  the shares of dz above TOL, within it ("same") and below -TOL for TOL in TOLS; the share of
  followers that leave after the leader; the median of dz over the leader's gain (thermals where
  it is at least MIN_GAIN); the medians of the difference of the entry altitudes and of the gains,
  which do not depend on a common offset of the altitudes of two loggers. The rows of thermals
  where the two climb in one run each are reported apart (single_run).
- dz by class of dt.
- Link with rotation, over the thermals with two climbers or more and a head after: the share where
  the member that leaves highest, and the one that leaves first, is the head after, against the
  mean of 1/n (members) and of 1/k (the k climbers); the share where the leader (before, after)
  leaves first and highest, against the mean of 1/k.

Bands: 5-95% of N_BOOT draws of the dependency blocks of the note (episodes).

    uv run python exits.py [--run NAME] [--sample N]    # from this folder, after cohesion.py
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from breathing import MIN_GROUP, N_BOOT, boot_median, observe, observed, stat
from hierarchy import MIN_RUN
from rotation import climb_runs, climbing
from tamsd import LABELS, OUT, SEED, save

# Tolerances of "same height", m; the middle one is quoted. The noise of an altitude on the 5 s grid is
# 1.8 m (robust sigma of its second difference / sqrt 6, an upper bound: it includes the real
# curvature; 30 dates), 2.5 m for a difference of two: 20 m is about 8 times that, and 10 and 40 m show
# how much the shares move with it. A common offset of two loggers is not noise: see dgain.
TOLS = (10, 20, 40)
MIN_GAIN = 100  # least height gain of the leader for dz over the gain, m (5% of the gains are below 70 m, 9% below 100 m)
DZ_EDGES = np.arange(-300, 301, 20)  # bins of dz in the figure, m
DT_EDGES = (-600, -300, -120, -60, 0, 60, 120, 300, 600)  # classes of dt, s
N_NULL = 20  # draws of the random leader of each thermal, stacked
LEADERS = ("before", "after")
COLORS = {"before": "#eb6834", "after": "#2a78d6", "random": "#999999"}
NO_RUNS = (np.zeros(0), np.zeros(0))
FIELDS = ("entry_t", "exit_t", "entry_z", "exit_z", "runs")


def thermals(f, step):
    """The group thermals of a series of climbing shares f: (g0, a, e, g1), the glide before as the
    instants [g0, a), the thermal [a, e), the glide after [e, g1)."""
    state = f == 0
    edge = np.flatnonzero(np.diff(state)) + 1
    first, end = np.r_[0, edge], np.r_[edge, len(f)]
    glide = np.flatnonzero(state[first] & ((end - first) * step >= MIN_RUN))
    return [(first[i], end[i], first[j], end[j]) for i, j in zip(glide[:-1], glide[1:])]


def head_over(x):
    """The member most often the head (smallest x) over the rows of x (T, n), the nearest to the
    thermal first, ties to the one that heads the earliest row; -1 where no row is ordered."""
    ok = np.isfinite(x[:, 0])
    if not ok.any():
        return -1
    h = x[ok].argmin(axis=1)
    count = np.bincount(h, minlength=x.shape[1])
    who, row = np.unique(h, return_index=True)
    return int(who[np.lexsort((row, -count[who]))[0]])


def exits(t, z, run, a, e):
    """Entry time, exit time, entry altitude, exit altitude and number of runs of one member in the
    thermal [a, e) of the instants `t`, its altitudes `z` (NaN without a fix) and its climb runs
    `run` (starts, ends): NaN and 0 if it does not climb there."""
    start, end = run
    mine = (end > t[a]) & (start <= t[e - 1])
    if not mine.any():
        return np.nan, np.nan, np.nan, np.nan, 0
    s, n = start[mine][0], end[mine][-1]
    return s, n, np.interp(s, t, z), np.interp(n, t, z), mine.sum()


def one_group(group, real, flight_of, step, runs, tracks, row_of):
    """The group thermals of one group: dicts with the block, the size n, the length in s, the heads
    before and after, and per member the FIELDS (n,)."""
    p, x = observe(group, real, flight_of, step, tracks, row_of)
    f = climbing(group, step, runs).mean(axis=1)
    t = group.t_start + step * np.arange(len(f))
    out = []
    for g0, a, e, g1 in thermals(f, step):
        got = np.array([exits(t, p[k, :, 3], runs.get(m, NO_RUNS), a, e) for k, m in enumerate(group.members)])
        out.append({"block": group.block, "n": len(group.members), "seconds": (e - a) * step,
                    "before": head_over(x[g0:a][::-1]), "after": head_over(x[e:g1])} | dict(zip(FIELDS, got.T)))
    return out


def measure(groups, flights, step, p):
    """The group thermals of all the groups of `groups` (see one_group)."""
    runs = climb_runs(flights, {f for m in groups["members"] for f in m})
    out = []
    for group, real, flight_of, tracks, row_of in observed(groups, flights, p):
        out += one_group(group, real, flight_of, step, runs, tracks, row_of)
    return out


def climbers(r):
    """The members that climb in the thermal r with a fix at their exit."""
    return np.flatnonzero(np.isfinite(r["exit_z"]))


def pairs(ths, lead):
    """One row per follower-thermal, with `lead` the leader of each thermal (-1 none): block, n,
    dz, dt, gain of the leader, difference of the entry altitudes, difference of the gains, and
    whether both climb in one run."""
    rows = []
    for r, l in zip(ths, lead):
        c = climbers(r)
        gain = r["exit_z"] - r["entry_z"]
        if l in c:
            rows += [(r["block"], r["n"], r["exit_z"][k] - r["exit_z"][l], r["exit_t"][k] - r["exit_t"][l], gain[l],
                      r["entry_z"][k] - r["entry_z"][l], gain[k] - gain[l], r["runs"][k] == 1 and r["runs"][l] == 1)
                     for k in c[c != l]]
    return np.array(rows, float).reshape(-1, 8)


def order(ths):
    """One row per thermal with two climbers or more: block, n, number of climbers, the member that
    leaves first, the one that leaves highest, the heads before and after (-1 none), and whether
    each of them climbs."""
    rows = []
    for r in ths:
        c = climbers(r)
        if len(c) >= 2:
            rows.append((r["block"], r["n"], len(c), c[r["exit_t"][c].argmin()], c[r["exit_z"][c].argmax()],
                         r["before"], r["after"], r["before"] in c, r["after"] in c))
    return np.array(rows, int).reshape(-1, 9)


def share(x, owner, w):
    """The mean of x and its draws over the blocks: `owner` is the block of each value, `w` the
    multiplicity of the blocks in each draw (draws, blocks)."""
    a, c = np.bincount(owner, x, w.shape[1]), np.bincount(owner, minlength=w.shape[1])
    return stat(float(a.sum() / c.sum()), (w @ a) / (w @ c))


def median(x, owner, w):
    ok = np.isfinite(x)
    return stat(*boot_median(x[ok], owner[ok], w)) if ok.any() else stat(np.nan, np.full(len(w), np.nan))


def leader_stats(rows, w):
    """The measures of the follower-thermal rows (see pairs)."""
    owner = rows[:, 0].astype(int)
    dz, dt, gain, entry, dgain = rows[:, 2:7].T
    ok = np.isfinite(dz)
    big = ok & (gain >= MIN_GAIN)
    shares = {}
    for tol in TOLS:
        sel = [dz[ok] > tol, abs(dz[ok]) <= tol, dz[ok] < -tol]
        shares[str(tol)] = {k: share(s.astype(float), owner[ok], w) for k, s in zip(("above", "same", "below"), sel)}
    return {"rows": int(ok.sum()), "dz": median(dz, owner, w), "dt": median(dt, owner, w),
            "dz_over_gain": median(np.where(big, dz / gain, np.nan), owner, w) | {"rows": int(big.sum())},
            "dz_entry": median(entry, owner, w), "dgain": median(dgain, owner, w), "shares": shares,
            "follower_later": share((dt[ok] > 0).astype(float), owner[ok], w)}


def by_size(rows, w):
    """leader_stats by size, and of the single-run rows of all sizes."""
    n = rows[:, 1]
    sel = {"2": n == 2, "3+": n >= 3, "all": n >= 2}
    return {s: leader_stats(rows[m], w) for s, m in sel.items()} | {"single_run": leader_stats(rows[rows[:, 7] == 1], w)}


def link_stats(T, w):
    """The link with rotation of the rows of order() `T`, by size."""
    out = {}
    for size, m in {"2": T[:, 1] == 2, "3+": T[:, 1] >= 3, "all": T[:, 1] >= 2}.items():
        t = T[m]
        owner = t[:, 0]
        use = t[:, 6] >= 0
        out[size] = {"thermals": int(len(t)), "with_head_after": int(use.sum())}
        for name, col in (("highest", 4), ("first", 3)):
            out[size][name + "_is_head_after"] = {
                "share": share((t[use, col] == t[use, 6]).astype(float), owner[use], w),
                "expected_members": float((1 / t[use, 1]).mean()), "expected_climbers": float((1 / t[use, 2]).mean())}
        for j, d in enumerate(LEADERS):
            ok = t[:, 7 + j] == 1
            out[size][f"leader_{d}"] = {"climbs": int(ok.sum()), "expected": float((1 / t[ok, 2]).mean()),
                                        "first": share((t[ok, 3] == t[ok, 5 + j]).astype(float), owner[ok], w),
                                        "highest": share((t[ok, 4] == t[ok, 5 + j]).astype(float), owner[ok], w)}
    return out


def summarize(ths, nb, rng):
    """The thermals, the rows by leader definition (and at random), the dz histogram and dz by dt
    class, and the link with rotation, with their bands."""
    np.seterr(divide="ignore", invalid="ignore")
    w = rng.multinomial(nb, np.full(nb, 1 / nb), size=N_BOOT).astype(float)
    n = np.array([r["n"] for r in ths])
    k = np.array([len(climbers(r)) for r in ths])
    out = {"thermals": {s: {"total": int(m.sum()), "climbers_1": int((m & (k == 1)).sum()), "climbers_2plus": int((m & (k >= 2)).sum()),
                            "median_min": float(np.median([r["seconds"] for r, i in zip(ths, m) if i]) / 60)}
                        for s, m in {"2": n == 2, "3+": n >= 3, "all": n >= 2}.items()},
           "leaders": {}, "hist": {}, "dz_by_dt": {}}
    rows = {d: pairs(ths, np.array([r[d] for r in ths])) for d in LEADERS}
    found = [climbers(r) for r in ths]
    rows["random"] = np.concatenate([pairs(ths, np.array([rng.choice(c) if len(c) else -1 for c in found])) for _ in range(N_NULL)])
    for d, r in rows.items():
        out["leaders"][d] = by_size(r, w)
        dz, dt = r[:, 2], r[:, 3]
        ok = np.isfinite(dz)
        out["hist"][d] = (np.histogram(dz[ok], DZ_EDGES)[0] / ok.sum()).tolist()
        c = np.digitize(dt, DT_EDGES)
        out["dz_by_dt"][d] = [median(np.where(c == i, dz, np.nan), r[:, 0].astype(int), w) | {"rows": int(((c == i) & ok).sum())}
                              for i in range(1, len(DT_EDGES))]
    out["link"] = link_stats(order(ths), w)
    out["params"] = {"tols_m": list(TOLS), "min_gain_m": MIN_GAIN, "min_run_s": MIN_RUN, "min_group_s": MIN_GROUP,
                     "dz_edges_m": DZ_EDGES.tolist(), "dt_edges_s": list(DT_EDGES), "n_null": N_NULL}
    return out


def plot(r, path):
    """Left: distribution of dz for the leader before and after the thermal, and at random; middle:
    shares of dz above, within TOL of and below the leader, by size; right: median dz by class of
    dt."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.9))
    mid = (DZ_EDGES[:-1] + DZ_EDGES[1:]) / 2
    tol = TOLS[len(TOLS) // 2]
    a.axvspan(-tol, tol, color="#999999", alpha=0.15, lw=0)
    for d, ls in zip((*LEADERS, "random"), ("-", "-", ":")):
        a.step(mid, r["hist"][d], where="mid", color=COLORS[d], ls=ls, lw=1.3, label=f"leader {d}" if d != "random" else "leader at random")
    a.axvline(0, color="#999999", lw=0.6)
    a.set(xlabel="$z_{exit}$ of the follower minus the leader (m)", ylabel="share of followers per bin")
    a.legend(frameon=False, fontsize=8)
    marks = {"2": "o", "3+": "^"}
    for i, d in enumerate(LEADERS):
        for j, size in enumerate(marks):
            s = r["leaders"][d][size]["shares"][str(tol)]
            v = np.array([s[k]["value"] for k in ("above", "same", "below")])
            bd = np.array([s[k]["band"] for k in ("above", "same", "below")]).T
            b.errorbar(np.arange(3) + (i - 0.5) * 0.3 + (j - 0.5) * 0.1, v, yerr=[v - bd[0], bd[1] - v], fmt=marks[size], ms=4, lw=1.1,
                       color=COLORS[d], label=f"leader {d}, {size} pilots")
    s = r["leaders"]["random"]["all"]["shares"][str(tol)]
    b.plot(range(3), [s[k]["value"] for k in ("above", "same", "below")], "_", color=COLORS["random"], ms=26, mew=1.5, label="leader at random")
    b.set_xticks(range(3), ["above", f"within {tol} m", "below"])
    b.set(ylabel="share of followers", ylim=(0, None))
    b.legend(frameon=False, fontsize=7)
    t = (np.array(DT_EDGES[:-1]) + np.array(DT_EDGES[1:])) / 2 / 60
    for d in LEADERS:
        rows = r["dz_by_dt"][d]
        c.fill_between(t, *np.array([x["band"] for x in rows]).T, color=COLORS[d], alpha=0.15, lw=0)
        c.plot(t, [x["value"] for x in rows], color=COLORS[d], marker="o", ms=3, lw=1.2, label=f"leader {d}")
    c.axhline(0, color="#999999", lw=0.6)
    c.axvline(0, color="#999999", lw=0.6)
    c.set(xlabel="$t_{exit}$ of the follower minus the leader (min, class midpoint)", ylabel="median $\\Delta z_{exit}$ (m)")
    c.legend(frameon=False, fontsize=8)
    save(fig, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="read by tamsd")
    parser.add_argument("--sample", type=int, help="a random sample of dates (seed 0)")
    args = parser.parse_args()
    params = json.loads((LABELS / "days" / "params.json").read_text())
    step = params["step"]
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
    ths = measure(groups, flights, step, params)
    result = summarize(ths, len(ids), np.random.default_rng(SEED))
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"exits_sample{args.sample}" if args.sample else "exits"
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / f"{name}.pdf")
    print(json.dumps({k: result[k] for k in ("thermals", "link")}), flush=True)
